import numpy as np
import torch
from torch import Tensor
import spacy
from typing import List, Union
from os.path import join as pjoin

from evaluators.Guo.utils.word_vectorizer import POS_enumerator
from evaluators.Guo.networks.evaluator_wrapper import build_models
from evaluators.Guo.utils.get_opt import get_opt
from evaluators.Guo.utils.word_vectorizer import WordVectorizer
from evaluators.base_evaluators import BaseEvaluator

from src.tools.smplrifke_to_guofeats import smpl_to_guofeats


class GuoEvaluatorHandler(BaseEvaluator):
    """
    Handler per Guo Evaluator compatibile con l'interfaccia unificata.
    """
    
    def __init__(
        self,
        dataset: str = "humanml3d",
        device: str = "cpu",
        checkpoints_dir: str = "evaluators/Guo/checkpoints",
        glove_path: str = "./evaluators/Guo/glove",
        meta_dir: str = None,
    ):
        # ... (tutto il codice __init__ rimane uguale) ...
        self.dataset = dataset
        self.device = torch.device(device)
        
        # Carica spacy
        self.nlp = spacy.load('en_core_web_sm')
        
        # Determina il path del checkpoint
        dataset_opt_paths = {
            "humanml3d": f'{checkpoints_dir}/t2m/Comp_v6_KLD005/opt.txt',
            "kit_22": f'{checkpoints_dir}/kit_22/Comp_v6_KLD005/opt.txt',
            "babel": f'{checkpoints_dir}/babel/Comp_v6_KLD005/opt.txt',
            "motionx": f'{checkpoints_dir}/motionx/Comp_v6_KLD005/opt.txt',
        }
        
        if dataset not in dataset_opt_paths:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_opt_paths.keys())}")
        
        dataset_opt_path = dataset_opt_paths[dataset]
        opt = get_opt(dataset_opt_path, str(self.device))
        
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        opt.checkpoints_dir = checkpoints_dir
        
        self.opt = opt
        
        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        
        self.text_encoder.to(self.device)
        self.motion_encoder.to(self.device)
        self.movement_encoder.to(self.device)
        
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        
        if meta_dir is None:
            meta_dir = pjoin("evaluators/Guo", opt.meta_dir)
        
        self.motion_mean = torch.from_numpy(np.load(pjoin(meta_dir, 'mean.npy'))).to(self.device)
        self.motion_std = torch.from_numpy(np.load(pjoin(meta_dir, 'std.npy'))).to(self.device)
        
        self.w_vectorizer = WordVectorizer(glove_path, 'our_vab')
    
    def process_text(self, sentence):
        # ... (uguale) ...
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list
    
    @staticmethod
    def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
        # ... (uguale) ...
        dims = batch[0].dim()
        max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
        size = (len(batch),) + tuple(max_size)
        canvas = batch[0].new_zeros(size=size)
        for i, b in enumerate(batch):
            sub_tensor = canvas[i]
            for d in range(dims):
                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
            sub_tensor.add_(b)
        return canvas
    
    def encode(self, texts: List[str], motions: Union[torch.Tensor, List[np.ndarray]], lengths: torch.Tensor = None) -> torch.Tensor:
        # ... (uguale, ma con type hint corretto) ...
        
        if motions[0].shape[-1] == 205:
            motions = smpl_to_guofeats(motions)

        motions = [torch.tensor(motion, dtype=torch.float32, device=self.device) for motion in motions]
        
        cap_lens, word_embs, pos_ohot, tokens_list = [], [], [], []        
        for text in texts:
            word_list, pos_list = self.process_text(text)
            tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
            
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

            pos_ohot.append(pos_one_hots)
            word_embs.append(word_embeddings)
            cap_lens.append(sent_len)
            tokens_list.append(tokens)

        cap_lens = torch.tensor(cap_lens, device=self.device)
        word_embs = self.collate_tensor_with_padding([torch.tensor(w, device=self.device) for w in word_embs])
        pos_ohot = self.collate_tensor_with_padding([torch.tensor(p, device=self.device) for p in pos_ohot])

        # Motion
        m_lens = torch.tensor([motion.shape[0] for motion in motions], device=self.device)
        dtype_ = torch.float32
        motions = [(torch.tensor( (motion - self.motion_mean) / self.motion_std, dtype=dtype_)) for motion in motions]
        motions = self.collate_tensor_with_padding([m for m in motions]) # tensor(#B, #F, 263)
        if motions.shape[1] < self.opt.max_motion_length: # nel paper originale considerano al massimo sequenze lunghe 196
            motions = torch.concatenate([motions,torch.zeros((motions.shape[0], self.opt.max_motion_length - motions.shape[1], motions.shape[2]))], axis=1)
        
        # non_zero_mask = motions.abs().sum(dim=2) != 0
        # m_lens = non_zero_mask.sum(dim=1)  # shape: (bs,)

        # Now we sort all the tensor based on cap lens otherwise the text_encoder will not work
        sorted_cap_lens, sorted_indices = torch.sort(cap_lens, descending=True)
        word_embs = word_embs[sorted_indices]
        pos_ohot = pos_ohot[sorted_indices]
        motions = motions[sorted_indices]
        m_lens = m_lens[sorted_indices]
        tokens_list = [tokens_list[i] for i in sorted_indices.tolist()]
        cap_lens = sorted_cap_lens  # optional, just to reflect the sorted state

        word_embs = word_embs.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()

        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        # Motion encoding
        movements = self.movement_encoder(motions[..., :-4]).detach()
        m_lens = m_lens // self.opt.unit_length
        motion_embedding = self.motion_encoder(movements, m_lens)
        motion_latents = motion_embedding # tensor(#B, 512)
        
        # Text encoding
        text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        text_embedding = text_embedding[align_idx]
        text_latents = text_embedding 

        return text_latents, motion_latents
    
    def to(self, device):
        """Sposta il modello sul device"""
        self.device = torch.device(device)
        self.text_encoder.to(self.device)
        self.motion_encoder.to(self.device)
        self.movement_encoder.to(self.device)
        return self
    
    def eval(self):
        """Mette il modello in eval mode"""
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        return self
    
    def load_state_dict(self, state_dict):
        """Carica lo stato del modello dal dizionario specificato"""
        print(" ⚠️   - Il caricamento del checkpoint di Guo è fato nell'init...rifare a modo")
        return self
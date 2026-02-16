import torch
import numpy as np
from torch.utils.data import Dataset


class GuoDatasetAdapter(Dataset):
    """
    Adapter per usare il dataset esistente con Guo Lightning.
    Converte il formato del dataset in quello atteso da Guo.
    """
    
    def __init__(self, base_dataset, w_vectorizer, mean, std, max_text_len=20):
        self.base_dataset = base_dataset
        self.w_vectorizer = w_vectorizer
        self.mean = mean
        self.std = std
        self.max_text_len = max_text_len
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get data from base dataset
        data = self.base_dataset[idx]
        
        text = data['text']
        if isinstance(text, list):
            text = text[0]
        
        motion = data['x']  # Assuming this is the motion data
        
        # Process text (simplified - you'll need the actual process_text from guo_handler)
        # For now, return dummy data - implement properly based on your needs
        word_list = text.split()
        tokens = ['%s/OTHER' % word for word in word_list]
        
        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        
        # Get word embeddings and POS
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh)
            word_embeddings.append(word_emb)
        
        word_embs = np.array(word_embeddings)
        pos_ohot = np.array(pos_one_hots)
        
        # Normalize motion
        motion_normalized = (motion - self.mean) / self.std
        
        return {
            'word_embs': word_embs,
            'pos_ohot': pos_ohot,
            'cap_len': sent_len,
            'motion': motion_normalized,
            'm_len': len(motion),
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for dataloader"""
        word_embs = torch.stack([torch.tensor(item['word_embs'], dtype=torch.float32) for item in batch])
        pos_ohot = torch.stack([torch.tensor(item['pos_ohot'], dtype=torch.float32) for item in batch])
        cap_lens = torch.tensor([item['cap_len'] for item in batch], dtype=torch.long)
        
        # Pad motions to same length
        max_len = max([item['motion'].shape[0] for item in batch])
        motions = []
        m_lens = []
        
        for item in batch:
            motion = item['motion']
            m_len = motion.shape[0]
            
            if m_len < max_len:
                padding = np.zeros((max_len - m_len, motion.shape[1]))
                motion = np.concatenate([motion, padding], axis=0)
            
            motions.append(motion)
            m_lens.append(m_len)
        
        motions = torch.tensor(np.array(motions), dtype=torch.float32)
        m_lens = torch.tensor(m_lens, dtype=torch.long)
        
        return word_embs, pos_ohot, cap_lens, motions, m_lens
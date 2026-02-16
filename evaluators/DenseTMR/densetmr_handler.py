import torch
from typing import List, Union
import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from evaluators.base_evaluators import BaseEvaluator
from evaluators.utils import transpose

from src.utils import check_tensor_stats, set_preload_false
from src.data.collate import collate_x_dict
from src.config import read_config


class DenseTMREvaluatorHandler(BaseEvaluator):
    """Handler per DenseTMR Evaluator che carica il modello da checkpoint."""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda", ckpt: str = "last"):
        self.device = torch.device(device)
        checkpoint_dir = Path(checkpoint_dir)
        
        cfg = read_config(checkpoint_dir)
        self.model = instantiate(cfg.model)
        self.cfg = cfg
        
        ckpt_path = checkpoint_dir / "logs/checkpoints" / f"{ckpt}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        print(f"✅  - Loading DenseTMR model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.to(self.device)
        self.model.eval()

        self.text_processor = instantiate(cfg.data.text_to_token_emb, preload=False, device=device)
        self.normalizer = instantiate(cfg.data.motion_loader.normalizer)
        self.normalizer.mean = self.normalizer.mean.to(self.device)
        self.normalizer.std = self.normalizer.std.to(self.device)

    def encode(self, texts: List[str], motions: Union[torch.Tensor, List[np.ndarray]], lengths: torch.Tensor = None) -> torch.Tensor:
        text_emb = self.encode_text(texts)
        motion_emb = self.encode_motion(motions, lengths)
        return text_emb, motion_emb
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        # Pre-processa i testi usando il text processor
        x_dict = collate_x_dict(self.text_processor(texts))
        text_emb = self.model.encode(x_dict, sample_mean=True)
        return text_emb
    
    def encode_motion(self, motions: Union[torch.Tensor, List[np.ndarray]], lengths: torch.Tensor = None, check_normalization=True, diffusion_steps=None) -> torch.Tensor:
        """
            motions: movimenti (non normalizzati, normali)
            lengths: lunghezze dei movimenti
            check_normalization: se True, controlla se i movimenti sembrano normalizzati prima di passarli al modello
            diffusion_steps: diffusione da usare per ogni movimento 
        """
        
        if isinstance(motions, list):
            motion_lengths = [len(m) for m in motions]
            max_len = max(motion_lengths)
            
            padded_motions = []
            for motion in motions:
                if isinstance(motion, torch.Tensor):
                    motion = motion.cpu().numpy()
                
                if len(motion) < max_len:
                    padding = np.zeros((max_len - len(motion), motion.shape[1]))
                    padded_motion = np.concatenate([motion, padding], axis=0)
                else:
                    padded_motion = motion
                padded_motions.append(padded_motion)
            
            motions = torch.tensor(np.array(padded_motions), dtype=torch.float32)
            lengths = torch.tensor(motion_lengths, dtype=torch.long)
        
        motions = motions.to(self.device)
        
        motions = self.normalizer(motions)
        if check_normalization: 
            check_tensor_stats(motions[0])

        if lengths is not None:
            lengths = lengths.to(self.device)
        else:
            lengths = torch.tensor([motions.shape[1]] * motions.shape[0], device=self.device)
        
        with torch.no_grad():
            x_dict = collate_x_dict(
                [
                    {
                        "x": motion,
                        "length": lengths[idx],
                        "t": diffusion_steps[idx] if diffusion_steps is not None else None,
                    }
                    for idx, motion in enumerate(motions)
                ]
            )
            
            motion_emb = self.model.encode(x_dict, sample_mean=True).cpu()

        return motion_emb
    
    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.text_processor.device = self.device
        self.normalizer.to(self.device)
        return self
    
    def eval(self):
        self.model.eval()
        return self

    def matching_score(self, text_emb: torch.Tensor, motion_emb: torch.Tensor) -> torch.Tensor:
        """Calcola il punteggio di similarità tra embedding di testo e movimento."""
        x_logits = torch.nn.functional.normalize(text_emb, dim=-1).to(self.device)
        y_logits = torch.nn.functional.normalize(motion_emb, dim=-1).to(self.device)
        sim_matrix = x_logits @ transpose(y_logits)

        res = (sim_matrix + 1) / 2
        res = res.diagonal()
        return res


        
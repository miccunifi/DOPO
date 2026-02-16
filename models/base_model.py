from abc import ABC, abstractmethod
from typing import List, Union
import torch
import numpy as np
import pytorch_lightning as pl


class BaseModel(ABC, pl.LightningModule):
    """
    Base class per tutti gli evaluator.
    Definisce l'interfaccia comune che tutti gli evaluator devono implementare.
    """
    
    def generate(self, texts: List[str], lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Genera motion data a partire da testi.
        
        Args:
            texts: Lista di stringhe
            motions: Tensor (B, T, D) o Lista di array numpy (T, D)
            lengths: Tensor (B,) con le lunghezze effettive (opzionale)
            
        Returns:
            generated_motions: Tensor (B, T, D) con le motion generate

        """
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):    
        pass
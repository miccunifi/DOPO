from abc import ABC, abstractmethod
from typing import List, Union
import torch
import numpy as np


class BaseEvaluator(ABC):
    """
    Base class per tutti gli evaluator.
    Definisce l'interfaccia comune che tutti gli evaluator devono implementare.
    """
    
    @abstractmethod
    def encode(self, texts: List[str], motions: Union[torch.Tensor, List[np.ndarray]], lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Encode una lista di testi in embeddings.
        Encode una lista/batch di motion in embeddings.
        
        Args:
            texts: Lista di stringhe
            motions: Tensor (B, T, D) o Lista di array numpy (T, D)
            lengths: Tensor (B,) con le lunghezze effettive (opzionale)

            
        Returns:
            text_embeddings: Tensor (B, D) con gli embeddings del testo
            motion_embeddings: Tensor (B, D) con gli embeddings delle motion

        """
        pass
        
    def __call__(self, motions, texts, normalize: bool = False):
        """
        Forward pass completo.
        
        Args:
            motions: Motion data
            texts: Lista di stringhe
            normalize: Se normalizzare gli embeddings
            
        Returns:
            text_embeddings, motion_embeddings: Tuple di tensori (B, D)
        """
        pass
    
    def to(self, device):
        """Sposta il modello sul device specificato"""
        return self
    
    def eval(self):
        """Mette il modello in modalità evaluation"""
        return self
    
    def load_state_dict(self, state_dict):
        """Carica lo stato del modello dal dizionario specificato"""
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return self
    
    def matching_score(self, text_emb: torch.Tensor, motion_emb: torch.Tensor) -> torch.Tensor:
        """Calcola il punteggio di similarità tra embedding di testo e movimento."""
        pass

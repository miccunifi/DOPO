import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from typing import Dict, Any

from evaluators.Guo.networks.modules import (
    MovementConvEncoder,
    TextEncoderBiGRUCo,
    MotionEncoderBiGRUCo,
)


class GuoLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper per Guo Text-Motion Matching (fase 4).
    
    Prerequisiti:
    - Fase 1 (motion autoencoder) deve essere già trainato
    - Fase 2 (text2length) opzionale
    - Fase 3 (text2motion) opzionale
    
    Questo modulo train solo i text e motion encoders per il matching.
    """
    
    def __init__(
        self,
        dim_pose: int = 263,
        dim_movement_enc_hidden: int = 512,
        dim_movement_latent: int = 512,
        dim_word: int = 300,
        dim_pos_ohot: int = 15,  # len(POS_enumerator)
        dim_text_hidden: int = 512,
        dim_motion_hidden: int = 1024,
        dim_coemb_hidden: int = 512,
        lr: float = 1e-4,
        gamma: float = 0.1,
        milestones: list = None,
        negative_margin: float = 1.0,
        # Path al movimento encoder pretrained (da fase 1)
        movement_enc_checkpoint: str = None,
        # Se True, freeze il movement encoder
        freeze_movement_enc: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if milestones is None:
            milestones = [50, 100, 150, 200]
        
        # Movement encoder (pretrained da fase 1)
        self.movement_encoder = MovementConvEncoder(
            dim_pose - 4,
            dim_movement_enc_hidden,
            dim_movement_latent
        )
        
        # Load pretrained weights se fornito
        if movement_enc_checkpoint is not None:
            checkpoint = torch.load(movement_enc_checkpoint, map_location='cpu')
            if 'movement_enc' in checkpoint:
                self.movement_encoder.load_state_dict(checkpoint['movement_enc'])
            elif 'state_dict' in checkpoint:
                self.movement_encoder.load_state_dict(checkpoint['state_dict'])
            else:
                self.movement_encoder.load_state_dict(checkpoint)
            print(f"Loaded movement encoder from {movement_enc_checkpoint}")
        
        if freeze_movement_enc:
            for param in self.movement_encoder.parameters():
                param.requires_grad = False
            self.movement_encoder.eval()
        
        # Text encoder (da trainare)
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=dim_word,
            pos_size=dim_pos_ohot,
            hidden_size=dim_text_hidden,
            output_size=dim_coemb_hidden,
            device=self.device,
        )
        
        # Motion encoder (da trainare)
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=dim_movement_latent,
            hidden_size=dim_motion_hidden,
            output_size=dim_coemb_hidden,
            device=self.device,
        )
    
    def forward(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        """Forward pass per ottenere text e motion embeddings"""
        # Text encoding
        text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        
        # Motion encoding
        with torch.set_grad_enabled(not self.hparams.freeze_movement_enc):
            movements = self.movement_encoder(motions[..., :-4])
        
        m_lens = m_lens // 4  # unit_length di solito è 4
        motion_embedding = self.motion_encoder(movements, m_lens)
        
        return text_embedding, motion_embedding
    
    def compute_loss(self, text_emb, motion_emb, batch_size):
        """
        Contrastive loss per text-motion matching.
        Ogni testo deve matchare con la sua motion corrispondente.
        """
        # Normalize embeddings
        text_emb_norm = F.normalize(text_emb, dim=1)
        motion_emb_norm = F.normalize(motion_emb, dim=1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(text_emb_norm, motion_emb_norm.t())
        
        # Positive pairs (diagonal)
        pos_sim = torch.diag(sim_matrix)
        
        # Negative loss: push away non-matching pairs
        # Text-to-motion negatives
        t2m_neg_sim = sim_matrix.clone()
        t2m_neg_sim.fill_diagonal_(-float('inf'))
        t2m_neg_loss = F.relu(self.hparams.negative_margin - pos_sim.unsqueeze(1) + t2m_neg_sim).sum(1)
        
        # Motion-to-text negatives
        m2t_neg_sim = sim_matrix.t().clone()
        m2t_neg_sim.fill_diagonal_(-float('inf'))
        m2t_neg_loss = F.relu(self.hparams.negative_margin - pos_sim.unsqueeze(1) + m2t_neg_sim).sum(1)
        
        # Total loss
        loss = (t2m_neg_loss + m2t_neg_loss).mean()
        
        # Accuracy (percentage of correct top-1 matches)
        t2m_acc = (sim_matrix.argmax(dim=1) == torch.arange(batch_size, device=self.device)).float().mean()
        m2t_acc = (sim_matrix.argmax(dim=0) == torch.arange(batch_size, device=self.device)).float().mean()
        
        return loss, t2m_acc, m2t_acc
    
    def training_step(self, batch, batch_idx):
        word_embs, pos_ohot, cap_lens, motions, m_lens = batch
        
        batch_size = word_embs.size(0)
        
        # Forward
        text_emb, motion_emb = self(word_embs, pos_ohot, cap_lens, motions, m_lens)
        
        # Loss
        loss, t2m_acc, m2t_acc = self.compute_loss(text_emb, motion_emb, batch_size)
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/t2m_acc', t2m_acc * 100, prog_bar=True)
        self.log('train/m2t_acc', m2t_acc * 100, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        word_embs, pos_ohot, cap_lens, motions, m_lens = batch
        
        batch_size = word_embs.size(0)
        
        # Forward
        text_emb, motion_emb = self(word_embs, pos_ohot, cap_lens, motions, m_lens)
        
        # Loss
        loss, t2m_acc, m2t_acc = self.compute_loss(text_emb, motion_emb, batch_size)
        
        # Log
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/t2m_acc', t2m_acc * 100, prog_bar=True)
        self.log('val/m2t_acc', m2t_acc * 100, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.milestones[0] if self.hparams.milestones else 50,
            gamma=self.hparams.gamma
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
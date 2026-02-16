# src/model/stable_mofusion_lightning.py
from models.base_model import BaseModel
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR
from diffusers import DDPMScheduler
from copy import deepcopy
from models.StableMoFusion.models.unet import T2MUnet
from typing import List, Union


class EMAModel:
    """Exponential Moving Average per i parametri del modello"""
    def __init__(self, model, decay=0.9999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)
    
    def to(self, device):
        """Sposta l'EMA model sul device specificato"""
        self.module = self.module.to(device)
        return self


class StableMoFusionLightning(BaseModel):
    def __init__(
        self,
        input_feats: int,
        text_latent_dim: int = 256,
        base_dim: int = 512,
        dim_mults: list = [2, 2, 2, 2],
        time_dim: int = 512,
        adagn: bool = True,
        no_eff: bool = False,
        cond_mask_prob: float = 0.1,
        diffusion_steps: int = 1000,
        beta_schedule: str = "linear",
        prediction_type: str = "sample",
        lr: float = 2e-4,
        weight_decay: float = 1e-2,
        decay_rate: float = 0.9,
        update_lr_steps: int = 5000,
        clip_grad_norm: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_every: int = 32,
        motion_normalizer = None,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        
        self.model = T2MUnet(
            input_feats=input_feats,
            text_latent_dim=text_latent_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_dim=time_dim,
            adagn=adagn,
            zero=True,
            no_eff=no_eff,
            cond_mask_prob=cond_mask_prob
        )
        
        self.ema_model = None
        self._use_ema = use_ema
        self._ema_decay = ema_decay
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
            variance_type="fixed_small",
            prediction_type=prediction_type,
            clip_sample=False
        )
        
        self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self._step_count = 0

        self.motion_normalizer = motion_normalizer
        self.num_inference_steps = num_inference_steps
    
    def on_fit_start(self):
        """Crea l'EMA model dopo che il modello è stato spostato su GPU"""
        if self._use_ema and self.ema_model is None:
            self.ema_model = EMAModel(self.model, decay=self._ema_decay)
            self.ema_model.to(self.device)
            print(f"EMA model created and moved to {self.device}")
        
    def forward(self, x, timesteps, text, use_ema=False):
        if use_ema and self.ema_model is not None:
            return self.ema_model.module(x, timesteps, text=text)
        return self.model(x, timesteps, text=text)
    
    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T, device=self.device)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask
    
    def _compute_loss(self, batch, use_ema=False):
        text = batch["text"]
        motion_x_dict = batch["motion_x_dict"]
        
        motions = motion_x_dict["x"]
        m_lens = motion_x_dict["length"]
        
        if isinstance(text[0], list):
            text = [t[0] for t in text]
        
        x_start = motions.detach().float()
        B, T = x_start.shape[:2]
        
        cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).to(self.device)
        src_mask = self.generate_src_mask(T, cur_len)
        
        real_noise = torch.randn_like(x_start)
        t = torch.randint(0, self.hparams.diffusion_steps, (B,), device=self.device)
        x_t = self.noise_scheduler.add_noise(x_start, real_noise, t)
        
        if use_ema and self.ema_model is not None:
            prediction = self.ema_model.module(x_t, t, text=text)
        else:
            prediction = self.model(x_t, t, text=text)
        
        if self.hparams.prediction_type == 'sample':
            target = x_start
        elif self.hparams.prediction_type == 'epsilon':
            target = real_noise
        elif self.hparams.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(x_start, real_noise, t)
        
        loss = self.mse_criterion(prediction, target).mean(dim=-1)
        loss = (loss * src_mask).sum(-1) / src_mask.sum(-1)
        loss = loss.mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        self._step_count += 1
        loss = self._compute_loss(batch)
        
        if self.ema_model is not None and self._step_count % self.hparams.ema_update_every == 0:
            self.ema_model.update(self.model)
        
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, use_ema=False)
        self.log('val/loss', loss, prog_bar=True)
        
        if self.ema_model is not None:
            loss_ema = self._compute_loss(batch, use_ema=True)
            self.log('val/loss_ema', loss_ema, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.decay_rate > 0:
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.decay_rate)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": self.hparams.update_lr_steps,
                },
            }
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        if self.hparams.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip_grad_norm)
    
    def on_save_checkpoint(self, checkpoint):
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        checkpoint['step_count'] = self._step_count
    
    def on_load_checkpoint(self, checkpoint):
        if 'ema_state_dict' in checkpoint and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.ema_model.to(self.device)
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']

    def to(self, device):
        super().to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        if self.motion_normalizer is not None:
            self.motion_normalizer.to(device)
        return self

    def generate(self, texts: List[str], lengths: torch.Tensor = None):
        B = len(texts)
        T = max(lengths)
        shape = (B, T, self.model.input_feats)

        # random sampling noise x_T
        sample = torch.randn(shape,device=self.device)

        # set timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = [ torch.tensor([t] * B, device=self.device).long() for t in self.noise_scheduler.timesteps]
        
        # cache text_embedded 
        enc_text = self.model.encode_text(texts, self.device)
            
        for i, t in enumerate(timesteps):
            # 1. model predict 
            with torch.no_grad():
                if  getattr(self.model, 'cond_mask_prob', 0) > 0 :
                    predict = self.model.forward_with_cfg(sample,t,enc_text=enc_text)
                else:
                    
                    predict = self.model(sample, t, enc_text=enc_text)

            # 2. compute less noisy motion and set x_t -> x_t-1
            sample = self.noise_scheduler.step(predict, t[0], sample).prev_sample

        sample = self.motion_normalizer.inverse(sample)

        return sample
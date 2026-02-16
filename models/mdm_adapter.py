import torch
from typing import List, Union

from models.base_model import BaseModel
from models.MDM.src.model.gaussian import GaussianDiffusion
from src.data.text import TextToEmb


class MDM_adapter(GaussianDiffusion, BaseModel):
    name = "gaussian"

    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
        lr: float = 2e-4,
    ):
        super().__init__(
            denoiser=denoiser,
            schedule=schedule,
            timesteps=timesteps,
            motion_normalizer=motion_normalizer,
            text_normalizer=text_normalizer,
            prediction=prediction,
            lr=lr,
        )

        self.save_hyperparameters()

        modelpath = "ViT-B/32"
        mean_pooling = False
        text_model = TextToEmb(
            modelpath=modelpath, mean_pooling=mean_pooling, device=self.device
        )
        for param in text_model.parameters():
            param.requires_grad = False
        text_model.eval()
        
        # Memorizza in un dizionario - non sarà tracciato
        self._frozen_models = {"text_model": text_model}
    
    @property
    def text_model(self):
        return self._frozen_models["text_model"]
    
    def to(self, *args, **kwargs):
        # Chiama il to() della classe padre per spostare parametri e buffer
        super().to(*args, **kwargs)
        
        # Sposta anche il text_model
        # self.text_normalizer = self.text_normalizer.to(self.device)
        # if hasattr(self, '_frozen_models') and 'text_model' in self._frozen_models:
        #     self._frozen_models['text_model'] = self._frozen_models['text_model'].to(*args, **kwargs)

        self.motion_normalizer = self.motion_normalizer.to(self.device)
        
        return self
    
    def generate(self, texts: List[str], lengths: torch.Tensor = None) -> torch.Tensor:
        infos = {
            "all_lengths": lengths,
            "all_texts": texts,
        }
        with torch.no_grad():
            tx_emb = self.text_model(infos["all_texts"]) # TODO questa fase è in CPU
            tx_emb_uncond = self.text_model(["" for _ in infos["all_texts"]])

            if isinstance(tx_emb, torch.Tensor):
                tx_emb = {
                    "x": tx_emb[:, None].to(self.device),
                    "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(self.device),
                }
                tx_emb_uncond = {
                    "x": tx_emb_uncond[:, None].to(self.device),
                    "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(self.device),
                }

            xstarts = self(tx_emb, tx_emb_uncond, infos)

        return xstarts
    
    def training_step(self, batch, batch_idx):
        batch["x"] = batch["motion_x_dict"]["x"]
        batch["length"] = batch["motion_x_dict"]["length"]
        batch["mask"] = batch["motion_x_dict"]["mask"]

        loss = super().training_step(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch["x"] = batch["motion_x_dict"]["x"]
        batch["length"] = batch["motion_x_dict"]["length"]
        batch["mask"] = batch["motion_x_dict"]["mask"]        

        loss = super().validation_step(batch, batch_idx)
        return loss
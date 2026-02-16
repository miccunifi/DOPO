# src/model/stable_mofusion_lightning.py
from pathlib import Path
import pytorch_lightning as pl
import torch
from copy import copy, deepcopy
from typing import List, Union
import logging
from peft import  LoraConfig,  get_peft_model

from torch.optim.lr_scheduler import ExponentialLR
from diffusers import DDPMScheduler

from models.stablemofusion_adapter import StableMoFusionLightning
from models.StableMoFusion.models.unet import T2MUnet

logger = logging.getLogger(__name__)


class StableMoFusionLightningSPO(StableMoFusionLightning):
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
        checkpoint_dir = None,
        ckpt = "last",
        lora = True,
        lora_rank = 4,
        lora_alpha = 16,
        lora_dropout = 0.1,
        lora_bias = "none",
    ):
        super().__init__(
            input_feats=input_feats,
            text_latent_dim=text_latent_dim,
            base_dim=base_dim,
            dim_mults=dim_mults,
            time_dim=time_dim,
            adagn=adagn,
            no_eff=no_eff,
            cond_mask_prob=cond_mask_prob,
            diffusion_steps=diffusion_steps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            lr=lr,
            weight_decay=weight_decay,
            decay_rate=decay_rate,
            update_lr_steps=update_lr_steps,
            clip_grad_norm=clip_grad_norm,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_update_every=ema_update_every,
            motion_normalizer = motion_normalizer,
            num_inference_steps=num_inference_steps,
        )

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            ckpt_path = checkpoint_dir / "logs/checkpoints" / f"{ckpt}.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded model weights from checkpoint: {ckpt_path}")
        
        self._create_reference_model()

    
        self.lora = lora
        if self.lora:
            self.apply_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                bias=lora_bias,
            )
        self.model.eval()
        
        self.save_hyperparameters()
        # Manual optimization
        self.automatic_optimization = False



    def apply_lora(
        self,
        rank: int = 4,
        alpha: int = 16,
        dropout: float = 0.1,       
        bias: str = "none",
    ):  
        for p in self.model.parameters():
            p.requires_grad = False

         # Apply LoRA only to UNet cross-attention
        lora_config = LoraConfig(
            inference_mode=False,
            r=rank,
            lora_alpha=alpha, 
            lora_dropout=dropout,
            bias=bias,
            target_modules=["query", "key", "value"],
        )
        self.model.unet = get_peft_model(self.model.unet, lora_config)
        
        # Manually unfreeze text components for full fine-tuning
        self.model.embed_text.requires_grad = True
        self.model.textTransEncoder.requires_grad = True
        self.model.text_ln.requires_grad = True

    
    def _create_reference_model(self):
        """Create a frozen copy of the current model as reference"""
        # Deep copy of the entire model
        pretrained_diffusion_rl = copy.deepcopy(self.model)
        pretrained_diffusion_rl.eval()
        for param in pretrained_diffusion_rl.parameters():
            param.requires_grad = False
        
        print(f"Reference model created and frozen on {self.device}")

    @torch.no_grad()
    def _generate_spo_dataset(self, train_iterator, global_iteration, c, device, infos, text_model, smplh, train_embedding_tmr, mean_norm, std_norm):
        self.model.unet.train()
        self.model.textTransEncoder.train()
        self.model.embed_text.train()
        self.model.text_ln.train()

        dataset = {
            "xt_best": [],
            "xt_worst": [],
            "xt": [],
            "t": [],
            "mask": [],
            "length": [],
            "enc_text": [],
            "caption": [],
            "prev_model_output": [],
        }

        with torch.no_grad():
            animations, results_by_timestep = self.model.generate_batch_rl_spo(
                texts, 
                torch.LongTensor([int(x) for x in motion_lens]),
                reward_model=reward_model_smpl,
                k=k,
                divert_start_step=divert_start_step,
                log=True,
                tmr_text=tmr_text,
                T=T,
                mean_norm=mean_norm,
                std_norm=std_norm,
            )

        animations = masked(animations, mask)
        denormed_animations = animations * (std_norm + 1e-12) + mean_norm 
        
        batch_size = animations.shape[0]
        t = torch.tensor(0).repeat(batch_size)
        reward = reward_model_smpl(denormed_animations, infos, tmr_text, t).mean()
        wandb.log({"reward": reward})
        
        # Filter timesteps that have SPO data (xt_best/xt_worst)
        spo_timesteps = sorted(
            [t for t in results_by_timestep.keys() if "xt_best" in results_by_timestep[t]],
            reverse=True
        )
        diff_step = len(spo_timesteps)
        
        if diff_step == 0:
            print("Warning: No SPO timesteps found, skipping batch")
            continue

        batch_size = animations.shape[0]
        seq_len = results_by_timestep[spo_timesteps[0]]["xt_best"].shape[1]
        nfeats = 205

        all_xt_best = []
        all_xt_worst = []
        all_xt_old = []
        all_t = []
        all_mask = []
        all_lengths = []
        all_enc_text = []
        all_caption = []
        all_prev_model_output = []
        all_step_index = []
        all_lower_order_nums = []

        for t in spo_timesteps:
            experiment = results_by_timestep[t]
            experiment = {k_: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k_, v in experiment.items()}

            all_xt_best.append(experiment["xt_best"])
            all_xt_worst.append(experiment["xt_worst"])
            all_xt_old.append(experiment["xt_old"])
            all_t.append(experiment["t"] if isinstance(experiment["t"], torch.Tensor) else torch.full((batch_size,), t).cpu())
            all_mask.append(experiment["mask"])
            all_lengths.append(experiment["length"])
            all_enc_text.append(experiment["enc_text"])
            all_caption.append(experiment["caption"])
            all_prev_model_output.append(experiment["prev_model_output"])
           
        # Concatenate
        dataset["xt_best"].append( torch.cat(all_xt_best, dim=0).view(diff_step, batch_size, seq_len, nfeats).permute(1, 0, 2, 3))
        dataset["xt_worst"].append(torch.cat(all_xt_worst, dim=0).view(diff_step, batch_size, seq_len, nfeats).permute(1, 0, 2, 3))
        dataset["xt"].append(torch.cat(all_xt_old, dim=0).view(diff_step, batch_size, seq_len, nfeats).permute(1, 0, 2, 3))
        dataset["t"].append(torch.cat(all_t, dim=0).view(diff_step, batch_size).T)
        dataset["mask"].append(torch.cat(all_mask, dim=0).view(diff_step, batch_size, seq_len).permute(1, 0, 2))
        dataset["length"].append(torch.cat(all_lengths, dim=0).view(diff_step, batch_size).T)
        dataset["caption"].extend(all_caption[0])
        
        # Handle enc_text
        enc_text_sample = all_enc_text[0]
        if isinstance(enc_text_sample, dict):
            dataset["enc_text"] = all_enc_text[0]
        else:
            enc_text_seq_len = enc_text_sample.shape[1] if enc_text_sample.ndim > 2 else 77
            enc_text_feat = enc_text_sample.shape[-1] if enc_text_sample.ndim > 1 else 256
            dataset["enc_text"].append(torch.cat(all_enc_text, dim=0).view(diff_step, batch_size, enc_text_seq_len, enc_text_feat).permute(1, 0, 2, 3))
        
        # # New fields
        dataset["prev_model_output"].append(torch.cat(all_prev_model_output, dim=0).view(diff_step, batch_size, seq_len, nfeats).permute(1, 0, 2, 3))

        for key in dataset:
            if key == "caption": continue
            dataset[key] = torch.cat(dataset[key], dim=0)

        mean_norm = mean_norm.to("cpu")
        std_norm = std_norm.to("cpu")

        torch.cuda.empty_cache()
        return dataset


    def training_step(self, batch, batch_idx):
        """
        Training step with SPO:
        1. Generate SPO dataset
        2. Train on it for train_epochs
        """
        train_datasets_rl = self._generate_spo_dataset(batch, batch_idx)
        
        # Train with SPO
        train(
            diffusion_rl, 
            optimizer, 
            train_datasets_rl, 
            global_iteration, 
            c, 
            infos, 
            device, 
            old_model=pretrained_diffusion_rl,
            scheduler=None
        )
                    
                    
        # Validation
        if (global_iteration + 1) % c.val_iter == 0:
            avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo, avg_guo_kit, avg_guo_babel, avg_guo_motionx = test(
                diffusion_rl, 
                val_dataloader, 
                device, 
                infos,
                text_model, 
                smplh, 
                joints_renderer,
                None,  # smpl_renderer
                c, 
                None,  # val_embedding_tmr
                path=folder_results + "/VAL/" + str(global_iteration + 1) + "/",
                mean_norm=mean_norm_stablemofusion, 
                std_norm=std_norm_stablemofusion
            )

from pathlib import Path
from peft import LoraConfig
import torch
import copy
from typing import Dict, Any
import pytorch_lightning as pl
from tqdm import tqdm
import logging
from peft import LoraModel, LoraConfig

from src.data.collate import length_to_mask
from models.mdm_adapter import MDM_adapter
from models.MDM.src.model.gaussian import masked

logger = logging.getLogger(__name__)


def expand_for_k(data, k):
    """Recursively expand data structures for k samples."""
    if isinstance(data, torch.Tensor):
        # Add k dimension at the beginning
        return data.unsqueeze(0).expand(k, *[-1] * data.ndim)
    elif isinstance(data, dict):
        return {key: expand_for_k(val, k) for key, val in data.items()}
    elif isinstance(data, list):
        # Replicate the list k times
        return [data for _ in range(k)]
    else:
        return data
    

def flatten_k_bs(data, k, bs):
    """Recursively flatten k and bs dimensions to (k*bs)."""
    if isinstance(data, torch.Tensor):
        if data.ndim > 1:
            # Reshape (k, bs, ...) -> (k*bs, ...)
            return data.reshape(k * bs, *data.shape[2:])
        else:
            return data
    elif isinstance(data, dict):
        return {key: flatten_k_bs(val, k, bs) for key, val in data.items()}
    elif isinstance(data, list):
        # Flatten list of lists
        return [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])]
    else:
        return data


# Create combined y dictionary
def combine_data(data1, data2, use_first_for_second=False):
    """Recursively combine two data structures by concatenating tensors."""
    if isinstance(data1, torch.Tensor):
        data2_to_use = data1 if use_first_for_second else data2
        return torch.cat([data1, data2_to_use], dim=0)
    elif isinstance(data1, dict):
        result = {}
        for key in data1.keys():
            result[key] = combine_data(data1[key], data2[key], use_first_for_second)
        return result
    else:
        # For other types, just create a list
        data2_to_use = data1 if use_first_for_second else data2
        return [data1, data2_to_use]
    

# helper to gather along k (dim=0)
def gather_k(x, idx, bs):  # idx: [BS]
    gather_idx = idx.view(1, bs, 1, 1).expand(1, bs, x.shape[-2], x.shape[-1])  # [1,BS,T,F]
    return x.gather(dim=0, index=gather_idx)[0]  # [BS,T,F]


def nan_masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = float('nan')
    return tensor


class MDM_SPO_adapter(MDM_adapter):
    """SPO adapter con reference model interno e manual optimization"""
    
    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        dense_reward_model,
        prediction: str = "x",
        lr: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        # SPO parameters
        beta: float = 1.0,
        advantage_clip_epsilon: float = 0.2,
        k_samples: int = 4,
        train_epochs: int = 20,
        train_iterations: int = 1000,
        train_batch_size: int = 32,
        grad_clip: float = 1.0,
        update_reference_every_n_steps: int = None,  # Optional: aggiorna reference model ogni N steps
        guidance_weight: float = 5.0,
        lora=True,
        lora_rank: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_bias: str = "none",
        checkpoint_dir: str = None,
        ckpt: str = None,
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
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            ckpt_path = checkpoint_dir / "logs/checkpoints" / f"{ckpt}.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            self.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded model weights from checkpoint: {ckpt_path}")

        # SPO-specific
        self.beta = beta
        self.advantage_clip_epsilon = advantage_clip_epsilon
        self.k_samples = k_samples
        self.train_epochs = train_epochs
        self.train_iterations = train_iterations
        self.train_batch_size = train_batch_size
        self.grad_clip = grad_clip
        self.update_reference_every_n_steps = update_reference_every_n_steps
        
        # Reference model will be created in setup()
        self.guidance_weight = guidance_weight 
        self.dense_reward_model = dense_reward_model
        self.divert_start_step = 5  # todo kill
        self.num_inference_steps = 50

        self._create_reference_model()

        self.lora = lora
        if self.lora:
            self.apply_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                bias=lora_bias,
            )

        # Salva hyperparameters DOPO super().__init__
        # Questo sovrascrive quelli di base con quelli SPO
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
        for name, param in self.denoiser.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=[
                "to_skel_layer",
                "skel_embedding",
                "tx_embedding.0",
                "tx_embedding.2",
                "seqTransEncoder.layers.0.self_attn.out_proj",
                "seqTransEncoder.layers.1.self_attn.out_proj",
                "seqTransEncoder.layers.2.self_attn.out_proj",
                "seqTransEncoder.layers.3.self_attn.out_proj",
                "seqTransEncoder.layers.4.self_attn.out_proj",
                "seqTransEncoder.layers.5.self_attn.out_proj",
                "seqTransEncoder.layers.6.self_attn.out_proj",
                "seqTransEncoder.layers.7.self_attn.out_proj",

            ],
            lora_dropout=dropout,
            bias=bias,
        )

        self.denoiser = LoraModel(self.denoiser, lora_config, "sus")

    def _create_reference_model(self):
        """Create a frozen copy of the current model as reference"""
        # Deep copy of the entire model
        self._reference_model = copy.deepcopy(self)
        
        # Freeze all parameters
        self._reference_model.eval()
        self._reference_model.requires_grad_(False)
        for param in self._reference_model.parameters():
            param.requires_grad = False
        
        # Move to same device
        self._reference_model = self._reference_model.to(self.device)
        
        print(f"Reference model created and frozen on {self.device}")

    def update_reference_model(self):
        """Update reference model with current model weights"""
        print("Updating reference model...")
        self._reference_model.load_state_dict(self.state_dict())
        self._reference_model.eval()

    def on_save_checkpoint(self, checkpoint):
        """Don't save reference model in checkpoint"""
        super().on_save_checkpoint(checkpoint)
        # Reference model will be recreated in setup()

    def on_load_checkpoint(self, checkpoint):
        """Reference model will be recreated in setup()"""
        super().on_load_checkpoint(checkpoint)

    def training_step(self, batch, batch_idx):
        """
        Training step with SPO:
        1. Generate SPO dataset
        2. Train on it for train_epochs
        """
        self.eval()
        # logger.info(f"Starting training step {batch_idx} with batch size {len(batch['keyid'])}")
        opt = self.optimizers()

        # Update reference model if needed
        if (self.update_reference_every_n_steps is not None and
            self.global_step > 0 and
            self.global_step % self.update_reference_every_n_steps == 0):
            self.update_reference_model()

        # === STEP 1: Generate SPO dataset ===
        # logger.info("Generating SPO dataset...")
        with torch.no_grad():
            self.dense_reward_model.to(self.device)
            spo_dataset = self._generate_spo_dataset(batch, batch_idx) # 3% -> 74%
            torch.cuda.empty_cache() 
            self.dense_reward_model.to("cpu")

        # === STEP 2: Train on SPO dataset ===
        # logger.info("Training on SPO dataset...")
        total_loss = 0.0
        total_ratio_best = 0.0
        total_ratio_worst = 0.0
        num_minibatches = 0

        tot_loss = 0.0
        iteration_ratio_best = 0.0
        iteration_ratio_worst = 0.0
        iteration_minibatches = 0

        spo_dataset = self._shuffle_spo_dataset(spo_dataset)
        num_total_samples = len(spo_dataset['t'])

        # Progress bar for minibatches
        pbar_batches = tqdm(
            range(0, num_total_samples, self.train_batch_size),
            desc="Training SPO",
            unit="batch"
        )

        for minibatch_idx in pbar_batches:
            minibatch = self._get_minibatch(spo_dataset, minibatch_idx, self.train_batch_size)
            loss, ratio_best, ratio_worst = self._compute_spo_loss(minibatch)
            self.manual_backward(loss)

            # Update metrics
            current_loss = loss.item()
            tot_loss += current_loss
            iteration_ratio_best += ratio_best.item()
            iteration_ratio_worst += ratio_worst.item()
            iteration_minibatches += 1

            # Update progress bar with current loss
            pbar_batches.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg_loss': f"{tot_loss/iteration_minibatches:.4f}"
            })

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            opt.step()
            opt.zero_grad()

        total_loss += tot_loss
        total_ratio_best += iteration_ratio_best
        total_ratio_worst += iteration_ratio_worst
        num_minibatches += iteration_minibatches

        # Log overall metrics
        if num_minibatches > 0:
            avg_loss = total_loss / num_minibatches
            avg_ratio_best = total_ratio_best / num_minibatches
            avg_ratio_worst = total_ratio_worst / num_minibatches

            logger.info(f"train_loss, {avg_loss}, train_ratio_best, {avg_ratio_best}, train_ratio_worst, {avg_ratio_worst}, on_step=True, on_epoch=True, batch_size={len(batch['keyid'])}")

            # NUOVO: Log su wandb invece di logger.info
            self.log_dict({
                'Training.loss': avg_loss,
                'Training.ratio_best': avg_ratio_best,
                'Training.ratio_worst': avg_ratio_worst,
            }, on_step=True, on_epoch=True, prog_bar=True)

        return None

    def _generate_spo_dataset(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Generate SPO dataset from a single batch"""
        # self.train() ?         
        # Generate with SPO diffusion
        # Note: You need to implement diffusionSPO and reward model integration
        dataset = {
            "xt_best": [],
            "xt_worst": [],
            "xt": [],
            "t": [],
            "mask": [],
            "length": [],
            "tx_x": [],
            "tx_mask": [],
            "tx_length": [],
            "tx_uncond_x": [],
            "tx_uncond_mask": [],
            "tx_uncond_length": [],
        }
        tmr_text_embeddings = self.dense_reward_model.encode_text(batch["text"])

        tx_emb = batch["tx"]
        tx_emb_uncond = batch["tx_uncond"]
        
        infos = {
            "all_lengths": batch["motion_x_dict"]["length"],
            "guidance_weight": self.guidance_weight,
        }

        sequences, results_by_timestep = self.diffusionSPO(tx_emb=tx_emb, 
                                                           tx_emb_uncond=tx_emb_uncond, 
                                                           infos=infos,
                                                           reward_model=self.dense_reward_model, 
                                                           tmr_text_emb=tmr_text_embeddings,
                                                           k=self.k_samples)
        batch_size = sequences.shape[0]
        t = torch.tensor(0).repeat(batch_size)
        sequences_norm = self.motion_normalizer.inverse(sequences)
        tmr_motion_emb = self.dense_reward_model.encode_motion(sequences_norm, diffusion_steps=t)
        metric = self.dense_reward_model.matching_score(tmr_text_embeddings, tmr_motion_emb).mean().item()

        logger.info(f"Mean metric reward for generated motions: {metric}")
        self.log_dict({'Training.tmr_generation': metric}, on_step=True, on_epoch=True, prog_bar=True)

        timesteps = sorted(results_by_timestep.keys(), reverse=True)
        diff_step = len(timesteps)

        seq_len = infos["all_lengths"].max()

        # Store text embeddings just once, with repeat handling during concatenation
        all_xt_best = []
        all_xt_worst = []
        all_xt_old = []
        all_t = []

        # y
        all_mask = []
        all_lengths = []
        all_tx_x = []
        all_tx_length = []
        all_tx_mask = []
        all_tx_uncond_x = []
        all_tx_uncond_length = []
        all_tx_uncond_mask = []

        for t in sorted(list(results_by_timestep.keys()), reverse=True):
            experiment = results_by_timestep[t]
            experiment = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in experiment.items()}

            all_xt_best.append(experiment["xt_best"])
            all_xt_worst.append(experiment["xt_worst"])
            all_xt_old.append(experiment["xt_old"])
            all_t.append(torch.full((batch_size,), t).cpu())
            # y
            all_mask.append(experiment["mask"])
            all_lengths.append(experiment["length"])
            all_tx_x.append(experiment["tx-x"])
            all_tx_length.append(experiment["tx-length"])
            all_tx_mask.append(experiment["tx-mask"])
            all_tx_uncond_x.append(experiment["tx_uncond-x"])
            all_tx_uncond_length.append(experiment["tx_uncond-length"])
            all_tx_uncond_mask.append(experiment["tx_uncond-mask"])

        # Concatenate all the results for this batch
        dataset["xt_best"].append(torch.cat(all_xt_best, dim=0).view(diff_step, batch_size, seq_len, 205).permute(1, 0, 2, 3))
        dataset["xt_worst"].append(torch.cat(all_xt_worst, dim=0).view(diff_step, batch_size, seq_len, 205).permute(1, 0, 2, 3))
        dataset["xt"].append(torch.cat(all_xt_old, dim=0).view(diff_step, batch_size, seq_len, 205).permute(1, 0, 2, 3))
        dataset["t"].append(torch.cat(all_t, dim=0).view(diff_step, batch_size).T)

        # y
        dataset["mask"].append(torch.cat(all_mask, dim=0).view(diff_step, batch_size, seq_len).permute(1, 0, 2))
        dataset["length"].append(torch.cat(all_lengths, dim=0).view(diff_step, batch_size).T)
        dataset["tx_x"].append(torch.cat(all_tx_x, dim=0).view(diff_step, batch_size, 1, 512).permute(1, 0, 2, 3))
        dataset["tx_length"].append(torch.cat(all_tx_length, dim=0).view(diff_step, batch_size).T)
        dataset["tx_mask"].append(torch.cat(all_tx_mask, dim=0).view(diff_step, batch_size, 1).permute(1, 0, 2))
        dataset["tx_uncond_x"].append(torch.cat(all_tx_uncond_x, dim=0).view(diff_step, batch_size, 1, 512).permute(1, 0, 2, 3))
        dataset["tx_uncond_length"].append(torch.cat(all_tx_uncond_length, dim=0).view(diff_step, batch_size).T)
        dataset["tx_uncond_mask"].append(torch.cat(all_tx_uncond_mask, dim=0).view(diff_step, batch_size, 1).permute(1, 0, 2))

        for key in dataset:
            dataset[key] = torch.cat(dataset[key], dim=0)
        
        # Prepare flat dataset
        dataset = self._prepare_spo_dataset(results_by_timestep)
        
        return dataset

    def _prepare_spo_dataset(self, results_by_timestep: Dict) -> Dict[str, torch.Tensor]:
        """Flatten results_by_timestep into training dataset"""
        dataset = {
            "xt_best": [],
            "xt_worst": [],
            "xt": [],
            "t": [],
            "mask": [],
            "length": [],
            "tx_x": [],
            "tx_mask": [],
            "tx_length": [],
            "tx_uncond_x": [],
            "tx_uncond_mask": [],
            "tx_uncond_length": [],
        }
        
        timesteps = sorted(results_by_timestep.keys(), reverse=True)
        
        for t in timesteps:
            exp = results_by_timestep[t]
            # Move to CPU to save GPU memory
            exp = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                   for k, v in exp.items()}
            
            batch_size = exp["xt_best"].shape[0]
            
            dataset["xt_best"].append(exp["xt_best"])
            dataset["xt_worst"].append(exp["xt_worst"])
            dataset["xt"].append(exp["xt_old"])
            dataset["t"].append(torch.full((batch_size,), t))
            dataset["mask"].append(exp["mask"])
            dataset["length"].append(exp["length"])
            dataset["tx_x"].append(exp["tx-x"])
            dataset["tx_mask"].append(exp["tx-mask"])
            dataset["tx_length"].append(exp["tx-length"])
            dataset["tx_uncond_x"].append(exp["tx_uncond-x"])
            dataset["tx_uncond_mask"].append(exp["tx_uncond-mask"])
            dataset["tx_uncond_length"].append(exp["tx_uncond-length"])
        
        # Concatenate all timesteps
        for key in dataset:
            dataset[key] = torch.cat(dataset[key], dim=0)
        
        return dataset

    def _shuffle_spo_dataset(self, dataset: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shuffle SPO dataset"""
        dataset_size = dataset["t"].shape[0]
        shuffle_indices = torch.randperm(dataset_size)
        
        shuffled = {}
        for key in dataset:
            shuffled[key] = dataset[key][shuffle_indices]
        
        return shuffled

    def _get_minibatch(self, dataset: Dict, start_idx: int, batch_size: int) -> Dict:
        """Extract minibatch from SPO dataset"""
        end_idx = min(start_idx + batch_size, len(dataset['t']))
        device = self.device
        
        minibatch = {}
        for key in dataset:
            minibatch[key] = dataset[key][start_idx:end_idx].to(device)
        
        return minibatch

    def _compute_spo_loss(self, minibatch: Dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SPO loss for a minibatch.
        Returns: (loss, mean_ratio_best, mean_ratio_worst)
        """
        # Prepare y
        y = {
            "length": minibatch["length"],
            "mask": minibatch["mask"],
            "tx": {
                "x": minibatch["tx_x"],
                "mask": minibatch["tx_mask"],
                "length": minibatch["tx_length"],
            },
            "tx_uncond": {
                "x": minibatch["tx_uncond_x"],
                "mask": minibatch["tx_uncond_mask"],
                "length": minibatch["tx_uncond_length"],
            },
            "infos": {},
        }
        
        # Compute log-likelihoods from current model
        new_log_like_best = self._get_loglike(
            minibatch["t"], minibatch["xt"], y, minibatch["xt_best"]
        )
        new_log_like_worst = self._get_loglike(
            minibatch["t"], minibatch["xt"], y, minibatch["xt_worst"]
        )
        
        # Compute log-likelihoods from reference model
        with torch.no_grad():
            ref_log_like_best = self._reference_model._get_loglike(
                minibatch["t"], minibatch["xt"], y, minibatch["xt_best"]
            )
            ref_log_like_worst = self._reference_model._get_loglike(
                minibatch["t"], minibatch["xt"], y, minibatch["xt_worst"]
            )
        
        # Compute importance sampling ratios with clipping
        eps = self.advantage_clip_epsilon
        ratio_best = torch.clamp(torch.exp(new_log_like_best - ref_log_like_best), 1 - eps, 1 + eps)
        ratio_worst = torch.clamp(torch.exp(new_log_like_worst - ref_log_like_worst), 1 - eps, 1 + eps)
        
        # SPO loss: -log(sigmoid(beta * (log(r_best) - log(r_worst))))
        loss = -torch.log(torch.sigmoid(self.beta * torch.log(ratio_best) - self.beta * torch.log(ratio_worst))).mean()
        
        return loss, ratio_best.mean(), ratio_worst.mean()

    def _get_loglike(self, t, xt, y, xt_target):
        """
        Compute log-likelihood for a target sample given current state.
        Matches the logic of get_loglike_SPO from GaussianDiffusion.
        
        Args:
            t: timestep tensor [BS]
            xt: current noisy sample [BS, T, F]
            y: conditioning dictionary
            xt_target: target sample to compute likelihood for [BS, T, F]
        
        Returns:
            log_prob: log probability for each sample [BS]
        """
        # Compute prediction with classifier-free guidance

        # Prepare conditional and unconditional inputs
        y_uncond = y.copy()
        y_uncond["tx"] = y_uncond["tx_uncond"]
        
        # Concatenate inputs for batch processing
        xt_batch = torch.cat([xt, xt], dim=0)  # [2*B, ...]
        t_batch = torch.cat([t, t], dim=0)  # [2*B, ...]
        
        # Create combined y dictionary
        y_batch = {}
        for key in y.keys():
            if key == "tx":
                y_batch[key] = combine_data(y[key], y_uncond[key])
            else:
                y_batch[key] = combine_data(y[key], y[key], use_first_for_second=True)
        
        # Single forward pass
        predict_batch = masked(self.denoiser(xt_batch, y_batch, t_batch), y_batch["mask"])

        # predict_batch_ref = masked(self._reference_model.denoiser(xt_batch, y_batch, t_batch), y_batch["mask"])
        
        # Split results
        predict_cond, predict_uncond = torch.chunk(predict_batch, 2, dim=0)
        
        # Apply classifier-free guidance
        
        predict = predict_uncond + self.guidance_weight * (predict_cond - predict_uncond)
        
        # Get posterior distribution parameters
        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            predict, xt, t.unsqueeze(-1) if t.dim() == 1 else t
        )
        
        # Mask and clamp sigma
        mean = masked(mean, y["mask"])
        sigma = torch.max(sigma, torch.tensor(0.1, device=sigma.device))
        
        # Compute log-likelihood using Gaussian distribution
        log_likelihood = self.log_likelihood(xt_target, mean, sigma)
        log_likelihood = nan_masked(log_likelihood, y["mask"])
        
        # Average over features and time, sum gives total log prob per sample
        log_prob = log_likelihood.nanmean(dim=[1, 2])
        
        return log_prob   

    def log_likelihood(self, x, mu, sigma):# TODO is this correct?
        var = sigma ** 2 + 1e-8  # Ensure variance is > 0
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((x - mu) ** 2) / var)
        return log_prob   

    def configure_optimizers(self):
        """Configure optimizer"""
        if self.lora:
            print("Optimizing only LoRA parameters")
            # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.denoiser.parameters()), lr=self.hparams.lr,
            #                           betas=(self.hparams.beta1, self.hparams.beta2), eps=self.hparams.eps,
            #                           weight_decay=self.hparams.weight_decay)
        
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.denoiser.parameters()), 
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2), 
                eps=self.hparams.eps,
                weight_decay=self.hparams.weight_decay
            )
        
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,  # Dimezza il LR
                patience=50,  # Dopo 50 validation step senza miglioramenti
                min_lr=self.hparams.lr * 1e-4
            )
        
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'Validation.tmr',
                    'interval': 'step',
                    'frequency': 1
                }
            }

        else:
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

            return optimizer

    def validation_step(self, batch, batch_idx):
        """TODO Validation: compute TMR matching score"""
        logger.info(f"Starting validation step {batch_idx} with batch size {len(batch['keyid'])}")
        with torch.no_grad():
            self.dense_reward_model.to(self.device)
            
            # Generate motions
            # sequences = self.diffusion(
            #     tx_emb=batch["tx"],
            #     tx_emb_uncond=batch["tx_uncond"],
            #     infos={
            #         "all_lengths": batch["motion_x_dict"]["length"],
            #         "guidance_weight": self.guidance_weight,
            #     }
            # )
            sequences = self.generate(batch["text"], batch["motion_x_dict"]["length"])
            
            # Compute TMR score
            batch_size = sequences.shape[0]
            t = torch.tensor(0).repeat(batch_size).to(self.device)
            
            tmr_text_emb = self.dense_reward_model.encode_text(batch["text"])
            tmr_motion_emb = self.dense_reward_model.encode_motion(sequences, diffusion_steps=t)
            tmr_score = self.dense_reward_model.matching_score(tmr_text_emb, tmr_motion_emb).mean()
            
            self.dense_reward_model.to("cpu")
            
            logger.info(f"Validation.tmr: {tmr_score}")
            self.log_dict({
                'Validation.tmr': tmr_score,
                'Validation.epoch': self.current_epoch,
                'Validation.global_step': self.global_step
            }, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return tmr_score
    
    def diffusionSPO(self, tx_emb=None, tx_emb_uncond=None, infos=None, reward_model=None, k=10, tmr_text_emb=None, log=True):
        device = self.device
        lengths = infos["all_lengths"][0:tx_emb["x"].shape[0]]
        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        duration = max(lengths)
        duration = int(duration.item()) if isinstance(duration, torch.Tensor) else duration
        nfeats = self.denoiser.nfeats

        shape = (bs, duration, nfeats)
        sample = masked(torch.randn(shape, device=device), mask)
        
        y_expanded = expand_for_k(y, k)

        results = {}
        best_scores_ = []
        worst_scores_ = []
        count = 0
        
        step = 100 // self.num_inference_steps 
        iterator = list(range(self.timesteps - 1, 0, -step)) + [0]

        x_start = None
        for i, t in enumerate(iterator):
            if t > self.divert_start_step:
                sample, predict, step_results = self._diffusion_step_with_k_sampling(
                    sample, y, y_expanded, t, k, bs, duration, nfeats, tmr_text_emb, infos
                )
                self._update_results(results, step_results, best_scores_, worst_scores_)
            else:
                sample, predict = self._diffusion_step_standard(
                    sample, y, t, bs, infos
                )
            
            if t == 0:
                x_start = predict

        results = {k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v for k, v in results.items()}

        return x_start, results


    def _diffusion_step_with_k_sampling(self, sample, y, y_expanded, t, k, bs, duration, nfeats, tmr_text_emb, infos):
        """Esegue uno step di diffusion con k-sampling per SPO"""
        t_tensor = torch.tensor(t).repeat(k * bs)
        old_sample = sample.clone()
        
        # Expand sample per k campioni
        sample = sample.unsqueeze(0).repeat(k, 1, 1, 1)
        sample = sample.reshape(k * bs, duration, nfeats)
        
        y_flat = flatten_k_bs(y_expanded, k, bs)
        
        # Compute prediction con guidance
        predict = self._compute_prediction_with_guidance(
            sample, y, y_expanded, t_tensor, k, bs, infos
        )
        
        # Update sample
        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            predict, sample, t_tensor.unsqueeze(-1)
        )
        mean = masked(mean, y_flat["mask"])
        sigma = torch.max(sigma, torch.tensor(0.1, device=sigma.device))
        noise = torch.randn_like(mean)
        sample = mean + sigma * noise
        
        # Calcola metriche e seleziona best/worst
        step_results = self._compute_k_sampling_metrics(
            sample, tmr_text_emb, t_tensor, old_sample, y, k, bs, duration, nfeats
        )
        
        # Random select next sample
        sample = self._random_select_next_sample(sample, k, bs, duration, nfeats)
        
        return sample, predict, step_results


    def _diffusion_step_standard(self, sample, y, t, bs, infos):
        """Esegue uno step di diffusion standard (senza k-sampling)"""
        t_tensor = torch.tensor(t).repeat(bs)
        
        # Compute prediction
        if infos["guidance_weight"] == 1.0:
            predict = masked(self.denoiser(sample, y, t_tensor), y["mask"])
        else:
            predict = self._compute_prediction_with_cfg(sample, y, t_tensor, infos)
        
        # Update sample
        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            predict, sample, t_tensor.unsqueeze(-1)
        )
        mean = masked(mean, y["mask"])
        sigma = torch.max(sigma, torch.tensor(0.1, device=sigma.device))
        noise = torch.randn_like(mean)
        sample = mean + sigma * noise
        
        return sample, predict


    def _compute_prediction_with_guidance(self, sample, y, y_expanded, t, k, bs, infos):
        """Calcola la predizione con classifier-free guidance per k-sampling"""
        y_flat = flatten_k_bs(y_expanded, k, bs)
        
        if self.guidance_weight == 1.0:
            predict = masked(self.denoiser(sample, y_flat, t), y_flat["mask"]).detach()
        else:
            y_uncond = y.copy()
            y_uncond["tx"] = y_uncond["tx_uncond"]

            y_cond_expanded = expand_for_k(y, k)
            y_uncond_expanded = expand_for_k(y_uncond, k)

            y_cond_flat = flatten_k_bs(y_cond_expanded, k, bs)
            y_uncond_flat = flatten_k_bs(y_uncond_expanded, k, bs)

            xt_batch = torch.cat([sample, sample], dim=0)

            y_batch = {}
            for key in y_cond_expanded.keys():
                if key == "tx":
                    y_batch[key] = combine_data(y_cond_flat[key], y_uncond_flat[key])
                else:
                    y_batch[key] = combine_data(
                        y_cond_flat[key], y_uncond_flat[key], use_first_for_second=True
                    )

            predict_batch = masked(
                self.denoiser(xt_batch, y_batch, t.repeat(2)), y_batch["mask"]
            ).detach()

            predict_cond, predict_uncond = torch.chunk(predict_batch, 2, dim=0)
            predict = predict_uncond + self.guidance_weight * (predict_cond - predict_uncond)
        
        return predict


    def _compute_prediction_with_cfg(self, sample, y, t, infos):
        """Calcola la predizione con classifier-free guidance (standard)"""
        y_uncond = y.copy()
        y_uncond["tx"] = y_uncond["tx_uncond"]

        xt_batch = torch.cat([sample, sample], dim=0)
        t_batch = torch.cat([t, t], dim=0)

        y_batch = {}
        for key in y.keys():
            if key == "tx":
                y_batch[key] = combine_data(y[key], y_uncond[key])
            else:
                y_batch[key] = combine_data(y[key], y[key], use_first_for_second=True)

        predict_batch = masked(self.denoiser(xt_batch, y_batch, t_batch), y_batch["mask"])
        predict_cond, predict_uncond = torch.chunk(predict_batch, 2, dim=0)
        
        predict = predict_uncond + infos["guidance_weight"] * (predict_cond - predict_uncond)
        
        return predict


    def _compute_k_sampling_metrics(self, sample, tmr_text_emb, t, old_sample, y, k, bs, duration, nfeats):
        """Calcola le metriche per k-sampling e ritorna i risultati"""
        sample_unnorm = self.motion_normalizer.inverse(sample)
        tmr_motion_emb = self.dense_reward_model.encode_motion(sample_unnorm, diffusion_steps=t)

        metric = self.dense_reward_model.matching_score(
            tmr_text_emb.repeat(self.k_samples, 1), tmr_motion_emb
        )
        metric = metric.reshape(k, bs)

        best_idx = metric.argmax(dim=0)
        worst_idx = metric.argmin(dim=0)

        sample_reshaped = sample.reshape(k, bs, duration, nfeats)
        best_samples = gather_k(sample_reshaped, best_idx, bs=bs)
        worst_samples = gather_k(sample_reshaped, worst_idx, bs=bs)
        best_scores = metric.max(dim=0).values
        worst_scores = metric.min(dim=0).values

        return {
            "t": t[0].item(),
            "xt_old": old_sample.detach().cpu(),
            "xt_best": best_samples.clone().detach().cpu(),
            "xt_worst": worst_samples.clone().detach().cpu(),
            "scores": (best_scores - worst_scores).mean(),
            "min_scores": (best_scores - worst_scores).min(),
            "best_scores": best_scores,
            "worst_scores": worst_scores,
            "length": torch.tensor(y["length"]).detach().cpu(),
            "mask": y["mask"].detach().cpu(),
            "tx-x": y["tx"]["x"],
            "tx-length": y["tx"]["length"],
            "tx-mask": y["tx"]["mask"],
            "tx_uncond-x": y["tx_uncond"]["x"],
            "tx_uncond-length": y["tx_uncond"]["length"],
            "tx_uncond-mask": y["tx_uncond"]["mask"],
        }


    def _update_results(self, results, step_results, best_scores_, worst_scores_):
        """Aggiorna results e le liste di scores"""
        t = step_results["t"]
        best_scores = step_results.pop("best_scores")
        worst_scores = step_results.pop("worst_scores")
        
        results[t] = step_results
        best_scores_.extend(best_scores)
        worst_scores_.extend(worst_scores)


    def _random_select_next_sample(self, sample, k, bs, duration, nfeats):
        """Seleziona randomicamente il prossimo sample da k campioni"""
        sample = sample.reshape(k, bs, duration, nfeats)
        idx = torch.randint(0, k, (bs,), device=sample.device)
        gather_idx = idx.view(1, bs, 1, 1).expand(1, bs, sample.shape[-2], sample.shape[-1])
        sample = sample.gather(dim=0, index=gather_idx)[0]
        return sample
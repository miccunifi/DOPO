import os
os.environ["WANDB_API_KEY"] = "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"

import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch 
from pytorch_lightning.loggers import WandbLogger

# Disabilita il Fast Path per coerenza numerica assoluta - altrimenti il primo ratio è != 1
torch.backends.mha.set_fastpath_enabled(False)

from src.config import read_config, save_config

logger = logging.getLogger(__name__)


def reserve_gpu_memory(enabled: bool, leave_free_gb: float = 0.3, device: str = "cuda"):
    """
    Alloca tutto tranne leave_free_gb sulla GPU.
    Più robusto che specificare quanta riserva: si adatta alla memoria disponibile.
    """
    if not enabled:
        return None

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    
    to_reserve_bytes = max(0, free_bytes - int(leave_free_gb * 1024**3))
    n_floats = to_reserve_bytes // 4

    if n_floats <= 0:
        print("Not enough free memory to reserve anything.")
        return None

    print(f"GPU: {total_bytes/1024**3:.1f} GB total, {free_bytes/1024**3:.2f} GB free")
    print(f"Reserving {to_reserve_bytes/1024**3:.2f} GB, leaving {leave_free_gb} GB free")
    
    _reserved = torch.zeros(n_floats, dtype=torch.float32, device=device)
    
    free_after, _ = torch.cuda.mem_get_info(device)
    print(f"After reservation: {free_after/1024**3:.2f} GB free")
    
    return _reserved


@hydra.main(config_path="configs", config_name="train_model_spo", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
        cfg = read_config(cfg.resume_dir)
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    # AGGIUNGI QUI, prima di istanziare il modello
    # _gpu_reservation = reserve_gpu_memory(
    #     enabled=cfg.get("reserve_gpu", True),
    #     leave_free_gb=0.3,  # margine di sicurezza
    # )

    logger.info("Loading the dataloaders")
    train_dataset = instantiate(cfg.data, split="train") #
    val_dataset = instantiate(cfg.data, split="val")

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info(f"Loading the model from '{cfg.model._target_}'")
    model = instantiate(cfg.model)

    wandb_logger = WandbLogger(
        project="DensePreferenceOptimization",
        name=cfg.run_dir.split("/")[-1],
        group=cfg.get("group_name", None), 
        log_model=False,
    )
    trainer = instantiate(cfg.trainer, logger=wandb_logger) 
    
    print(f"Callbacks: {[type(cb).__name__ for cb in trainer.callbacks]}")
    print(f"Checkpoint dir: {trainer.log_dir}")

    print("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
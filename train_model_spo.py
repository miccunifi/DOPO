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


@hydra.main(config_path="configs", config_name="train_model_spo", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

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
        log_model=False,
    )
    trainer = instantiate(cfg.trainer, logger=wandb_logger) 
    
    print(f"Callbacks: {[type(cb).__name__ for cb in trainer.callbacks]}")
    print(f"Checkpoint dir: {trainer.log_dir}")

    print("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
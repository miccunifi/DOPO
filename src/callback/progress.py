import logging
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class ProgressLogger(Callback):
    def __init__(self, precision: int = 2, log_every_n_steps: int = 500):
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training done")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ):
        # Logga ogni N steps durante il training (per step-based training)
        if trainer.global_step % self.log_every_n_steps == 0 and trainer.global_step > 0:
            self._log_step_metrics(trainer)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        # Logga alla fine di ogni epoch (per epoch-based training) - versione originale
        self._log_epoch_metrics(trainer, pl_module)

    def _log_step_metrics(self, trainer: Trainer):
        """Logga metriche per step-based training (formato train/loss)"""
        metric_format = f"{{:.{self.precision}e}}"
        metrics_dict = trainer.callback_metrics
        
        # Raccogli tutte le metriche (non solo train/)
        metrics_to_log = []
        
        for key in sorted(metrics_dict.keys()):
            # Ignora metriche con _step o _epoch nel nome per evitare duplicati
            if "_step" in key or "_epoch" in key:
                continue
                
            value = metrics_dict[key]
            
            try:
                if hasattr(value, 'item'):
                    value = value.item()
                
                # Formattazione speciale per metriche contrastive/retrieval
                if any(x in key for x in ["t2m", "m2t", "R1", "R5", "R10", "MedR"]):
                    if "len" in key:
                        formatted_value = str(int(value))
                    elif any(x in key for x in ["MedR", "R1", "R5", "R10"]):
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = metric_format.format(value)
                
                # Rimuovi prefix train/ o val/ per display più pulito
                display_name = key.replace("train/", "").replace("val/", "v_")
                metrics_to_log.append(f"{display_name}: {formatted_value}")
            except:
                pass
        
        if metrics_to_log:
            line = f"Step {trainer.global_step} | " + " | ".join(metrics_to_log)
            logger.info(line)

    def _log_epoch_metrics(self, trainer: Trainer, pl_module: LightningModule):
        """Logga metriche per epoch-based training (formato originale train_loss_epoch)"""
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        metrics_str = []

        losses_dict = trainer.callback_metrics

        def is_contrastive_metrics(x):
            return "t2m" in x or "m2t" in x

        # Trova tutte le metriche con formato split_name_epoch
        losses_to_print = [
            x
            for x in losses_dict.keys()
            for y in [x.split("_")]
            if len(y) == 3
            and y[2] == "epoch"
            and (
                (hasattr(pl_module, 'lmd') and y[1] in pl_module.lmd) 
                or y[1] == "loss" 
                or is_contrastive_metrics(y[1])
            )
        ]

        # Se non ci sono metriche in formato epoch, prova formato con slash
        if not losses_to_print:
            for key in losses_dict.keys():
                if "/" in key and "_step" not in key and "_epoch" not in key:
                    losses_to_print.append(key)

        # Natural order for contrastive
        letters = "0123456789"
        mapping = str.maketrans(letters, letters[::-1])

        def sort_losses(x):
            if "_epoch" in x:
                split, name, epoch_step = x.split("_")
                if is_contrastive_metrics(x):
                    # put them at the end
                    name = "a" + name.translate(mapping)
                return (name, split)
            else:
                # Formato con slash
                return (x, "")

        losses_to_print = sorted(losses_to_print, key=sort_losses, reverse=True)
        
        for metric_name in losses_to_print:
            value = losses_dict[metric_name]
            
            try:
                if hasattr(value, 'item'):
                    value = value.item()
            except:
                continue

            # Formato originale: train_loss_epoch
            if "_epoch" in metric_name:
                split, name, _ = metric_name.split("_")

                if is_contrastive_metrics(metric_name):
                    if "len" in metric_name:
                        metric = str(int(value))
                    elif "MedR" in metric_name:
                        metric = str(int(value * 100) / 100) + "%"
                    else:
                        metric = str(int(value * 100) / 100) + "%"
                else:
                    metric = metric_format.format(value)

                if split == "train":
                    mname = name
                else:
                    mname = f"v_{name}"

                metric = f"{mname} {metric}"
            
            # Formato con slash: train/loss
            else:
                if is_contrastive_metrics(metric_name):
                    if "len" in metric_name:
                        metric = str(int(value))
                    elif any(x in metric_name for x in ["MedR", "R1", "R5", "R10"]):
                        metric = f"{value:.2f}%"
                    else:
                        metric = f"{value:.2f}%"
                else:
                    metric = metric_format.format(value)
                
                display_name = metric_name.replace("train/", "").replace("val/", "v_")
                metric = f"{display_name} {metric}"

            metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        line = line + ": " + "  ".join(metrics_str)
        logger.info(line)
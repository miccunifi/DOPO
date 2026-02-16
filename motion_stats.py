import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from src.data.motion import Normalizer

from src.data.text import TextToEmb

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="motion_stats", version_base="1.3")
def motion_stats(cfg: DictConfig):
    logger.info("Computing motion stats")
    import src.prepare  # noqa

    train_dataset = instantiate(cfg.data, split="train")
    import torch

    feats = torch.cat([x["motion_x_dict"]["x"] for x in tqdm(train_dataset)])
    mean = feats.mean(0)
    std = feats.std(0)

    motion_stats_dir = train_dataset.motion_loader.normalizer.base_dir
    normalizer = Normalizer(base_dir=motion_stats_dir, disable=True) # train_dataset.motion_loader.normalizer
    logger.info(f"Saving them in {normalizer.base_dir}")
    # normalizer.save(mean, std)

    modelpath = "ViT-B/32"
    mean_pooling = False
    text_model = TextToEmb(
        modelpath=modelpath, mean_pooling=mean_pooling, device="cuda:0"
    )

    textfeats = torch.cat([text_model(x["text"]) for x in tqdm(train_dataset)])
    mean_textfeats = textfeats.mean(0)
    std_textfeats = textfeats.std(0)

    text_stats_dir = train_dataset.motion_loader.normalizer.base_dir.replace("smplrifke", f"text_stats_{modelpath}")
    text_normalizer = Normalizer(base_dir=text_stats_dir, disable=True)
    logger.info(f"Saving them in {text_normalizer.base_dir}")
    text_normalizer.save(mean_textfeats, std_textfeats)



if __name__ == "__main__":
    motion_stats()
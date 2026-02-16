from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .text_motion import TextMotionDataset
from .augmented_text_motion import AugmentedTextMotionDataset


class TextMotionDatasetHandler:
    """
    Handler class to dynamically instantiate either TextMotionDataset or 
    AugmentedTextMotionDataset based on the 'augmented' parameter.
    """
    
    def __new__(
        cls,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        clip_embedder,
        augmented: bool = False,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        # AugmentedTextMotionDataset specific parameters
        paraphrase_filename: str = None,
        summary_filename: str = None,
        paraphrase_prob: float = 0.2,
        summary_prob: float = 0.2,
        averaging_prob: float = 0.4,
        text_sampling_nbr: int = 4,
        with_noise: bool = False,
        **kwargs
    ):
        """
        Instantiate the appropriate dataset class based on the 'augmented' flag.
        
        Args:
            augmented: If True, instantiate AugmentedTextMotionDataset, 
                      otherwise TextMotionDataset
            All other args are passed to the respective dataset class
        """
        
        # Common arguments for both dataset types
        common_args = {
            "path": path,
            "motion_loader": motion_loader,
            "text_to_sent_emb": text_to_sent_emb,
            "text_to_token_emb": text_to_token_emb,
            "clip_embedder": clip_embedder,
            "split": split,
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
            "preload": preload,
            "tiny": tiny,
            "with_noise": with_noise,
        }
        print("✅  - Loading dataset from path:", path)
        
        if augmented:
            # Add augmentation-specific parameters
            augmented_args = {
                "paraphrase_filename": paraphrase_filename,
                "summary_filename": summary_filename,
                "paraphrase_prob": paraphrase_prob,
                "summary_prob": summary_prob,
                "averaging_prob": averaging_prob,
                "text_sampling_nbr": text_sampling_nbr,
            }
            return AugmentedTextMotionDataset(**common_args, **augmented_args)
        else:
            return TextMotionDataset(**common_args)
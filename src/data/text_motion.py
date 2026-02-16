import os
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from functools import partial

from .collate import collate_text_motion, collate_text_motion_with_noise
from src.data.text import TextToEmb


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        clip_embedder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        with_noise : bool = False,
    ):
        if tiny:
            split = split + "_tiny"

        self.split = split
        self.keyids = read_split(path, split)

        # TODO fix all this code that is working but it is a mess
        self.text_to_sent_emb = text_to_sent_emb if text_to_sent_emb else None # 'sentence-transformers/all-mpnet-base-v2'
        self.text_to_token_emb = text_to_token_emb if text_to_token_emb else None # 'distilbert-base-uncased'
        self.clip_embedder = clip_embedder if clip_embedder else None
        # self.clip_embedder = TextToEmb(
        #     modelpath="ViT-B/32", mean_pooling=False, device="cpu"
        # ) if clip_embedder else None

        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split or "val" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = split == "train"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats

        self.with_noise = with_noise

        print(f"⚠️  - Motion loader Noise schedule: {self.with_noise}")

        print(f"⚠️  - TODO check if sqrt_alphas_cumprod and the other is valid")
        self.sqrt_alphas_cumprod = torch.tensor(np.load("/deck/groups/MotionRL/DensePreferenceOptimization/evaluators/DenseTMR/sqrt_alphas_cumprod.npy"))
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.load("/deck/groups/MotionRL/DensePreferenceOptimization/evaluators/DenseTMR/sqrt_one_minus_alphas_cumprod.npy"))
        self.linear_scale = 1

        if self.with_noise:
            self.collate_fn = partial(
                collate_text_motion_with_noise,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                linear_scale=self.linear_scale
            )
        else:
            self.collate_fn = collate_text_motion

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )

        text = annotation["text"]
        text_x_dict = self.text_to_token_emb(text) if self.text_to_token_emb is not None else {"x":torch.tensor([0]), "length": torch.tensor(1)}
        sent_emb = self.text_to_sent_emb(text) if self.text_to_sent_emb is not None else  {"x":torch.tensor([0]), "length": torch.tensor(1)}
        clip_emb = self.clip_embedder(text) if self.clip_embedder is not None else  {"x":torch.tensor([0]), "length": torch.tensor(1)}
        clip_emb_uncond = self.clip_embedder("") if self.clip_embedder is not None else  {"x":torch.tensor([0]), "length": torch.tensor(1)}

        output = {
            "motion_x_dict": motion_x_dict,
            "text": text,
            "keyid": keyid,
            "text_x_dict": text_x_dict,
            "sent_emb": sent_emb,
            "tx": clip_emb,
            "tx_uncond": clip_emb_uncond,
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            annots = val.pop("annotations")
            filtered_annots = []

            if "humanact12" in val["path"]:
                continue  # skip humanact12 

            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))

import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def extract(coef, t, tensor):
    shape = tensor.shape
    return coef[t].reshape(shape[0], *((1,) * (len(shape) - 1))).to(tensor)


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len.to(device)
    ) < length.unsqueeze(1).to(device)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = torch.tensor([x_dict["length"] for x_dict in lst_x_dict])
    mask = length_to_mask(length, device=x.device)
    t = None
    if "t" in lst_x_dict[0]:
        t = torch.tensor([x_dict["t"] for x_dict in lst_x_dict])
    batch = {"x": x, "length": length, "mask": mask, "t":t}
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)
        
    return batch

def collate_text_motion_with_noise(
    lst_elements: List,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    linear_scale,
    *,
    device: Optional[str] = None
) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)

    t = torch.randint(0, 100, (1,), device=batch["motion_x_dict"]["x"].device).repeat(batch["motion_x_dict"]["x"].shape[0])
    batch["motion_x_dict"]["t"] = t
    noise = torch.randn_like(batch["motion_x_dict"]['x'])
    mean = linear_scale * extract(sqrt_alphas_cumprod, t, batch["motion_x_dict"]['x']) * batch["motion_x_dict"]['x']
    sigma = extract(sqrt_one_minus_alphas_cumprod, t, batch["motion_x_dict"]['x'])
    batch["motion_x_dict"]['x'] = mean + sigma * noise

    return batch


def collate_text_motion_multiple_texts(lst_elements: List, *, device: Optional[str] = None):
    other_keys = ['keyid', 'sent_emb']

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    batch["text"] = [elt["text"] for elt in lst_elements]

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    batch["motion_x_dict"] = collate_x_dict([x["motion_x_dict"] for x in lst_elements], device=device)

    batch["text_slices"] = []
    current_index = 0
    for elt in lst_elements:
        batch["text_slices"].append((current_index, current_index + len(elt["text"])))
        current_index += len(elt["text"])

    texts_concat = [x_dict for x in lst_elements for x_dict in x["text_x_dict"]]
    batch["text_x_dict"] = collate_x_dict(
        texts_concat,
        device=device
    )
    return batch


def collate_text_motion_multiple_texts_with_noise(
    lst_elements: List,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    linear_scale,
    *,
    device: Optional[str] = None
) -> Dict:
    other_keys = ['keyid', 'sent_emb']

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    batch["text"] = [elt["text"] for elt in lst_elements]

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    batch["motion_x_dict"] = collate_x_dict([x["motion_x_dict"] for x in lst_elements], device=device)

    t = torch.randint(0, 100, (1,), device=batch["motion_x_dict"]["x"].device).repeat(batch["motion_x_dict"]["x"].shape[0])
    batch["motion_x_dict"]["t"] = t
    noise = torch.randn_like(batch["motion_x_dict"]['x'])
    mean = linear_scale * extract(sqrt_alphas_cumprod, t, batch["motion_x_dict"]['x']) * batch["motion_x_dict"]['x']
    sigma = extract(sqrt_one_minus_alphas_cumprod, t, batch["motion_x_dict"]['x'])
    batch["motion_x_dict"]['x'] = mean + sigma * noise

    batch["text_slices"] = []
    current_index = 0
    for elt in lst_elements:
        batch["text_slices"].append((current_index, current_index + len(elt["text"])))
        current_index += len(elt["text"])

    texts_concat = [x_dict for x in lst_elements for x_dict in x["text_x_dict"]]
    batch["text_x_dict"] = collate_x_dict(
        texts_concat,
        device=device
    )
    return batch

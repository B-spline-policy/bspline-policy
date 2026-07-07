from typing import Callable, Dict, Iterable, Optional

import torch
from tqdm import tqdm


def normalize_rgb_cache_dtype(dtype: str) -> str:
    dtype = str(dtype).lower()
    if dtype not in {"float32", "uint8"}:
        raise ValueError(
            "cache_preprocessed_rgb_dtype must be 'float32' or 'uint8', "
            f"got {dtype!r}"
        )
    return dtype


def validate_preprocessed_cache_options(device: torch.device, share_memory: bool):
    if str(device).startswith("cuda") and share_memory:
        raise ValueError("cache_preprocessed_share_memory is only valid for CPU caches.")


def _to_tensor_on_device(value, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def make_empty_preprocessed_sample_cache(
    shape_meta: dict,
    rgb_keys: Iterable[str],
    lowdim_keys: Iterable[str],
    action_shape: tuple,
    obs_steps: int,
    device: torch.device,
    share_memory: bool = False,
    rgb_dtype: str = "float32",
    length: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_dtype = normalize_rgb_cache_dtype(rgb_dtype)
    validate_preprocessed_cache_options(device, share_memory)
    obs_tensors = {}
    for key in rgb_keys:
        c, h, w = tuple(shape_meta["obs"][key]["shape"])
        obs_tensors[key] = torch.empty(
            (length, obs_steps, c, h, w),
            dtype=torch.uint8 if rgb_dtype == "uint8" else torch.float32,
            device=device,
        )
    for key in lowdim_keys:
        shape = tuple(shape_meta["obs"][key]["shape"])
        obs_tensors[key] = torch.empty(
            (length, obs_steps) + shape,
            dtype=torch.float32,
            device=device,
        )

    action_tensor = torch.empty(
        (length,) + tuple(action_shape),
        dtype=torch.float32,
        device=device,
    )

    if share_memory:
        for tensor in obs_tensors.values():
            tensor.share_memory_()
        action_tensor.share_memory_()

    return {"obs": obs_tensors, "action": action_tensor}


def build_preprocessed_sample_cache(
    length: int,
    sample_fn: Callable[[int], Dict[str, Dict[str, torch.Tensor]]],
    device: str = "cpu",
    share_memory: bool = False,
    empty_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    desc: str = "Precomputing samples",
) -> Dict[str, Dict[str, torch.Tensor]]:
    cache_device = torch.device(device)
    validate_preprocessed_cache_options(cache_device, share_memory)
    if length == 0:
        if empty_cache is None:
            empty_cache = {"obs": {}, "action": torch.empty((0,), device=cache_device)}
        return empty_cache

    first = sample_fn(0)
    obs_tensors = {
        key: torch.empty(
            (length,) + tuple(value.shape),
            dtype=torch.as_tensor(value).dtype,
            device=cache_device,
        )
        for key, value in first["obs"].items()
    }
    action_tensor = torch.empty(
        (length,) + tuple(first["action"].shape),
        dtype=torch.float32,
        device=cache_device,
    )

    if share_memory:
        for tensor in obs_tensors.values():
            tensor.share_memory_()
        action_tensor.share_memory_()

    for idx in tqdm(range(length), desc=desc):
        sample = first if idx == 0 else sample_fn(idx)
        for key, value in sample["obs"].items():
            obs_tensors[key][idx].copy_(_to_tensor_on_device(value, cache_device))
        action_tensor[idx].copy_(_to_tensor_on_device(sample["action"], cache_device))

    return {"obs": obs_tensors, "action": action_tensor}


def get_preprocessed_cached_item(
    cache: Dict[str, Dict[str, torch.Tensor]],
    idx: int,
    rgb_keys: Iterable[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    obs = {key: value[idx] for key, value in cache["obs"].items()}
    for key in rgb_keys:
        value = obs.get(key)
        if value is not None and value.dtype == torch.uint8:
            obs[key] = value.to(dtype=torch.float32).mul_(1.0 / 255.0)
    return {
        "obs": obs,
        "action": cache["action"][idx],
    }

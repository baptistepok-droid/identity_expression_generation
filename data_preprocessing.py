from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image


def resize_image(image: Image.Image, height: int, width: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.BICUBIC)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def frames_to_video_tensor(
    frames: Iterable[Image.Image],
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    tensors = [image_to_tensor(resize_image(frame, height, width)) for frame in frames]
    if not tensors:
        raise ValueError("frames must contain at least one image")
    video = torch.stack(tensors, dim=1).unsqueeze(0)
    return video.to(device=device, dtype=dtype)


def image_to_video_tensor(
    image: Image.Image,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return frames_to_video_tensor([image], height, width, dtype, device)


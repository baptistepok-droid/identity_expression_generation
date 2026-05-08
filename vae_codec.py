from __future__ import annotations

import torch
from PIL import Image

from .data_preprocessing import frames_to_video_tensor, image_to_video_tensor


class VAECodec:
    """Small VAE wrapper owned by this project.

    The wrapped `vae` object only needs to expose `encode(video, device=...)`.
    The wrapper owns image/frame preprocessing and calls the VAE directly.
    """

    def __init__(
        self,
        vae,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.vae = vae
        self.height = height
        self.width = width
        self.dtype = dtype
        self.device = device

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        video = image_to_video_tensor(
            image=image,
            height=self.height,
            width=self.width,
            dtype=self.dtype,
            device=self.device,
        )
        return self.vae.encode(video, device=self.device).to(
            device=self.device,
            dtype=self.dtype,
        )

    def encode_frames(self, frames: list[Image.Image]) -> torch.Tensor:
        video = frames_to_video_tensor(
            frames=frames,
            height=self.height,
            width=self.width,
            dtype=self.dtype,
            device=self.device,
        )
        return self.vae.encode(video, device=self.device).to(
            device=self.device,
            dtype=self.dtype,
        )

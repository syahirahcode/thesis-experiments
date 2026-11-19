"""Shared CNN image preprocessing used by both experiment architectures."""

from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SIZE = (224, 224)


def preprocess_image(image_path: str | Path, device: torch.device) -> torch.Tensor:
    """Convert an OpenCV BGR image to a normalized batched RGB tensor."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Gambar tidak dapat dibaca atau path salah: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, IMAGE_SIZE)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32)
    return image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

import io
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as functional
from omegaconf import ListConfig, OmegaConf


def build_attack_pipeline(config: object | None) -> "AttackPipeline":
    if config is None:
        return AttackPipeline([])
    entries = OmegaConf.to_container(config, resolve=True)
    if entries is None:
        return AttackPipeline([])
    if not isinstance(entries, list):
        raise TypeError("attacks config must be a list")
    return AttackPipeline(entries)


class AttackPipeline:
    def __init__(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self._attacks: list[tuple[float, str, dict[str, Any]]] = []
        total_probability = 0.0
        for entry in entries:
            attack_type = str(entry["type"])
            if attack_type not in _ATTACKS:
                raise ValueError(f"unknown attack type: {attack_type}")
            probability = float(entry["probability"])
            if probability < 0.0:
                raise ValueError(f"attack probability must be >= 0, got {probability}")
            args = dict(entry.get("args") or {})
            self._attacks.append((probability, attack_type, args))
            total_probability += probability
        if total_probability > 1.0 + 1e-6:
            raise ValueError(f"attack probabilities sum to {total_probability}, must be <= 1")
        self._identity_probability = 1.0 - total_probability

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        if not self._attacks:
            return images
        outputs = [self._apply_one(images[index]) for index in range(images.shape[0])]
        return torch.stack(outputs, dim=0)

    def _apply_one(self, image: torch.Tensor) -> torch.Tensor:
        choice = float(torch.rand(()).item())
        cumulative = self._identity_probability
        if choice < cumulative:
            return image
        for probability, attack_type, args in self._attacks:
            cumulative += probability
            if choice < cumulative:
                resolved = {key: _sample_arg(value) for key, value in args.items()}
                return _ATTACKS[attack_type](image, **resolved)
        return image


def jpeg(image: torch.Tensor, *, quality: float) -> torch.Tensor:
    quality_int = int(round(quality))
    quality_int = max(1, min(95, quality_int))
    compressed = _jpeg_roundtrip(image.detach(), quality_int)
    return image + (compressed - image).detach()


def crop(image: torch.Tensor, *, scale: float) -> torch.Tensor:
    scale = float(min(1.0, max(1e-3, scale)))
    _, height, width = image.shape
    crop_h = max(1, int(round(height * scale)))
    crop_w = max(1, int(round(width * scale)))
    top = int(torch.randint(0, height - crop_h + 1, (1,)).item()) if crop_h < height else 0
    left = int(torch.randint(0, width - crop_w + 1, (1,)).item()) if crop_w < width else 0
    cropped = image[:, top : top + crop_h, left : left + crop_w]
    return _resize(cropped, height, width)


def rotate(image: torch.Tensor) -> torch.Tensor:
    k = int(torch.randint(1, 4, (1,)).item())
    return image.rot90(k, dims=(-2, -1))


def blur(image: torch.Tensor, *, radius: float) -> torch.Tensor:
    radius = max(0.0, float(radius))
    if radius < 1e-6:
        return image
    sigma = radius
    kernel_size = max(3, int(math.ceil(sigma * 3)) * 2 + 1)
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    gauss = torch.exp(-(coords.pow(2)) / (2.0 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.sum()
    weight = kernel.expand(3, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    blurred = functional.conv2d(image.unsqueeze(0), weight, padding=pad, groups=3)
    return blurred.squeeze(0).clamp(0.0, 1.0)


def resize(image: torch.Tensor, *, scale: float) -> torch.Tensor:
    scale = float(min(1.0, max(1e-3, scale)))
    _, height, width = image.shape
    small_h = max(1, int(round(height * scale)))
    small_w = max(1, int(round(width * scale)))
    downscaled = _resize(image, small_h, small_w)
    return _resize(downscaled, height, width)


def brightness(image: torch.Tensor, *, factor: float) -> torch.Tensor:
    return (image * float(factor)).clamp(0.0, 1.0)


def hflip(image: torch.Tensor) -> torch.Tensor:
    return image.flip(-1)


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    resized = functional.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).clamp(0.0, 1.0)


def _jpeg_roundtrip(image: torch.Tensor, quality: int) -> torch.Tensor:
    array = (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_image = PIL.Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decoded = PIL.Image.open(buffer)
    decoded.load()
    decoded_array = np.asarray(decoded, dtype=np.float32) / 255.0
    return torch.from_numpy(decoded_array.copy()).permute(2, 0, 1).to(device=image.device, dtype=image.dtype)


def _sample_arg(value: object) -> float:
    if isinstance(value, (list, tuple, ListConfig)):
        if len(value) != 2:
            raise ValueError(f"attack arg range must have length 2, got {value}")
        low = float(value[0])
        high = float(value[1])
        if low == high:
            return low
        return float(torch.empty(()).uniform_(low, high).item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"unsupported attack arg: {value!r}")


_ATTACKS: dict[str, Callable[..., torch.Tensor]] = {
    "jpeg": jpeg,
    "crop": crop,
    "rotate": rotate,
    "blur": blur,
    "resize": resize,
    "brightness": brightness,
    "hflip": hflip,
}

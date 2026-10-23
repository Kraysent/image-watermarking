from collections.abc import Callable, Mapping
from pathlib import Path

import imagehash
import PIL.Image
import PIL.ImageStat

Feature = Callable[[Path, PIL.Image.Image], Mapping[str, int | float | str]]


def file_size(path: Path, image: PIL.Image.Image) -> dict[str, int]:
    return {"file_size": path.stat().st_size}


def dimensions(path: Path, image: PIL.Image.Image) -> dict[str, int]:
    width, height = image.size
    return {"width": width, "height": height}


def mean_color(path: Path, image: PIL.Image.Image) -> dict[str, float]:
    stats = PIL.ImageStat.Stat(image)
    return {"mean_r": stats.mean[0], "mean_g": stats.mean[1], "mean_b": stats.mean[2]}


def std_color(path: Path, image: PIL.Image.Image) -> dict[str, float]:
    stats = PIL.ImageStat.Stat(image)
    return {"std_r": stats.stddev[0], "std_g": stats.stddev[1], "std_b": stats.stddev[2]}


def perceptual_hash(path: Path, image: PIL.Image.Image) -> dict[str, str]:
    return {"image_hash": str(imagehash.phash(image))}


FEATURES: list[Feature] = [
    file_size,
    dimensions,
    mean_color,
    std_color,
    perceptual_hash,
]

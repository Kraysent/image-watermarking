from pathlib import Path

import numpy as np
import PIL.Image

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_COUNT = 8
DEFAULT_SIZE = 64
DEFAULT_SEED = 0


def generate(
    *,
    output_dir: Path,
    count: int = DEFAULT_COUNT,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> Path:
    if count < 1:
        raise ValueError("count must be at least 1")
    if size < 1:
        raise ValueError("size must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        PIL.Image.fromarray(pixels, mode="RGB").save(output_dir / f"{index:04d}.png")
    return output_dir.resolve()

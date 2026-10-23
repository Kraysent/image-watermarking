from pathlib import Path

import numpy as np
import PIL.Image
import polars
import torch
import torch.utils.data


class WatermarkDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        dataset_path: Path,
        images_dir: Path,
        split: str,
        image_size: int,
        message_bits: int,
    ) -> None:
        frame = polars.read_parquet(dataset_path).filter(polars.col("split") == split)
        self.paths = [images_dir / relative_path for relative_path in frame["relative_path"].to_list()]
        self.image_size = image_size
        self.message_bits = message_bits

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with PIL.Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size), PIL.Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        message = torch.randint(0, 2, (self.message_bits,), dtype=torch.float32)
        return image_tensor, message

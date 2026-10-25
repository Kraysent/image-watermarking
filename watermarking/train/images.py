import numpy as np
import PIL.Image
import torch


def tensor_to_pil(image: torch.Tensor) -> PIL.Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    uint8 = (array * 255.0).round().clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(uint8)

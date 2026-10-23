from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from omegaconf import DictConfig, OmegaConf

import watermarking.demo.messages as messages
import watermarking.train.models as models


@dataclass
class DemoSession:
    encoder: models.Encoder
    decoder: models.Decoder
    train_cfg: DictConfig
    device: torch.device
    run_id: str

    @property
    def message_bits(self) -> int:
        return int(self.train_cfg.data.message_bits)

    @property
    def image_size(self) -> int:
        return int(self.train_cfg.data.image_size)

    @property
    def residual_scale(self) -> float:
        return float(self.train_cfg.encoder.residual_scale)


def load_session(*, run_id: str, checkpoint_dir: Path, device: str) -> DemoSession:
    run_dir = checkpoint_dir / run_id
    checkpoint_path = _latest_checkpoint_path(run_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"checkpoint not found in: {run_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_cfg = OmegaConf.create(checkpoint["config"])
    torch_device = torch.device(device)

    encoder = models.Encoder(
        message_bits=train_cfg.data.message_bits,
        channels=list(train_cfg.encoder.channels),
        residual_scale=train_cfg.encoder.residual_scale,
    ).to(torch_device)
    decoder = models.Decoder(
        message_bits=train_cfg.data.message_bits,
        channels=list(train_cfg.decoder.channels),
    ).to(torch_device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return DemoSession(
        encoder=encoder,
        decoder=decoder,
        train_cfg=train_cfg,
        device=torch_device,
        run_id=run_id,
    )


def _latest_checkpoint_path(run_dir: Path) -> Path | None:
    checkpoints = list(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None

    def epoch_number(path: Path) -> int:
        return int(path.stem.removeprefix("epoch_"))

    return max(checkpoints, key=epoch_number)


def pil_to_tensor(image: PIL.Image.Image, image_size: int) -> torch.Tensor:
    rgb = image.convert("RGB").resize((image_size, image_size), PIL.Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_pil(image: torch.Tensor) -> PIL.Image.Image:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    uint8 = (array * 255.0).round().clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(uint8)


def side_by_side(*images: torch.Tensor) -> PIL.Image.Image:
    arrays = [
        (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        for image in images
    ]
    return PIL.Image.fromarray(np.concatenate(arrays, axis=1))


def embed(
    session: DemoSession,
    image: PIL.Image.Image,
    *,
    mode: str,
    message_value: str,
) -> tuple[PIL.Image.Image, PIL.Image.Image, str, str, float]:
    message = messages.bits_from_mode(mode, message_value, session.message_bits)
    image_batch = pil_to_tensor(image, session.image_size).unsqueeze(0).to(session.device)
    message_batch = message.unsqueeze(0).to(session.device)

    with torch.no_grad():
        diff = session.encoder(image_batch, message_batch)
        watermarked = (image_batch + diff).clamp(0.0, 1.0)
        prediction = session.decoder(watermarked)
        amplified = (0.5 + diff / (2.0 * session.residual_scale)).clamp(0.0, 1.0)

    grid = side_by_side(image_batch[0], watermarked[0], amplified[0])
    watermarked_image = tensor_to_pil(watermarked[0])
    recovered = messages.threshold_bits(prediction[0])
    accuracy = messages.bit_accuracy(prediction[0], message)
    return (
        grid,
        watermarked_image,
        messages.format_bits(message),
        messages.format_bits(recovered),
        accuracy,
    )


def extract(session: DemoSession, image: PIL.Image.Image) -> tuple[str, str, str, float]:
    image_batch = pil_to_tensor(image, session.image_size).unsqueeze(0).to(session.device)
    with torch.no_grad():
        prediction = session.decoder(image_batch)[0]
    bits = messages.threshold_bits(prediction)
    return (
        messages.format_bits(bits),
        messages.bits_to_text(bits),
        messages.bits_to_hex(bits),
        messages.confidence(prediction),
    )

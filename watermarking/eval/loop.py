from pathlib import Path

import mlflow
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as functional
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

import watermarking.train.dataset as dataset_module
import watermarking.train.models as models


def evaluate(cfg: DictConfig, *, run_id: str) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    checkpoint_path = Path(cfg.train.checkpoint_dir) / run_id / "last.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_cfg = OmegaConf.create(checkpoint["config"])
    device = torch.device(cfg.eval.device)

    encoder = models.Encoder(
        message_bits=train_cfg.data.message_bits,
        channels=list(train_cfg.encoder.channels),
        residual_scale=train_cfg.encoder.residual_scale,
    ).to(device)
    decoder = models.Decoder(
        message_bits=train_cfg.data.message_bits,
        channels=list(train_cfg.decoder.channels),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    dataset = dataset_module.WatermarkDataset(
        dataset_path=Path(train_cfg.data.dataset_path),
        images_dir=Path(train_cfg.data.images_dir),
        split="test",
        image_size=train_cfg.data.image_size,
        message_bits=train_cfg.data.message_bits,
    )
    if len(dataset) == 0:
        raise RuntimeError("test split is empty")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg.data.batch_size,
        shuffle=False,
        num_workers=train_cfg.data.num_workers,
    )

    torch.manual_seed(cfg.eval.seed)
    total_message_loss = 0.0
    total_image_loss = 0.0
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for image, message in loader:
            image = image.to(device)
            message = message.to(device)

            diff = encoder(image, message)
            watermarked = (image + diff).clamp(0.0, 1.0)
            prediction = decoder(watermarked)

            message_loss = functional.mse_loss(prediction, message)
            image_loss = functional.mse_loss(diff, torch.zeros_like(diff))
            loss = train_cfg.loss.w_msg * message_loss + train_cfg.loss.w_img * image_loss

            total_message_loss += message_loss.item()
            total_image_loss += image_loss.item()
            total_loss += loss.item()
            steps += 1

    mean_message_loss = total_message_loss / steps
    mean_image_loss = total_image_loss / steps
    mean_loss = total_loss / steps

    n_images = min(cfg.eval.n_images, len(dataset))
    generator = torch.Generator().manual_seed(cfg.eval.seed)
    sample_indices = torch.randperm(len(dataset), generator=generator)[:n_images].tolist()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "eval_loss": mean_loss,
                "eval_message_loss": mean_message_loss,
                "eval_image_loss": mean_image_loss,
            }
        )
        with torch.no_grad():
            for sample_index, dataset_index in enumerate(sample_indices):
                image, message = dataset[dataset_index]
                image_batch = image.unsqueeze(0).to(device)
                message_batch = message.unsqueeze(0).to(device)
                diff = encoder(image_batch, message_batch)
                watermarked = (image_batch + diff).clamp(0.0, 1.0)
                residual_scale = float(train_cfg.encoder.residual_scale)
                amplified = (0.5 + diff / (2.0 * residual_scale)).clamp(0.0, 1.0)
                grid = _side_by_side(image_batch[0], watermarked[0], amplified[0])
                mlflow.log_image(grid, artifact_file=f"eval/sample_{sample_index:02d}.png")

    print(
        f"run_id={run_id} "
        f"eval_loss={mean_loss:.6f} "
        f"eval_message_loss={mean_message_loss:.6f} "
        f"eval_image_loss={mean_image_loss:.6f}"
    )


def _side_by_side(*images: torch.Tensor) -> PIL.Image.Image:
    arrays = [(_tensor_to_uint8(image)) for image in images]
    return PIL.Image.fromarray(np.concatenate(arrays, axis=1))


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return (array * 255.0).round().clip(0, 255).astype(np.uint8)

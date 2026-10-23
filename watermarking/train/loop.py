from pathlib import Path

import torch
import torch.nn.functional as functional
import torch.utils.data
from omegaconf import DictConfig

import watermarking.train.dataset as dataset_module
import watermarking.train.models as models


def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    dataset = dataset_module.WatermarkDataset(
        dataset_path=Path(cfg.data.dataset_path),
        images_dir=Path(cfg.data.images_dir),
        split="train",
        image_size=cfg.data.image_size,
        message_bits=cfg.data.message_bits,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    encoder = models.Encoder(
        message_bits=cfg.data.message_bits,
        channels=list(cfg.encoder.channels),
        residual_scale=cfg.encoder.residual_scale,
    ).to(device)
    decoder = models.Decoder(
        message_bits=cfg.data.message_bits,
        channels=list(cfg.decoder.channels),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.train.lr,
    )

    max_steps = cfg.train.max_steps
    global_step = 0

    for epoch in range(cfg.train.epochs):
        epoch_message_loss = 0.0
        epoch_image_loss = 0.0
        epoch_total_loss = 0.0
        steps = 0

        for image, message in loader:
            image = image.to(device)
            message = message.to(device)

            diff = encoder(image, message)
            watermarked = (image + diff).clamp(0.0, 1.0)
            prediction = decoder(watermarked)

            message_loss = functional.mse_loss(prediction, message)
            image_loss = functional.mse_loss(diff, torch.zeros_like(diff))
            loss = cfg.loss.w_msg * message_loss + cfg.loss.w_img * image_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_message_loss += message_loss.item()
            epoch_image_loss += image_loss.item()
            epoch_total_loss += loss.item()
            steps += 1
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        if steps == 0:
            raise RuntimeError("train split is empty")

        print(
            f"epoch={epoch + 1} "
            f"loss={epoch_total_loss / steps:.6f} "
            f"message_loss={epoch_message_loss / steps:.6f} "
            f"image_loss={epoch_image_loss / steps:.6f}"
        )

        if max_steps is not None and global_step >= max_steps:
            break

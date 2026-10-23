from collections.abc import Mapping
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


def train(cfg: DictConfig) -> str:
    _prepare_output_dirs(cfg)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _ensure_experiment(cfg.mlflow.experiment_name, Path(cfg.mlflow.artifact_root))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"run_id={run_id}")
        mlflow.log_params(_flatten_params(OmegaConf.to_container(cfg, resolve=True)))

        torch.manual_seed(cfg.train.seed)
        device = torch.device(cfg.train.device)

        train_dataset = dataset_module.WatermarkDataset(
            dataset_path=Path(cfg.data.dataset_path),
            images_dir=Path(cfg.data.images_dir),
            split="train",
            image_size=cfg.data.image_size,
            message_bits=cfg.data.message_bits,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        )
        eval_dataset = dataset_module.WatermarkDataset(
            dataset_path=Path(cfg.data.dataset_path),
            images_dir=Path(cfg.data.images_dir),
            split="test",
            image_size=cfg.data.image_size,
            message_bits=cfg.data.message_bits,
        )
        if len(eval_dataset) == 0:
            raise RuntimeError("test split is empty")
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
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
        finished_epoch = 0
        sample_indices = _sample_indices(len(eval_dataset), cfg.eval.n_images, cfg.eval.seed)

        for epoch in range(cfg.train.epochs):
            encoder.train()
            decoder.train()
            epoch_message_loss = 0.0
            epoch_image_loss = 0.0
            epoch_total_loss = 0.0
            steps = 0

            for image, message in train_loader:
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

            finished_epoch = epoch + 1
            mean_loss = epoch_total_loss / steps
            mean_message_loss = epoch_message_loss / steps
            mean_image_loss = epoch_image_loss / steps
            eval_metrics = _evaluate_epoch(
                encoder=encoder,
                decoder=decoder,
                loader=eval_loader,
                dataset=eval_dataset,
                sample_indices=sample_indices,
                device=device,
                residual_scale=float(cfg.encoder.residual_scale),
                w_msg=float(cfg.loss.w_msg),
                w_img=float(cfg.loss.w_img),
                seed=int(cfg.eval.seed),
                step=finished_epoch,
            )
            mlflow.log_metrics(
                {
                    "loss": mean_loss,
                    "message_loss": mean_message_loss,
                    "image_loss": mean_image_loss,
                    **eval_metrics,
                },
                step=finished_epoch,
            )
            print(
                f"epoch={finished_epoch} "
                f"loss={mean_loss:.6f} "
                f"message_loss={mean_message_loss:.6f} "
                f"image_loss={mean_image_loss:.6f} "
                f"eval_loss={eval_metrics['eval_loss']:.6f} "
                f"eval_message_loss={eval_metrics['eval_message_loss']:.6f} "
                f"eval_image_loss={eval_metrics['eval_image_loss']:.6f}"
            )

            if max_steps is not None and global_step >= max_steps:
                break

        checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "last.pt"
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "epoch": finished_epoch,
                "global_step": global_step,
                "run_id": run_id,
            },
            checkpoint_path,
        )
        mlflow.log_artifact(str(checkpoint_path))
        print(f"checkpoint={checkpoint_path.resolve()}")
        return run_id


def _evaluate_epoch(
    *,
    encoder: models.Encoder,
    decoder: models.Decoder,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    dataset: dataset_module.WatermarkDataset,
    sample_indices: list[int],
    device: torch.device,
    residual_scale: float,
    w_msg: float,
    w_img: float,
    seed: int,
    step: int,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    torch.manual_seed(seed)
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
            loss = w_msg * message_loss + w_img * image_loss
            total_message_loss += message_loss.item()
            total_image_loss += image_loss.item()
            total_loss += loss.item()
            steps += 1

        for sample_index, dataset_index in enumerate(sample_indices):
            image, message = dataset[dataset_index]
            image_batch = image.unsqueeze(0).to(device)
            message_batch = message.unsqueeze(0).to(device)
            diff = encoder(image_batch, message_batch)
            watermarked = (image_batch + diff).clamp(0.0, 1.0)
            amplified = (0.5 + diff / (2.0 * residual_scale)).clamp(0.0, 1.0)
            grid = _side_by_side(image_batch[0], watermarked[0], amplified[0])
            mlflow.log_image(grid, key=f"eval/sample_{sample_index:02d}", step=step)

    return {
        "eval_loss": total_loss / steps,
        "eval_message_loss": total_message_loss / steps,
        "eval_image_loss": total_image_loss / steps,
    }


def _sample_indices(dataset_size: int, n_images: int, seed: int) -> list[int]:
    count = min(n_images, dataset_size)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator)[:count].tolist()


def _side_by_side(*images: torch.Tensor) -> PIL.Image.Image:
    arrays = [
        (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        for image in images
    ]
    return PIL.Image.fromarray(np.concatenate(arrays, axis=1))


def _prepare_output_dirs(cfg: DictConfig) -> None:
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.mlflow.artifact_root).mkdir(parents=True, exist_ok=True)
    tracking_uri = str(cfg.mlflow.tracking_uri)
    if tracking_uri.startswith("sqlite:///"):
        Path(tracking_uri.removeprefix("sqlite:///")).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _ensure_experiment(experiment_name: str, artifact_root: Path) -> None:
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=str(artifact_root.resolve()),
        )
    mlflow.set_experiment(experiment_name)


def _flatten_params(value: object, prefix: str = "") -> dict[str, str | int | float | bool]:
    if isinstance(value, Mapping):
        params: dict[str, str | int | float | bool] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            params.update(_flatten_params(child, path))
        return params
    if isinstance(value, list):
        return {prefix: str(value)}
    if value is None:
        return {prefix: "null"}
    if isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    return {prefix: str(value)}

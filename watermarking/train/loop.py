import logging
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import mlflow
import mlflow.pyfunc
import structlog
import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

import watermarking.train.attacks as attacks_module
import watermarking.train.dataset as dataset_module
import watermarking.train.metrics as metrics
import watermarking.train.models as models

log = structlog.get_logger()


class WatermarkCheckpointModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        raise NotImplementedError("load artifacts['checkpoint'] with torch.load")


def train(cfg: DictConfig) -> str:
    _configure_logging()
    _prepare_output_dirs(cfg)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _ensure_experiment(cfg.mlflow.experiment_name, Path(cfg.mlflow.artifact_root))

    with mlflow.start_run(log_system_metrics=True) as run:
        run_id = run.info.run_id
        log.info("run started", run_id=run_id)
        mlflow.log_params(
            {
                **_flatten_params(OmegaConf.to_container(cfg, resolve=True)),
                "command": shlex.join(sys.argv),
            }
        )

        torch.manual_seed(cfg.train.seed)
        device = torch.device(cfg.train.device)

        message_seed = cfg.data.get("message_seed")
        if message_seed is not None:
            message_seed = int(message_seed)
        train_dataset = dataset_module.WatermarkDataset(
            dataset_path=Path(cfg.data.dataset_path),
            images_dir=Path(cfg.data.images_dir),
            split="train",
            image_size=cfg.data.image_size,
            message_bits=cfg.data.message_bits,
            message_seed=message_seed,
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
            message_seed=message_seed,
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
        train_attacks = attacks_module.build_attack_pipeline(cfg.train.get("attacks"))
        eval_attacks = attacks_module.build_attack_pipeline(cfg.eval.get("attacks"))

        max_steps = cfg.train.max_steps
        checkpoint_every = cfg.train.checkpoint_every
        global_step = 0
        finished_epoch = 0
        total_epochs = int(cfg.train.epochs)
        checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_start = time.perf_counter()

        try:
            for epoch in range(total_epochs):
                encoder.train()
                decoder.train()
                epoch_message_loss = 0.0
                epoch_image_loss = 0.0
                epoch_total_loss = 0.0
                epoch_bit_accuracy = 0.0
                steps = 0

                for image, message in train_loader:
                    image = image.to(device)
                    message = message.to(device)

                    diff = encoder(image, message)
                    watermarked = (image + diff).clamp(0.0, 1.0)
                    attacked = train_attacks.apply(watermarked)
                    prediction = decoder(attacked)

                    message_loss = metrics.message_loss(
                        prediction,
                        message,
                        cfg.loss.message_loss_type,
                    )
                    image_loss = metrics.image_loss(
                        image,
                        watermarked,
                        diff,
                        cfg.loss.image_loss_type,
                    )
                    loss = cfg.loss.w_msg * message_loss + cfg.loss.w_img * image_loss
                    bit_accuracy = metrics.bit_accuracy(prediction, message)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_message_loss += message_loss.item()
                    epoch_image_loss += image_loss.item()
                    epoch_total_loss += loss.item()
                    epoch_bit_accuracy += bit_accuracy.item()
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
                mean_bit_accuracy = epoch_bit_accuracy / steps
                eval_metrics = _evaluate_epoch(
                    encoder=encoder,
                    decoder=decoder,
                    loader=eval_loader,
                    device=device,
                    w_msg=float(cfg.loss.w_msg),
                    w_img=float(cfg.loss.w_img),
                    message_loss_type=cfg.loss.message_loss_type,
                    image_loss_type=cfg.loss.image_loss_type,
                    attacks=eval_attacks,
                )
                mlflow.log_metrics(
                    {
                        "train/loss": mean_loss,
                        "train/message_loss": mean_message_loss,
                        "train/image_loss": mean_image_loss,
                        "train/bit_accuracy": mean_bit_accuracy,
                        **eval_metrics,
                    },
                    step=finished_epoch,
                )
                elapsed = time.perf_counter() - train_start
                remaining_epochs = total_epochs - finished_epoch
                eta = (elapsed / finished_epoch) * remaining_epochs
                log.info(
                    "epoch finished",
                    epoch=finished_epoch,
                    train_loss=round(mean_loss, 6),
                    eval_loss=round(eval_metrics["eval/loss"], 6),
                    elapsed=_format_duration(elapsed),
                    eta=_format_duration(eta),
                )

                if checkpoint_every is not None and finished_epoch % int(checkpoint_every) == 0:
                    _save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        encoder=encoder,
                        decoder=decoder,
                        optimizer=optimizer,
                        cfg=cfg,
                        finished_epoch=finished_epoch,
                        global_step=global_step,
                        run_id=run_id,
                    )

                if max_steps is not None and global_step >= max_steps:
                    break
        except KeyboardInterrupt:
            log.info("interrupted, saving checkpoint")
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                cfg=cfg,
                finished_epoch=finished_epoch,
                global_step=global_step,
                run_id=run_id,
            )
            mlflow.end_run(status="KILLED")
            return run_id

        checkpoint_path = _save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            cfg=cfg,
            finished_epoch=finished_epoch,
            global_step=global_step,
            run_id=run_id,
        )
        _register_model(checkpoint_path=checkpoint_path, model_name=run.info.run_name)
        return run_id


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    encoder: models.Encoder,
    decoder: models.Decoder,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    finished_epoch: int,
    global_step: int,
    run_id: str,
) -> Path:
    checkpoint_path = checkpoint_dir / f"epoch_{finished_epoch}.pt"
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
    log.info("checkpoint saved", checkpoint=str(checkpoint_path.resolve()))
    return checkpoint_path


def _register_model(*, checkpoint_path: Path, model_name: str) -> None:
    mlflow.pyfunc.log_model(
        name=model_name,
        python_model=WatermarkCheckpointModel(),
        artifacts={"checkpoint": str(checkpoint_path)},
        registered_model_name=model_name,
    )


def _evaluate_epoch(
    *,
    encoder: models.Encoder,
    decoder: models.Decoder,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    w_msg: float,
    w_img: float,
    message_loss_type: str,
    image_loss_type: str,
    attacks: attacks_module.AttackPipeline,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    total_message_loss = 0.0
    total_image_loss = 0.0
    total_loss = 0.0
    total_bit_accuracy = 0.0
    total_message_accuracy = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    steps = 0

    with torch.no_grad():
        for image, message in loader:
            image = image.to(device)
            message = message.to(device)
            diff = encoder(image, message)
            watermarked = (image + diff).clamp(0.0, 1.0)
            attacked = attacks.apply(watermarked)
            prediction = decoder(attacked)
            message_loss = metrics.message_loss(prediction, message, message_loss_type)
            image_loss = metrics.image_loss(image, watermarked, diff, image_loss_type)
            loss = w_msg * message_loss + w_img * image_loss
            bit_accuracy = metrics.bit_accuracy(prediction, message)
            batch_size = image.shape[0]
            total_message_loss += message_loss.item()
            total_image_loss += image_loss.item()
            total_loss += loss.item()
            total_bit_accuracy += bit_accuracy.item()
            total_message_accuracy += metrics.message_accuracy(prediction, message).sum().item()
            total_psnr += metrics.psnr(image, watermarked).sum().item()
            total_ssim += metrics.ssim(image, watermarked).sum().item()
            total_samples += batch_size
            steps += 1

    return {
        "eval/loss": total_loss / steps,
        "eval/message_loss": total_message_loss / steps,
        "eval/image_loss": total_image_loss / steps,
        "eval/bit_accuracy": total_bit_accuracy / steps,
        "eval/message_accuracy": total_message_accuracy / total_samples,
        "eval/psnr": total_psnr / total_samples,
        "eval/ssim": total_ssim / total_samples,
    }


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


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

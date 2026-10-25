import io
import logging
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.tracing.attachments as attachments
import structlog
import torch
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

import watermarking.demo.messages as messages
import watermarking.train.attacks as attacks_module
import watermarking.train.checkpoints as checkpoints
import watermarking.train.dataset as dataset_module
import watermarking.train.images as images
import watermarking.train.metrics as metrics
import watermarking.train.models as models
import watermarking.train.system_metrics as system_metrics_module

log = structlog.get_logger()


class WatermarkCheckpointModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        raise NotImplementedError("load artifacts['checkpoint'] with torch.load")


def train(cfg: DictConfig, resume_run_id: str | None = None) -> str:
    _configure_logging()
    _prepare_output_dirs(cfg)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _ensure_experiment(cfg.mlflow.experiment_name, Path(cfg.mlflow.artifact_root))

    with mlflow.start_run(log_system_metrics=False) as run:
        run_id = run.info.run_id
        system_monitor = system_metrics_module.create_system_metrics_monitor(run_id)
        system_monitor.start()
        try:
            return _train_run(cfg=cfg, run=run, run_id=run_id, resume_run_id=resume_run_id)
        finally:
            system_monitor.finish()


def _train_run(
    *,
    cfg: DictConfig,
    run: mlflow.ActiveRun,
    run_id: str,
    resume_run_id: str | None = None,
) -> str:
    log.info("run started", run_id=run_id)
    params = {
        **_flatten_params(OmegaConf.to_container(cfg, resolve=True)),
        "command": shlex.join(sys.argv),
    }
    if resume_run_id is not None:
        params["resumed_from"] = resume_run_id
    mlflow.log_params(params)

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
        channels=list(cfg.model.encoder.channels),
        residual_scale=cfg.model.encoder.residual_scale,
    ).to(device)
    decoder = models.Decoder(
        message_bits=cfg.data.message_bits,
        channels=list(cfg.model.decoder.channels),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.train.lr,
    )
    attacks = attacks_module.build_attack_pipeline(cfg.train.get("attacks"))

    max_steps = cfg.train.max_steps
    checkpoint_every = cfg.train.checkpoint_every
    trace_sample_pct = float(cfg.train.get("trace_sample_pct", 0.0))
    trace_sample_prob = trace_sample_pct / 100.0
    early_stop_cfg = cfg.train.get("early_stop")
    early_stop_patience = None if early_stop_cfg is None else early_stop_cfg.get("patience")
    early_stop_min_delta_pct = 0.0 if early_stop_cfg is None else float(early_stop_cfg.get("min_delta_pct", 0.0))
    early_stop_min_epoch = 0 if early_stop_cfg is None else int(early_stop_cfg.get("min_epoch", 0))
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience)
    global_step = 0
    finished_epoch = 0
    if resume_run_id is not None:
        finished_epoch, global_step = _load_checkpoint(
            resume_run_id=resume_run_id,
            checkpoint_dir=Path(cfg.train.checkpoint_dir),
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
        )
    total_epochs = int(cfg.train.max_epochs)
    start_epoch = finished_epoch
    checkpoint_dir = Path(cfg.train.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.perf_counter()
    eval_loss_history: list[float] = []

    try:
        for epoch in range(start_epoch, total_epochs):
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
                should_trace = trace_sample_prob > 0.0 and torch.rand(()).item() < trace_sample_prob
                message_loss_value, image_loss_value, total_loss_value, bit_accuracy_value = _train_step(
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    image=image,
                    message=message,
                    attacks=attacks,
                    w_msg=float(cfg.train.loss.w_msg),
                    w_img=float(cfg.train.loss.w_img),
                    message_loss_type=cfg.train.loss.message_loss_type,
                    image_loss_type=cfg.train.loss.image_loss_type,
                    trace=should_trace,
                    epoch=epoch + 1,
                    global_step=global_step + 1,
                    run_id=run_id,
                )

                epoch_message_loss += message_loss_value
                epoch_image_loss += image_loss_value
                epoch_total_loss += total_loss_value
                epoch_bit_accuracy += bit_accuracy_value
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
                w_msg=float(cfg.train.loss.w_msg),
                w_img=float(cfg.train.loss.w_img),
                message_loss_type=cfg.train.loss.message_loss_type,
                image_loss_type=cfg.train.loss.image_loss_type,
                attacks=attacks,
            )
            eval_loss = eval_metrics["eval/loss"]
            eval_loss_history.append(eval_loss)
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
            epochs_this_run = finished_epoch - start_epoch
            remaining_epochs = total_epochs - finished_epoch
            eta = (elapsed / epochs_this_run) * remaining_epochs
            log.info(
                "epoch finished",
                epoch=finished_epoch,
                train_loss=round(mean_loss, 6),
                eval_loss=round(eval_loss, 6),
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

            if finished_epoch >= early_stop_min_epoch and _loss_plateaued(
                eval_loss_history,
                patience=early_stop_patience,
                min_delta_pct=early_stop_min_delta_pct,
            ):
                log.info(
                    "early stop",
                    epoch=finished_epoch,
                    patience=early_stop_patience,
                    min_delta_pct=early_stop_min_delta_pct,
                    min_epoch=early_stop_min_epoch,
                    eval_loss=round(eval_loss, 6),
                )
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


def _train_step(
    *,
    encoder: models.Encoder,
    decoder: models.Decoder,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    message: torch.Tensor,
    attacks: attacks_module.AttackPipeline,
    w_msg: float,
    w_img: float,
    message_loss_type: str,
    image_loss_type: str,
    trace: bool,
    epoch: int,
    global_step: int,
    run_id: str,
) -> tuple[float, float, float, float]:
    if not trace:
        return _run_train_step(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            image=image,
            message=message,
            attacks=attacks,
            w_msg=w_msg,
            w_img=w_img,
            message_loss_type=message_loss_type,
            image_loss_type=image_loss_type,
        )

    with mlflow.start_span(name="train_step", run_id=run_id) as span:
        span.set_inputs(
            {
                "epoch": epoch,
                "global_step": global_step,
                "batch_size": int(image.shape[0]),
            }
        )
        message_bits = message[0].detach()
        input_image = _image_attachment(image[0])
        bytes_in = messages.bits_to_hex(message_bits)
        text_in = messages.bits_to_text(message_bits)
        with mlflow.start_span(name="encoder") as encoder_span:
            encoder_span.set_inputs(
                {
                    "image": input_image,
                    "bytes_in": bytes_in,
                    "text_in": text_in,
                }
            )
            diff = encoder(image, message)
            watermarked = (image + diff).clamp(0.0, 1.0)
            watermarked_image = _image_attachment(watermarked[0])
            encoder_span.set_outputs(
                {
                    "image": watermarked_image,
                    "diff_abs_mean": float(diff.detach().abs().mean()),
                }
            )
        with mlflow.start_span(name="attack") as attack_span:
            attack_span.set_inputs({"image": watermarked_image})
            attacked = attacks.apply(watermarked)
            attacked_image = _image_attachment(attacked[0])
            attack_span.set_outputs({"image": attacked_image})
        with mlflow.start_span(name="decoder") as decoder_span:
            decoder_span.set_inputs({"image": attacked_image})
            prediction = decoder(attacked)
            recovered_bits = messages.threshold_bits(prediction[0].detach())
            bytes_out = messages.bits_to_hex(recovered_bits)
            text_out = messages.bits_to_text(recovered_bits)
            decoder_span.set_outputs(
                {
                    "bytes_out": bytes_out,
                    "text_out": text_out,
                }
            )
        with mlflow.start_span(name="loss") as loss_span:
            message_loss = metrics.message_loss(prediction, message, message_loss_type)
            image_loss = metrics.image_loss(image, watermarked, diff, image_loss_type)
            loss = w_msg * message_loss + w_img * image_loss
            bit_accuracy = metrics.bit_accuracy(prediction, message)
            loss_span.set_inputs({"w_msg": w_msg, "w_img": w_img})
            loss_span.set_outputs(
                {
                    "message_loss": float(message_loss.detach()),
                    "image_loss": float(image_loss.detach()),
                    "loss": float(loss.detach()),
                    "bit_accuracy": float(bit_accuracy.detach()),
                }
            )
        optimizer.zero_grad()
        with mlflow.start_span(name="backward") as backward_span:
            loss.backward()
            backward_span.set_inputs({"loss": float(loss.detach())})
            backward_span.set_outputs({"status": "ok"})
        with mlflow.start_span(name="optimizer") as optimizer_span:
            optimizer.step()
            optimizer_span.set_inputs({"global_step": global_step})
            optimizer_span.set_outputs({"status": "ok"})
        outputs = (
            float(message_loss.detach()),
            float(image_loss.detach()),
            float(loss.detach()),
            float(bit_accuracy.detach()),
        )
        span.set_outputs(
            {
                "message_loss": outputs[0],
                "image_loss": outputs[1],
                "loss": outputs[2],
                "bit_accuracy": outputs[3],
            }
        )
        return outputs


def _image_attachment(image: torch.Tensor) -> attachments.Attachment:
    buffer = io.BytesIO()
    images.tensor_to_pil(image).save(buffer, format="PNG")
    return attachments.Attachment(content_type="image/png", content_bytes=buffer.getvalue())


def _run_train_step(
    *,
    encoder: models.Encoder,
    decoder: models.Decoder,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    message: torch.Tensor,
    attacks: attacks_module.AttackPipeline,
    w_msg: float,
    w_img: float,
    message_loss_type: str,
    image_loss_type: str,
) -> tuple[float, float, float, float]:
    diff = encoder(image, message)
    watermarked = (image + diff).clamp(0.0, 1.0)
    attacked = attacks.apply(watermarked)
    prediction = decoder(attacked)
    message_loss = metrics.message_loss(prediction, message, message_loss_type)
    image_loss = metrics.image_loss(image, watermarked, diff, image_loss_type)
    loss = w_msg * message_loss + w_img * image_loss
    bit_accuracy = metrics.bit_accuracy(prediction, message)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (
        float(message_loss.detach()),
        float(image_loss.detach()),
        float(loss.detach()),
        float(bit_accuracy.detach()),
    )


def _load_checkpoint(
    *,
    resume_run_id: str,
    checkpoint_dir: Path,
    encoder: models.Encoder,
    decoder: models.Decoder,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    run_dir = checkpoint_dir / resume_run_id
    checkpoint_path = checkpoints.latest_checkpoint_path(run_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"checkpoint not found in: {run_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    finished_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    log.info(
        "checkpoint loaded",
        checkpoint=str(checkpoint_path.resolve()),
        resumed_from=resume_run_id,
        epoch=finished_epoch,
        global_step=global_step,
    )
    return finished_epoch, global_step


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


def _loss_plateaued(
    losses: list[float],
    *,
    patience: int | None,
    min_delta_pct: float,
) -> bool:
    if patience is None or patience <= 0 or len(losses) < patience:
        return False
    window = losses[-patience:]
    mean_loss = sum(window) / len(window)
    return max(window) - min(window) <= (min_delta_pct / 100.0) * abs(mean_loss)


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

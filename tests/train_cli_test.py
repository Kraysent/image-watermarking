from pathlib import Path

import click.testing
import hydra
import mlflow
import PIL.Image
import polars
import torch

import watermarking.cli as cli_module
import watermarking.train.cli as train_cli


def test_cli_help_lists_train() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "demo" in result.output


def test_cli_has_no_eval_command() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(cli_module.cli, ["eval", "--help"])
    assert result.exit_code != 0


def test_hydra_overrides_compose() -> None:
    with hydra.initialize_config_dir(config_dir=str(train_cli.CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "encoder.channels=[32,64]",
                "decoder.channels=[8]",
                "train.epochs=3",
                "loss.w_img=2.5",
                "loss.image_loss_type=ssim",
            ],
        )
    assert cfg.train.epochs == 3
    assert list(cfg.encoder.channels) == [32, 64]
    assert list(cfg.decoder.channels) == [8]
    assert cfg.loss.w_img == 2.5
    assert cfg.loss.image_loss_type == "ssim"


def test_hydra_test_config_compose() -> None:
    with hydra.initialize_config_dir(config_dir=str(train_cli.CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="test")
    assert cfg.data.image_size == 32
    assert cfg.data.message_bits == 4
    assert cfg.train.device == "cpu"
    assert cfg.mlflow.experiment_name == "watermarking-test"


def test_train_runs_one_step(tmp_path: Path) -> None:
    images_dir = tmp_path / "raw"
    images_dir.mkdir(parents=True)
    for name in ("train.png", "test.png"):
        PIL.Image.new("RGB", (40, 40), color=(12, 34, 56)).save(images_dir / name)

    dataset_path = tmp_path / "dataset.parquet"
    polars.DataFrame(
        {
            "relative_path": ["train.png", "test.png"],
            "split": ["train", "test"],
        }
    ).write_parquet(dataset_path)
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    tracking_uri = f"sqlite:///{tmp_path / 'outputs' / 'mlflow.db'}"
    artifact_root = tmp_path / "outputs" / "mlartifacts"

    runner = click.testing.CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "train",
            f"data.dataset_path={dataset_path}",
            f"data.images_dir={images_dir}",
            "data.image_size=32",
            "data.message_bits=4",
            "data.batch_size=1",
            "encoder.channels=[4,8]",
            "decoder.channels=[4]",
            "train.max_steps=1",
            "train.device=cpu",
            f"train.checkpoint_dir={checkpoint_dir}",
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.artifact_root={artifact_root}",
            "mlflow.experiment_name=test",
        ],
    )
    assert result.exit_code == 0, result.output

    run_id = next(line.removeprefix("run_id=") for line in result.output.splitlines() if line.startswith("run_id="))
    checkpoint_path = checkpoint_dir / run_id / "epoch_1.pt"
    assert checkpoint_path.is_file()
    assert "checkpoint=" in result.output
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint) >= {"encoder", "decoder", "optimizer", "config", "epoch", "global_step", "run_id"}
    assert checkpoint["run_id"] == run_id
    assert checkpoint["epoch"] == 1

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert "command" in run.data.params
    assert run.data.params["command"]
    assert "eval/loss" in run.data.metrics
    assert "eval/message_loss" in run.data.metrics
    assert "eval/image_loss" in run.data.metrics
    assert "train/bit_accuracy" in run.data.metrics
    assert "eval/bit_accuracy" in run.data.metrics
    assert "eval/message_accuracy" in run.data.metrics
    assert "eval/psnr" in run.data.metrics
    assert "eval/ssim" in run.data.metrics
    artifact_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="epoch_1.pt"))
    assert artifact_path.is_file()

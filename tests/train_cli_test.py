import re
from pathlib import Path

import click.testing
import hydra
import mlflow
import PIL.Image
import polars
import torch

import watermarking.cli as cli_module
import watermarking.train.cli as train_cli


def _run_id_from_output(output: str) -> str:
    match = re.search(r"\brun_id=([a-f0-9]+)\b", output)
    assert match is not None, output
    return match.group(1)


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
                "model.encoder.channels=[32,64]",
                "model.decoder.channels=[8]",
                "train.max_epochs=3",
                "train.loss.w_img=2.5",
                "train.loss.image_loss_type=ssim",
            ],
        )
    assert cfg.train.max_epochs == 3
    assert list(cfg.model.encoder.channels) == [32, 64]
    assert list(cfg.model.decoder.channels) == [8]
    assert cfg.train.loss.w_img == 2.5
    assert cfg.train.loss.image_loss_type == "ssim"


def test_hydra_deep_model_compose() -> None:
    with hydra.initialize_config_dir(config_dir=str(train_cli.CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["model=deep"])
    assert list(cfg.model.encoder.channels) == [64, 64, 64, 64]
    assert cfg.model.encoder.message_layer == 3
    assert list(cfg.model.decoder.channels) == [64, 64, 64, 64, 64]


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
            "model.encoder.channels=[4,8]",
            "model.decoder.channels=[4]",
            "train.max_steps=1",
            "train.device=cpu",
            f"train.checkpoint_dir={checkpoint_dir}",
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.artifact_root={artifact_root}",
            "mlflow.experiment_name=test",
        ],
    )
    assert result.exit_code == 0, result.output

    run_id = _run_id_from_output(result.output)
    checkpoint_path = checkpoint_dir / run_id / "epoch_1.pt"
    assert checkpoint_path.is_file()
    assert "checkpoint=" in result.output
    assert "elapsed=" in result.output
    assert "eta=" in result.output
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(checkpoint) >= {"encoder", "decoder", "optimizer", "config", "epoch", "global_step", "run_id"}
    assert checkpoint["run_id"] == run_id
    assert checkpoint["epoch"] == 1

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
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

    client = mlflow.tracking.MlflowClient(tracking_uri)
    registered = client.get_registered_model(run.info.run_name)
    assert registered.name == run.info.run_name
    versions = client.search_model_versions(f"name='{run.info.run_name}'")
    assert len(versions) == 1
    assert versions[0].run_id == run_id


def test_train_resumes_from_run_id(tmp_path: Path) -> None:
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
    train_args = [
        "train",
        f"data.dataset_path={dataset_path}",
        f"data.images_dir={images_dir}",
        "data.image_size=32",
        "data.message_bits=4",
        "data.batch_size=1",
        "model.encoder.channels=[4,8]",
        "model.decoder.channels=[4]",
        "train.max_epochs=2",
        "train.max_steps=1",
        "train.device=cpu",
        f"train.checkpoint_dir={checkpoint_dir}",
        f"mlflow.tracking_uri={tracking_uri}",
        f"mlflow.artifact_root={artifact_root}",
        "mlflow.experiment_name=test",
    ]

    runner = click.testing.CliRunner()
    first = runner.invoke(cli_module.cli, train_args)
    assert first.exit_code == 0, first.output
    source_run_id = _run_id_from_output(first.output)
    source_checkpoint = torch.load(
        checkpoint_dir / source_run_id / "epoch_1.pt",
        map_location="cpu",
        weights_only=False,
    )

    resumed = runner.invoke(
        cli_module.cli,
        [
            "train",
            "--run-id",
            source_run_id,
            *train_args[1:],
            "train.max_epochs=2",
            "train.max_steps=2",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    new_run_id = _run_id_from_output(resumed.output)
    assert new_run_id != source_run_id

    resumed_checkpoint_path = checkpoint_dir / new_run_id / "epoch_2.pt"
    assert resumed_checkpoint_path.is_file()
    resumed_checkpoint = torch.load(resumed_checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed_checkpoint["run_id"] == new_run_id
    assert resumed_checkpoint["epoch"] == 2
    assert resumed_checkpoint["global_step"] == 2
    assert resumed_checkpoint["encoder"].keys() == source_checkpoint["encoder"].keys()

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(new_run_id)
    assert run.data.params["resumed_from"] == source_run_id
    assert "resumed_from" not in mlflow.get_run(source_run_id).data.params

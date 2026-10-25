import re
from pathlib import Path

import click.testing
import mlflow
import polars
import pytest

import watermarking.cli as cli_module


def _run_id_from_output(output: str) -> str:
    match = re.search(r"\brun_id=([a-f0-9]+)\b", output)
    assert match is not None, output
    return match.group(1)


def _prepare_overfit_dataset(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "raw"
    catalog_path = tmp_path / "catalog.parquet"
    dataset_path = tmp_path / "dataset.parquet"
    runner = click.testing.CliRunner()

    synthetic_result = runner.invoke(
        cli_module.cli,
        [
            "ingest",
            "synthetic",
            f"--output={images_dir}",
            "--count=2",
            "--size=48",
            "--seed=0",
        ],
    )
    assert synthetic_result.exit_code == 0, synthetic_result.output

    catalog_result = runner.invoke(
        cli_module.cli,
        ["ingest", "catalog", f"--input={images_dir}", f"--output={catalog_path}"],
    )
    assert catalog_result.exit_code == 0, catalog_result.output

    validate_result = runner.invoke(
        cli_module.cli,
        ["ingest", "validate", f"--input={catalog_path}"],
    )
    assert validate_result.exit_code == 0, validate_result.output

    paths = polars.read_parquet(catalog_path)["relative_path"].to_list()
    assert len(paths) == 2
    polars.DataFrame(
        {
            "relative_path": paths + paths,
            "split": ["train", "train", "test", "test"],
        }
    ).write_parquet(dataset_path)
    return images_dir, dataset_path


def _overfit_train_args(
    *,
    images_dir: Path,
    dataset_path: Path,
    checkpoint_dir: Path,
    tracking_uri: str,
    artifact_root: Path,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "train",
        "model=deep",
        f"data.dataset_path={dataset_path}",
        f"data.images_dir={images_dir}",
        "data.batch_size=2",
        "data.message_seed=0",
        f"train.checkpoint_dir={checkpoint_dir}",
        "train.early_stop=null",
        f"mlflow.tracking_uri={tracking_uri}",
        f"mlflow.artifact_root={artifact_root}",
        "mlflow.experiment_name=watermarking-test",
    ]
    if extra:
        args.extend(extra)
    return args


@pytest.mark.integration
def test_ingest_and_train_overfits_to_near_zero_loss(tmp_path: Path) -> None:
    images_dir, dataset_path = _prepare_overfit_dataset(tmp_path)
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    tracking_uri = f"sqlite:///{tmp_path / 'outputs' / 'mlflow.db'}"
    artifact_root = tmp_path / "outputs" / "mlartifacts"
    runner = click.testing.CliRunner()

    train_result = runner.invoke(
        cli_module.cli,
        _overfit_train_args(
            images_dir=images_dir,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
        ),
    )
    assert train_result.exit_code == 0, train_result.output

    run_id = _run_id_from_output(train_result.output)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    message_losses = client.get_metric_history(run_id, "train/message_loss")
    bit_accuracies = client.get_metric_history(run_id, "train/bit_accuracy")
    assert len(message_losses) >= 2
    assert message_losses[0].value > 0.5
    assert message_losses[-1].value < 1e-2
    assert bit_accuracies[-1].value == 1.0


@pytest.mark.integration
def test_zero_image_loss_weight_makes_detection_trivially_perfect(tmp_path: Path) -> None:
    images_dir, dataset_path = _prepare_overfit_dataset(tmp_path)
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    tracking_uri = f"sqlite:///{tmp_path / 'outputs' / 'mlflow.db'}"
    artifact_root = tmp_path / "outputs" / "mlartifacts"
    runner = click.testing.CliRunner()

    train_result = runner.invoke(
        cli_module.cli,
        _overfit_train_args(
            images_dir=images_dir,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            extra=["train.loss.w_img=0"],
        ),
    )
    assert train_result.exit_code == 0, train_result.output

    run_id = _run_id_from_output(train_result.output)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    bit_accuracies = client.get_metric_history(run_id, "train/bit_accuracy")
    assert bit_accuracies[-1].value == 1.0


@pytest.mark.integration
def test_zero_message_loss_weight_collapses_to_null_diff(tmp_path: Path) -> None:
    images_dir, dataset_path = _prepare_overfit_dataset(tmp_path)
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    tracking_uri = f"sqlite:///{tmp_path / 'outputs' / 'mlflow.db'}"
    artifact_root = tmp_path / "outputs" / "mlartifacts"
    runner = click.testing.CliRunner()

    train_result = runner.invoke(
        cli_module.cli,
        _overfit_train_args(
            images_dir=images_dir,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            extra=["train.loss.w_msg=0"],
        ),
    )
    assert train_result.exit_code == 0, train_result.output

    run_id = _run_id_from_output(train_result.output)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    train_image_losses = client.get_metric_history(run_id, "train/image_loss")
    assert train_image_losses[-1].value < 1e-4

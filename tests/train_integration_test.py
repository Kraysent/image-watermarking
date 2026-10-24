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


@pytest.mark.integration
def test_ingest_and_train_overfits_to_near_zero_loss(tmp_path: Path) -> None:
    images_dir = tmp_path / "raw"
    catalog_path = tmp_path / "catalog.parquet"
    dataset_path = tmp_path / "dataset.parquet"
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    tracking_uri = f"sqlite:///{tmp_path / 'outputs' / 'mlflow.db'}"
    artifact_root = tmp_path / "outputs" / "mlartifacts"
    runner = click.testing.CliRunner()

    synthetic_result = runner.invoke(
        cli_module.cli,
        [
            "ingest",
            "synthetic",
            f"--output={images_dir}",
            "--count=3",
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

    split_result = runner.invoke(
        cli_module.cli,
        [
            "ingest",
            "split",
            f"--input={catalog_path}",
            f"--output={dataset_path}",
            "--seed=0",
            "--train-ratio=0.75",
        ],
    )
    assert split_result.exit_code == 0, split_result.output
    splits = set(polars.read_parquet(dataset_path)["split"].to_list())
    assert splits == {"train", "test"}

    train_result = runner.invoke(
        cli_module.cli,
        [
            "train",
            "--config-name=test",
            f"data.dataset_path={dataset_path}",
            f"data.images_dir={images_dir}",
            f"train.checkpoint_dir={checkpoint_dir}",
            f"mlflow.tracking_uri={tracking_uri}",
            f"mlflow.artifact_root={artifact_root}",
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    run_id = _run_id_from_output(train_result.output)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)
    message_losses = client.get_metric_history(run_id, "train/message_loss")
    bit_accuracies = client.get_metric_history(run_id, "train/bit_accuracy")
    assert len(message_losses) >= 2
    assert message_losses[0].value > 0.05
    assert message_losses[-1].value < 1e-3
    assert bit_accuracies[-1].value == 1.0

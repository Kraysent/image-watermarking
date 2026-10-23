from pathlib import Path

import click.testing
import mlflow
import PIL.Image
import polars

import watermarking.cli as cli_module


def test_cli_help_lists_eval() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.output


def test_eval_logs_metrics_and_images(tmp_path: Path) -> None:
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
    common = [
        f"data.dataset_path={dataset_path}",
        f"data.images_dir={images_dir}",
        "data.image_size=32",
        "data.message_bits=4",
        "data.batch_size=1",
        "encoder.channels=[4,8]",
        "decoder.channels=[4]",
        "train.device=cpu",
        f"train.checkpoint_dir={checkpoint_dir}",
        f"mlflow.tracking_uri={tracking_uri}",
        f"mlflow.artifact_root={artifact_root}",
        "mlflow.experiment_name=test",
    ]

    runner = click.testing.CliRunner()
    train_result = runner.invoke(
        cli_module.cli,
        ["train", "train.max_steps=1", *common],
    )
    assert train_result.exit_code == 0, train_result.output
    run_id = next(
        line.removeprefix("run_id=") for line in train_result.output.splitlines() if line.startswith("run_id=")
    )

    eval_result = runner.invoke(
        cli_module.cli,
        ["eval", f"--run-id={run_id}", "eval.n_images=1", "eval.device=cpu", *common],
    )
    assert eval_result.exit_code == 0, eval_result.output
    assert "eval_loss=" in eval_result.output
    assert "eval_message_loss=" in eval_result.output
    assert "eval_image_loss=" in eval_result.output

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert "eval_loss" in run.data.metrics
    assert "eval_message_loss" in run.data.metrics
    assert "eval_image_loss" in run.data.metrics

    image_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="eval/sample_00.png"))
    assert image_path.is_file()

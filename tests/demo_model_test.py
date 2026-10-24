import re
from pathlib import Path

import click.testing
import PIL.Image
import polars

import watermarking.cli as cli_module
import watermarking.demo.app as app_module
import watermarking.demo.model as model_module


def _run_id_from_output(output: str) -> str:
    match = re.search(r"\brun_id=([a-f0-9]+)\b", output)
    assert match is not None, output
    return match.group(1)


def test_cli_help_lists_demo() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "demo" in result.output


def test_embed_and_extract_roundtrip(tmp_path: Path) -> None:
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
    train_result = runner.invoke(
        cli_module.cli,
        [
            "train",
            f"data.dataset_path={dataset_path}",
            f"data.images_dir={images_dir}",
            "data.image_size=32",
            "data.message_bits=32",
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
    assert train_result.exit_code == 0, train_result.output
    run_id = _run_id_from_output(train_result.output)

    session = model_module.load_session(run_id=run_id, checkpoint_dir=checkpoint_dir, device="cpu")
    image = PIL.Image.new("RGB", (40, 40), color=(200, 100, 50))
    grid, watermarked, embedded, recovered, accuracy = model_module.embed(
        session,
        image,
        mode="text",
        message_value="hi",
    )
    assert grid.size[0] == 32 * 3
    assert watermarked.size == (32, 32)
    assert len(embedded.replace(" ", "")) == 32
    assert len(recovered.replace(" ", "")) == 32
    assert 0.0 <= accuracy <= 1.0

    bits, text, hex_value, confidence = model_module.extract(session, watermarked)
    assert bits
    assert isinstance(text, str)
    assert isinstance(hex_value, str)
    assert confidence >= 0.0

    app = app_module.build_app(session)
    assert app is not None

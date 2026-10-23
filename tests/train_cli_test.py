from pathlib import Path

import click.testing
import hydra
import PIL.Image
import polars

import watermarking.cli as cli_module
import watermarking.train.cli as train_cli


def test_cli_help_lists_train() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output


def test_hydra_overrides_compose() -> None:
    with hydra.initialize_config_dir(config_dir=str(train_cli.CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "encoder.channels=[32,64]",
                "decoder.channels=[8]",
                "train.epochs=3",
                "loss.w_img=2.5",
            ],
        )
    assert cfg.train.epochs == 3
    assert list(cfg.encoder.channels) == [32, 64]
    assert list(cfg.decoder.channels) == [8]
    assert cfg.loss.w_img == 2.5


def test_train_runs_one_step(tmp_path: Path) -> None:
    images_dir = tmp_path / "raw"
    image_path = images_dir / "sample.png"
    image_path.parent.mkdir(parents=True)
    PIL.Image.new("RGB", (40, 40), color=(12, 34, 56)).save(image_path)

    dataset_path = tmp_path / "dataset.parquet"
    polars.DataFrame({"relative_path": ["sample.png"], "split": ["train"]}).write_parquet(dataset_path)

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
        ],
    )
    assert result.exit_code == 0, result.output
    assert "epoch=1" in result.output

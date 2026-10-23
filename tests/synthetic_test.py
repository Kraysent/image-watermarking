from pathlib import Path

import click.testing
import PIL.Image
import pytest

import watermarking.cli as cli_module
import watermarking.data.synthetic as synthetic_module


def test_generate_writes_noise_pngs(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    result = synthetic_module.generate(output_dir=output_dir, count=3, size=24, seed=1)

    assert result == output_dir.resolve()
    paths = sorted(output_dir.glob("*.png"))
    assert [path.name for path in paths] == ["0000.png", "0001.png", "0002.png"]
    with PIL.Image.open(paths[0]) as image:
        assert image.size == (24, 24)
        assert image.mode == "RGB"


def test_generate_rejects_invalid_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="count"):
        synthetic_module.generate(output_dir=tmp_path, count=0)


def test_cli_synthetic_writes_images(tmp_path: Path) -> None:
    output_dir = tmp_path / "raw"
    runner = click.testing.CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["ingest", "synthetic", "--output", str(output_dir), "--count", "2", "--size", "32", "--seed", "0"],
    )
    assert result.exit_code == 0, result.output
    assert len(list(output_dir.glob("*.png"))) == 2

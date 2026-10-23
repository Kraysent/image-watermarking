from pathlib import Path

import pandera.errors
import polars
import pytest

import watermarking.data.validate as validate_module


def _row(**overrides: int | float | str) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = {
        "relative_path": "tacos/food.png",
        "file_size": 1000,
        "width": 100,
        "height": 50,
        "mean_r": 120.0,
        "mean_g": 110.0,
        "mean_b": 100.0,
        "std_r": 10.0,
        "std_g": 10.0,
        "std_b": 10.0,
        "image_hash": "abc123",
    }
    row.update(overrides)
    return row


def _write_catalog(path: Path, rows: list[dict[str, int | float | str]]) -> Path:
    polars.DataFrame(rows).write_parquet(path)
    return path


def test_validate_accepts_valid_catalog(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "catalog.parquet", [_row()])
    result = validate_module.validate(input_path=path)
    assert result.height == 1


def test_validate_rejects_small_dimensions(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "catalog.parquet", [_row(width=10, height=50)])
    with pytest.raises(pandera.errors.SchemaError):
        validate_module.validate(input_path=path)


def test_validate_rejects_extreme_aspect_ratio(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "catalog.parquet", [_row(width=200, height=21)])
    with pytest.raises(pandera.errors.SchemaError):
        validate_module.validate(input_path=path)


def test_validate_rejects_duplicate_image_hash(tmp_path: Path) -> None:
    path = _write_catalog(
        tmp_path / "catalog.parquet",
        [_row(relative_path="a.png", image_hash="same"), _row(relative_path="b.png", image_hash="same")],
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_module.validate(input_path=path)


def test_validate_rejects_flat_color_images(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "catalog.parquet", [_row(std_r=0.0, std_g=0.0, std_b=0.0)])
    with pytest.raises(pandera.errors.SchemaError):
        validate_module.validate(input_path=path)

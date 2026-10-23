from pathlib import Path

import PIL.Image
import polars

import watermarking.data.catalog as catalog_module


def test_catalog_writes_image_features(tmp_path: Path) -> None:
    image_path = tmp_path / "tacos" / "red.png"
    image_path.parent.mkdir()
    PIL.Image.new("RGB", (10, 20), (255, 0, 0)).save(image_path)
    (tmp_path / "dataset_stats.csv").write_text("class,count\n", encoding="utf-8")
    output_path = tmp_path / "catalog.parquet"

    result = catalog_module.catalog(input_dir=tmp_path, output_path=output_path)

    frame = polars.read_parquet(result)
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["relative_path"] == "tacos/red.png"
    assert row["file_size"] == image_path.stat().st_size
    assert row["width"] == 10
    assert row["height"] == 20
    assert row["mean_r"] == 255
    assert row["mean_g"] == 0
    assert row["mean_b"] == 0
    assert row["std_r"] == 0
    assert row["std_g"] == 0
    assert row["std_b"] == 0
    assert row["image_hash"]

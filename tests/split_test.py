from pathlib import Path

import polars

import watermarking.data.split as split_module


def test_split_writes_path_and_split_only(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    polars.DataFrame(
        {
            "relative_path": ["a.png", "b.png", "c.png"],
            "image_hash": ["hash-a", "hash-b", "hash-c"],
            "width": [100, 100, 100],
        }
    ).write_parquet(catalog_path)
    output_path = tmp_path / "splits.parquet"

    result = split_module.split(input_path=catalog_path, output_path=output_path, seed=0)

    frame = polars.read_parquet(result)
    assert frame.columns == ["relative_path", "split"]
    assert frame.height == 3
    assert set(frame["split"].to_list()) <= {"train", "test"}


def test_split_is_deterministic(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    polars.DataFrame(
        {
            "relative_path": [f"{i}.png" for i in range(20)],
            "image_hash": [f"hash-{i}" for i in range(20)],
        }
    ).write_parquet(catalog_path)

    first = split_module.split(input_path=catalog_path, output_path=tmp_path / "a.parquet", seed=7)
    second = split_module.split(input_path=catalog_path, output_path=tmp_path / "b.parquet", seed=7)

    assert polars.read_parquet(first).equals(polars.read_parquet(second))


def test_split_respects_train_ratio(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    polars.DataFrame(
        {
            "relative_path": [f"{i}.png" for i in range(200)],
            "image_hash": [f"hash-{i}" for i in range(200)],
        }
    ).write_parquet(catalog_path)

    result = split_module.split(
        input_path=catalog_path,
        output_path=tmp_path / "splits.parquet",
        train_ratio=0.8,
    )
    frame = polars.read_parquet(result)
    train_fraction = frame.filter(polars.col("split") == "train").height / frame.height
    assert 0.7 < train_fraction < 0.9

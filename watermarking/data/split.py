from pathlib import Path

import polars

import watermarking.data.catalog as catalog_module

DEFAULT_INPUT_PATH = catalog_module.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path("data/dataset.parquet")
DEFAULT_SEED = 0
DEFAULT_TRAIN_RATIO = 0.8
BUCKETS = 100


def split(
    *,
    input_path: Path,
    output_path: Path,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Path:
    threshold = int(train_ratio * BUCKETS)
    frame = polars.read_parquet(input_path).select(
        "relative_path",
        polars.when(polars.col("image_hash").hash(seed) % BUCKETS < threshold)
        .then(polars.lit("train"))
        .otherwise(polars.lit("test"))
        .alias("split"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    return output_path.resolve()

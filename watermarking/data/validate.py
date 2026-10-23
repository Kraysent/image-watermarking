from pathlib import Path

import pandera.polars as pa
import polars

DEFAULT_INPUT_PATH = Path("data/catalog/popular-street-foods.parquet")
MIN_SIZE = 20
MAX_ASPECT_RATIO = 8.0
MIN_STD = 1.0


class CatalogSchema(pa.DataFrameModel):
    width: int = pa.Field(gt=MIN_SIZE)
    height: int = pa.Field(gt=MIN_SIZE)
    std_r: float
    std_g: float
    std_b: float
    image_hash: str = pa.Field(unique=True)

    @pa.dataframe_check
    def aspect_ratio_ok(self, data: pa.PolarsData) -> polars.LazyFrame:
        width = polars.col("width")
        height = polars.col("height")
        return data.lazyframe.select(
            polars.max_horizontal(width, height) / polars.min_horizontal(width, height) <= MAX_ASPECT_RATIO
        )

    @pa.dataframe_check
    def has_color_variance(self, data: pa.PolarsData) -> polars.LazyFrame:
        return data.lazyframe.select(polars.max_horizontal("std_r", "std_g", "std_b") > MIN_STD)


def validate(*, input_path: Path) -> polars.DataFrame:
    frame = polars.read_parquet(input_path)
    return CatalogSchema.validate(frame)

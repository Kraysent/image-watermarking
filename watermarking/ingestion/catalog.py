from pathlib import Path

import PIL.Image
import polars

import watermarking.ingestion.features as features

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_OUTPUT_PATH = Path("data/catalog/popular-street-foods.parquet")


def catalog(*, input_dir: Path, output_path: Path) -> Path:
    rows: list[dict[str, int | float | str]] = []

    for path in _image_paths(input_dir):
        with PIL.Image.open(path) as image:
            rgb = image.convert("RGB")
            row: dict[str, int | float | str] = {"relative_path": path.relative_to(input_dir).as_posix()}
            for feature in features.FEATURES:
                row.update(feature(path, rgb))
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    polars.DataFrame(rows).write_parquet(output_path)
    return output_path.resolve()


def _image_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

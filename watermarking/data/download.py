from pathlib import Path

import kagglehub

DATASET_HANDLE = "nikolasgegenava/popular-street-foods"
DEFAULT_OUTPUT_DIR = Path("data/raw")


def download(*, output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(output_dir),
        force_download=force,
    )
    return output_dir.resolve()

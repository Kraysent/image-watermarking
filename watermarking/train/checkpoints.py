from pathlib import Path


def latest_checkpoint_path(run_dir: Path) -> Path | None:
    checkpoints = list(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None

    def epoch_number(path: Path) -> int:
        return int(path.stem.removeprefix("epoch_"))

    return max(checkpoints, key=epoch_number)

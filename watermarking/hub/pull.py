from pathlib import Path

import huggingface_hub


def latest_checkpoint_filename(repo_id: str) -> str:
    api = huggingface_hub.HfApi()
    filenames = [
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type="model")
        if Path(path).name.startswith("epoch_") and path.endswith(".pt")
    ]
    if not filenames:
        raise FileNotFoundError(f"checkpoint not found in Hugging Face repo: {repo_id}")

    def epoch_number(path: str) -> int:
        return int(Path(path).stem.removeprefix("epoch_"))

    return max(filenames, key=epoch_number)


def download_latest_checkpoint(repo_id: str) -> Path:
    filename = latest_checkpoint_filename(repo_id)
    return Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
    )

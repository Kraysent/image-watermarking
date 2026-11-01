from pathlib import Path
from typing import Any

import pytest

import watermarking.hub.pull as pull_module


def test_latest_checkpoint_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            assert repo_id == "org/model"
            assert repo_type == "model"
            return [".gitattributes", "epoch_3.pt", "epoch_12.pt", "readme.md"]

    monkeypatch.setattr(pull_module.huggingface_hub, "HfApi", FakeApi)
    assert pull_module.latest_checkpoint_filename("org/model") == "epoch_12.pt"


def test_latest_checkpoint_filename_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            return [".gitattributes"]

    monkeypatch.setattr(pull_module.huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(FileNotFoundError, match="org/model"):
        pull_module.latest_checkpoint_filename("org/model")


def test_download_latest_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "epoch_12.pt"
    checkpoint_path.write_bytes(b"ckpt")

    class FakeApi:
        def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
            return ["epoch_3.pt", "epoch_12.pt"]

    def fake_download(**kwargs: Any) -> str:
        assert kwargs["repo_id"] == "org/model"
        assert kwargs["filename"] == "epoch_12.pt"
        assert kwargs["repo_type"] == "model"
        return str(checkpoint_path)

    monkeypatch.setattr(pull_module.huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(pull_module.huggingface_hub, "hf_hub_download", fake_download)
    assert pull_module.download_latest_checkpoint("org/model") == checkpoint_path

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest
from omegaconf import DictConfig

import watermarking.train.loop as loop_module
import watermarking.train.tracking.prefect as prefect_module


@pytest.fixture(autouse=True)
def _run_training_without_prefect_server(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.get_closest_marker("prefect_api") is not None:
        return

    def run_train(
        cfg: DictConfig,
        resume_run_id: str | None = None,
        continue_run_id: str | None = None,
        continue_epochs: int | None = None,
    ) -> str:
        return loop_module.train(
            cfg,
            resume_run_id=resume_run_id,
            continue_run_id=continue_run_id,
            continue_epochs=continue_epochs,
        )

    monkeypatch.setattr(prefect_module, "run_train", run_train)


@pytest.fixture(scope="session")
def tracking_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("mlflow")
    artifacts = root / "mlartifacts"
    artifacts.mkdir()
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{root / 'mlflow.db'}",
            "--artifacts-destination",
            str(artifacts),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_tracking_server(url, process)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tracking_server(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30
    health = f"{url}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"mlflow server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise RuntimeError("mlflow server did not become ready")

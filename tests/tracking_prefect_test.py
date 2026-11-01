import ssl
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import watermarking.train.tracking.prefect as prefect_module


def test_httpx_settings_empty_without_certs() -> None:
    cfg = OmegaConf.create({"tracking": {"client_cert_path": None, "server_cert_path": None}})
    assert prefect_module.httpx_settings_from_tracking(cfg) is None


def test_httpx_settings_loads_client_cert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client_cert = tmp_path / "client.pem"
    client_cert.write_text("cert")
    loaded: list[str] = []

    class _Context:
        def load_cert_chain(self, certfile: str, keyfile: str | None = None, password: object = None) -> None:
            loaded.append(certfile)

    def create_default_context(*args: object, **kwargs: object) -> _Context:
        assert kwargs.get("cafile") is None
        assert args == ()
        return _Context()

    monkeypatch.setattr(ssl, "create_default_context", create_default_context)
    cfg = OmegaConf.create(
        {
            "tracking": {
                "client_cert_path": str(client_cert),
                "server_cert_path": None,
            }
        }
    )
    settings = prefect_module.httpx_settings_from_tracking(cfg)
    assert settings is not None
    assert loaded == [str(client_cert)]


def test_httpx_settings_uses_server_cafile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server_cert = tmp_path / "ca.pem"
    server_cert.write_text("ca")
    seen: dict[str, object] = {}

    class _Context:
        def load_cert_chain(self, certfile: str, keyfile: str | None = None, password: object = None) -> None:
            raise AssertionError("client cert should not be loaded")

    def create_default_context(*args: object, **kwargs: object) -> _Context:
        seen["kwargs"] = kwargs
        return _Context()

    monkeypatch.setattr(ssl, "create_default_context", create_default_context)
    cfg = OmegaConf.create(
        {
            "tracking": {
                "client_cert_path": None,
                "server_cert_path": str(server_cert),
            }
        }
    )
    settings = prefect_module.httpx_settings_from_tracking(cfg)
    assert settings is not None
    assert seen["kwargs"] == {"cafile": str(server_cert)}


@pytest.mark.prefect_api
def test_run_train_requires_prefect_api_url() -> None:
    cfg = OmegaConf.create(
        {
            "tracking": {
                "prefect_api_url": None,
                "client_cert_path": None,
                "server_cert_path": None,
            }
        }
    )
    with pytest.raises(ValueError, match="prefect_api_url"):
        prefect_module.run_train(cfg)


@pytest.mark.prefect_api
def test_run_train_points_prefect_at_tracking_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def train_flow(
        cfg: object,
        resume_run_id: str | None = None,
        continue_run_id: str | None = None,
        continue_epochs: int | None = None,
    ) -> str:
        seen["api_url"] = prefect_module.prefect.settings.PREFECT_API_URL.value()
        seen["resume_run_id"] = resume_run_id
        seen["continue_run_id"] = continue_run_id
        seen["continue_epochs"] = continue_epochs
        return "run-id"

    monkeypatch.setattr(prefect_module, "train_flow", train_flow)
    cfg = OmegaConf.create(
        {
            "tracking": {
                "prefect_api_url": "https://prefect.example/api",
                "client_cert_path": None,
                "server_cert_path": None,
            }
        }
    )
    assert prefect_module.run_train(cfg, resume_run_id="abc", continue_epochs=None) == "run-id"
    assert seen == {
        "api_url": "https://prefect.example/api",
        "resume_run_id": "abc",
        "continue_run_id": None,
        "continue_epochs": None,
    }

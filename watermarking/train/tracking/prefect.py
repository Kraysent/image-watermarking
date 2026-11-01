import ssl
from collections.abc import Iterator
from contextlib import contextmanager

import prefect
import prefect.settings
from omegaconf import DictConfig

import watermarking.train.loop as loop_module


def httpx_settings_from_tracking(cfg: DictConfig) -> dict[str, ssl.SSLContext] | None:
    client_cert_path = _cert_path(cfg, "client_cert_path")
    server_cert_path = _cert_path(cfg, "server_cert_path")
    if client_cert_path is None and server_cert_path is None:
        return None
    if server_cert_path is None:
        ctx = ssl.create_default_context()
    else:
        ctx = ssl.create_default_context(cafile=server_cert_path)
    if client_cert_path is not None:
        ctx.load_cert_chain(client_cert_path)
    return {"verify": ctx}


def run_train(
    cfg: DictConfig,
    resume_run_id: str | None = None,
    continue_run_id: str | None = None,
    continue_epochs: int | None = None,
) -> str:
    api_url = _require_api_url(cfg)
    with _bootstrap_client_ssl(cfg):
        with prefect.settings.temporary_settings(updates={prefect.settings.PREFECT_API_URL: api_url}):
            return train_flow(
                cfg,
                resume_run_id=resume_run_id,
                continue_run_id=continue_run_id,
                continue_epochs=continue_epochs,
            )


@prefect.flow(name="train", log_prints=True)
def train_flow(
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


def _require_api_url(cfg: DictConfig) -> str:
    value = cfg.tracking.get("prefect_api_url")
    if value is None or str(value).strip() == "":
        raise ValueError("tracking.prefect_api_url is required")
    return str(value)


def _cert_path(cfg: DictConfig, key: str) -> str | None:
    value = cfg.tracking.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


@contextmanager
def _bootstrap_client_ssl(cfg: DictConfig) -> Iterator[None]:
    client_cert_path = _cert_path(cfg, "client_cert_path")
    server_cert_path = _cert_path(cfg, "server_cert_path")
    if client_cert_path is None and server_cert_path is None:
        yield
        return

    original = ssl.create_default_context

    def create_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | bytes | None = None,
        capath: str | bytes | None = None,
        cadata: str | bytes | None = None,
    ) -> ssl.SSLContext:
        if server_cert_path is not None:
            cafile = server_cert_path
            capath = None
            cadata = None
        ctx = original(purpose, cafile=cafile, capath=capath, cadata=cadata)
        if client_cert_path is not None:
            ctx.load_cert_chain(client_cert_path)
        return ctx

    ssl.create_default_context = create_default_context
    try:
        yield
    finally:
        ssl.create_default_context = original

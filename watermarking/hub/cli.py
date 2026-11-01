from pathlib import Path

import click
import hydra

import watermarking.hub.push as push_module

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@click.command(
    "push",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
@click.option("--run-id", required=True, type=str)
@click.option("--private", is_flag=True)
@click.option("--config-name", default="config", show_default=True)
@click.pass_context
def push(ctx: click.Context, run_id: str, private: bool, config_name: str) -> None:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(ctx.args))
    repo_id = cfg.tracking.get("huggingface_repo")
    if repo_id is None:
        raise click.UsageError("tracking.huggingface_repo is required")
    client_cert_path = cfg.tracking.get("client_cert_path")
    server_cert_path = cfg.tracking.get("server_cert_path")
    url = push_module.push_checkpoint(
        run_id=run_id,
        repo_id=str(repo_id),
        tracking_uri=str(cfg.tracking.tracking_uri),
        private=private,
        client_cert_path=None if client_cert_path is None else str(client_cert_path),
        server_cert_path=None if server_cert_path is None else str(server_cert_path),
    )
    click.echo(url)

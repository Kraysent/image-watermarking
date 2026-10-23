from pathlib import Path

import click
import hydra
from omegaconf import OmegaConf

import watermarking.demo.app as app_module
import watermarking.demo.model as model_module

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@click.command(
    "demo",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
@click.option("--run-id", required=True, type=str)
@click.pass_context
def demo(ctx: click.Context, run_id: str) -> None:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=list(ctx.args))
    click.echo(OmegaConf.to_yaml(cfg))
    session = model_module.load_session(
        run_id=run_id,
        checkpoint_dir=Path(cfg.train.checkpoint_dir),
        device=cfg.demo.device,
    )
    app = app_module.build_app(session)
    app.launch(server_name=cfg.demo.host, server_port=cfg.demo.port)

from pathlib import Path

import click
import hydra
from omegaconf import OmegaConf

import watermarking.train.loop as loop_module

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option("--config-name", default="config", show_default=True)
@click.option("--run-id", default=None, type=str)
@click.pass_context
def train(ctx: click.Context, config_name: str, run_id: str | None) -> None:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(ctx.args))
    click.echo(OmegaConf.to_yaml(cfg))
    loop_module.train(cfg, resume_run_id=run_id)

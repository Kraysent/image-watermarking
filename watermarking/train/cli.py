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
@click.pass_context
def train(ctx: click.Context) -> None:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=list(ctx.args))
    click.echo(OmegaConf.to_yaml(cfg))
    loop_module.train(cfg)

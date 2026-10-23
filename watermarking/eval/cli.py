from pathlib import Path

import click
import hydra
from omegaconf import OmegaConf

import watermarking.eval.loop as loop_module

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@click.command(
    "eval",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
@click.option("--run-id", required=True, type=str)
@click.pass_context
def evaluate(ctx: click.Context, run_id: str) -> None:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=list(ctx.args))
    click.echo(OmegaConf.to_yaml(cfg))
    loop_module.evaluate(cfg, run_id=run_id)

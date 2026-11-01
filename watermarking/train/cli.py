from pathlib import Path

import click
import hydra
from omegaconf import OmegaConf

import watermarking.train.loop as loop_module
import watermarking.train.tracking.prefect as prefect_tracking

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option("--config-name", default="config", show_default=True)
@click.option("--run-id", default=None, type=str)
@click.option(
    "--continue",
    "continue_from",
    nargs=2,
    type=click.Tuple([str, int]),
    default=None,
    help="Continue an existing run for more epochs: RUN_ID EPOCHS",
)
@click.pass_context
def train(
    ctx: click.Context,
    config_name: str,
    run_id: str | None,
    continue_from: tuple[str, int] | None,
) -> None:
    if run_id is not None and continue_from is not None:
        raise click.UsageError("--run-id and --continue are mutually exclusive")
    if continue_from is not None and continue_from[1] <= 0:
        raise click.UsageError("--continue epochs must be positive")

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(ctx.args))

    if continue_from is not None:
        continue_run_id, continue_epochs = continue_from
        cfg = loop_module.cfg_from_checkpoint(cfg, continue_run_id)
        click.echo(OmegaConf.to_yaml(cfg))
        prefect_tracking.run_train(
            cfg,
            continue_run_id=continue_run_id,
            continue_epochs=continue_epochs,
        )
        return

    click.echo(OmegaConf.to_yaml(cfg))
    prefect_tracking.run_train(cfg, resume_run_id=run_id)

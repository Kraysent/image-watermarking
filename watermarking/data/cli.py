from pathlib import Path

import click

import watermarking.data.catalog as catalog_module
import watermarking.data.download as download_module
import watermarking.data.split as split_module
import watermarking.data.validate as validate_module


@click.group()
def ingest() -> None:
    pass


@ingest.command()
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=download_module.DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option("--force", is_flag=True)
def download(output_dir: Path, force: bool) -> None:
    path = download_module.download(output_dir=output_dir, force=force)
    click.echo(path)


@ingest.command()
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=download_module.DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=catalog_module.DEFAULT_OUTPUT_PATH,
    show_default=True,
)
def catalog(input_dir: Path, output_path: Path) -> None:
    path = catalog_module.catalog(input_dir=input_dir, output_path=output_path)
    click.echo(path)


@ingest.command("validate")
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=validate_module.DEFAULT_INPUT_PATH,
    show_default=True,
)
def validate(input_path: Path) -> None:
    validate_module.validate(input_path=input_path)
    click.echo(f"OK: {input_path.resolve()}")


@ingest.command("split")
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=split_module.DEFAULT_INPUT_PATH,
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=split_module.DEFAULT_OUTPUT_PATH,
    show_default=True,
)
@click.option("--seed", type=int, default=split_module.DEFAULT_SEED, show_default=True)
@click.option(
    "--train-ratio",
    type=float,
    default=split_module.DEFAULT_TRAIN_RATIO,
    show_default=True,
)
def split(input_path: Path, output_path: Path, seed: int, train_ratio: float) -> None:
    path = split_module.split(
        input_path=input_path,
        output_path=output_path,
        seed=seed,
        train_ratio=train_ratio,
    )
    click.echo(path)

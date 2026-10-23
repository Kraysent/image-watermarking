from pathlib import Path

import click

import watermarking.ingestion.catalog as catalog_module
import watermarking.ingestion.download as download_module


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

from pathlib import Path

import click

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

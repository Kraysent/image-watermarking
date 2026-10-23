import click

import watermarking.ingestion.cli as ingestion_cli


@click.group()
def cli() -> None:
    pass


cli.add_command(ingestion_cli.ingest)

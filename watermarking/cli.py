import click

import watermarking.data.cli as data_cli


@click.group()
def cli() -> None:
    pass


cli.add_command(data_cli.ingest)

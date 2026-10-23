import click

import watermarking.data.cli as data_cli
import watermarking.train.cli as train_cli


@click.group()
def cli() -> None:
    pass


cli.add_command(data_cli.ingest)
cli.add_command(train_cli.train)

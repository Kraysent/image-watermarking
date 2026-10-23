import click

import watermarking.data.cli as data_cli
import watermarking.demo.cli as demo_cli
import watermarking.eval.cli as eval_cli
import watermarking.train.cli as train_cli


@click.group()
def cli() -> None:
    pass


cli.add_command(data_cli.ingest)
cli.add_command(train_cli.train)
cli.add_command(eval_cli.evaluate)
cli.add_command(demo_cli.demo)

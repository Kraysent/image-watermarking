import click.testing

import watermarking.demo.cli as demo_cli


def test_cli_help_shows_usage() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(demo_cli.demo, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output

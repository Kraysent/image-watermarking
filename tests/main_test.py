import click.testing

import watermarking.cli as cli_module


def test_cli_help_shows_usage() -> None:
    runner = click.testing.CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output

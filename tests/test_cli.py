from typer.testing import CliRunner

from rheology_paper_ocr.cli import app


def test_cli_help_lists_primary_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "serve" in result.output

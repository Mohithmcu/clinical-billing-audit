"""Unit tests for the CLI interface."""

from __future__ import annotations

import click
from click.testing import CliRunner
import pytest

from clinical_billing_audit.cli import main


def test_cli_help():
    """Test top-level CLI help command."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Clinical Billing Audit" in result.output
    assert "data-quality" in result.output
    assert "extract-entities" in result.output
    assert "train" in result.output
    assert "detect-anomalies" in result.output
    assert "evaluate" in result.output


def test_cli_subcommand_help():
    """Test help for subcommands."""
    runner = CliRunner()
    for subcmd in ["data-quality", "extract-entities", "train", "detect-anomalies", "evaluate", "export-powerbi"]:
        result = runner.invoke(main, [subcmd, "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

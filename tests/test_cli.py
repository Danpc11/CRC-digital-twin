"""
Tests para cli.py -- verifica que cada subcomando parsea sus argumentos
correctamente y despacha la llamada esperada, SIN correr de verdad los
scripts pesados subyacentes (eso ya esta cubierto por los tests de cada
script individual). subprocess.run se mockea para capturar que
comando se habria ejecutado, no para correrlo.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli


def test_build_parser_has_all_expected_subcommands():
    parser = cli.build_parser()
    subcommands = {a.dest for a in parser._subparsers._group_actions[0]._choices_actions}
    esperados = {"demo", "calibrate", "classify", "validate-external", "pooled-cox",
                 "prognosis", "simulate-treatment", "app", "test"}
    assert esperados.issubset(subcommands)


def test_demo_subcommand_parses_default_output():
    parser = cli.build_parser()
    args = parser.parse_args(["demo"])
    assert args.output == "results_demo"
    assert args.func == cli.cmd_demo


def test_calibrate_requires_input_and_output():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate"])  # faltan --input/--output requeridos


def test_calibrate_parses_skip_survival_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["calibrate", "--input", "a.tsv", "--output", "out/", "--skip-survival"])
    assert args.skip_survival is True
    assert args.input == "a.tsv"


def test_pooled_cox_accepts_multiple_cohort_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "pooled-cox", "--cohort", "A", "a.tsv", "--cohort", "B", "b.tsv",
    ])
    assert args.cohort == [["A", "a.tsv"], ["B", "b.tsv"]]


@patch("cli.subprocess.run")
def test_cmd_calibrate_dispatches_run_pipeline_with_correct_args(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    args = cli.build_parser().parse_args(
        ["calibrate", "--input", "in.tsv", "--output", "out/", "--skip-survival"])
    cli.cmd_calibrate(args)

    called_cmd = mock_run.call_args[0][0]
    assert "run_pipeline.py" in called_cmd[1]
    assert "--input" in called_cmd and "in.tsv" in called_cmd
    assert "--skip-survival" in called_cmd


@patch("cli.subprocess.run")
def test_cmd_calibrate_exits_cleanly_on_nonzero_returncode(mock_run):
    """Si el subproceso falla, cli.py debe propagar el codigo de salida
    con sys.exit limpio -- ANTES de este fix, cmd_calibrate usaba
    subprocess.run(check=True) directo, que en fallo real lanza un
    traceback crudo de CalledProcessError en vez de salida limpia,
    inconsistente con el resto de los subcomandos (que usan
    _run_module/_run_command, con sys.exit(returncode))."""
    mock_run.return_value = MagicMock(returncode=1)
    args = cli.build_parser().parse_args(["calibrate", "--input", "in.tsv", "--output", "out/"])
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_calibrate(args)
    assert exc_info.value.code == 1


@patch("cli.subprocess.run")
def test_cmd_pooled_cox_expands_multiple_cohort_flags_correctly(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    args = cli.build_parser().parse_args([
        "pooled-cox", "--cohort", "A", "a.tsv", "--cohort", "B", "b.tsv", "--output", "out/",
    ])
    cli.cmd_pooled_cox(args)
    called_cmd = mock_run.call_args[0][0]
    assert called_cmd.count("--cohort") == 2
    assert "A" in called_cmd and "a.tsv" in called_cmd
    assert "B" in called_cmd and "b.tsv" in called_cmd


def test_app_subcommand_default_port_and_address():
    parser = cli.build_parser()
    args = parser.parse_args(["app"])
    assert args.port == 8501
    assert args.address == "0.0.0.0"


@patch("cli.subprocess.run")
def test_cmd_test_propagates_pytest_failure_exit_code(mock_run):
    """
    Regresion de un hallazgo real de revision externa: cmd_test usaba
    subprocess.run(check=False) y DESCARTABA el resultado por completo
    -- si pytest fallaba, cli.py test igual reportaba exito (codigo 0)
    al shell/CI que lo invocara. Debe propagar el codigo real.
    """
    mock_run.return_value = MagicMock(returncode=1)
    args = cli.build_parser().parse_args(["test"])
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_test(args)
    assert exc_info.value.code == 1


@patch("cli.subprocess.run")
def test_cmd_test_exits_zero_when_pytest_passes(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    args = cli.build_parser().parse_args(["test"])
    cli.cmd_test(args)  # no debe lanzar SystemExit


def test_dynamics_diagnostics_subcommand_is_registered():
    parser = cli.build_parser()
    subcommands = {a.dest for a in parser._subparsers._group_actions[0]._choices_actions}
    assert "dynamics-diagnostics" in subcommands


@patch("cli.subprocess.run")
def test_modern_hopfield_dispatches_stabilized_sweep(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    args = cli.build_parser().parse_args([
        "modern-hopfield", "--patterns", "p.tsv", "--compare-stabilized-sweep",
        "--forcing-candidates", "0.7", "1.5", "3.0",
    ])
    cli.cmd_modern_hopfield(args)
    called_cmd = mock_run.call_args[0][0]
    assert "--compare-stabilized-sweep" in called_cmd
    start = called_cmd.index("--forcing-candidates") + 1
    end = called_cmd.index("--withdrawal-time")
    assert called_cmd[start:end] == ["0.7", "1.5", "3.0"]
    assert called_cmd[end + 1] == "30.0"


@patch("cli.subprocess.run")
def test_cmd_dynamics_diagnostics_dispatches_correct_args(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    args = cli.build_parser().parse_args([
        "dynamics-diagnostics", "--patterns", "p.tsv", "--beta", "1.5", "--n-samples", "100",
    ])
    cli.cmd_dynamics_diagnostics(args)
    called_cmd = mock_run.call_args[0][0]
    assert "dynamics_diagnostics.py" in called_cmd[1]
    assert "--patterns" in called_cmd and "p.tsv" in called_cmd
    assert "--beta" in called_cmd and "1.5" in called_cmd


@patch("cli.subprocess.run")
def test_cmd_dynamics_diagnostics_dispatches_full_flag(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    args = cli.build_parser().parse_args([
        "dynamics-diagnostics", "--patterns", "p.tsv", "--full",
    ])
    cli.cmd_dynamics_diagnostics(args)
    assert "--full" in mock_run.call_args[0][0]

"""
cli.py -- punto de entrada unico para el pipeline ColoQ / crc-digital-twin.

Envuelve los scripts individuales de src/ en subcomandos consistentes,
para no tener que recordar la ruta y los flags de cada uno.

USO:
    python3 cli.py --help
    python3 cli.py demo                      # genera datos sinteticos y corre todo
    python3 cli.py calibrate --input X.tsv --output results/
    python3 cli.py classify --patterns P.tsv --input X.tsv --output results/
    python3 cli.py validate-external --patterns P.tsv --input X.tsv --output results/
    python3 cli.py pooled-cox --cohort NOMBRE ruta.tsv [--cohort ...] --output results/
    python3 cli.py cox-diagnostics --input scored.tsv [--input ...] --adjust-stage --output results/
    python3 cli.py dynamics-diagnostics --patterns P.tsv --output results/
    python3 cli.py prognosis --patterns P.tsv
    python3 cli.py simulate-treatment --patterns P.tsv --treatment immunotherapy_antiPD1
    python3 cli.py app                       # lanza la interfaz Streamlit

Cada subcomando delega en el modulo correspondiente de src/ -- este
archivo no duplica logica, solo orquesta.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _run_command(cmd: list):
    """Ejecuta un comando como subproceso, propagando su codigo de salida
    de forma limpia (sys.exit), sin traceback crudo de CalledProcessError."""
    print(f"$ {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _run_module(module_name: str, args: list):
    """Ejecuta un script de src/ como subproceso, propagando su codigo de salida."""
    _run_command([sys.executable, str(SRC / module_name)] + args)


def cmd_demo(args):
    """Pipeline completo sobre datos sinteticos -- no requiere credenciales ni descargas."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    print("=== 1/4 Generando cohorte sintetica ===")
    _run_module("synthetic_data.py", [])

    synth = data_dir / "synthetic_cohort.tsv"
    print("\n=== 2/4 Calibracion + validacion de supervivencia ===")
    subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"),
                     "--input", str(synth), "--output", str(out)], check=True)

    patterns = out / "calibrated_patterns.tsv"
    print("\n=== 3/4 Concordancia con etiquetas de referencia ===")
    _run_module("concordance_analysis.py", ["--input", str(out / "scored_cohort.tsv")])

    print("\n=== 4/4 Pronostico longitudinal (con evidencia y tratamientos aplicables) ===")
    _run_module("prognosis_demo.py", ["--patterns", str(patterns),
                                       "--output", str(out / "prognosis_demo.png")])

    print(f"\nDemo completa. Resultados en: {out}")


def cmd_calibrate(args):
    extra = ["--input", args.input, "--output", args.output]
    if args.skip_survival:
        extra.append("--skip-survival")
    _run_command([sys.executable, str(ROOT / "run_pipeline.py")] + extra)


def cmd_classify(args):
    _run_module("external_validation.py", [
        "--patterns", args.patterns, "--input", args.input, "--output", args.output,
    ])


def cmd_validate_external(args):
    argv = ["--patterns", args.patterns, "--input", args.input, "--output", args.output]
    if args.duration_col:
        argv += ["--duration-col", args.duration_col]
    if args.event_col:
        argv += ["--event-col", args.event_col]
    _run_module("external_validation.py", argv)


def cmd_pooled_cox(args):
    argv = []
    for name, path in args.cohort:
        argv += ["--cohort", name, path]
    argv += ["--output", args.output]
    _run_module("pooled_cox_validation.py", argv)


def cmd_cox_diagnostics(args):
    argv = []
    for path in args.input:
        argv += ["--input", path]
    if args.reference:
        argv += ["--reference", args.reference]
    if args.adjust_stage:
        argv.append("--adjust-stage")
    if args.no_stratify:
        argv.append("--no-stratify")
    if args.output:
        argv += ["--output", args.output]
    _run_module("cox_diagnostics.py", argv)


def cmd_dynamics_diagnostics(args):
    argv = ["--patterns", args.patterns, "--beta", str(args.beta), "--n-samples", str(args.n_samples)]
    if args.find_interval:
        argv.append("--find-interval")
        argv += ["--beta-min-busqueda", str(args.beta_min_busqueda)]
        argv += ["--beta-max-busqueda", str(args.beta_max_busqueda)]
        argv += ["--n-steps-busqueda", str(args.n_steps_busqueda)]
        argv += ["--correlation-threshold", str(args.correlation_threshold)]
        argv += ["--origin-threshold", str(args.origin_threshold)]
    if args.full:
        argv.append("--full")
    if args.output:
        argv += ["--output", args.output]
    _run_module("dynamics_diagnostics.py", argv)


def cmd_prognosis(args):
    argv = ["--patterns", args.patterns]
    if args.recurrence_target:
        argv += ["--recurrence-target", args.recurrence_target]
    if args.output:
        argv += ["--output", args.output]
    _run_module("prognosis_demo.py", argv)


def cmd_simulate_treatment(args):
    argv = ["--patterns", args.patterns, "--treatment", args.treatment]
    if args.recurrence_target:
        argv += ["--recurrence-target", args.recurrence_target]
    if args.ras_braf_wildtype:
        argv += ["--ras-braf-wildtype", args.ras_braf_wildtype]
    if args.output:
        argv += ["--output", args.output]
    _run_module("treatment_simulation_demo.py", argv)


def cmd_app(args):
    app_path = ROOT / "app.py"
    if not app_path.exists():
        sys.exit(f"No se encontro {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.port", str(args.port), "--server.address", args.address]
    print(f"$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd)


def cmd_test(args):
    _run_command([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-v"])


def build_parser():
    p = argparse.ArgumentParser(
        prog="coloq",
        description="ColoQ / crc-digital-twin -- gemelo digital mecanicista de cancer colorrectal "
                    "para paneles RT-qPCR. Ver README.md y PROJECT_STATUS.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("demo", help="Pipeline completo sobre datos sinteticos (sin credenciales)")
    s.add_argument("--output", default="results_demo")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("calibrate", help="Calibrar patrones de atractor contra una cohorte etiquetada")
    s.add_argument("--input", required=True, help="TSV con esquema de calibration.py")
    s.add_argument("--output", required=True)
    s.add_argument("--skip-survival", action="store_true")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("classify", help="Clasificar una cohorte con patrones YA calibrados (sin recalibrar)")
    s.add_argument("--patterns", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_classify)

    s = sub.add_parser("validate-external", help="Validacion externa con patrones congelados")
    s.add_argument("--patterns", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--duration-col")
    s.add_argument("--event-col")
    s.set_defaults(func=cmd_validate_external)

    s = sub.add_parser("pooled-cox", help="Cox estratificado combinando varias cohortes externas")
    s.add_argument("--cohort", action="append", nargs=2, metavar=("NOMBRE", "SCORED_TSV"), required=True)
    s.add_argument("--output", default="results_pooled_cox")
    s.set_defaults(func=cmd_pooled_cox)

    s = sub.add_parser("cox-diagnostics",
                        help="Diagnosticos formales del modelo de Cox (Schoenfeld, influyentes, heterogeneidad)")
    s.add_argument("--input", action="append", required=True, metavar="SCORED_TSV")
    s.add_argument("--reference")
    s.add_argument("--adjust-stage", action="store_true",
                    help="Incluir estadio armonizado como covariable")
    s.add_argument("--no-stratify", action="store_true",
                    help="NO estratificar por cohorte (solo comparacion/debug)")
    s.add_argument("--output")
    s.set_defaults(func=cmd_cox_diagnostics)

    s = sub.add_parser("dynamics-diagnostics",
                        help="Equilibrios y estabilidad reales de la dinamica no lineal (jacobiano, cuencas de atraccion)")
    s.add_argument("--patterns", required=True)
    s.add_argument("--beta", type=float, default=2.0)
    s.add_argument("--n-samples", type=int, default=300)
    s.add_argument("--find-interval", action="store_true",
                    help="Buscar automaticamente un intervalo de beta con 4 atractores CMS genuinos")
    s.add_argument("--beta-min-busqueda", type=float, default=0.1)
    s.add_argument("--beta-max-busqueda", type=float, default=15.0,
                    help="Si el resultado anterior dijo 'limite superior NO determinado', "
                         "subir este valor y repetir")
    s.add_argument("--n-steps-busqueda", type=int, default=150)
    s.add_argument("--correlation-threshold", type=float, default=0.8,
                    help="ARBITRARIO -- hallazgo real: cambia la conclusion cualitativa "
                         "de 'existe/no existe un beta valido', reportar siempre junto al valor")
    s.add_argument("--origin-threshold", type=float, default=0.5)
    s.add_argument("--full", action="store_true",
                    help="Correr y guardar analisis avanzados de cuencas y atractores espurios")
    s.add_argument("--output")
    s.set_defaults(func=cmd_dynamics_diagnostics)

    s = sub.add_parser("prognosis", help="Demo de pronostico longitudinal post-quirurgico")
    s.add_argument("--patterns", required=True)
    s.add_argument("--recurrence-target")
    s.add_argument("--output")
    s.set_defaults(func=cmd_prognosis)

    s = sub.add_parser("simulate-treatment", help="Simulacion contrafactual con/sin tratamiento")
    s.add_argument("--patterns", required=True)
    s.add_argument("--treatment", required=True)
    s.add_argument("--recurrence-target")
    s.add_argument("--ras-braf-wildtype", choices=["true", "false", "unknown"])
    s.add_argument("--output")
    s.set_defaults(func=cmd_simulate_treatment)

    s = sub.add_parser("app", help="Lanzar la interfaz web (Streamlit)")
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--address", default="0.0.0.0")
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("test", help="Correr la suite de regresion")
    s.set_defaults(func=cmd_test)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
app.py -- interfaz web (Streamlit) para ColoQ / crc-digital-twin.

Cuatro pestanas:
  1. Clasificar    -- sube un TSV de expresion, clasifica contra patrones calibrados
  2. Pronostico    -- trayectoria longitudinal post-quirurgica + alerta + evidencia
  3. Tratamiento   -- simulacion contrafactual con/sin intervencion
  4. Acerca de     -- panel, evidencia acumulada y limitaciones explicitas

Lanzar con:  python3 cli.py app     (o: streamlit run app.py)

NOTA DE ALCANCE: herramienta de investigacion. No es un dispositivo
medico ni una ayuda a la decision clinica validada -- ver pestana
"Acerca de" y PROJECT_STATUS.md.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from attractor_model import build_model_from_patterns
from calibration import (
    calibrate_patterns_from_data,
    infer_gene_columns,
    load_calibrated_patterns,
    zscore_genes,
)
from prognosis import detect_recurrence_signal, hazard_from_trajectory
from prognosis_demo import (
    EVIDENCE_STRENGTH,
    applicable_treatments,
    classify_current_state,
    simulate_longitudinal_patient,
)
from survival_validation import score_cohort
from treatment_perturbation import TREATMENT_MECHANISMS, describe_treatment
from treatment_simulation_demo import simulate_with_optional_treatment

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
EVIDENCE_COLOR = {"fuerte": "🟢", "debil": "🟡", "sin evidencia": "🔴", "referencia": "⚪"}

st.set_page_config(page_title="ColoQ — gemelo digital CRC", page_icon="🧬", layout="wide")


@st.cache_data
def load_patterns_cached(path_str: str):
    return load_calibrated_patterns(path_str)


def find_default_patterns() -> str | None:
    """Busca un calibrated_patterns.tsv en las rutas de resultados tipicas."""
    for candidate in sorted(ROOT.glob("results*/calibrated_patterns.tsv")):
        return str(candidate)
    return None


def patterns_selector(key_suffix: str):
    """Widget compartido: elegir archivo de patrones (por ruta o subida)."""
    default = find_default_patterns()
    source = st.radio(
        "Patrones calibrados",
        ["Usar ruta en el servidor", "Subir archivo"],
        horizontal=True,
        key=f"src_{key_suffix}",
    )
    if source == "Subir archivo":
        up = st.file_uploader("calibrated_patterns.tsv", type=["tsv", "txt"], key=f"up_{key_suffix}")
        if up is None:
            return None, None
        df = pd.read_csv(up, sep="\t", index_col="gene")
        gene_order = list(df.index)
        patterns = {col: df[col].to_numpy() for col in df.columns}
        return patterns, gene_order
    else:
        path = st.text_input(
            "Ruta a calibrated_patterns.tsv",
            value=default or "results_demo/calibrated_patterns.tsv",
            key=f"path_{key_suffix}",
        )
        if not Path(path).exists():
            st.warning(f"No se encontró `{path}`. Corre `python3 cli.py demo` para generar unos de ejemplo.")
            return None, None
        return load_patterns_cached(path)


st.title("🧬 ColoQ — gemelo digital de cáncer colorrectal")
st.caption(
    "Subtipos moleculares consensuados (CMS1–4) como atractores de una red tipo Hopfield · "
    "panel de 10 genes compatible con RT-qPCR · herramienta de investigación, no dispositivo médico"
)

tab_clasificar, tab_pronostico, tab_tratamiento, tab_about = st.tabs(
    ["Clasificar", "Pronóstico longitudinal", "Simulación de tratamiento", "Acerca de"]
)

# ----------------------------------------------------------------------
# 1. CLASIFICAR
# ----------------------------------------------------------------------
with tab_clasificar:
    st.header("Clasificar muestras por subtipo CMS")
    st.markdown(
        "Sube un TSV con una fila por muestra y una columna por gen del panel "
        "(ver esquema en `README.md`). Se aplican patrones **ya calibrados**, sin recalibrar."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        patterns, gene_order = patterns_selector("clf")
    with col2:
        data_file = st.file_uploader("TSV de expresión", type=["tsv", "txt"], key="clf_data")

    if patterns and data_file is not None:
        df = pd.read_csv(data_file, sep="\t")
        missing = set(gene_order) - set(df.columns)
        if missing:
            st.error(f"Al TSV le faltan genes de los patrones: {sorted(missing)}")
        else:
            z = zscore_genes(df, gene_order)
            scored = score_cohort(z, gene_order, patterns)

            st.subheader("Resultados")
            counts = scored["predicted_cms"].value_counts()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Muestras clasificadas", len(scored))
                st.dataframe(counts.rename("n"), use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(counts.index, counts.values, color=WONG[:len(counts)])
                ax.set_ylabel("n muestras")
                plt.xticks(rotation=20, ha="right", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

            low_conf = (scored["classification_confidence"] < 0.4).sum()
            if low_conf:
                st.warning(
                    f"{low_conf} de {len(scored)} muestras tienen confianza < 0.40. "
                    "En la cohorte de validación externa, las muestras con baja confianza "
                    "concentraron la mayoría de los errores de clasificación — interpretarlas con cautela."
                )

            st.dataframe(
                scored[["predicted_cms", "classification_confidence"] + gene_order].head(50),
                use_container_width=True,
            )
            st.download_button(
                "Descargar resultados (TSV)",
                scored.to_csv(sep="\t", index=False).encode(),
                file_name="scored_cohort.tsv",
            )
    else:
        st.info("Selecciona los patrones calibrados y sube un TSV de expresión para empezar.")

# ----------------------------------------------------------------------
# 2. PRONOSTICO LONGITUDINAL
# ----------------------------------------------------------------------
with tab_pronostico:
    st.header("Pronóstico longitudinal post-quirúrgico")
    st.markdown(
        "Simula mediciones periódicas post-quirúrgicas y detecta el momento en que la señal "
        "molecular reaparece (movimiento desde el origen hacia un atractor)."
    )

    patterns, gene_order = patterns_selector("prog")

    if patterns:
        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox("Atractor de recaída simulada", list(patterns.keys()), index=len(patterns) - 1)
        with c2:
            onset = st.slider("Inicio de recaída (mes)", 3, 24, 15, step=3)
        with c3:
            n_checks = st.slider("Número de chequeos", 4, 16, 10)

        W, labels, _ = build_model_from_patterns(patterns)
        t_checks, x_series = simulate_longitudinal_patient(
            W, gene_order, patterns[target], len(gene_order),
            n_timepoints=n_checks, recurrence_onset_month=onset,
        )
        hazard = hazard_from_trajectory(x_series)
        alert, alert_idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for i, gene in enumerate(gene_order):
            ax1.plot(t_checks, x_series[i], color=WONG[i % len(WONG)], marker="o", markersize=3,
                     label=gene, linewidth=1.4)
        ax1.axvline(onset, color="grey", linestyle="--", linewidth=1)
        ax1.set_ylabel("Expresión (z-score)")
        ax1.legend(fontsize=6, ncol=5, loc="upper left")
        ax2.plot(t_checks, hazard, color="#D55E00", marker="o", linewidth=1.8)
        ax2.axvline(onset, color="grey", linestyle="--", linewidth=1)
        if alert:
            ax2.axvline(t_checks[alert_idx], color="red", linestyle=":", linewidth=1.5)
        ax2.set_xlabel("Meses desde cirugía")
        ax2.set_ylabel("Hazard ordinal")
        fig.tight_layout()
        st.pyplot(fig)

        if alert:
            x_at_alert = x_series[:, alert_idx]
            attractor, corr = classify_current_state(x_at_alert, patterns)
            st.error(f"**Alerta de recurrencia detectada en el mes {t_checks[alert_idx]}**")

            ev = EVIDENCE_STRENGTH.get(attractor, {})
            icon = EVIDENCE_COLOR.get(ev.get("level", ""), "")
            st.markdown(f"**Dirección:** {attractor} (correlación = {corr:.3f})")
            st.markdown(f"**Evidencia externa de este atractor:** {icon} {ev.get('level', 'desconocida').upper()}")
            st.caption(ev.get("detail", ""))

            st.subheader("Tratamientos con mecanismo aplicable")
            treatments = applicable_treatments(x_at_alert, gene_order, patterns)
            if treatments:
                for name, eff in treatments:
                    with st.expander(f"{name} — eficacia relativa {eff:.3f}"):
                        st.write(describe_treatment(name))
                        if name == "anti_egfr":
                            st.warning(
                                "Estatus RAS/BRAF real no disponible: este número usa el proxy débil "
                                "por RNA (cercanía a CMS3). **No sustituye la prueba de mutación** "
                                "(qPCR alelo-específico / HRM)."
                            )
            else:
                st.info("Ninguno de los mecanismos modelados tiene eficacia no trivial en este estado.")

            st.caption(
                "Dirección de efecto fundamentada en literatura clínica; magnitud NO calibrada. "
                "Exploración in silico — no usar para decisiones clínicas."
            )
        else:
            st.success("No se detectó alerta en la ventana simulada.")

# ----------------------------------------------------------------------
# 3. SIMULACION DE TRATAMIENTO
# ----------------------------------------------------------------------
with tab_tratamiento:
    st.header("Simulación contrafactual de tratamiento")
    st.markdown("Misma trayectoria de recaída, con y sin intervención — *¿qué hubiera pasado si...?*")

    patterns, gene_order = patterns_selector("tx")

    if patterns:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            target = st.selectbox("Atractor de recaída", list(patterns.keys()), key="tx_target")
        with c2:
            treatment = st.selectbox("Tratamiento", list(TREATMENT_MECHANISMS.keys()))
        with c3:
            tx_onset = st.slider("Inicio del tratamiento (mes)", 3, 30, 18, step=3)
        with c4:
            ras_status = st.selectbox("Estatus RAS/BRAF", ["unknown", "true", "false"],
                                       help="Solo relevante para anti_egfr. 'true' = wild-type confirmado.")

        ras_map = {"true": True, "false": False, "unknown": None}
        W, labels, _ = build_model_from_patterns(patterns)
        n_genes = len(gene_order)

        _, x_base = simulate_with_optional_treatment(
            W, n_genes, gene_order, patterns[target], patterns, treatment=None,
            n_timepoints=10, recurrence_onset_month=15,
        )
        t_checks, x_tx = simulate_with_optional_treatment(
            W, n_genes, gene_order, patterns[target], patterns, treatment=treatment,
            treatment_onset_month=tx_onset, ras_braf_wildtype=ras_map[ras_status],
            n_timepoints=10, recurrence_onset_month=15,
        )
        h_base, h_tx = hazard_from_trajectory(x_base), hazard_from_trajectory(x_tx)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t_checks, h_base, color="#D55E00", marker="o", linewidth=1.8, label="Sin tratamiento")
        ax.plot(t_checks, h_tx, color="#0072B2", marker="o", linewidth=1.8, label=f"Con {treatment}")
        ax.axvline(15, color="grey", linestyle="--", linewidth=1, label="Inicio recaída")
        ax.axvline(tx_onset, color="#0072B2", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Meses desde cirugía")
        ax.set_ylabel("Hazard ordinal")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

        delta = h_base[-1] - h_tx[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Hazard final sin tx", f"{h_base[-1]:.2f}")
        c2.metric("Hazard final con tx", f"{h_tx[-1]:.2f}")
        c3.metric("Diferencia", f"{delta:.2f}", delta=f"{-delta:.2f}")

        if abs(delta) < 0.01:
            st.info(
                f"Sin efecto: el mecanismo de **{treatment}** no aplica a un paciente que se dirige "
                f"a **{target}**. Esto es el comportamiento esperado, no un error — los gates de "
                "eficacia reflejan qué pacientes se benefician según la evidencia clínica."
            )

        st.write(describe_treatment(treatment))
        st.caption(
            "Dirección de efecto fundamentada en literatura; magnitud NO calibrada contra datos "
            "reales de tratamiento. Exploración in silico únicamente."
        )

# ----------------------------------------------------------------------
# 4. ACERCA DE
# ----------------------------------------------------------------------
with tab_about:
    st.header("Acerca de ColoQ")

    st.subheader("Panel (10 genes, todos compatibles con RT-qPCR)")
    st.table(pd.DataFrame({
        "Eje CMS": ["CMS1 (MSI/inmune)", "CMS2 (canónico/WNT)", "CMS3 (metabólico)", "CMS4 (mesenquimal)"],
        "Genes": ["MLH1, GNLY, USP18", "MYC, AXIN2", "FABP1, CPS1, SI", "VIM, TGFB1"],
    }))

    st.subheader("Evidencia acumulada")
    st.table(pd.DataFrame({
        "Cohorte": ["GSE39582", "GSE17536", "GSE17537", "GSE14333", "Cox combinado (3 cohortes)"],
        "Rol": ["Entrenamiento", "Externa", "Externa (nunca usada para ajustar)", "Externa", "n=326"],
        "n": [557, 145, 55, 126, 326],
        "p-valor": ["0.00039", "0.090", "0.71", "0.59", "0.045 (global)"],
    }))
    st.markdown(
        "- Concordancia con la clasificación oficial del consorcio (GSE39582): **kappa = 0.679**\n"
        "- **CMS4** es el eje con evidencia más fuerte y consistente (HR=1.88, p=0.03 en el Cox combinado)\n"
        "- **CMS3** no muestra señal de supervivencia distinguible de CMS2 (p=0.70)"
    )

    st.subheader("Limitaciones (leer antes de usar)")
    st.warning(
        "**Herramienta de investigación — no es un dispositivo médico ni una ayuda a la decisión "
        "clínica validada.**\n\n"
        "- El *hazard* es **ordinal**, no una probabilidad calibrada de recurrencia\n"
        "- La magnitud del efecto de tratamiento **no está calibrada** contra datos reales; solo la "
        "dirección está fundamentada en literatura clínica\n"
        "- La validación se hizo sobre datos de expresión (microarray/RNA-seq); el panel está "
        "*diseñado* para RT-qPCR pero la transferencia analítica a esa plataforma está pendiente\n"
        "- Concordance = 0.57 en el Cox combinado: significativo no equivale a buen discriminador individual\n"
        "- GSE17536 se usó iterativamente para decidir genes, lo que compromete parcialmente su "
        "independencia estadística como cohorte externa"
    )

    st.caption("Detalle completo en `PROJECT_STATUS.md` y `CHANGELOG.md` del repositorio.")

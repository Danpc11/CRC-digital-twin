"""
app.py -- interfaz web (Streamlit) para ColoQ / crc-digital-twin.

DIRECCION DE DISENO
-------------------
El proyecto tiene una tesis: la fuerza de la evidencia NO es uniforme
entre los cuatro atractores (CMS4 solido, CMS1 tendencia, CMS3 sin
senal). La mayoria de herramientas de diagnostico esconden esa
asimetria; aqui es el sistema visual principal -- cada afirmacion
carga su propio marcador de confianza.

  - Paleta: los cuatro colores CMS vienen de la paleta Wong
    (colorblind-safe) y son LOS MISMOS que usan las figuras de
    matplotlib del proyecto. Coherencia entre pantalla y publicacion.
  - Tipografia: monoespaciada para todo valor numerico (correlaciones,
    riesgos, p-valores) -- son datos medidos, no prosa.
  - Estructura: panel de instrumento. Los patrones calibrados se cargan
    UNA vez en la barra lateral y persisten entre pestanas.

Lanzar con:  python3 cli.py app     (o: streamlit run app.py)

ALCANCE: herramienta de investigacion. No es un dispositivo medico ni
una ayuda a la decision clinica validada -- ver pestana "Metodo".
"""

import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from calibration import load_calibrated_patterns, load_gene_reference_stats, zscore_genes
from clinical_selection import assess_molecular_eligibility, hybrid_mechanism_hypotheses
from modern_hopfield import (
    patterns_to_matrix,
    score_cohort_modern_hopfield,
    validate_modern_hopfield_beta,
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

# --- sistema de color: identidad por subtipo, consistente con las figuras ---
CMS_COLOR = {
    "CMS1_MSI_immune": "#0072B2",      # azul       (Wong)
    "CMS2_canonical_WNT": "#E69F00",   # ambar      (Wong)
    "CMS3_metabolic": "#009E73",       # verde      (Wong)
    "CMS4_mesenchymal": "#D55E00",     # bermellon  (Wong)
    "none": "#8A8F98",
}
CMS_SHORT = {
    "CMS1_MSI_immune": "CMS1",
    "CMS2_canonical_WNT": "CMS2",
    "CMS3_metabolic": "CMS3",
    "CMS4_mesenchymal": "CMS4",
    "none": "n/c",
}
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# --- sistema de evidencia: el elemento firma de la interfaz ---
EVIDENCE_STYLE = {
    "fuerte":        {"bars": 3, "color": "#1B7F5A", "label": "Evidencia solida"},
    "debil":         {"bars": 1, "color": "#B77A00", "label": "Tendencia, no concluyente"},
    "sin evidencia": {"bars": 0, "color": "#B03A2E", "label": "Sin senal detectada"},
    "referencia":    {"bars": 0, "color": "#6C737F", "label": "Grupo de referencia"},
}

st.set_page_config(page_title="ColoQ", page_icon="◧", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
          font-variant-numeric: tabular-nums; }

  .eyebrow { font-family: ui-monospace, Menlo, Consolas, monospace;
             font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
             color: #6C737F; margin-bottom: 0.35rem; }

  .readout { border: 1px solid #E3E6EA; border-left-width: 4px; border-radius: 3px;
             padding: 0.9rem 1.1rem; background: #FCFCFD; margin-bottom: 0.6rem; }
  .readout .value { font-family: ui-monospace, Menlo, Consolas, monospace;
                    font-size: 1.65rem; font-weight: 600; line-height: 1.15;
                    font-variant-numeric: tabular-nums; }
  .readout .unit { font-size: 0.8rem; color: #6C737F; font-weight: 400; }

  .ev-meter { display: inline-flex; gap: 3px; vertical-align: middle; margin-right: 0.5rem; }
  .ev-bar { width: 7px; height: 15px; border-radius: 1px; background: #E3E6EA; }
  .ev-note { font-size: 0.82rem; color: #4A5058; line-height: 1.5; }

  .cms-tag { display: inline-block; padding: 0.12rem 0.5rem; border-radius: 3px;
             font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.78rem;
             font-weight: 600; color: #fff; letter-spacing: 0.02em; }

  .scope { border-top: 1px solid #E3E6EA; margin-top: 1.4rem; padding-top: 0.7rem;
           font-size: 0.76rem; color: #6C737F; line-height: 1.55; }

  section[data-testid="stSidebar"] { border-right: 1px solid #E3E6EA; }
  div[data-testid="stMetricValue"] { font-family: ui-monospace, Menlo, Consolas, monospace; }

  /* Vista de impresion -- para cuando un medico imprime el resumen del
     paciente directo desde el navegador (Ctrl+P), sin la barra
     lateral, pestanas ni controles interactivos que no aplican en
     papel. */
  @media print {
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="stTabs"] [role="tablist"] { display: none !important; }
    .stButton, .stDownloadButton, .stSlider, .stSelectbox, .stRadio { display: none !important; }
    .readout { break-inside: avoid; }
  }
</style>
""", unsafe_allow_html=True)


def cms_tag(label: str) -> str:
    return (f'<span class="cms-tag" style="background:{CMS_COLOR.get(label, "#8A8F98")}">'
            f'{CMS_SHORT.get(label, label)}</span>')


def evidence_meter(level: str) -> str:
    """Tres barras, llenas segun la fuerza de evidencia externa del atractor."""
    style = EVIDENCE_STYLE.get(level, EVIDENCE_STYLE["sin evidencia"])
    bars = "".join(
        f'<span class="ev-bar" style="background:'
        f'{style["color"] if i < style["bars"] else "#E3E6EA"}"></span>'
        for i in range(3)
    )
    return (f'<span class="ev-meter">{bars}</span>'
            f'<span class="mono" style="font-size:0.8rem;color:{style["color"]};font-weight:600">'
            f'{style["label"]}</span>')


def readout(eyebrow: str, value: str, unit: str = "", accent: str = "#4A5058") -> str:
    return (f'<div class="readout" style="border-left-color:{accent}">'
            f'<div class="eyebrow">{eyebrow}</div>'
            f'<div class="value" style="color:{accent}">{value} '
            f'<span class="unit">{unit}</span></div></div>')


# --- simulacion cacheada -- sin esto, cada interaccion con un slider
# (arrastrar, no solo soltar) vuelve a integrar las ODEs desde cero.
# Se siente lento/entrecortado. st.cache_data hashea los argumentos
# (arrays de numpy y diccionarios de patrones incluidos) y reusa el
# resultado si nada cambio -- verificado que invalida correctamente
# cuando el vector del paciente o los patrones cambian, no solo
# cuando cambian los parametros "cosmeticos" (onset, n_checks).
@st.cache_data(show_spinner=False)
def cached_trajectory(model_matrix, gene_order, driver_vector, n_genes, n_timepoints,
                      recurrence_onset_month, dynamics_model="modern_hopfield",
                      dynamics_beta=3.0, max_forcing_strength=5.0):
    return simulate_longitudinal_patient(
        model_matrix, gene_order, driver_vector, n_genes,
        n_timepoints=n_timepoints, recurrence_onset_month=recurrence_onset_month,
        dynamics_model=dynamics_model, beta=dynamics_beta,
        max_forcing_strength=max_forcing_strength)


@st.cache_data(show_spinner=False)
def cached_treatment_sim(model_matrix, n_genes, gene_order, driver_vector, patterns, treatment,
                          treatment_onset_month, ras_braf_wildtype, n_timepoints,
                          recurrence_onset_month, dynamics_model, dynamics_beta,
                          max_forcing_strength):
    return simulate_with_optional_treatment(
        model_matrix, n_genes, gene_order, driver_vector, patterns, treatment=treatment,
        treatment_onset_month=treatment_onset_month, ras_braf_wildtype=ras_braf_wildtype,
        n_timepoints=n_timepoints, recurrence_onset_month=recurrence_onset_month,
        dynamics_model=dynamics_model, beta=dynamics_beta,
        max_forcing_strength=max_forcing_strength)


@st.cache_data(show_spinner=False)
def cached_validate_modern_beta(patterns, beta):
    return validate_modern_hopfield_beta(patterns, beta, correlation_threshold=0.9)


def evaluate_all_treatments(model_matrix, n_genes, gene_order, driver_vector, patterns,
                             recurrence_onset_month=15, treatment_onset_month=18,
                             ras_braf_wildtype=None, dynamics_model="modern_hopfield",
                             dynamics_beta=3.0, max_forcing_strength=5.0):
    """
    Corre los mecanismos de tratamiento disponibles contra el vector de
    UN paciente, devuelve un resumen ordenado de mayor a menor
    beneficio simulado. Es la logica central de la pestana "Paciente"
    y del PDF -- responde "de estos mecanismos, cual (si alguno) tiene
    efecto no trivial para este paciente especifico" en un solo lugar,
    en vez de que el usuario tenga que probar cada tratamiento a mano
    uno por uno en la pestana Intervencion.
    """
    _, x_base = cached_treatment_sim(
        model_matrix, n_genes, gene_order, driver_vector, patterns, None,
        treatment_onset_month, ras_braf_wildtype, 10, recurrence_onset_month,
        dynamics_model, dynamics_beta, max_forcing_strength)
    h_base = hazard_from_trajectory(x_base)[-1]

    results = []
    for name in TREATMENT_MECHANISMS:
        _, x_tx = cached_treatment_sim(
            model_matrix, n_genes, gene_order, driver_vector, patterns, name,
            treatment_onset_month, ras_braf_wildtype, 10, recurrence_onset_month,
            dynamics_model, dynamics_beta, max_forcing_strength)
        h_tx = hazard_from_trajectory(x_tx)[-1]
        results.append({
            "treatment": name, "h_base": h_base, "h_tx": h_tx,
            "delta": h_base - h_tx, "aplica": abs(h_base - h_tx) > 0.01,
        })
    return sorted(results, key=lambda r: -r["delta"])


def build_patient_pdf(sample_id, predicted_cms, confidence, evidence, t_checks, hazard,
                       alert, alert_idx, treatment_results, gene_order) -> bytes:
    """
    Arma el PDF de reporte de un paciente: clasificacion, evidencia,
    grafica de riesgo, y tabla de tratamientos evaluados. Usa reportlab
    (puro Python, sin dependencias de sistema -- compatible con Docker
    y con el ejecutable de PyInstaller, a diferencia de herramientas
    HTML-a-PDF como weasyprint que necesitan cairo/pango).
    """
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    body_style = styles["Normal"]
    caution_style = ParagraphStyle(
        "caution", parent=body_style, fontSize=8, textColor=colors.HexColor("#6C737F"))

    # grafica de riesgo como imagen embebida
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(t_checks, hazard, color=CMS_COLOR.get(predicted_cms, "#D55E00"),
            marker="o", markersize=4, linewidth=2)
    if alert:
        ax.axvline(t_checks[alert_idx], color="#B03A2E", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Meses desde la cirugía")
    ax.set_ylabel("Riesgo (ordinal)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_buf.seek(0)

    story = [
        Paragraph("ColoQ &mdash; Reporte de paciente", title_style),
        Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", caution_style),
        Spacer(1, 14),
        Paragraph(f"<b>Paciente:</b> {sample_id}", body_style),
        Paragraph(f"<b>Subtipo predicho:</b> {CMS_SHORT.get(predicted_cms, predicted_cms)} "
                  f"(confianza {confidence:.2f})", body_style),
        Paragraph(f"<b>Respaldo de evidencia externa de este eje:</b> "
                  f"{evidence.get('level', 'desconocida').upper()}", body_style),
        Paragraph(evidence.get("detail", ""), caution_style),
        Spacer(1, 10),
    ]

    if alert:
        story.append(Paragraph(
            f"<b>Alerta de recurrencia simulada en el mes {t_checks[alert_idx]}.</b>", body_style))
    else:
        story.append(Paragraph("Sin alerta en la ventana simulada.", body_style))
    story.append(Spacer(1, 8))
    story.append(Image(img_buf, width=15 * cm, height=7 * cm))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Mecanismos de tratamiento evaluados</b>", styles["Heading3"]))
    story.append(Paragraph(
        "Los valores de intensidad NO son una estimación de eficacia clínica ni un "
        "porcentaje de beneficio -- ver aviso al final.", caution_style))
    table_data = [["Mecanismo", "Riesgo ordinal sin tx", "Riesgo ordinal con tx",
                    "Intensidad simulada (arbitraria)", "Dirección"]]
    for r in treatment_results:
        table_data.append([
            r["treatment"], f"{r['h_base']:.2f}", f"{r['h_tx']:.2f}",
            f"{r['delta']:+.2f}", "Reduce riesgo" if r["aplica"] else "Sin dirección de efecto",
        ])
    tbl = Table(table_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E3E6EA")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6F7")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))

    story.append(Paragraph(
        "Herramienta de investigacion. No es un dispositivo medico ni una ayuda a la "
        "decision clinica validada. El riesgo mostrado es ordinal (ordena momentos "
        "dentro de la misma trayectoria), no una probabilidad calibrada de recurrencia. "
        "La direccion del efecto de tratamiento esta fundamentada en literatura clinica; "
        "la magnitud NO esta calibrada contra datos reales de tratamiento. "
        "Panel de referencia: " + ", ".join(gene_order) + ".",
        caution_style))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    doc.build(story)
    return buf.getvalue()


# ======================================================================
# BARRA LATERAL -- los patrones se cargan una sola vez
# ======================================================================
with st.sidebar:
    st.markdown('<div class="eyebrow">Gemelo digital · CRC</div>', unsafe_allow_html=True)
    st.markdown("## ◧ ColoQ")
    st.caption("Subtipificación molecular CMS1–4 sobre panel de 10 genes RT-qPCR")

    st.divider()
    st.markdown('<div class="eyebrow">Patrones calibrados</div>', unsafe_allow_html=True)

    default = next((str(p) for p in sorted(ROOT.glob("results*/calibrated_patterns.tsv"))), "")
    mode = st.radio("Origen", ["Ruta local", "Subir archivo"], horizontal=True,
                     label_visibility="collapsed")

    patterns, gene_order, gene_stats = None, None, None
    if mode == "Subir archivo":
        up = st.file_uploader("calibrated_patterns.tsv", type=["tsv", "txt"])
        if up is not None:
            _df = pd.read_csv(up, sep="\t", index_col="gene")
            gene_order = list(_df.index)
            pattern_cols = [c for c in _df.columns if not c.startswith("_ref_")]
            patterns = {c: _df[c].to_numpy() for c in pattern_cols}
            if "_ref_mean" in _df.columns and "_ref_std" in _df.columns:
                gene_stats = {g: (float(_df.loc[g, "_ref_mean"]), float(_df.loc[g, "_ref_std"]))
                               for g in gene_order}
    else:
        path = st.text_input("Ruta", value=default or "results_demo/calibrated_patterns.tsv",
                              label_visibility="collapsed")
        if Path(path).exists():
            patterns, gene_order = load_calibrated_patterns(path)
            gene_stats = load_gene_reference_stats(path)
        else:
            st.info("Sin patrones cargados. Genera unos con `python3 cli.py demo`, "
                    "o sube un archivo.")

    if patterns:
        st.success(f"{len(gene_order)} genes · {len(patterns)} subtipos")
        with st.expander("Ver panel"):
            st.markdown('<span class="mono" style="font-size:0.8rem">'
                        + " · ".join(gene_order) + "</span>", unsafe_allow_html=True)
        if gene_stats is None:
            st.caption("⚠️ Sin estadísticas de referencia congeladas — no se podrán "
                      "clasificar muestras individuales (n=1), solo cohortes. "
                      "Recalibra con `run_pipeline.py` actualizado para habilitarlo.")
        st.session_state["gene_stats"] = gene_stats

        st.divider()
        st.markdown('<div class="eyebrow">Motor dinámico</div>', unsafe_allow_html=True)
        # Modern Hopfield V2 es el UNICO motor -- reemplaza por completo la
        # dinamica de proyeccion anterior (decision explicita, no un toggle
        # con ambas opciones disponibles). El codigo de "projection_legacy"
        # sigue existiendo en prognosis_demo.py/treatment_simulation_demo.py
        # para reproducibilidad cientifica de figuras/resultados previos
        # via CLI, pero ya no se expone como eleccion en la app.
        dynamics_model = "modern_hopfield"
        dynamics_beta = st.number_input(
            "β dinámico", 0.1, 20.0, 3.0, 0.1, key="modern_hopfield_beta",
            help="Verificado con datos reales de GSE39582: los 4 subtipos son atractores "
                 "estables desde β≈3.0. Valores mas bajos pueden no conservar los 4.")
        max_forcing_strength = st.number_input(
            "Fuerza máxima del driver", 0.1, 20.0, 5.0, 0.1,
            key="modern_hopfield_forcing",
            help="Verificado: fuerza≥5.0 alcanza los 4 patrones objetivo (incl. el mas dificil, "
                 "CMS4) con el criterio mas estricto -- exito medido DESPUES de retirar el "
                 "forzamiento, no solo mientras esta activo. Magnitud experimental especifica "
                 "de esta calibracion; no es una dosis clinica.")
        st.caption("Reposo estabilizado, transición suave y driver normalizado (V2).")

    st.divider()
    st.markdown(
        '<div class="scope" style="border:0;margin:0;padding:0">'
        'Herramienta de investigación. No es un dispositivo médico ni una ayuda '
        'a la decisión clínica validada.</div>', unsafe_allow_html=True)

    # Boton de salida -- cerrar la pestana del navegador NO detiene el
    # servidor (queda corriendo en segundo plano). Esto si lo mata:
    # os._exit(0) es una terminacion dura del proceso completo, la
    # unica forma confiable de "cerrar la app" desde dentro de la UI
    # cuando se corre como ejecutable de un solo archivo (sin consola
    # visible con la que interactuar en Windows/Mac empacado).
    st.divider()
    if st.session_state.get("confirm_exit", False):
        st.warning("¿Cerrar ColoQ? Se pierde cualquier resultado no descargado.")
        col_yes, col_no = st.columns(2)
        if col_yes.button("Sí, salir", type="primary", use_container_width=True):
            st.markdown("Cerrando ColoQ...")
            os._exit(0)
        if col_no.button("Cancelar", use_container_width=True):
            st.session_state["confirm_exit"] = False
            st.rerun()
    else:
        if st.button("⏻  Salir de ColoQ", use_container_width=True):
            st.session_state["confirm_exit"] = True
            st.rerun()


# ======================================================================
# ENCABEZADO
# ======================================================================
st.markdown('<div class="eyebrow">Cáncer colorrectal · subtipos moleculares consensuados</div>',
            unsafe_allow_html=True)
st.title("Clasificación y trayectoria molecular")

if not patterns:
    st.warning("Carga los patrones calibrados en la barra lateral para empezar.")
    st.markdown(
        "**¿No tienes patrones todavía?** Genera un conjunto de ejemplo sobre datos "
        "sintéticos, sin credenciales ni descargas:\n\n```bash\npython3 cli.py demo\n```"
    )
    st.stop()

st.caption(
    "Motor dinámico activo: Modern Hopfield V2"
    f" · β={dynamics_beta:.2f} · fuerza={max_forcing_strength:.2f}")

beta_validation = cached_validate_modern_beta(patterns, dynamics_beta)
if not beta_validation["valid"]:
    invalid = ", ".join(beta_validation["invalid_patterns"])
    st.error(
        f"β={dynamics_beta:.2f} no conserva los cuatro atractores CMS con "
        f"correlación ≥0.90. No califican: {invalid}. Las trayectorias con este β "
        "son exploratorias y no deben interpretarse como recuperación CMS estable.")
else:
    st.success("β validado: los cuatro centroides convergen a atractores estables.")

dynamics_matrix, _labels = patterns_to_matrix(patterns)
n_genes = len(gene_order)

tab_muestras, tab_paciente, tab_traj, tab_tx, tab_metodo = st.tabs(
    ["Muestras", "Paciente", "Trayectoria", "Intervención", "Método"]
)

# ======================================================================
# 1. MUESTRAS
# ======================================================================
with tab_muestras:
    st.markdown("Asigna subtipo CMS a cada muestra aplicando los patrones cargados. "
                "No se recalibra: los centroides se mantienen fijos.")

    data_file = st.file_uploader(
        "Tabla de expresión (TSV: una fila por muestra, una columna por gen)",
        type=["tsv", "txt"])

    if data_file is None:
        st.markdown(
            '<div class="scope" style="border:0">Esquema esperado en <code>README.md</code>. '
            'Las columnas de genes deben coincidir con el panel cargado.</div>',
            unsafe_allow_html=True)
    else:
        df = pd.read_csv(data_file, sep="\t")
        missing = sorted(set(gene_order) - set(df.columns))
        if missing:
            st.error(
                f"La tabla no incluye {len(missing)} genes del panel: "
                f"`{', '.join(missing)}`. Agrégalos o carga un panel que coincida."
            )
        else:
            gene_stats = st.session_state.get("gene_stats")
            n_muestras = len(df)

            # Con pocas muestras, normalizar contra la propia tabla es
            # estadisticamente inestable (n chico) o directamente
            # imposible (n=1, varianza de un punto es cero). Usar
            # estadisticas congeladas de la calibracion en ese caso --
            # PERO para cohortes grandes (validacion externa), NO forzar
            # esto: cada cohorte auto-normalizandose contra si misma es
            # la practica correcta ahi.
            usar_stats_congeladas = n_muestras < 10

            if usar_stats_congeladas and gene_stats is None:
                if n_muestras == 1:
                    st.error(
                        "Estás clasificando **una sola muestra** (un paciente), pero los "
                        "patrones cargados no traen estadísticas de referencia congeladas "
                        "-- con n=1 es matemáticamente imposible calcular varianza propia. "
                        "Recalibra con `run_pipeline.py` (versión actual) para generar un "
                        "`calibrated_patterns.tsv` que sí las incluya, y vuelve a cargarlo "
                        "en la barra lateral."
                    )
                    st.stop()
                else:
                    st.warning(
                        f"Solo {n_muestras} muestras, sin estadísticas de referencia "
                        "congeladas disponibles -- la normalización contra esta tabla "
                        "chica puede ser inestable. Recomendado: recalibra con "
                        "`run_pipeline.py` actual para habilitar referencia congelada."
                    )

            stats_a_usar = gene_stats if usar_stats_congeladas else None
            if usar_stats_congeladas and gene_stats is not None:
                st.caption(f"Normalizando con estadísticas de referencia congeladas "
                          f"de la calibración (apropiado para n={n_muestras} muestra(s)).")

            try:
                z = zscore_genes(df, gene_order, stats=stats_a_usar)
            except ValueError as err:
                st.error(
                    f"No se puede normalizar la tabla: {err}\n\n"
                    "Ocurre cuando un gen tiene el mismo valor en todas las muestras "
                    "(varianza cero) o contiene celdas vacías. Revisa esa columna antes "
                    "de volver a cargarla."
                )
                st.stop()

            scored = score_cohort(z, gene_order, patterns)

            with st.expander("Recuperación dinámica Modern Hopfield (experimental)"):
                st.caption(
                    "No reemplaza la clasificación CMS validada por correlación. "
                    "Solo acepta una recuperación si converge a un equilibrio estable, "
                    "con residuo pequeño y correlación suficiente.")
                run_modern = st.checkbox(
                    "Calcular recuperación dinámica para esta cohorte",
                    key="run_modern_hopfield")
                mh_c1, mh_c2, mh_c3 = st.columns(3)
                mh_beta = mh_c1.number_input(
                    "β Modern Hopfield", min_value=0.1, max_value=20.0,
                    value=3.0, step=0.1, key="modern_beta")
                mh_corr = mh_c2.number_input(
                    "Correlación mínima", min_value=0.0, max_value=1.0,
                    value=0.8, step=0.05, key="modern_corr")
                mh_input_margin = mh_c3.number_input(
                    "Margen mínimo pre-relajación", min_value=0.0, max_value=1.0,
                    value=0.15, step=0.05, key="modern_input_margin",
                    help="Por debajo de este margen el estado se reporta híbrido/ambiguo y el modelo se abstiene.")
            if run_modern:
                with st.spinner("Recuperando equilibrios Modern Hopfield..."):
                    scored = score_cohort_modern_hopfield(
                        scored, gene_order, patterns, beta=float(mh_beta),
                        corr_threshold=float(mh_corr),
                        input_margin_threshold=float(mh_input_margin))

            # sample_id ya viene incluida en 'scored' si el archivo la
            # traia (zscore_genes/score_cohort hacen df.copy(), preservan
            # todas las columnas originales) -- solo agregar un
            # identificador de respaldo si de verdad falta.
            if "sample_id" not in scored.columns:
                scored.insert(0, "sample_id", [f"muestra_{i+1}" for i in range(len(scored))])
            else:
                scored["sample_id"] = scored["sample_id"].astype(str)

            # Persistir para las pestanas de Trayectoria e Intervencion --
            # permite elegir un paciente real de la cohorte cargada en vez
            # de simular solo hacia un atractor puro generico.
            st.session_state["scored_cohort"] = scored
            st.session_state["scored_cohort_gene_order"] = gene_order

            counts = scored["predicted_cms"].value_counts()
            conf = scored["classification_confidence"]

            c1, c2, c3 = st.columns(3)
            c1.markdown(readout("Muestras", f"{len(scored)}"), unsafe_allow_html=True)
            c2.markdown(readout("Confianza mediana", f"{conf.median():.2f}"),
                        unsafe_allow_html=True)
            low = int((conf < 0.40).sum())
            c3.markdown(readout("Confianza baja", f"{low}", f"de {len(scored)}",
                                 accent="#B77A00" if low else "#4A5058"),
                        unsafe_allow_html=True)

            st.markdown('<div class="eyebrow">Distribución por subtipo</div>',
                        unsafe_allow_html=True)
            cols = st.columns(len(counts))
            for col, (label, n) in zip(cols, counts.items()):
                pct = 100 * n / len(scored)
                col.markdown(
                    f'{cms_tag(label)}<div class="mono" style="font-size:1.4rem;font-weight:600;'
                    f'color:{CMS_COLOR.get(label)};margin-top:.3rem">{n}'
                    f'<span style="font-size:.8rem;color:#6C737F;font-weight:400">'
                    f' · {pct:.0f}%</span></div>',
                    unsafe_allow_html=True)

            if low:
                st.warning(
                    f"{low} muestras con confianza inferior a 0.40. En la validación externa, "
                    "las muestras de baja confianza concentraron la mayoría de los errores "
                    "de clasificación — conviene revisarlas caso por caso."
                )

            display_cols = ["sample_id", "predicted_cms", "classification_confidence"]
            display_cols += [c for c in (
                "modern_hopfield_cms", "modern_hopfield_correlation",
                "modern_hopfield_input_margin", "modern_hopfield_abstention_reason",
                "modern_hopfield_margin", "modern_hopfield_residual",
                "modern_hopfield_stable", "modern_hopfield_concordant") if c in scored.columns]
            st.dataframe(
                scored[display_cols + gene_order],
                use_container_width=True, height=280,
            )
            st.download_button("Descargar tabla clasificada",
                                scored.to_csv(sep="\t", index=False).encode(),
                                file_name="scored_cohort.tsv")

# ======================================================================
# 2. PACIENTE -- vista clinica consolidada: un paciente, una pantalla.
# No requiere que el usuario entienda "atractor" ni "vector de
# estado" para llegar a una respuesta accionable.
# ======================================================================
with tab_paciente:
    if "scored_cohort" not in st.session_state:
        st.info("Sube y clasifica una cohorte en la pestaña **Muestras** primero "
                "para ver el resumen de un paciente aquí.")
    else:
        cohort_p = st.session_state["scored_cohort"]
        cohort_genes_p = st.session_state["scored_cohort_gene_order"]

        sel_id_p = st.selectbox("Paciente", cohort_p["sample_id"].tolist(), key="paciente_sel")
        fila = cohort_p[cohort_p["sample_id"] == sel_id_p].iloc[0]
        driver_p = fila[cohort_genes_p].to_numpy(dtype=float)
        pred_p = fila["predicted_cms"]
        conf_p = float(fila["classification_confidence"])

        with st.spinner("Simulando trayectoria..."):
            t_p, x_p = cached_trajectory(
                dynamics_matrix, cohort_genes_p, driver_p, n_genes, 10, 15,
                dynamics_model, dynamics_beta, max_forcing_strength)
            hazard_p = hazard_from_trajectory(x_p)
            alert_p, idx_p = detect_recurrence_signal(hazard_p, baseline_window=2, threshold_sigma=3.0)

        h1, h2, h3 = st.columns([2, 1, 1])
        h1.markdown(
            f'<div class="eyebrow">Paciente</div>'
            f'<div class="mono" style="font-size:1.3rem;font-weight:600">{sel_id_p}</div>',
            unsafe_allow_html=True)
        h2.markdown(readout("Subtipo predicho", CMS_SHORT.get(pred_p, pred_p),
                             accent=CMS_COLOR.get(pred_p, "#4A5058")), unsafe_allow_html=True)
        h3.markdown(readout("Confianza", f"{conf_p:.2f}"), unsafe_allow_html=True)

        if "modern_hopfield_cms" in fila.index:
            mh_label = fila["modern_hopfield_cms"]
            mh_corr_value = float(fila["modern_hopfield_correlation"])
            mh_input_margin_value = float(fila.get("modern_hopfield_input_margin", float("nan")))
            concordant = bool(fila.get("modern_hopfield_concordant", False))
            if mh_label == "indeterminado":
                reason = fila.get("modern_hopfield_abstention_reason", "incertidumbre_no_especificada")
                st.warning("Modern Hopfield experimental se abstiene: "
                           f"{reason} (margen pre-relajación={mh_input_margin_value:.3f}). "
                           "Se conserva el perfil continuo y la clasificación principal por correlación.")
            elif concordant:
                st.success(f"Modern Hopfield experimental concuerda: "
                           f"{CMS_SHORT.get(mh_label, mh_label)} (r={mh_corr_value:.3f}).")
                st.caption(f"Margen pre-relajación: {mh_input_margin_value:.3f}; "
                           "la correlación final solo audita convergencia.")
            else:
                st.warning(f"Discordancia experimental: correlación={CMS_SHORT.get(pred_p, pred_p)}, "
                           f"Modern Hopfield={CMS_SHORT.get(mh_label, mh_label)} "
                           f"(r={mh_corr_value:.3f}). Requiere revisión; no se fuerza una decisión.")
                st.caption(f"Margen pre-relajación: {mh_input_margin_value:.3f}; "
                           "la correlación final no representa confianza clínica.")

        ev_p = EVIDENCE_STRENGTH.get(pred_p, {})
        st.markdown(evidence_meter(ev_p.get("level", "sin evidencia")), unsafe_allow_html=True)
        st.markdown(f'<div class="ev-note">{ev_p.get("detail", "")}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="eyebrow">Perfil CMS continuo</div>', unsafe_allow_html=True)
        cms_score_cols = [c for c in ("cms1_tendency", "cms2_tendency", "cms3_tendency", "cms4_tendency")
                          if c in fila.index]
        if cms_score_cols:
            cms_chart = pd.DataFrame({
                "CMS": [c.split("_")[0].upper() for c in cms_score_cols],
                "tendencia": [float(fila[c]) for c in cms_score_cols],
            }).set_index("CMS")
            st.bar_chart(cms_chart)
            interpretation = fila.get("cms_interpretation", "indeterminado")
            primary = CMS_SHORT.get(fila.get("cms_primary_tendency", "none"), "n/c")
            secondary = CMS_SHORT.get(fila.get("cms_secondary_tendency", "none"), "n/c")
            if interpretation == "hibrido":
                st.info(f"Perfil híbrido: mayor tendencia {primary}, secundaria {secondary}. "
                        "No se fuerza una etiqueta única.")
            else:
                st.success(f"Perfil dominante {primary}; tendencia secundaria {secondary}.")
            profile_for_hypotheses = {
                "scores": {
                    "CMS1_MSI_immune": float(fila.get("cms1_tendency", 0.0)),
                    "CMS2_canonical_WNT": float(fila.get("cms2_tendency", 0.0)),
                    "CMS3_metabolic": float(fila.get("cms3_tendency", 0.0)),
                    "CMS4_mesenchymal": float(fila.get("cms4_tendency", 0.0)),
                }
            }
            with st.expander("Hipótesis mecanísticas del perfil híbrido"):
                st.dataframe(pd.DataFrame(hybrid_mechanism_hypotheses(profile_for_hypotheses)),
                             use_container_width=True, hide_index=True)
                st.caption("Pesos transcriptómicos exploratorios; no justifican combinar tratamientos.")

        st.markdown('<div class="eyebrow">Evaluación molecular de elegibilidad</div>',
                    unsafe_allow_html=True)
        st.caption("Las reglas requieren biomarcadores confirmados. CMS nunca sustituye MSI/MMR, RAS, BRAF, HER2, KRAS G12C o NTRK.")
        with st.expander("Ingresar contexto clínico y biomarcadores", expanded=False):
            metastatic = st.checkbox("Enfermedad avanzada/metastásica confirmada", key="clinical_metastatic")
            side = st.selectbox("Localización primaria", ["unknown", "left", "right"], key="clinical_side")
            msi = st.selectbox("MSI/MMR", ["unknown", "msi_h_dmmr", "mss_pmmr"], key="clinical_msi")
            ras = st.selectbox("KRAS/NRAS", ["unknown", "wild_type", "mutated"], key="clinical_ras")
            braf = st.selectbox("BRAF", ["unknown", "wild_type", "v600e"], key="clinical_braf")
            her2 = st.selectbox("HER2", ["unknown", "positive", "negative"], key="clinical_her2")
            kras_g12c = st.selectbox("KRAS G12C", ["unknown", "positive", "negative"], key="clinical_g12c")
            ntrk = st.selectbox("NTRK", ["unknown", "fusion_positive", "negative"], key="clinical_ntrk")

        molecular_results = assess_molecular_eligibility(
            {"msi_mmr": msi, "ras": ras, "braf": braf, "her2": her2,
             "kras_g12c": kras_g12c, "ntrk": ntrk},
            {"metastatic": metastatic, "primary_side": side},
        )
        st.dataframe(pd.DataFrame(molecular_results), use_container_width=True, hide_index=True)
        st.warning("Resultado para apoyo a investigación. La decisión requiere expediente completo, regulación local y comité/oncólogo tratante.")

        st.divider()

        st.markdown('<div class="eyebrow">Riesgo simulado en el tiempo</div>', unsafe_allow_html=True)
        fig_p, ax_p = plt.subplots(figsize=(8, 3))
        ax_p.plot(t_p, hazard_p, color=CMS_COLOR.get(pred_p, "#D55E00"), marker="o",
                  markersize=4, linewidth=2)
        if alert_p:
            ax_p.axvline(t_p[idx_p], color="#B03A2E", linestyle=":", linewidth=1.6)
        ax_p.set_xlabel("Meses desde la cirugía", fontsize=9)
        ax_p.set_ylabel("Riesgo (ordinal)", fontsize=9)
        ax_p.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax_p.spines[s].set_visible(False)
        fig_p.tight_layout()
        st.pyplot(fig_p)

        if alert_p:
            st.error(f"Alerta de recurrencia simulada en el mes {t_p[idx_p]}.")
        else:
            st.success("Sin alerta en la ventana simulada.")

        with st.expander("Ver detalle molecular (10 genes)"):
            fig_g, ax_g = plt.subplots(figsize=(8, 3))
            for i, gene in enumerate(cohort_genes_p):
                ax_g.plot(t_p, x_p[i], color=WONG[i % len(WONG)], marker="o",
                         markersize=2.5, linewidth=1.2, label=gene)
            ax_g.legend(fontsize=6, ncol=5, loc="upper left", frameon=False)
            ax_g.set_ylabel("Expresión (z-score)", fontsize=9)
            ax_g.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax_g.spines[s].set_visible(False)
            fig_g.tight_layout()
            st.pyplot(fig_g)

        st.divider()

        st.markdown('<div class="eyebrow">¿Qué mecanismos de tratamiento tienen dirección de efecto simulada para este paciente?</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Los números de abajo son una **intensidad simulada arbitraria**, no una "
            "estimación de eficacia clínica ni un porcentaje de beneficio real -- ver "
            "el aviso completo más abajo."
        )
        with st.spinner("Evaluando mecanismos de tratamiento..."):
            resultados_tx = evaluate_all_treatments(
                dynamics_matrix, n_genes, cohort_genes_p, driver_p, patterns,
                dynamics_model=dynamics_model, dynamics_beta=dynamics_beta,
                max_forcing_strength=max_forcing_strength)

        aplican = [r for r in resultados_tx if r["aplica"]]
        if aplican:
            for r in aplican:
                st.markdown(
                    f'<div class="readout" style="border-left-color:#1B7F5A">'
                    f'<div class="eyebrow">{r["treatment"]} · dirección de efecto: reduce el riesgo simulado</div>'
                    f'<span class="mono" style="font-size:0.85rem;color:#4A5058">'
                    f'intensidad simulada arbitraria: {r["delta"]:+.2f} '
                    f'(riesgo ordinal {r["h_base"]:.2f} → {r["h_tx"]:.2f})</span>'
                    f'</div>', unsafe_allow_html=True)
                st.caption(describe_treatment(r["treatment"]))
        else:
            st.info("Ninguno de los mecanismos modelados muestra dirección de efecto no "
                    "trivial para este paciente, según su clasificación actual.")

        sin_efecto = [r["treatment"] for r in resultados_tx if not r["aplica"]]
        if sin_efecto:
            st.caption(f"Sin dirección de efecto simulada: {', '.join(sin_efecto)}.")

        st.markdown(
            '<div class="scope">Estos valores son una <strong>intensidad simulada '
            'arbitraria</strong>, no una eficacia clínica ni una probabilidad de '
            'beneficio -- la dirección del efecto está fundamentada en literatura '
            'clínica, pero la magnitud NO está calibrada contra datos reales de '
            'tratamiento y no debe leerse como un porcentaje de mejora esperado. '
            'Exploración in silico, nunca una recomendación clínica.</div>',
            unsafe_allow_html=True)

        st.divider()

        col_pdf, col_print = st.columns(2)
        pdf_bytes = build_patient_pdf(
            sel_id_p, pred_p, conf_p, ev_p, t_p, hazard_p, alert_p, idx_p,
            resultados_tx, cohort_genes_p)
        col_pdf.download_button(
            "📄 Descargar reporte (PDF)", pdf_bytes,
            file_name=f"coloq_reporte_{sel_id_p}.pdf", mime="application/pdf",
            use_container_width=True)
        col_print.caption("También puedes imprimir esta pestaña directo desde el "
                          "navegador (Ctrl+P / Cmd+P) — los controles se ocultan "
                          "automáticamente en la vista de impresión.")


# ======================================================================
# 3. TRAYECTORIA
# ======================================================================
with tab_traj:
    st.markdown("Simula el seguimiento post-quirúrgico: el estado parte del origen "
                "(sin enfermedad residual) y se detecta el momento en que la señal "
                "molecular reaparece.")

    has_cohort = "scored_cohort" in st.session_state
    modo = st.radio(
        "Fuente de la trayectoria", ["Atractor genérico", "Paciente de mi cohorte"],
        horizontal=True, key="traj_modo",
        help="'Paciente de mi cohorte' usa la expresión medida de un paciente real "
             "(subido en la pestaña Muestras) como dirección hipotética de recaída, "
             "en vez de un centroide CMS puro." if has_cohort else None,
    )

    driver_vector = None
    target_color_key = None

    if modo == "Paciente de mi cohorte":
        if not has_cohort:
            st.info("Sube y clasifica una cohorte en la pestaña **Muestras** primero "
                     "para poder elegir un paciente aquí.")
        else:
            cohort = st.session_state["scored_cohort"]
            cohort_genes = st.session_state["scored_cohort_gene_order"]
            sel_id = st.selectbox("Paciente", cohort["sample_id"].tolist())
            paciente = cohort[cohort["sample_id"] == sel_id].iloc[0]
            driver_vector = paciente[cohort_genes].to_numpy(dtype=float)
            target_color_key = paciente["predicted_cms"]

            st.markdown(
                f'{cms_tag(target_color_key)} <span class="mono" style="font-size:.85rem">'
                f'clasificado con confianza {paciente["classification_confidence"]:.2f} · '
                f'la trayectoria usa la expresión MEDIDA de este paciente como dirección '
                f'hipotética de recaída, no un centroide puro</span>',
                unsafe_allow_html=True)

    if driver_vector is None:
        # modo generico -- atractor puro (comportamiento original)
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Atractor de la recaída", list(patterns.keys()),
                               index=len(patterns) - 1,
                               format_func=lambda x: CMS_SHORT.get(x, x))
        onset = c2.slider("Inicio de la recaída (mes)", 3, 24, 15, step=3)
        n_checks = c3.slider("Chequeos de seguimiento", 4, 16, 10)
        driver_vector = patterns[target]
        target_color_key = target
    else:
        c2, c3 = st.columns(2)
        onset = c2.slider("Inicio de la recaída (mes)", 3, 24, 15, step=3)
        n_checks = c3.slider("Chequeos de seguimiento", 4, 16, 10)

    t_checks, x_series = simulate_longitudinal_patient(
        dynamics_matrix, gene_order, driver_vector, n_genes,
        n_timepoints=n_checks, recurrence_onset_month=onset,
        dynamics_model=dynamics_model, beta=dynamics_beta,
        max_forcing_strength=max_forcing_strength)
    hazard = hazard_from_trajectory(x_series)
    alert, alert_idx = detect_recurrence_signal(hazard, baseline_window=2, threshold_sigma=3.0)

    left, right = st.columns([3, 2])

    with left:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
        for i, gene in enumerate(gene_order):
            ax1.plot(t_checks, x_series[i], color=WONG[i % len(WONG)], marker="o",
                     markersize=2.5, linewidth=1.3, label=gene)
        ax1.axvline(onset, color="#9AA0A6", linestyle="--", linewidth=1)
        ax1.set_ylabel("Expresión (z-score)", fontsize=9)
        ax1.legend(fontsize=6, ncol=5, loc="upper left", frameon=False)
        ax1.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax1.spines[s].set_visible(False)

        ax2.plot(t_checks, hazard, color=CMS_COLOR.get(target_color_key, "#D55E00"), marker="o",
                 markersize=3.5, linewidth=2)
        ax2.axvline(onset, color="#9AA0A6", linestyle="--", linewidth=1)
        if alert:
            ax2.axvline(t_checks[alert_idx], color="#B03A2E", linestyle=":", linewidth=1.6)
        ax2.set_xlabel("Meses desde la cirugía", fontsize=9)
        ax2.set_ylabel("Riesgo (ordinal)", fontsize=9)
        ax2.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    with right:
        if alert:
            x_at = x_series[:, alert_idx]
            attractor, corr = classify_current_state(x_at, patterns)
            ev = EVIDENCE_STRENGTH.get(attractor, {})

            st.markdown(readout("Alerta de recurrencia", f"mes {t_checks[alert_idx]}",
                                 accent="#B03A2E"), unsafe_allow_html=True)
            st.markdown(
                f'<div class="eyebrow">Dirección</div>{cms_tag(attractor)} '
                f'<span class="mono" style="font-size:.85rem;color:#6C737F">'
                f'r = {corr:.3f}</span>', unsafe_allow_html=True)

            st.markdown('<div class="eyebrow" style="margin-top:1rem">Respaldo de este atractor</div>',
                        unsafe_allow_html=True)
            st.markdown(evidence_meter(ev.get("level", "sin evidencia")), unsafe_allow_html=True)
            st.markdown(f'<div class="ev-note">{ev.get("detail", "")}</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="eyebrow" style="margin-top:1rem">Mecanismos aplicables</div>',
                        unsafe_allow_html=True)
            treatments = applicable_treatments(x_at, gene_order, patterns)
            if treatments:
                for name, eff in treatments:
                    with st.expander(f"{name} — {eff:.2f}"):
                        st.caption(describe_treatment(name))
                        if name == "anti_egfr":
                            st.warning(
                                "Sin estatus RAS/BRAF real: este valor usa el proxy por RNA. "
                                "No sustituye la prueba de mutación (qPCR alelo-específico / HRM).")
            else:
                st.caption("Ningún mecanismo modelado tiene efecto no trivial en este estado.")
        else:
            st.markdown(readout("Seguimiento", "sin alerta", accent="#1B7F5A"),
                        unsafe_allow_html=True)
            st.caption("El estado permanece cerca del origen durante toda la ventana simulada.")

# ======================================================================
# 4. INTERVENCION
# ======================================================================
with tab_tx:
    st.markdown("Compara la misma trayectoria de recaída con y sin intervención — "
                "el contrafactual que un score estático no puede producir. Útil para "
                "explorar si un mecanismo de tratamiento (ej. quimioterapia) tiene "
                "efecto no trivial para un paciente dado, según su propia expresión medida.")

    has_cohort = "scored_cohort" in st.session_state
    modo_tx = st.radio(
        "Fuente de la trayectoria", ["Atractor genérico", "Paciente de mi cohorte"],
        horizontal=True, key="tx_modo",
        help="'Paciente de mi cohorte' usa la expresión medida de un paciente real "
             "como dirección hipotética de recaída." if has_cohort else None,
    )

    tx_driver_vector = None
    tx_color_key = None

    if modo_tx == "Paciente de mi cohorte":
        if not has_cohort:
            st.info("Sube y clasifica una cohorte en la pestaña **Muestras** primero "
                     "para poder elegir un paciente aquí.")
        else:
            cohort = st.session_state["scored_cohort"]
            cohort_genes = st.session_state["scored_cohort_gene_order"]
            sel_id_tx = st.selectbox("Paciente", cohort["sample_id"].tolist(), key="tx_patient")
            paciente_tx = cohort[cohort["sample_id"] == sel_id_tx].iloc[0]
            tx_driver_vector = paciente_tx[cohort_genes].to_numpy(dtype=float)
            tx_color_key = paciente_tx["predicted_cms"]
            st.markdown(
                f'{cms_tag(tx_color_key)} <span class="mono" style="font-size:.85rem">'
                f'clasificado con confianza {paciente_tx["classification_confidence"]:.2f}</span>',
                unsafe_allow_html=True)

    if tx_driver_vector is None:
        c1 = st.columns(1)[0]
        tx_target = c1.selectbox("Atractor de la recaída", list(patterns.keys()),
                                  key="tx_t", format_func=lambda x: CMS_SHORT.get(x, x))
        tx_driver_vector = patterns[tx_target]
        tx_color_key = tx_target

    c2, c3, c4 = st.columns(3)
    treatment = c2.selectbox("Tratamiento", list(TREATMENT_MECHANISMS.keys()))
    tx_onset = c3.slider("Inicio del tratamiento (mes)", 3, 30, 18, step=3)
    ras = c4.selectbox("Estatus RAS/BRAF", ["desconocido", "wild-type", "mutante"],
                        help="Solo afecta a anti_egfr.")

    ras_map = {"wild-type": True, "mutante": False, "desconocido": None}
    _, x_base = simulate_with_optional_treatment(
        dynamics_matrix, n_genes, gene_order, tx_driver_vector, patterns,
        treatment=None, n_timepoints=10, recurrence_onset_month=15,
        dynamics_model=dynamics_model, beta=dynamics_beta,
        max_forcing_strength=max_forcing_strength)
    t_checks, x_tx = simulate_with_optional_treatment(
        dynamics_matrix, n_genes, gene_order, tx_driver_vector, patterns, treatment=treatment,
        treatment_onset_month=tx_onset, ras_braf_wildtype=ras_map[ras],
        n_timepoints=10, recurrence_onset_month=15,
        dynamics_model=dynamics_model, beta=dynamics_beta,
        max_forcing_strength=max_forcing_strength)
    h_base, h_tx = hazard_from_trajectory(x_base), hazard_from_trajectory(x_tx)
    delta = h_base[-1] - h_tx[-1]

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.plot(t_checks, h_base, color="#8A8F98", marker="o", markersize=3.5,
                linewidth=2, label="Sin tratamiento")
        ax.plot(t_checks, h_tx, color=CMS_COLOR.get(tx_color_key, "#0072B2"), marker="o",
                markersize=3.5, linewidth=2, label=f"Con {treatment}")
        ax.fill_between(t_checks, h_tx, h_base, where=(h_base >= h_tx),
                        color=CMS_COLOR.get(tx_color_key, "#0072B2"), alpha=0.10)
        ax.axvline(15, color="#9AA0A6", linestyle="--", linewidth=1)
        ax.axvline(tx_onset, color=CMS_COLOR.get(tx_color_key, "#0072B2"),
                   linestyle=":", linewidth=1.4)
        ax.set_xlabel("Meses desde la cirugía", fontsize=9)
        ax.set_ylabel("Riesgo (ordinal)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    with right:
        st.markdown(readout("Riesgo ordinal final · sin tratamiento", f"{h_base[-1]:.2f}",
                             accent="#8A8F98"), unsafe_allow_html=True)
        st.markdown(readout("Riesgo ordinal final · con tratamiento", f"{h_tx[-1]:.2f}",
                             accent=CMS_COLOR.get(tx_color_key, "#0072B2")), unsafe_allow_html=True)
        st.markdown(
            f'<div class="eyebrow" style="margin-top:.6rem">Intensidad simulada arbitraria '
            f'(no eficacia clínica)</div>'
            f'<span class="mono" style="font-size:1rem;font-weight:600;'
            f'color:{"#1B7F5A" if delta > 0.01 else "#8A8F98"}">{delta:+.2f}</span>',
            unsafe_allow_html=True)

        if abs(delta) < 0.01:
            st.info(f"**{treatment}** no muestra dirección de efecto no trivial para esta "
                    f"trayectoria "
                    f"{'de este paciente' if modo_tx == 'Paciente de mi cohorte' else f'hacia {CMS_SHORT.get(tx_color_key)}'}. "
                    "Es el comportamiento esperado: los criterios de aplicabilidad reflejan qué "
                    "pacientes tendrían dirección de efecto según la evidencia clínica "
                    "publicada — un mecanismo que no aplica no debería mostrar intensidad "
                    "simulada.")
        elif modo_tx == "Paciente de mi cohorte":
            direccion = "reduce" if delta > 0 else "no reduce"
            st.success(f"Con este mecanismo, el modelo {direccion} el riesgo ordinal simulado "
                       f"para este paciente (intensidad simulada arbitraria: {delta:+.2f}). "
                       "Recordatorio: esto es dirección de efecto, no una recomendación "
                       "clínica ni una magnitud de beneficio real — ver el aviso de alcance abajo.")
        st.caption(describe_treatment(treatment))

    st.markdown(
        '<div class="scope">La <strong>dirección</strong> del efecto está fundamentada en '
        'literatura clínica; la <strong>magnitud</strong> (mostrada como "intensidad '
        'simulada arbitraria") no está calibrada contra datos reales de tratamiento y no '
        'debe leerse como un porcentaje de eficacia ni como una probabilidad de beneficio. '
        'Exploración in silico, nunca una recomendación clínica.</div>', unsafe_allow_html=True)

# ======================================================================
# 5. METODO
# ======================================================================
with tab_metodo:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="eyebrow">Panel</div>', unsafe_allow_html=True)
        st.markdown("**10 genes, todos amplificables por RT-qPCR**")
        for lab, genes in [("CMS1_MSI_immune", "MLH1 · GNLY · USP18"),
                            ("CMS2_canonical_WNT", "MYC · AXIN2"),
                            ("CMS3_metabolic", "FABP1 · CPS1 · SI"),
                            ("CMS4_mesenchymal", "VIM · TGFB1")]:
            st.markdown(
                f'{cms_tag(lab)} <span class="mono" style="font-size:.85rem">{genes}</span>',
                unsafe_allow_html=True)

        st.markdown('<div class="eyebrow" style="margin-top:1.4rem">Cohortes</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Cohorte": ["GSE39582", "GSE17536", "GSE17537", "GSE14333", "GSE33113"],
            "Rol": ["Entrenamiento", "Externa (iterativa)", "Externa (intacta)",
                    "Externa", "Externa (estadio II)"],
            "n": [557, 145, 55, 126, 89],
            "p": ["0.00039", "0.090", "0.71", "0.59", "0.031"],
        }), hide_index=True, use_container_width=True)
        st.caption("Cox estratificado combinando las cuatro externas (n=415), ajustado por "
                   "estadio (n=388): p < 0.001 global. CMS4 HR=2.06 (p=0.018), robusto al "
                   "ajuste; CMS1 HR=2.09 (p=0.016), resultado nuevo por confirmar. "
                   "Concordancia con la clasificación del consorcio: kappa = 0.679.")

    with c2:
        st.markdown('<div class="eyebrow">Respaldo por atractor</div>', unsafe_allow_html=True)
        for label, ev in EVIDENCE_STRENGTH.items():
            st.markdown(f'{cms_tag(label)} {evidence_meter(ev["level"])}',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="ev-note" style="margin-bottom:.8rem">{ev["detail"]}</div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="eyebrow">Qué no hace esta herramienta</div>',
                unsafe_allow_html=True)
    st.markdown("""
- El riesgo es **ordinal**: ordena momentos dentro de un mismo paciente, no es una probabilidad de recurrencia.
- La **magnitud** del efecto de tratamiento no está calibrada; solo la dirección tiene respaldo en literatura.
- La validación se hizo sobre datos de expresión (microarray). El panel está **diseñado** para RT-qPCR,
  pero la transferencia analítica a esa plataforma está pendiente.
- Concordance = 0.57 en el Cox combinado: significativo no equivale a buen discriminador individual.
- GSE17536 se usó de forma iterativa para elegir genes, lo que compromete parcialmente su independencia.
    """)
    st.caption("Detalle completo en `PROJECT_STATUS.md` y `CHANGELOG.md`.")

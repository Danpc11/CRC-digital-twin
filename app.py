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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from attractor_model import build_model_from_patterns
from calibration import load_calibrated_patterns, zscore_genes
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

    patterns, gene_order = None, None
    if mode == "Subir archivo":
        up = st.file_uploader("calibrated_patterns.tsv", type=["tsv", "txt"])
        if up is not None:
            _df = pd.read_csv(up, sep="\t", index_col="gene")
            gene_order = list(_df.index)
            patterns = {c: _df[c].to_numpy() for c in _df.columns}
    else:
        path = st.text_input("Ruta", value=default or "results_demo/calibrated_patterns.tsv",
                              label_visibility="collapsed")
        if Path(path).exists():
            patterns, gene_order = load_calibrated_patterns(path)
        else:
            st.info("Sin patrones cargados. Genera unos con `python3 cli.py demo`, "
                    "o sube un archivo.")

    if patterns:
        st.success(f"{len(gene_order)} genes · {len(patterns)} subtipos")
        with st.expander("Ver panel"):
            st.markdown('<span class="mono" style="font-size:0.8rem">'
                        + " · ".join(gene_order) + "</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div class="scope" style="border:0;margin:0;padding:0">'
        'Herramienta de investigación. No es un dispositivo médico ni una ayuda '
        'a la decisión clínica validada.</div>', unsafe_allow_html=True)


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

W, _labels, _ = build_model_from_patterns(patterns)
n_genes = len(gene_order)

tab_muestras, tab_traj, tab_tx, tab_metodo = st.tabs(
    ["Muestras", "Trayectoria", "Intervención", "Método"]
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
            try:
                z = zscore_genes(df, gene_order)
            except ValueError as err:
                st.error(
                    f"No se puede normalizar la tabla: {err}\n\n"
                    "Ocurre cuando un gen tiene el mismo valor en todas las muestras "
                    "(varianza cero) o contiene celdas vacías. Revisa esa columna antes "
                    "de volver a cargarla."
                )
                st.stop()

            scored = score_cohort(z, gene_order, patterns)
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

            st.dataframe(
                scored[["predicted_cms", "classification_confidence"] + gene_order],
                use_container_width=True, height=280,
            )
            st.download_button("Descargar tabla clasificada",
                                scored.to_csv(sep="\t", index=False).encode(),
                                file_name="scored_cohort.tsv")

# ======================================================================
# 2. TRAYECTORIA
# ======================================================================
with tab_traj:
    st.markdown("Simula el seguimiento post-quirúrgico: el estado parte del origen "
                "(sin enfermedad residual) y se detecta el momento en que la señal "
                "molecular reaparece.")

    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Atractor de la recaída", list(patterns.keys()),
                           index=len(patterns) - 1,
                           format_func=lambda x: CMS_SHORT.get(x, x))
    onset = c2.slider("Inicio de la recaída (mes)", 3, 24, 15, step=3)
    n_checks = c3.slider("Chequeos de seguimiento", 4, 16, 10)

    t_checks, x_series = simulate_longitudinal_patient(
        W, gene_order, patterns[target], n_genes,
        n_timepoints=n_checks, recurrence_onset_month=onset)
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

        ax2.plot(t_checks, hazard, color=CMS_COLOR.get(target, "#D55E00"), marker="o",
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
# 3. INTERVENCION
# ======================================================================
with tab_tx:
    st.markdown("Compara la misma trayectoria de recaída con y sin intervención — "
                "el contrafactual que un score estático no puede producir.")

    c1, c2, c3, c4 = st.columns(4)
    tx_target = c1.selectbox("Atractor de la recaída", list(patterns.keys()),
                              key="tx_t", format_func=lambda x: CMS_SHORT.get(x, x))
    treatment = c2.selectbox("Tratamiento", list(TREATMENT_MECHANISMS.keys()))
    tx_onset = c3.slider("Inicio del tratamiento (mes)", 3, 30, 18, step=3)
    ras = c4.selectbox("Estatus RAS/BRAF", ["desconocido", "wild-type", "mutante"],
                        help="Solo afecta a anti_egfr.")

    ras_map = {"wild-type": True, "mutante": False, "desconocido": None}
    _, x_base = simulate_with_optional_treatment(
        W, n_genes, gene_order, patterns[tx_target], patterns,
        treatment=None, n_timepoints=10, recurrence_onset_month=15)
    t_checks, x_tx = simulate_with_optional_treatment(
        W, n_genes, gene_order, patterns[tx_target], patterns, treatment=treatment,
        treatment_onset_month=tx_onset, ras_braf_wildtype=ras_map[ras],
        n_timepoints=10, recurrence_onset_month=15)
    h_base, h_tx = hazard_from_trajectory(x_base), hazard_from_trajectory(x_tx)
    delta = h_base[-1] - h_tx[-1]

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.plot(t_checks, h_base, color="#8A8F98", marker="o", markersize=3.5,
                linewidth=2, label="Sin tratamiento")
        ax.plot(t_checks, h_tx, color=CMS_COLOR.get(tx_target, "#0072B2"), marker="o",
                markersize=3.5, linewidth=2, label=f"Con {treatment}")
        ax.fill_between(t_checks, h_tx, h_base, where=(h_base >= h_tx),
                        color=CMS_COLOR.get(tx_target, "#0072B2"), alpha=0.10)
        ax.axvline(15, color="#9AA0A6", linestyle="--", linewidth=1)
        ax.axvline(tx_onset, color=CMS_COLOR.get(tx_target, "#0072B2"),
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
        st.markdown(readout("Riesgo final · sin tratamiento", f"{h_base[-1]:.2f}",
                             accent="#8A8F98"), unsafe_allow_html=True)
        st.markdown(readout("Riesgo final · con tratamiento", f"{h_tx[-1]:.2f}",
                             accent=CMS_COLOR.get(tx_target, "#0072B2")), unsafe_allow_html=True)
        st.markdown(readout("Diferencia", f"{delta:+.2f}",
                             accent="#1B7F5A" if delta > 0.01 else "#8A8F98"),
                    unsafe_allow_html=True)

        if abs(delta) < 0.01:
            st.info(f"**{treatment}** no aplica a un paciente que se dirige a "
                    f"**{CMS_SHORT.get(tx_target)}**. Es el comportamiento esperado: los "
                    "criterios de eficacia reflejan qué pacientes se benefician según la "
                    "evidencia clínica publicada.")
        st.caption(describe_treatment(treatment))

    st.markdown(
        '<div class="scope">La <strong>dirección</strong> del efecto está fundamentada en '
        'literatura clínica; la <strong>magnitud</strong> no está calibrada contra datos '
        'reales de tratamiento. Exploración in silico.</div>', unsafe_allow_html=True)

# ======================================================================
# 4. METODO
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
            "Cohorte": ["GSE39582", "GSE17536", "GSE17537", "GSE14333"],
            "Rol": ["Entrenamiento", "Externa", "Externa (intacta)", "Externa"],
            "n": [557, 145, 55, 126],
            "p": ["0.00039", "0.090", "0.71", "0.59"],
        }), hide_index=True, use_container_width=True)
        st.caption("Cox estratificado combinando las tres externas (n=326): p = 0.045 global. "
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

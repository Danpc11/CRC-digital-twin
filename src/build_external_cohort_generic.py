"""
build_external_cohort_generic.py

Version generalizada para agregar cohortes CRCSC adicionales (mas alla
de GSE39582/GSE17536/GSE17537) al analisis de validacion externa
combinada. Ya tienen etiqueta CMS oficial en cms_labels_public_all.txt:
    gse2109, gse14333, gse13294, gse37892, gse33113, gse20916,
    gse13067, gse35896, gse23878
(petacc3 y kfsyscc tambien aparecen en el dump del consorcio, pero no
son accesiones GSE literales -- PETACC-3 en particular es un ensayo
clinico europeo que probablemente requiere acceso controlado, no esta
en GEO publico sin mas. Se dejan fuera de este script; investigar
acceso por separado si se quieren incluir.)

A diferencia de los scripts especificos por cohorte (build_gse17536_
dataset.py, etc.), este NO asume plataforma ni nombres de columna de
supervivencia -- las distintas cohortes usan cosas distintas. Flujo de
dos pasos obligatorio:

PASO 1 -- diagnostico (no descarga nada nuevo si ya tienes el series_matrix):
    curl -L -o data/raw_geo/GSE13294_series_matrix.txt.gz \\
      "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13294/matrix/GSE13294_series_matrix.txt.gz"

    python3 src/build_external_cohort_generic.py --gse GSE13294 --diagnose

PASO 2 -- construccion, con los nombres reales de columna que el diagnostico revelo:
    python3 src/build_external_cohort_generic.py --gse GSE13294 \\
        --dataset gse13294 \\
        --duration-col "characteristics__xxx" \\
        --event-col "characteristics__yyy" \\
        --event-map "recurrence=1,no recurrence=0"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW_GEO = Path(__file__).resolve().parents[1] / "data" / "raw_geo"
RAW_SYNAPSE = Path(__file__).resolve().parents[1] / "data" / "raw_synapse"

TARGET_SYMBOLS = ["MLH1", "GNLY", "USP18", "MYC", "AXIN2", "GALNT8", "CPS1", "AGR2", "VIM", "TGFB1"]

CMS_RENAME = {
    "CMS1": "CMS1_MSI_immune",
    "CMS2": "CMS2_canonical_WNT",
    "CMS3": "CMS3_metabolic",
    "CMS4": "CMS4_mesenchymal",
    "NOLBL": "none",
}
CMS_LABEL_COLUMN = "CMS_final_network_plus_RFclassifier_in_nonconsensus_samples"


def get_platform_id(path):
    """Escanea el series_matrix por la linea '!Series_platform_id'."""
    import gzip
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if line.startswith("!Series_platform_id"):
                return line.split("\t")[1].strip().strip('"')
    return None


def parse_platform_annotation(path):
    table_lines, header, in_table = [], None, False
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                in_table = False
                continue
            if in_table:
                if header is None:
                    header = line.split("\t")
                else:
                    table_lines.append(line)
    if header is None:
        raise ValueError(f"No se encontro tabla de plataforma en {path}")
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", names=header)


def derive_survival_from_dates(
    pheno: "pd.DataFrame", start_col: str, event_date_col: str, censor_date_col: str,
    na_tokens: tuple = ("NA", "N/A", "na", "", "none", "None"),
) -> "pd.DataFrame":
    """
    Deriva duracion_meses/evento a partir de tres columnas de fecha --
    patron comun en GEO (fecha de cirugia, fecha del evento de interes,
    fecha de ultimo contacto) en vez de una duracion ya calculada.

    CUIDADO real encontrado en produccion (GSE37892): el token "NA"
    llega como TEXTO LITERAL, no como valor faltante -- pd.isna()/notna()
    no lo detectan por defecto, lo que hace que TODAS las muestras
    parezcan tener el evento (100% en vez de la proporcion real). Hay
    que reemplazar los tokens de texto por NaN real ANTES de cualquier
    chequeo de nulidad o parseo de fecha.

    evento=1 si event_date_col tiene una fecha real (no nulo tras la
    limpieza de tokens) -- duracion = event_date - start_date.
    evento=0 si event_date_col es nulo -- duracion = censor_date - start_date
    (censura por ultimo contacto sin el evento).
    """
    import pandas as pd

    df = pheno.copy()
    for col in (start_col, event_date_col, censor_date_col):
        df[col] = df[col].replace(list(na_tokens), pd.NA)
        df[col] = pd.to_datetime(df[col], errors="coerce")

    n_total = len(df)
    n_sin_fecha_inicio = df[start_col].isna().sum()
    if n_sin_fecha_inicio:
        print(f"AVISO: {n_sin_fecha_inicio}/{n_total} muestras sin fecha de inicio valida -- "
              "se excluiran (no se puede calcular duracion sin punto de partida).")

    evento = df[event_date_col].notna().astype(int)
    fecha_fin = df[event_date_col].where(evento == 1, df[censor_date_col])

    n_sin_fecha_fin = (evento == 0) & fecha_fin.isna()
    if n_sin_fecha_fin.sum():
        print(f"AVISO: {n_sin_fecha_fin.sum()}/{n_total} muestras censuradas sin fecha de "
              "ultimo contacto valida -- se excluiran.")

    duracion_meses = (fecha_fin - df[start_col]).dt.days / 30.4375  # promedio dias/mes

    n_negativa = (duracion_meses < 0).sum()
    if n_negativa:
        print(f"AVISO: {n_negativa} muestras con duracion NEGATIVA (fecha de fin antes que "
              "inicio) -- probable error de captura de datos, se excluiran.")
        duracion_meses = duracion_meses.where(duracion_meses >= 0)

    df["_derived_duration_months"] = duracion_meses
    df["_derived_event"] = evento

    n_evento = int((evento == 1).sum())
    n_valido = duracion_meses.notna().sum()
    print(f"Supervivencia derivada de fechas: {n_valido}/{n_total} muestras validas, "
          f"{n_evento} eventos ({100*n_evento/n_total:.1f}% tasa cruda, antes de excluir invalidas).")

    return df


# Rango plausible para RFS/DFS en MESES en cohortes de CRC (seguimiento
# habitual 5-10 anios). Una mediana > 240 casi seguro son DIAS; una
# mediana < 1.5 casi seguro son ANIOS. Sin esta comprobacion, una
# cohorte en dias entraba al Cox agrupado como "meses" y solo se notaba
# por HR raros (2026-09-12).
DURATION_MONTHS_MEDIAN_MAX = 240.0
DURATION_MAX_IF_YEARS = 15.0


EXPR_LOG2_MEDIAN_MAX = 50.0
EXPR_LOG2_MAX_MAX = 100.0


def ensure_log2_scale(expr: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Comprueba que la matriz de expresion (probes x muestras) este en
    escala log2 y, si parece lineal, la transforma con log2(x+1).

    Criterio: expresion log2 de microarreglo tiene mediana ~5-9 y maximo
    ~14-16. Mediana > 50 o maximo > 100 son valores de escala LINEAL
    (MAS5, RMA sin log). El z-score por gen amortigua la diferencia de
    ESCALA pero no la de FORMA de la distribucion: en lineal, los genes
    muy expresados dominan y la correlacion con los centroides cambia.
    Devuelve (matriz, "log2" | "lineal->log2").
    """
    vals = expr.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        print("AVISO: matriz de expresion vacia; no se puede validar la escala.")
        return expr, "sin_datos"
    med, mx, mn = float(np.median(finite)), float(finite.max()), float(finite.min())
    print(f"Escala de expresion: mediana={med:.2f}, min={mn:.2f}, max={mx:.2f}")
    if med > EXPR_LOG2_MEDIAN_MAX or mx > EXPR_LOG2_MAX_MAX:
        msg = ("La expresion parece estar en escala LINEAL (no log2). Los centroides "
               "calibrados y las estadisticas de referencia estan en log2.")
        if strict:
            print("AVISO: " + msg + " Se aplica log2(x+1).")
            if mn < 0:
                raise ValueError("Valores negativos con escala aparentemente lineal: revisa la matriz.")
            return np.log2(expr.astype(float) + 1.0), "lineal->log2"
        print("AVISO: " + msg + " NO se transforma (--no-auto-log2).")
        return expr, "lineal_sin_transformar"
    return expr, "log2"


DURATION_UNIT_FACTORS = {"months": 1.0, "days": 1.0 / 30.4375, "years": 12.0}


def convert_duration_units(duration: pd.Series, units: str) -> pd.Series:
    """
    Convierte la duracion a MESES desde la unidad declarada.

    Existe porque GSE33113 anota "time to meta or recurrence" en DIAS:
    sin convertir, una mediana de 1179 dias entraba al pipeline como
    1179 "meses" y los horizontes de calibracion a 36/60 meses (y el
    Cox agrupado junto a cohortes en meses) quedaban sin sentido.
    """
    if units not in DURATION_UNIT_FACTORS:
        raise ValueError(f"--duration-units debe ser uno de {sorted(DURATION_UNIT_FACTORS)}")
    factor = DURATION_UNIT_FACTORS[units]
    if units != "months":
        print(f"Convirtiendo duracion de {units} a meses (factor {factor:.6g}).")
    return pd.to_numeric(duration, errors="coerce") * factor


EVENT_RATE_MAX_PLAUSIBLE = 0.60


def check_event_coding(event: pd.Series, duration: pd.Series, strict: bool = True) -> dict:
    """
    Detecta una columna de evento invertida (indicador de CENSURA leido
    como evento).

    Dos senales, ambas vistas con GSE14333 (columna 'DFS_Cens', donde 1
    = censurado): (1) una tasa de "evento" implausible para RFS en
    estadio I-III (>60%; lo normal es 20-35%), y (2) los supuestos
    eventos con tiempos de seguimiento MAS LARGOS que los censurados,
    cuando por definicion una recidiva ocurre antes del fin de
    seguimiento. La combinacion de ambas es casi diagnostica.
    """
    ev = pd.to_numeric(event, errors="coerce")
    dur = pd.to_numeric(duration, errors="coerce")
    ok = ev.notna() & dur.notna()
    ev, dur = ev[ok], dur[ok]
    if len(ev) == 0 or ev.nunique() < 2:
        return {"event_rate": float("nan"), "sospechoso": False}
    rate = float(ev.mean())
    t_event, t_cens = float(dur[ev == 1].mean()), float(dur[ev == 0].mean())
    print(f"Codificacion de evento: tasa={rate:.1%}; tiempo medio con evento={t_event:.1f} "
          f"meses vs sin evento={t_cens:.1f} meses")
    alta = rate > EVENT_RATE_MAX_PLAUSIBLE
    invertido = t_event > t_cens
    res = {"event_rate": rate, "tiempo_medio_evento": t_event,
           "tiempo_medio_censurado": t_cens, "sospechoso": bool(alta and invertido)}
    if alta and invertido:
        msg = (f"La columna de evento parece INVERTIDA: tasa de eventos {rate:.1%} "
               f"(>{EVENT_RATE_MAX_PLAUSIBLE:.0%}) y los 'eventos' tienen seguimiento mas "
               "largo que los censurados. Columnas llamadas '*_Cens' suelen codificar "
               "1=censurado. Reconstruye con --event-map \"0=1,1=0\" o usa "
               "--allow-suspicious-event-coding si ya lo verificaste.")
        if strict:
            raise ValueError(msg)
        print("AVISO: " + msg)
    elif alta:
        print(f"AVISO: tasa de eventos {rate:.1%}, alta para RFS en estadio I-III. Verifica.")
    return res


def check_duration_units(duration_months: pd.Series, strict: bool = True) -> str:
    """Comprueba que la duracion tenga pinta de estar en meses.

    Devuelve "meses", "dias?" o "anios?". Con strict=True lanza
    ValueError si no parece meses; con strict=False solo avisa.
    """
    vals = pd.to_numeric(duration_months, errors="coerce").dropna()
    if len(vals) == 0:
        print("AVISO: no hay duraciones numericas para validar unidades.")
        return "sin_datos"
    med, mx = float(vals.median()), float(vals.max())
    if med > DURATION_MONTHS_MEDIAN_MAX:
        verdict, hint = "dias?", "divide entre 30.4375"
    elif mx <= DURATION_MAX_IF_YEARS:
        # un seguimiento cuyo MAXIMO no llega a 15 "meses" no es un estudio
        # de RFS en CRC; casi seguro son anios
        verdict, hint = "anios?", "multiplica por 12"
    else:
        verdict, hint = "meses", ""
    print(f"Unidades de duracion: mediana={med:.1f}, max={mx:.1f} -> {verdict}")
    if verdict != "meses":
        msg = (f"La duracion no parece estar en MESES ({verdict}; {hint}). "
               "Todo el pipeline (Cox agrupado, horizontes de calibracion a 36/60 meses) "
               "asume meses. Corrige la columna o usa --allow-suspicious-units si ya lo verificaste.")
        if strict:
            raise ValueError(msg)
        print("AVISO: " + msg)
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gse", required=True, help="Ej. GSE13294")
    parser.add_argument("--diagnose", action="store_true",
                         help="Solo inspeccionar plataforma y columnas disponibles, sin construir nada")
    parser.add_argument("--dataset", default=None,
                         help="Valor de la columna 'dataset' en cms_labels_public_all.txt (default: gse en minusculas)")
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--event-col", default=None)
    parser.add_argument("--chemo-col", default=None,
                        help="Columna de quimioterapia adyuvante (p. ej. AdjCTX en GSE14333). "
                             "Se guarda como adjuvant_chemo para cox_treatment_interaction.py.")
    parser.add_argument("--stage-col", default=None,
                         help="Columna de estadio clinico (Dukes/TNM/AJCC). Necesaria para el "
                              "modelo de Cox ajustado (pooled_cox_validation.py --adjust-stage).")
    parser.add_argument("--allow-suspicious-event-coding", action="store_true",
                        help="Degradar a aviso la deteccion de columna de evento invertida")
    parser.add_argument("--duration-units", default="months",
                        choices=["months", "days", "years"],
                        help="Unidad de --duration-col en el fenotipo original. GSE33113 usa days.")
    parser.add_argument("--no-auto-log2", action="store_true",
                        help="No transformar a log2 aunque la expresion parezca lineal")
    parser.add_argument("--allow-suspicious-units", action="store_true",
                        help="No abortar si la duracion parece estar en dias o anios en vez "
                             "de meses (ver check_duration_units). Usar solo si ya lo verificaste.")
    parser.add_argument("--event-map", default=None,
                         help="Mapeo texto->numero si event-col no es ya 0/1, formato 'valorA=1,valorB=0'")
    parser.add_argument("--derive-survival-from-dates", default=None,
                         help="Para cohortes que reportan fechas en vez de duracion ya calculada -- "
                              "formato 'columna_inicio,columna_fecha_evento,columna_fecha_censura'. "
                              "Si se usa, --duration-col/--event-col se ignoran (se generan "
                              "automaticamente como _derived_duration_months/_derived_event).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    matrix_path = RAW_GEO / f"{args.gse}_series_matrix.txt.gz"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"No se encontro {matrix_path}. Descargalo primero con curl "
            f"(ver docstring de este script para el patron de URL)."
        )

    from parse_geo_series_matrix import parse_series_matrix
    pheno, expr = parse_series_matrix(matrix_path)
    expr, expr_scale = ensure_log2_scale(expr, strict=not args.no_auto_log2)

    if args.derive_survival_from_dates:
        start_col, event_date_col, censor_date_col = [
            c.strip() for c in args.derive_survival_from_dates.split(",")]
        pheno = derive_survival_from_dates(pheno, start_col, event_date_col, censor_date_col)
        args.duration_col = "_derived_duration_months"
        args.event_col = "_derived_event"

    platform_id = get_platform_id(matrix_path)
    print(f"Plataforma: {platform_id}")
    annot_path = RAW_GEO / f"{platform_id}.txt"
    has_annot = annot_path.exists()
    print(f"Anotacion local ({annot_path.name}): {'encontrada' if has_annot else 'NO encontrada'}")
    if not has_annot:
        print(
            f"  Descargala con:\n"
            f"  curl -L -o {annot_path} \\\n"
            f'    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=self&acc={platform_id}&form=text&view=full"'
        )

    if args.diagnose:
        print(f"\nMuestras: {len(pheno)}")
        print(f"Columnas de fenotipo disponibles:\n{list(pheno.columns)}")
        print(
            "\nBusca columnas relacionadas a supervivencia (rfs/dfs/relapse/recur/event/delay/time/survival) "
            "e identifica cuales usar como --duration-col y --event-col para el paso de construccion."
        )
        return

    if not (args.duration_col and args.event_col):
        raise ValueError("Faltan --duration-col y/o --event-col. Corre primero con --diagnose.")
    if not has_annot:
        raise FileNotFoundError(f"Falta la anotacion de plataforma en {annot_path}. Descargala primero.")

    dataset_name = args.dataset or args.gse.lower()
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1] / "data" / f"{dataset_name}_cms_labeled.tsv"
    )

    def keep_cols():
        cols = [args.duration_col, args.event_col]
        for flag, col in (("--stage-col", args.stage_col), ("--chemo-col", args.chemo_col)):
            if col:
                if col in pheno.columns:
                    cols.append(col)
                else:
                    raise ValueError(
                        f"{flag} '{col}' no existe en el fenotipo. "
                        f"Corre con --diagnose para ver las columnas disponibles.")
        return cols

    print("\nCargando anotacion de plataforma...")
    annot = parse_platform_annotation(annot_path)
    symbol_col_candidates = [c for c in annot.columns if "symbol" in c.lower()]
    if not symbol_col_candidates:
        raise ValueError(f"No se encontro columna de simbolo de gen. Columnas: {list(annot.columns)}")
    symbol_col = symbol_col_candidates[0]
    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    probe_to_symbol = annot.set_index(id_col)[symbol_col]

    print(f"Mapeando probes a simbolos: {TARGET_SYMBOLS}")
    gene_data = {}
    for symbol in TARGET_SYMBOLS:
        matching_probes = probe_to_symbol[probe_to_symbol == symbol].index
        matching_probes = [p for p in matching_probes if p in expr.index]
        if not matching_probes:
            print(f"  AVISO: {symbol} no disponible en esta plataforma -- se omite")
            continue
        gene_data[symbol] = expr.loc[matching_probes].mean(axis=0)
        if len(matching_probes) > 1:
            print(f"  {symbol}: promediado sobre {len(matching_probes)} probes")

    if not gene_data:
        raise ValueError("Ningun gen del panel se pudo mapear en esta plataforma.")

    gene_expr = pd.DataFrame(gene_data)

    labels_path = RAW_SYNAPSE / "tcga_cms_labels" / "cms_labels_public_all.txt"
    labels = pd.read_csv(labels_path, sep="\t")
    gse_labels = labels[labels["dataset"] == dataset_name].set_index("sample")
    print(f"\nEtiquetas CMS para dataset='{dataset_name}': {len(gse_labels)} muestras")

    if len(gse_labels) == 0:
        print("AVISO: sin etiqueta CMS oficial -- se continua con cms_label='none' para todas las muestras.")
        merged = gene_expr.join(pheno[keep_cols()], how="left")
        merged["cms_label"] = "none"
    else:
        direct_overlap = len(set(gene_expr.index) & set(gse_labels.index))
        pheno_for_join = pheno  # por default, pheno sigue indexado por GSM

        if direct_overlap > 0:
            print(f"Cruce directo por GSM: {direct_overlap}/{len(gse_labels)} coinciden.")
        else:
            # El GSM de GEO no coincide con el ID que usa el consorcio --
            # visto en GSE33113: cms_labels usa 'col001' mientras la
            # expresion viene indexada por 'GSM820048'. El puente esta en
            # Sample_title o Sample_description (confirmado en GSE33113:
            # Sample_title = 'col001'). Probar ambos campos antes de
            # rendirse.
            print("AVISO: 0 coincidencias por GSM directo -- probando puente via "
                  "Sample_title / Sample_description...")
            bridge_col = None
            for candidate in ("Sample_title", "Sample_description"):
                if candidate not in pheno.columns:
                    continue
                bridge_overlap = len(set(pheno[candidate]) & set(gse_labels.index))
                print(f"  {candidate}: {bridge_overlap}/{len(gse_labels)} coincidencias")
                if bridge_overlap > 0:
                    bridge_col = candidate
                    break

            if bridge_col is None:
                raise ValueError(
                    f"No se encontro ningun puente de ID entre la expresion (GSM) y las "
                    f"etiquetas CMS de '{dataset_name}'. Revisa manualmente los campos "
                    f"disponibles en el fenotipo: {list(pheno.columns)}"
                )

            print(f"  Usando '{bridge_col}' como puente de ID.")
            gsm_to_bridge = pheno[bridge_col].to_dict()
            gene_expr = gene_expr.rename(index=gsm_to_bridge)
            pheno_for_join = pheno.rename(index=gsm_to_bridge)

        merged = gene_expr.join(gse_labels[[CMS_LABEL_COLUMN]], how="inner")
        merged = merged.join(pheno_for_join[keep_cols()], how="left")
        merged = merged.rename(columns={CMS_LABEL_COLUMN: "cms_label"})
        merged["cms_label"] = merged["cms_label"].replace(CMS_RENAME)

    rename_map = {args.duration_col: "relapse_free_months",
                  args.event_col: "relapse_event"}
    if args.stage_col:
        rename_map[args.stage_col] = "stage"
    if args.chemo_col:
        rename_map[args.chemo_col] = "adjuvant_chemo"
    merged = merged.rename(columns=rename_map)

    # GEO a veces codifica valores faltantes como texto literal ("NA",
    # "N/A", etc.) en vez de celda vacia -- normalizar a NaN real ANTES
    # de mapear/validar, o el validador de --event-map los marca como
    # "no cubiertos" y truena innecesariamente.
    NA_TOKENS = {"NA", "N/A", "n/a", "na", "NaN", "nan", ""}
    merged["relapse_free_months"] = merged["relapse_free_months"].replace(NA_TOKENS, pd.NA)
    merged["relapse_event"] = merged["relapse_event"].replace(NA_TOKENS, pd.NA)

    merged["relapse_free_months"] = pd.to_numeric(merged["relapse_free_months"], errors="coerce")
    merged["relapse_free_months"] = convert_duration_units(
        merged["relapse_free_months"], args.duration_units)
    check_duration_units(merged["relapse_free_months"], strict=not args.allow_suspicious_units)
    check_event_coding(merged["relapse_event"], merged["relapse_free_months"],
                       strict=not args.allow_suspicious_event_coding)

    if args.event_map:
        event_map = dict(pair.split("=") for pair in args.event_map.split(","))
        event_map = {k: int(v) for k, v in event_map.items()}
        unmapped = set(merged["relapse_event"].dropna().unique()) - set(event_map.keys())
        if unmapped:
            raise ValueError(f"Valores no cubiertos por --event-map: {unmapped}. Mapeo actual: {event_map}")
        merged["relapse_event"] = merged["relapse_event"].map(event_map)
    else:
        merged["relapse_event"] = pd.to_numeric(merged["relapse_event"], errors="coerce")

    merged.index.name = "sample_id"
    merged = merged.reset_index()

    print(f"\n{len(merged)} muestras con expresion. "
          f"{merged['relapse_free_months'].isna().sum()} sin duracion, "
          f"{merged['relapse_event'].isna().sum()} sin evento.")
    print("\nDistribucion CMS:")
    print(merged["cms_label"].value_counts())

    merged.to_csv(output_path, sep="\t", index=False)
    print(f"\nGuardado: {output_path} ({len(merged)} muestras, {len(gene_data)} genes)")
    print(
        f"\nSiguiente paso -- validacion externa con patrones congelados:\n"
        f"  python3 src/external_validation.py \\\n"
        f"    --patterns results_gse39582_final/calibrated_patterns.tsv \\\n"
        f"    --input {output_path} \\\n"
        f"    --output results_external_{dataset_name}/\n"
        f"\nY luego agregar '{output_path.name} scored' al pooled_cox_validation.py junto con las demas cohortes."
    )


if __name__ == "__main__":
    main()

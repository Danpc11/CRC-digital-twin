"""
Shared utilities for the community-detection scripts (format_for_infomap.py,
run_leiden.py) operating on ARACNe3's consolidated network. Centralizes
edge loading/filtering so both algorithms see exactly the same network --
required for a fair Infomap vs. Leiden comparison.
"""

import sys

import pandas as pd

# 10-gene panel from the CRC-digital-twin project.
PANEL_GENES = [
    "MLH1", "GNLY", "USP18", "MYC", "AXIN2",
    "FABP1", "CPS1", "SI", "VIM", "TGFB1",
]

WEIGHT_COLUMNS = {
    "mi": "mi.values",
    "count": "count.values",
}


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_edges(input_path, weight="mi", min_count=1, directed=False):
    """Read ARACNe3's consolidated-net_<runid>.tsv and return a
    ['source', 'target', 'weight'] DataFrame ready for graph
    construction.

    - Filters by count.values >= min_count (number of ARACNe3
      subnetworks supporting the edge -- a reproducibility signal,
      distinct from and complementary to ARACNe3's own --alpha).
    - If directed=False (default), collapses each reciprocal A<->B pair
      into a single undirected edge (mean weight) -- with -r = all
      genes, ARACNe3 tests every pair in both directions and the weight
      is essentially identical either way.
    """
    if not input_path.exists():
        fail(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, sep="\t")
    required = {"regulator.values", "target.values", "mi.values", "count.values"}
    missing = required - set(df.columns)
    if missing:
        fail(
            f"Missing expected columns from ARACNe3's consolidated "
            f"network: {sorted(missing)}. Columns present: "
            f"{df.columns.tolist()}"
        )

    n_total = len(df)
    df = df[df["count.values"] >= min_count].copy()
    print(f"Edges: {n_total} total, {len(df)} after filtering "
          f"count.values >= {min_count}")

    weight_col = WEIGHT_COLUMNS[weight]
    df = df.rename(columns={
        "regulator.values": "source",
        "target.values": "target",
        weight_col: "weight",
    })[["source", "target", "weight"]]

    if not directed:
        pair = pd.DataFrame({
            "a": df[["source", "target"]].min(axis=1),
            "b": df[["source", "target"]].max(axis=1),
            "weight": df["weight"],
        })
        n_before = len(pair)
        pair = pair.groupby(["a", "b"], as_index=False)["weight"].mean()
        print(f"Undirected network: {n_before} directed edges -> "
              f"{len(pair)} undirected edges (reciprocal pairs "
              f"collapsed by mean).")
        df = pair.rename(columns={"a": "source", "b": "target"})

    return df


def report_panel_modules(modules_df, name_col, module_col, panel_genes=PANEL_GENES):
    """Print each panel gene's module and its co-members ("friends").
    modules_df must have columns [name_col, module_col].
    """
    present = [g for g in panel_genes if g in set(modules_df[name_col])]
    missing = [g for g in panel_genes if g not in present]
    if missing:
        print(f"WARNING: dt panel genes absent from this network: {missing}")

    panel_rows = modules_df[modules_df[name_col].isin(present)]
    print("\nModule for each dt panel gene:")
    print(panel_rows.to_string(index=False))

    print("\n'Friends' (module co-members) per panel gene:")
    for gene in present:
        row = modules_df[modules_df[name_col] == gene]
        if row.empty:
            continue
        mod_id = row[module_col].iloc[0]
        friends = modules_df[
            (modules_df[module_col] == mod_id) & (modules_df[name_col] != gene)
        ][name_col].tolist()
        print(f"  {gene} (module {mod_id}, {len(friends)} co-members): "
              f"{friends[:15]}{' ...' if len(friends) > 15 else ''}")

#!/usr/bin/env python3
"""
Runs the Leiden algorithm (Traag, van Eck & Waltman 2019, via
python-igraph + leidenalg) on ARACNe3's consolidated network, as a
comparison point against Infomap (format_for_infomap.py). Uses the same
network_utils.load_edges() as the Infomap script, so both algorithms
see exactly the same network.

Why 'modularity' instead of igraph's default ('CPM')
--------------------------------------------------------
Graph.community_leiden defaults to the Constant Potts Model (CPM) with
resolution=1.0. For CPM, resolution is a literal density threshold (a
module only counts as good if its internal density exceeds it) --
resolution=1.0 demands near-clique modules, absurdly strict for a real
co-expression/MI network, and in practice yields mostly 1-2 gene
modules. This script defaults to objective_function=modularity, where
resolution=1.0 is classic Newman-Girvan modularity -- a reasonable
default, comparable in spirit to Infomap's markov_time=1.0. If using
CPM, choose --resolution based on the network's density.

Usage
-----
    python3 run_leiden.py \\
        --input ../results/crc_net/consolidated-net_crc_net.tsv \\
        --min-count 2 \\
        --output-modules ../results/crc_net/crc_net_leiden_modules.tsv
"""

import argparse
from pathlib import Path

import igraph

from network_utils import PANEL_GENES, WEIGHT_COLUMNS, load_edges, report_panel_modules


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path,
                    help="ARACNe3 consolidated-net_<runid>.tsv")
    p.add_argument("--output-modules", required=True, type=Path,
                    help="output TSV for gene/module")
    p.add_argument("--weight", choices=list(WEIGHT_COLUMNS), default="mi",
                    help="edge weight column (default 'mi').")
    p.add_argument("--min-count", type=int, default=1,
                    help="drop edges supported by fewer than N ARACNe3 "
                         "subnetworks (default 1 = no filter).")
    p.add_argument("--directed", action="store_true",
                    help="keep regulator->target. Default: undirected "
                         "(see network_utils.load_edges docstring).")
    p.add_argument("--objective", choices=["modularity", "CPM"],
                    default="modularity",
                    help="Leiden objective function (default "
                         "'modularity', see script docstring).")
    p.add_argument("--resolution", type=float, default=1.0,
                    help="resolution parameter (default 1.0). Higher = "
                         "smaller/more numerous modules. With "
                         "--objective CPM this number is a literal "
                         "density, don't use 1.0 without thinking "
                         "about it (see docstring).")
    p.add_argument("--n-iterations", type=int, default=-1,
                    help="Leiden refinement iterations (default -1 = "
                         "iterate until stable; igraph's default of 2 "
                         "may not have converged).")
    p.add_argument("--seed", type=int, default=9001)
    return p.parse_args()


def main():
    args = parse_args()
    df = load_edges(args.input, args.weight, args.min_count, args.directed)

    print(f"Nodes (genes): {len(set(df['source']) | set(df['target']))}")

    edges = list(df.itertuples(index=False, name=None))
    g = igraph.Graph.TupleList(
        edges, directed=args.directed, edge_attrs=["weight"]
    )

    print(f"Running Leiden ({args.objective}, resolution={args.resolution}, "
          f"n_iterations={args.n_iterations}) on {g.vcount()} nodes and "
          f"{g.ecount()} edges ...")

    import random
    random.seed(args.seed)
    clustering = g.community_leiden(
        objective_function=args.objective,
        weights="weight",
        resolution=args.resolution,
        n_iterations=args.n_iterations,
    )
    print(f"  quality: {clustering.quality:.4f}, "
          f"{len(clustering)} modules")

    import pandas as pd
    modules = pd.DataFrame({
        "name": g.vs["name"],
        "module_id": clustering.membership,
    })
    modules["module_size"] = modules.groupby("module_id")["module_id"].transform("count")
    modules = modules.sort_values(["module_id", "name"]).reset_index(drop=True)

    args.output_modules.parent.mkdir(parents=True, exist_ok=True)
    modules.to_csv(args.output_modules, sep="\t", index=False)
    print(f"Modules written to: {args.output_modules}")

    missing_panel = [g_ for g_ in PANEL_GENES if g_ not in set(modules["name"])]
    if missing_panel:
        print(f"WARNING: dt panel genes absent from this network: {missing_panel}")

    report_panel_modules(modules, name_col="name", module_col="module_id")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Converts ARACNe3's consolidated network (consolidated-net_<runid>.tsv)
to Pajek format (.net), also readable by igraph and Infomap's R package
-- useful if this pipeline ever moves from Python to R. Optionally runs
Infomap directly (Python API) and writes each gene's module assignment.

See network_utils.py for the shared edge loading/filtering logic (used
by run_leiden.py too).

Usage
-----
Reformat only (to use later with the infomap CLI, or from R):

    python3 format_for_infomap.py \\
        --input ../results/crc_net/consolidated-net_crc_net.tsv \\
        --output-net ../results/crc_net/crc_net.net

Reformat and run Infomap in one step:

    python3 format_for_infomap.py \\
        --input ../results/crc_net/consolidated-net_crc_net.tsv \\
        --output-net ../results/crc_net/crc_net.net \\
        --run-infomap --min-count 2 --num-trials 10
"""

import argparse
from pathlib import Path

from network_utils import PANEL_GENES, WEIGHT_COLUMNS, load_edges, report_panel_modules


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path,
                    help="ARACNe3 consolidated-net_<runid>.tsv")
    p.add_argument("--output-net", required=True, type=Path,
                    help="output path in Pajek format (.net)")
    p.add_argument("--weight", choices=list(WEIGHT_COLUMNS), default="mi",
                    help="edge weight column: 'mi' = mutual information "
                         "(default), 'count' = number of ARACNe3 "
                         "subnetworks supporting the edge (confidence).")
    p.add_argument("--min-count", type=int, default=1,
                    help="drop edges supported by fewer than N ARACNe3 "
                         "subnetworks (count.values column). Default 1 "
                         "= no filter.")
    p.add_argument("--directed", action="store_true",
                    help="keep regulator->target as reported by ARACNe3 "
                         "(*Arcs in the .net). Default: undirected "
                         "(*Edges), collapsing each reciprocal A<->B "
                         "pair into one edge.")
    p.add_argument("--run-infomap", action="store_true",
                    help="in addition to writing the .net, run Infomap "
                         "(Python API) on the same network and write "
                         "the gene -> module table.")
    p.add_argument("--output-modules", type=Path, default=None,
                    help="output TSV for gene/module/flow if "
                         "--run-infomap is used (default: same name as "
                         "--output-net with a _modules.tsv suffix).")
    p.add_argument("--depth", type=int, default=-1,
                    help="Infomap module hierarchy level to report: -1 "
                         "= finest (default), 1 = coarsest (top level).")
    p.add_argument("--num-trials", type=int, default=10,
                    help="run N independent optimizations and keep the "
                         "one with lowest codelength (default 10) -- "
                         "Infomap is a stochastic local search, a "
                         "single trial can land in a mediocre local "
                         "optimum.")
    p.add_argument("--markov-time", type=float, default=1.0,
                    help="scales flow to change the cost of moving "
                         "between modules (default 1.0). Higher = "
                         "fewer/larger modules. Raise this if the "
                         "result is over-fragmented.")
    p.add_argument("--seed", type=int, default=9001)
    return p.parse_args()


def write_pajek(df, genes, output_net, directed):
    gene_to_id = {g: i + 1 for i, g in enumerate(genes)}  # Pajek is 1-indexed

    with open(output_net, "w") as fh:
        fh.write(f"*Vertices {len(genes)}\n")
        for g, gid in gene_to_id.items():
            fh.write(f'{gid} "{g}"\n')
        fh.write("*Arcs\n" if directed else "*Edges\n")
        for row in df.itertuples(index=False):
            fh.write(f"{gene_to_id[row.source]} {gene_to_id[row.target]} "
                      f"{row.weight:.6g}\n")

    return gene_to_id


def run_infomap(df, gene_to_id, directed, seed, depth, num_trials, markov_time):
    import infomap

    im = infomap.Infomap(
        directed=directed,
        seed=seed,
        num_trials=num_trials,
        markov_time=markov_time,
        silent=True,
    )
    for g, gid in gene_to_id.items():
        im.add_node(gid, g)
    for row in df.itertuples(index=False):
        im.add_link(gene_to_id[row.source], gene_to_id[row.target], row.weight)

    print(f"Running Infomap on {len(gene_to_id)} nodes and {len(df)} "
          f"edges ({'directed' if directed else 'undirected'}), "
          f"{num_trials} trials, markov_time={markov_time} ...")
    result = im.run()
    print(f"  codelength: {result.codelength:.4f} bits, "
          f"{result.num_top_modules} top-level modules")

    modules = result.to_dataframe(
        columns=["node_id", "name", "module_id", "flow", "path"],
        depth=depth,
        sort=True,
    )
    return modules


def main():
    args = parse_args()
    df = load_edges(args.input, args.weight, args.min_count, args.directed)

    genes = sorted(set(df["source"]) | set(df["target"]))
    print(f"Nodes (genes): {len(genes)}")

    args.output_net.parent.mkdir(parents=True, exist_ok=True)
    gene_to_id = write_pajek(df, genes, args.output_net, args.directed)
    print(f"Network written in Pajek format: {args.output_net}")

    missing_panel = [g for g in PANEL_GENES if g not in gene_to_id]
    if missing_panel:
        print(f"WARNING: dt panel genes absent from this network: {missing_panel}")

    if not args.run_infomap:
        return

    modules = run_infomap(
        df, gene_to_id, args.directed, args.seed, args.depth,
        args.num_trials, args.markov_time,
    )

    out_modules = args.output_modules
    if out_modules is None:
        out_modules = args.output_net.with_name(
            args.output_net.stem + "_modules.tsv"
        )
    modules.to_csv(out_modules, sep="\t", index=False)
    print(f"Modules written to: {out_modules}")

    report_panel_modules(modules, name_col="name", module_col="module_id")


if __name__ == "__main__":
    main()

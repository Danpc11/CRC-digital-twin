"""
Define network neighborhoods for the 10-gene dt panel, for downstream ORA
(over-representation analysis).

Resolves the "consensus vs. single-algorithm" question left open in
CLAUDE.md ("Proximos pasos" #6) by applying ONE uniform rule to every
panel gene present in the min-count-3 network, rather than treating
high- and low-agreement genes differently:

    neighborhood(gene) = Infomap module members(gene) INTERSECT
                          Leiden module members(gene)

Rationale: for genes where the two algorithms already agree closely
(e.g. VIM, Jaccard=1.00) the intersection is ~ the full module either
way, so nothing is lost. For genes where they disagree (e.g. CPS1,
Jaccard=0.05) the intersection keeps only the "core" co-membership both
objective functions (map equation vs. modularity) independently
recovered -- the most defensible set to hand to an enrichment test,
at the cost of a smaller (possibly very small) gene set.

Also reports each algorithm's module alone and the union, so the
trade-off is visible rather than hidden behind the consensus number.

TGFB1 has zero edges surviving min-count>=3 (see CLAUDE.md) and is
therefore absent from both module files -- no community-based
neighborhood can be defined for it from this network. Flagged
separately, not silently dropped.

Final per-gene resolution (decided after inspecting the consensus
numbers below -- see CLAUDE.md "Proximos pasos" and
results/crc_net/panel_neighborhoods_final.tsv for the write-up):

- MLH1, MYC, SI, VIM, FABP1 (Jaccard 0.10-1.00, consensus n=1-9):
  consensus (Infomap cap Leiden module members, mc3). Both algorithms
  agree enough that the intersection is a meaningful, non-empty core.
- GNLY, USP18, AXIN2, CPS1 (Jaccard 0.00-0.13): consensus is empty
  (CPS1) or too small (n=1-3) for ORA to have any power. Fall back to
  the UNION of the two mc3 modules instead -- larger, noisier, and
  explicitly flagged as lower-confidence exploratory input for ORA.
- TGFB1: absent from the mc3 network entirely. Re-running Infomap/
  Leiden on the min-count>=2 network (crc_net_infomap_modules.tsv /
  crc_net_leiden_modules.tsv, already computed) does NOT help -- TGFB1
  falls into the known giant/near-giant module at that threshold
  (Infomap module of 11,382 genes = 56% of the network; Leiden module
  of 2,578 genes), which is uninformative as a "neighborhood" for
  enrichment. Instead, TGFB1 uses its 7 DIRECT edges with
  count.values>=2 in the consolidated network (not a community
  assignment -- local adjacency only), collapsed to undirected. This
  is a different, weaker kind of evidence than the module-based
  neighborhoods used for the other 9 genes and should be reported as
  such.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from network_utils import PANEL_GENES  # noqa: E402

# Genes where the mc3 Infomap/Leiden consensus (intersection) is non-empty
# and large enough to be worth using as-is.
USE_CONSENSUS = {"MLH1", "MYC", "SI", "VIM", "FABP1"}
# Genes where consensus is empty/too small (n<=3) -- use the union instead,
# flagged as lower-confidence.
USE_UNION = {"GNLY", "USP18", "AXIN2", "CPS1"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--infomap-modules", required=True, type=Path,
                    help="crc_net_infomap_modules_mc<N>.tsv (from infomap_modules rule)")
    p.add_argument("--leiden-modules", required=True, type=Path,
                    help="crc_net_leiden_modules_mc<N>_res<R>.tsv (from leiden_modules rule)")
    p.add_argument("--consolidated", required=True, type=Path,
                    help="ARACNe3 consolidated-net_<runid>.tsv (unfiltered) -- "
                         "used as a direct-neighbor fallback for panel genes "
                         "absent from the min-count-filtered community network.")
    p.add_argument("--out-summary", required=True, type=Path,
                    help="per-gene infomap/leiden/consensus/union summary TSV")
    p.add_argument("--out-genesets-dir", required=True, type=Path,
                    help="dir for per-gene <GENE>_{infomap,leiden,consensus,union}.txt")
    p.add_argument("--out-final-summary", required=True, type=Path,
                    help="final per-gene method/confidence resolution TSV")
    p.add_argument("--out-final-dir", required=True, type=Path,
                    help="dir for per-gene <GENE>.txt, ready for ORA")
    p.add_argument("--fallback-min-count", type=int, default=2,
                    help="count.values threshold for the direct-neighbor "
                         "fallback used by panel genes absent from the "
                         "community network (default 2).")
    return p.parse_args()


def module_members(df, name_col, module_col, gene):
    row = df[df[name_col] == gene]
    if row.empty:
        return None, set()
    mod_id = row[module_col].iloc[0]
    members = set(df[df[module_col] == mod_id][name_col]) - {gene}
    return mod_id, members


def direct_neighbors(consolidated_path, gene, min_count):
    df = pd.read_csv(consolidated_path, sep="\t")
    sub = df[(df["regulator.values"] == gene) | (df["target.values"] == gene)]
    sub = sub[sub["count.values"] >= min_count]
    genes = (set(sub["regulator.values"]) | set(sub["target.values"])) - {gene}
    return genes


def main():
    args = parse_args()
    im = pd.read_csv(args.infomap_modules, sep="\t")
    le = pd.read_csv(args.leiden_modules, sep="\t")

    args.out_genesets_dir.mkdir(parents=True, exist_ok=True)
    args.out_final_dir.mkdir(parents=True, exist_ok=True)

    final_rows = []
    rows = []
    for gene in PANEL_GENES:
        im_mod, im_members = module_members(im, "name", "module_id", gene)
        le_mod, le_members = module_members(le, "name", "module_id", gene)

        if im_mod is None and le_mod is None:
            rows.append({
                "gene": gene, "status": "absent_from_mc3_network",
                "infomap_module": None, "leiden_module": None,
                "n_infomap": 0, "n_leiden": 0,
                "n_consensus": 0, "n_union": 0, "jaccard": None,
                "consensus_genes": "",
            })
            print(f"{gene}: ABSENT from min-count-3 network (no edges "
                  f"survived reproducibility filter) -- no module-based "
                  f"neighborhood defined. See TGFB1 handling note.")

            if gene == "TGFB1":
                direct = direct_neighbors(args.consolidated, gene, args.fallback_min_count)
                out = args.out_final_dir / f"{gene}.txt"
                out.write_text("\n".join(sorted(direct)) + "\n")
                final_rows.append({
                    "gene": gene, "method": f"direct_neighbors_mincount{args.fallback_min_count}",
                    "n": len(direct), "confidence": "low (not module-based)",
                    "genes": ",".join(sorted(direct)),
                })
                print(f"  -> fallback: {len(direct)} direct neighbors "
                      f"(count.values>={args.fallback_min_count}): {sorted(direct)}")
            continue

        consensus = im_members & le_members
        union = im_members | le_members
        jaccard = len(consensus) / len(union) if union else 0.0

        rows.append({
            "gene": gene, "status": "ok",
            "infomap_module": im_mod, "leiden_module": le_mod,
            "n_infomap": len(im_members), "n_leiden": len(le_members),
            "n_consensus": len(consensus), "n_union": len(union),
            "jaccard": round(jaccard, 4),
            "consensus_genes": ",".join(sorted(consensus)),
        })

        for label, geneset in [
            ("infomap", im_members),
            ("leiden", le_members),
            ("consensus", consensus),
            ("union", union),
        ]:
            out = args.out_genesets_dir / f"{gene}_{label}.txt"
            out.write_text("\n".join(sorted(geneset)) + ("\n" if geneset else ""))

        print(f"{gene}: infomap module {im_mod} (n={len(im_members)}), "
              f"leiden module {le_mod} (n={len(le_members)}), "
              f"consensus n={len(consensus)}, union n={len(union)}, "
              f"jaccard={jaccard:.3f}")

        if gene in USE_CONSENSUS:
            method, chosen, confidence = "consensus", consensus, "high"
        elif gene in USE_UNION:
            method, chosen, confidence = "union", union, "low (consensus too small/empty)"
        else:
            raise ValueError(f"{gene} not assigned to USE_CONSENSUS or USE_UNION")

        out = args.out_final_dir / f"{gene}.txt"
        out.write_text("\n".join(sorted(chosen)) + ("\n" if chosen else ""))
        final_rows.append({
            "gene": gene, "method": method, "n": len(chosen),
            "confidence": confidence, "genes": ",".join(sorted(chosen)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_summary, sep="\t", index=False)
    print(f"\nWrote summary: {args.out_summary}")
    print(f"Wrote per-gene gene sets (infomap/leiden/consensus/union): "
          f"{args.out_genesets_dir}/")

    final_summary = pd.DataFrame(final_rows).set_index("gene").loc[PANEL_GENES].reset_index()
    final_summary.to_csv(args.out_final_summary, sep="\t", index=False)
    print(f"\nWrote FINAL neighborhood resolution: {args.out_final_summary}")
    print(f"Wrote final per-gene neighborhood files (ready for ORA): {args.out_final_dir}/")
    print(final_summary[["gene", "method", "n", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()

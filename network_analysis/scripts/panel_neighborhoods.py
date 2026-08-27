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

Any panel gene with zero edges surviving min-count>=3 is absent from
both module files -- no community-based neighborhood can be defined for
it from this network (happened for TGFB1 on the 263-sample network; not
guaranteed to stay that way on a different network, so this is checked
per gene at runtime, not assumed). Flagged separately, not silently
dropped.

Per-gene resolution rule (general, computed fresh from each run's own
Jaccard values -- NOT a hardcoded per-gene list. An earlier version of
this script hardcoded which genes counted as "consensus" vs. "union"
based on one specific run's numbers; that broke the instant the network
changed -- TGFB1 moved from "absent" to "present" on a rerun and had
nowhere to go, ValueError):

- If a gene's Infomap/Leiden consensus (intersection) is non-empty AND
  its Jaccard index (consensus / union) is >= JACCARD_THRESHOLD, use
  the consensus: both algorithms agree enough that the intersection is
  a meaningful, non-empty core. Confidence: high.
- Otherwise (consensus empty, or Jaccard below threshold -- the overlap
  is small enough to plausibly be incidental rather than real
  agreement), fall back to the UNION of the two mc3 modules instead --
  larger, noisier, explicitly flagged as lower-confidence exploratory
  input for ORA.
- Any gene absent from the mc3 network entirely (no edges survived the
  count.values reproducibility filter) has no module-based neighborhood
  to fall back to at all. Instead it uses its DIRECT edges at a relaxed
  count.values>=fallback_min_count threshold in the unfiltered
  consolidated network (local adjacency, not a community assignment) --
  a different, weaker kind of evidence than the module-based
  neighborhoods used for the rest of the panel, reported as such.

JACCARD_THRESHOLD=0.2 was picked by inspecting the 263-sample run's own
values: MLH1/MYC/SI/VIM (0.35-1.00) clearly agreed, GNLY/USP18/AXIN2/CPS1
(0.05-0.19) clearly didn't -- 0.2 falls cleanly in the gap. Revisit if a
future network's Jaccard distribution doesn't have as clean a gap.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from network_utils import PANEL_GENES  # noqa: E402

# Minimum Jaccard(Infomap module, Leiden module) to trust the consensus
# (intersection) as real agreement rather than incidental overlap -- see
# module docstring for how this was picked.
JACCARD_THRESHOLD = 0.2


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
                  f"survived reproducibility filter) -- falling back to "
                  f"direct neighbors.")

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

        if consensus and jaccard >= JACCARD_THRESHOLD:
            method, chosen, confidence = "consensus", consensus, "high"
        else:
            method, chosen, confidence = "union", union, "low (consensus too small/empty)"

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

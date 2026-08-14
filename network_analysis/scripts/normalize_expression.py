#!/usr/bin/env python3
"""
Normalizes the TCGA-CRC expression matrix for ARACNe3 in one step.

ARACNe3 only formally requires per-sample sequencing-depth normalization
(CPM, TPM, etc.). Internally it rank-transforms each gene before
estimating mutual information, so a monotonic transform applied equally
to all values (e.g. log2) does not change network topology -- only
distortions in the relative sample ordering per gene matter.

TCGACRC_expression.tsv already comes as log2(RSEM normalized_count + 1)
(non-negative, ~16-19 in housekeeping genes like ACTB/GAPDH, max ~20.8),
and RSEM normalized_count is already depth-corrected. So this script
does NOT re-apply CPM/TPM/log2 -- that would be wrong on already
transformed data. Instead it:
  1. Fails on missing values (ARACNe3 crashes cryptically via stof() on
     an empty/non-numeric field).
  2. Collapses duplicate gene symbols (median, in log space) -- ARACNe3
     aborts entirely if it finds two rows with the same gene name.
  3. Drops zero-variance genes: ARACNe3's rank transform can't
     meaningfully rank a constant vector.
  4. Sanity-checks the value range looks like already-normalized log2
     data (aborts otherwise, instead of silently building a bad network).
  5. Writes the clean matrix and the full gene list (for -r, an
     all-vs-all network) in ARACNe3's expected format.

Usage:
    python3 normalize_expression.py \
        --input ../../data/TCGACRC_expression.tsv \
        --output-matrix ../data/processed/crc_expression_aracne3.tsv \
        --output-regulators ../data/processed/regulators_full.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Housekeeping genes used only for the value-range sanity check (should
# look like log2(RSEM normalized_count+1), typically 14-20 in TCGA
# RNA-seq).
SANITY_GENES = ["ACTB", "GAPDH"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                    help="Raw expression matrix (genes x samples, TSV).")
    p.add_argument("--output-matrix", required=True, type=Path,
                    help="Output path for the ARACNe3-ready matrix (-e).")
    p.add_argument("--output-regulators", required=True, type=Path,
                    help="Output path for the full gene list (-r).")
    p.add_argument("--min-variance", type=float, default=1e-8,
                    help="Minimum across-sample variance to keep a gene "
                         "(default: 1e-8, i.e. requires variance > 0).")
    p.add_argument("--max-sane-value", type=float, default=40.0,
                    help="Maximum plausible value for already-normalized "
                         "log2 data; if exceeded, the input is assumed "
                         "unnormalized and the script aborts (default: 40.0).")
    return p.parse_args()


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()

    if not args.input.exists():
        fail(f"Input file not found: {args.input}")

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, sep="\t", index_col=0)
    print(f"  {df.shape[0]} genes x {df.shape[1]} samples")

    # 1. No missing values.
    n_missing = int(df.isna().sum().sum())
    if n_missing > 0:
        genes_con_na = df.index[df.isna().any(axis=1)].tolist()
        fail(
            f"{n_missing} missing values found in {len(genes_con_na)} "
            f"genes (e.g.: {genes_con_na[:10]}). ARACNe3 crashes on an "
            f"empty/non-numeric field -- impute or drop these "
            f"genes/samples before continuing."
        )

    # 2. Duplicate gene symbols -> median in log space.
    if df.index.duplicated().any():
        dup_genes = sorted(df.index[df.index.duplicated()].unique().tolist())
        print(
            f"WARNING: {len(dup_genes)} duplicate gene symbols detected "
            f"(e.g.: {dup_genes[:10]}). Collapsing by median (ARACNe3 "
            f"aborts if it finds 2 rows with the same name)."
        )
        df = df.groupby(df.index).median()

    # Duplicate samples would be a data error, not something to
    # silently fix.
    if df.columns.duplicated().any():
        dup_samples = sorted(df.columns[df.columns.duplicated()].unique().tolist())
        fail(
            f"{len(dup_samples)} duplicate sample columns (e.g.: "
            f"{dup_samples[:10]}). Check the source matrix."
        )

    # 3. Sanity-check the value range (already normalized + log2).
    global_min = float(df.values.min())
    global_max = float(df.values.max())
    print(f"  Value range: [{global_min:.3f}, {global_max:.3f}]")
    if global_min < -10 or global_max > args.max_sane_value:
        fail(
            f"Value range [{global_min:.3f}, {global_max:.3f}] doesn't "
            f"look like already depth-normalized, log2-scale data "
            f"(expected approx [0, {args.max_sane_value}]). This script "
            f"assumes the input is already log2(depth-normalized value "
            f"+ 1) (e.g. log2(RSEM normalized_count+1)); if the input "
            f"is raw counts, normalize for depth (CPM/TPM) and apply "
            f"log2 before running this script -- ARACNe3 requires it "
            f"as its only formal normalization step (see ARACNe3's "
            f"README)."
        )
    present_sanity = [g for g in SANITY_GENES if g in df.index]
    if present_sanity:
        medians = df.loc[present_sanity].median(axis=1)
        print(f"  Housekeeping gene medians (sanity check): {medians.to_dict()}")

    # 4. Drop zero (or near-zero) variance genes across samples.
    variances = df.var(axis=1, ddof=1)
    zero_var_genes = variances[variances <= args.min_variance].index.tolist()
    if zero_var_genes:
        print(
            f"WARNING: dropping {len(zero_var_genes)} genes with "
            f"variance <= {args.min_variance} across samples (they "
            f"carry no signal and ARACNe3's copula transform can't "
            f"rank a constant vector)."
        )
        df = df.drop(index=zero_var_genes)

    print(f"  Final matrix: {df.shape[0]} genes x {df.shape[1]} samples")

    # 5. Write the matrix and gene list (regulators = all genes, for an
    #    all-vs-all network).
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    args.output_regulators.parent.mkdir(parents=True, exist_ok=True)

    df.index.name = df.index.name or "gene"
    df.to_csv(args.output_matrix, sep="\t", float_format="%.5f")
    print(f"Normalized matrix written to: {args.output_matrix}")

    with open(args.output_regulators, "w") as fh:
        for gene in df.index:
            fh.write(f"{gene}\n")
    print(f"Full gene list (-r, all-vs-all) written to: "
          f"{args.output_regulators} ({df.shape[0]} genes)")


if __name__ == "__main__":
    main()

# network_analysis

A Snakemake pipeline that builds a genome-wide gene regulatory network
(GRN) from TCGA colorectal cancer (CRC) expression data, partitions it
into modules, and characterizes the network neighborhood of the 10-gene
digital-twin panel (`MLH1`, `GNLY`, `USP18`, `MYC`, `AXIN2`, `FABP1`,
`CPS1`, `SI`, `VIM`, `TGFB1`) via over-representation analysis (ORA).

Several of these panel genes are not transcription factors, so the
network is built all-genes-as-regulators rather than restricted to a
curated TF list — otherwise most of the panel would never appear as a
regulator, and the point is to see what each gene associates with, not
only what it might drive.

## Pipeline

```
normalize_expression
        │
        ▼
     aracne3  ──(fallback if the shared binary lacks +x)── prepare_aracne3_binary
        │
        ├──▶ infomap_modules ─┐
        └──▶ leiden_modules ──┤
                               ▼
                     panel_neighborhoods
                               │
                               ▼
                   ora_panel_neighborhoods
                               │
                               ▼
                       ora_sankey_figure
```

1. **`normalize_expression`** (`scripts/normalize_expression.py`) — QC
   and formatting of the raw expression matrix for ARACNe3 (missing
   values, duplicate gene symbols, zero-variance genes), and writes the
   full gene list to use as the regulator set.
2. **`aracne3`** — runs [ARACNe3](https://github.com/califano-lab/ARACNe3)
   in adaptive mode with all genes as candidate regulators, producing a
   consolidated mutual-information network.
3. **`infomap_modules`** / **`leiden_modules`**
   (`scripts/format_for_infomap.py`, `scripts/run_leiden.py`, sharing
   edge-loading logic in `scripts/network_utils.py`) — two independent
   community-detection algorithms (map equation and modularity) run on
   the same filtered, undirected network, to assess how robust module
   assignments are to algorithm choice.
4. **`panel_neighborhoods`** (`scripts/panel_neighborhoods.py`) —
   defines each panel gene's final network neighborhood from its
   Infomap/Leiden module assignments (consensus where the two
   algorithms agree, union where they don't; a direct-neighbor fallback
   for any panel gene absent from the filtered network).
5. **`ora_panel_neighborhoods`** (`scripts/ora_panel_neighborhoods.R`)
   — over-representation analysis of each neighborhood against GO
   Biological Process, GO Molecular Function, KEGG, Reactome, and
   MSigDB Hallmark gene sets.
6. **`ora_sankey_figure`** (`scripts/generate_ora_sankey.py`) — renders
   a static publication figure (SVG + PNG) summarizing neighborhood
   confidence, panel gene, and recovered pathways.

## Requirements

- [Snakemake](https://snakemake.readthedocs.io/) ≥ 8.
- Python: `pandas`, `numpy`, `python-igraph`, `leidenalg`, `infomap`.
- R (only for the `ora_panel_neighborhoods` step): `clusterProfiler`,
  `org.Hs.eg.db`, `ReactomePA`, `msigdbr`.
- A built [ARACNe3](https://github.com/califano-lab/ARACNe3) binary.
- Google Chrome or Chromium, for rasterizing the Sankey figure to PNG
  (optional — the SVG is written regardless; PNG export is skipped
  with a warning if no browser is found).

None of these are pinned in a repo-wide `environment.yml`/
`requirements.txt` yet — install them in whatever environment you run
`snakemake` from.

## Configuration

All parameters live in `config.yaml`, with inline comments explaining
each choice (the community-detection and ORA defaults were tuned
empirically — see the comments before changing them). Two things to
check before your first run:

- **`raw_expression_matrix`** points at
  `data/raw_synapse/tcga_rnaseq/TCGACRC_expression-merged.tsv` (not
  versioned, per the root `.gitignore` policy) — path, filename, and
  format confirmed directly from `src/build_tcga_rnaseq_dataset.py`. A
  local copy is in place for testing, sourced from a byte-identical
  file the repo's maintainer left on shared storage; still worth a
  final confirmation from him that this specific copy matches his
  original.
- **`aracne3.binary`** points at a shared cluster install by default.
  Point it at your own ARACNe3 build; if it isn't executable, the
  pipeline falls back to copying it locally and `chmod +x`-ing the
  copy (see the comments in `Snakefile` and `config.yaml`).

## Running

```bash
cd network_analysis
snakemake --cores N -n     # dry run — check the plan first
snakemake --cores N        # full run
```

Everything is resumable: Snakemake only re-runs steps whose inputs
changed. The ARACNe3 step is the expensive one (hours, depending on
`aracne3.threads` and `aracne3.x_target_per_regulator` — see the
comments in `config.yaml`); the community-detection, neighborhood, ORA,
and figure steps are fast (seconds to a few minutes) once a consolidated
network exists.

## Outputs

All outputs are written under `results/<aracne3.runid>/` and are
**not** committed to this repo (same policy as the rest of the project
— see the root `.gitignore`): the consolidated network, module
assignments, per-gene neighborhoods, ORA result tables, and the Sankey
figure are all regenerable by re-running the pipeline. `bin/`, `data/`,
`logs/`, and `results/` under this directory are gitignored.

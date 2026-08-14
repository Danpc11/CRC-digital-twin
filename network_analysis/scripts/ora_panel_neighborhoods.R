#!/usr/bin/env Rscript
# ORA (over-representation analysis) on the final network neighborhoods of
# the 10 dt panel genes (see panel_neighborhoods.py /
# results/crc_net/neighborhoods_final/<GENE>.txt).
#
# Databases: GO Biological Process, GO Molecular Function, KEGG, Reactome,
# MSigDB Hallmarks.
#
# Universe/background: the 20,260 genes that passed expression QC and were
# submitted to ARACNe3 as candidate network nodes
# (data/processed/regulators_full.txt) -- NOT the 19,031-gene min-count-3
# filtered network, which is already a non-neutral, connectivity-biased
# subset (decided with the user 2026-08-13).

suppressMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(ReactomePA)
  library(msigdbr)
})

# ---- CLI args (no optparse dependency -- just `--flag value` pairs) -----
parse_args <- function() {
  raw <- commandArgs(trailingOnly = TRUE)
  if (length(raw) %% 2 != 0) stop("Expected --flag value pairs, got: ", paste(raw, collapse = " "))
  flags <- raw[c(TRUE, FALSE)]
  values <- raw[c(FALSE, TRUE)]
  if (!all(grepl("^--", flags))) stop("Expected --flag value pairs, got: ", paste(raw, collapse = " "))
  args <- setNames(as.list(values), sub("^--", "", flags))
  required <- c("neighborhoods-dir", "universe", "out-dir")
  missing <- setdiff(required, names(args))
  if (length(missing) > 0) {
    stop("Missing required --", paste(missing, collapse = ", --"), "\n",
         "Usage: Rscript ora_panel_neighborhoods.R --neighborhoods-dir DIR ",
         "--universe FILE --out-dir DIR")
  }
  args
}
args <- parse_args()

NEIGHBORHOODS_DIR <- args[["neighborhoods-dir"]]
UNIVERSE_FILE <- args[["universe"]]
OUT_DIR <- args[["out-dir"]]
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

PANEL_GENES <- c("MLH1", "GNLY", "USP18", "MYC", "AXIN2",
                  "FABP1", "CPS1", "SI", "VIM", "TGFB1")

# ---- Universe ---------------------------------------------------------
bg_symbols <- readLines(UNIVERSE_FILE)
bg_symbols <- bg_symbols[nzchar(bg_symbols)]
cat(sprintf("Universe: %d gene symbols (%s)\n", length(bg_symbols), UNIVERSE_FILE))

bg_map <- suppressMessages(bitr(bg_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db))
bg_entrez <- unique(bg_map$ENTREZID)
cat(sprintf("Universe mapped to Entrez: %d / %d symbols (%.1f%%)\n",
            length(bg_entrez), length(bg_symbols), 100 * length(bg_entrez) / length(bg_symbols)))

# ---- Hallmark gene sets (MSigDB, symbol-based) -------------------------
hallmark_t2g <- as.data.frame(msigdbr(species = "Homo sapiens", collection = "H"))[, c("gs_name", "gene_symbol")]

# ---- Helpers ------------------------------------------------------------
safe_run <- function(label, expr) {
  tryCatch(expr, error = function(e) {
    cat(sprintf("  [%s] FAILED: %s\n", label, conditionMessage(e)))
    NULL
  })
}

# Some neighborhood gene symbols (from the 2011-era TCGA expression matrix)
# are outdated HGNC aliases no longer present as a SYMBOL key in current
# org.Hs.eg.db (e.g. FAM101B -> current symbol RFLNB). Resolve those via
# ALIAS lookup before enrichment so a single stale symbol doesn't silently
# (or, for a length-1 query, fatally) drop out of the analysis. Only
# applied to the small per-gene query lists, not the 20,260-gene universe
# (there, unmapped symbols are mostly genuinely non-canonical loci, not
# stale aliases, and 84% mapping is an expected/acceptable ORA background
# rate).
current_symbols <- keys(org.Hs.eg.db, keytype = "SYMBOL")
current_aliases <- keys(org.Hs.eg.db, keytype = "ALIAS")

resolve_symbols <- function(symbols) {
  vapply(symbols, function(s) {
    if (s %in% current_symbols) return(s)
    if (!(s %in% current_aliases)) return(s)
    hit <- suppressMessages(AnnotationDbi::select(
      org.Hs.eg.db, keys = s, keytype = "ALIAS", columns = "SYMBOL"
    ))
    if (nrow(hit) >= 1 && !is.na(hit$SYMBOL[1])) {
      cat(sprintf("  [alias] %s -> %s\n", s, hit$SYMBOL[1]))
      hit$SYMBOL[1]
    } else {
      s
    }
  }, character(1), USE.NAMES = FALSE)
}

safe_bitr <- function(symbols) {
  if (length(symbols) == 0) return(character(0))
  res <- tryCatch(
    suppressMessages(bitr(symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)),
    error = function(e) data.frame(ENTREZID = character(0))
  )
  unique(res$ENTREZID)
}

n_sig <- function(res, cutoff = 0.05) {
  if (is.null(res)) return(NA_integer_)
  sum(as.data.frame(res)$p.adjust < cutoff)
}

write_result <- function(res, gene, db) {
  if (is.null(res) || nrow(as.data.frame(res)) == 0) return(invisible(NULL))
  out <- file.path(OUT_DIR, sprintf("%s_%s.csv", gene, db))
  write.csv(as.data.frame(res), out, row.names = FALSE)
}

# ---- Main loop ------------------------------------------------------------
summary_rows <- list()

for (gene in PANEL_GENES) {
  neigh_file <- file.path(NEIGHBORHOODS_DIR, sprintf("%s.txt", gene))
  query_symbols <- readLines(neigh_file)
  query_symbols <- query_symbols[nzchar(query_symbols)]
  n_query <- length(query_symbols)
  cat(sprintf("\n=== %s (neighborhood n=%d) ===\n", gene, n_query))

  query_symbols <- resolve_symbols(query_symbols)
  query_entrez <- safe_bitr(query_symbols)

  go_bp <- safe_run("GO_BP", enrichGO(
    gene = query_symbols, universe = bg_symbols, OrgDb = org.Hs.eg.db,
    keyType = "SYMBOL", ont = "BP", pAdjustMethod = "BH",
    pvalueCutoff = 0.05, qvalueCutoff = 0.2
  ))
  go_mf <- safe_run("GO_MF", enrichGO(
    gene = query_symbols, universe = bg_symbols, OrgDb = org.Hs.eg.db,
    keyType = "SYMBOL", ont = "MF", pAdjustMethod = "BH",
    pvalueCutoff = 0.05, qvalueCutoff = 0.2
  ))
  kegg <- if (length(query_entrez) > 0) safe_run("KEGG", enrichKEGG(
    gene = query_entrez, universe = bg_entrez, organism = "hsa",
    pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.2
  )) else NULL
  reactome <- if (length(query_entrez) > 0) safe_run("Reactome", enrichPathway(
    gene = query_entrez, universe = bg_entrez, organism = "human",
    pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.2,
    readable = TRUE
  )) else NULL
  hallmark <- safe_run("Hallmark", enricher(
    gene = query_symbols, universe = bg_symbols, TERM2GENE = hallmark_t2g,
    pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.2
  ))

  write_result(go_bp, gene, "GO_BP")
  write_result(go_mf, gene, "GO_MF")
  write_result(kegg, gene, "KEGG")
  write_result(reactome, gene, "Reactome")
  write_result(hallmark, gene, "Hallmark")

  for (db_label in c("GO_BP", "GO_MF", "KEGG", "Reactome", "Hallmark")) {
    res <- get(switch(db_label,
      GO_BP = "go_bp", GO_MF = "go_mf", KEGG = "kegg",
      Reactome = "reactome", Hallmark = "hallmark"))
    # NOTE: enrichGO/enrichKEGG/enrichPathway/enricher already filter their
    # returned object to rows passing (raw p < pvalueCutoff) AND
    # (qvalue < qvalueCutoff) -- nrow(as.data.frame(res)) is "terms
    # reported", not "terms tested" (the full ontology/pathway set tested
    # against the universe is much larger and isn't retained in the
    # object). n_significant below re-filters that already-short list by
    # BH-adjusted p < 0.05, so it's usually close to n_terms_reported.
    n_reported <- if (is.null(res)) 0L else nrow(as.data.frame(res))
    n_significant <- n_sig(res)
    top_term <- if (!is.null(res) && n_reported > 0) as.data.frame(res)$Description[1] else NA_character_
    top_padj <- if (!is.null(res) && n_reported > 0) as.data.frame(res)$p.adjust[1] else NA_real_
    cat(sprintf("  %-8s: %3d terms reported, %3d significant (padj<0.05)%s\n",
                db_label, n_reported, ifelse(is.na(n_significant), 0, n_significant),
                if (!is.na(top_term)) sprintf(" | top: %s (padj=%.2e)", top_term, top_padj) else ""))
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      gene = gene, n_neighborhood = n_query, database = db_label,
      n_terms_reported = n_reported, n_significant = ifelse(is.na(n_significant), 0, n_significant),
      top_term = top_term, top_padj = top_padj
    )
  }
}

summary_df <- do.call(rbind, summary_rows)
out_summary <- file.path(OUT_DIR, "ora_summary.tsv")
write.table(summary_df, out_summary, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("\nWrote ORA summary: %s\n", out_summary))
cat(sprintf("Wrote per-gene-per-database result tables: %s/\n", OUT_DIR))

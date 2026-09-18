# figures/ — figuras del trabajo Rosenkranz 2026 (ColoQ, panel v0.3 desde 2026-09-18)

Cada script genera una figura como PDF vectorial (Arial embebida, TrueType) y PNG a 300 dpi.
Salida por defecto: `gdrive:premio_Roche/figuras/` (si el Drive está montado) o `figures/out/`.
Convenciones comunes en `_common.py`: Arial ≥ 7 pt, paleta Okabe-Ito para los CMS
(CMS1 naranja, CMS2 azul, CMS3 verde azulado, CMS4 bermellón), gris para todo lo demás,
etiquetas de panel en negrita, sin títulos dentro de la figura, ancho 180 mm.

| Script | Figura | Insumos |
|---|---|---|
| `fig1_flujo.py` | Esquema del flujo (180×100). `--numbered` da el PNG con cajas numeradas. Coordenadas en `BOXES`. | ninguno |
| `fig2_desempeno.py` | Desempeño del clasificador (180×70): matriz de confusión, MI/AUC por gen, forest Cox ajustado. `--pct-by row|col`. | `results_gse39582/scored_cohort.tsv`; `network_analysis/results/crc_net_577/predictive_panel/`; `results_pooled_cox_mismamuestra/` |
| `fig3_regulacion.py` | Arquitectura regulatoria (180×110): heatmap z_meta TMR × CMS en 4 bloques + hallmarks replicados. | `crc_mra_results/{regulators,hallmarks}/`; `predictive_panel/panel_genes_vs_crc_mra_networks.tsv` |
| `fig4_clinica.py` | Valor clínico (180×80): KM pMMR con números en riesgo + forest modelo D. | `results_gse39582/scored_cohort.tsv`; `results_cox_clinical/` |

Los insumos que no viven en el repo (`respaldo_v0.2.0`, `crc_mra_results`) se buscan en el Drive
(`premio_Roche/respaldo_v0.2.0`) y en `/STORAGE/genut/calixto/crc_mra_results`; se pueden fijar con
`--respaldo`, `--mra-dir`, `--predictive-panel`. Cada script imprime al final las cifras que dibujó,
para cotejarlas con las tablas del manuscrito.

```bash
cd CRC-digital-twin
for s in fig1_flujo fig2_desempeno fig3_regulacion fig4_clinica; do python3 figures/$s.py; done
python3 figures/fig1_flujo.py --numbered
```

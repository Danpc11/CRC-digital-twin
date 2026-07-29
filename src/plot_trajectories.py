"""
plot_trajectories.py

Genera una figura de 2x2 mostrando, para cada perfil de mutacion
conductora, la evolucion temporal del vector de estado (8 genes
marcadores) desde una condicion inicial neutra hasta el atractor CMS
correspondiente.

Paleta Wong (colorblind-safe), sin estilos decorativos adicionales.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attractor_model import GENES, simulate_patient

# Paleta Wong (colorblind-safe), 8 colores para 8 genes
WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

DRIVERS = [
    ("MSI_high", "MSI alta (perdida MLH1/MSH2) -> CMS1"),
    ("APC_mut", "APC mutante (WNT constitutiva) -> CMS2"),
    ("KRAS_mut", "KRAS mutante -> CMS3"),
    ("SMAD4_loss", "Perdida SMAD4 (EMT) -> CMS4"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

for ax, (driver, title) in zip(axes.flat, DRIVERS):
    result = simulate_patient(driver)
    t, x = result["t"], result["x"]
    for i, gene in enumerate(GENES):
        ax.plot(t, x[i], color=WONG[i], label=gene, linewidth=1.6)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
    ax.set_ylim(-1.1, 1.1)

for ax in axes[-1, :]:
    ax.set_xlabel("tiempo (u.a.)")
for ax in axes[:, 0]:
    ax.set_ylabel("expresion (z-score)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Convergencia del gemelo digital a atractores CMS segun mutacion conductora",
             fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])

out_path = Path(__file__).resolve().parents[1] / "figures" / "cms_attractor_trajectories.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figura guardada en: {out_path}")

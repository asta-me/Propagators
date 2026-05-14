"""
===============================================================================
 2D WAVE FIELD RECONSTRUCTION: AMPLITUDE ANALYSIS & ARTIFACT SUPPRESSION
===============================================================================
Author:      Marco Astarita
Reference:   K. Matsushima, "Introduction to Computer Holography", Springer (2020).
             Figs 6.11 (Band-limiting) & 6.12 (Noise Reduction).

DESCRIPTION:
  High-resolution 2D analysis of numerical propagation errors.
  Comparison of three numerical schemes to isolate specific artifact sources:

  1. Standard Angular Spectrum (AS):
     - Baseline method.
     - Exhibits "Transfer Function Aliasing" (high-frequency noise).
     - Exhibits "Field Invasion" (wrap-around error).

  2. Band-Limited Angular Spectrum (BLAS) [No Extension]:
     - Applies frequency filtering (Matsushima Eq. 6.44).
     - Result: Noise suppression (Smooth field), but persistent wrap-around.

  3. BLAS + Quadruple Extension:
     - Combines band-limiting with spatial domain padding.
     - Result: Physically correct free-space propagation (Noise-free, Linear Conv).

OUTPUT:
  Minimalist grayscale visualization (2D maps and 1D cross-sections) suitable
  for academic publication.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from Propagators import prop_AS, prop_BLAS

# =============================================================================
# 1. PARAMETRI DI SIMULAZIONE (Configurazione Matsushima)
# =============================================================================
lam = 633e-9          # Lunghezza d'onda [m] (HeNe)
N = 2048              # Alta risoluzione spaziale
dx = 1.0e-6           # Campionamento [m] (1 micron)
L = N * dx            # Dimensione totale campo (~2 mm)

D = 0.5e-3            # Diametro Apertura Circolare [m] (0.5 mm)
z_test = 0.1          # Distanza di propagazione [m] (10 cm)

# =============================================================================
# 2. GENERAZIONE SORGENTE (Piano z=0)
# =============================================================================
x = dx * (np.arange(N) - N/2)
y = dx * (np.arange(N) - N/2)
X, Y = np.meshgrid(x, y)

u_in = np.zeros((N, N), dtype=complex)
mask = (X**2 + Y**2) < (D/2)**2
u_in[mask] = 1.0

print(f"--- ANALISI NUMERICA (AMPIEZZA) ---")
print(f"Risoluzione: {N}x{N}, dx = {dx*1e6:.1f} um")
print(f"Distanza z : {z_test*100:.1f} cm")
print("Calcolo in corso...")


# =============================================================================
# 3. CALCOLO PROPAGAZIONI
# =============================================================================

# 1. Standard AS (Metodo Naive)
u_std = prop_AS(u_in, lam, dx, dx, z_test)

# 2. BLAS No Ext (Solo Filtro)
u_blas_noext = prop_BLAS(u_in, lam, dx, dx, z_test, quadruple_extension=False)

# 3. BLAS + Ext (Reference / Corretto)
u_blas_ext = prop_BLAS(u_in, lam, dx, dx, z_test, quadruple_extension=True)


# =============================================================================
# 4. PLOTTING (Stile Pubblicazione)
# =============================================================================
fields = [u_std, u_blas_noext, u_blas_ext]
titles = ["Standard AS", "BLAS (No Ext)", "BLAS + Ext (Correct)"]

# Configurazione Figura: 3 Righe x 3 Colonne
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.subplots_adjust(wspace=0.15, hspace=0.25)

# Reference Amplitude Max (usata per normalizzare tutte le scale)
Amp_ref = np.abs(u_blas_ext)
A_max = np.max(Amp_ref)

mid_idx = N // 2 

for i in range(3):
    # Estrazione Dati Ampiezza
    u_curr = fields[i]
    Amp_2d = np.abs(u_curr)
    Amp_1d = Amp_2d[mid_idx, :]
    
    # --- RIGA 1: MAPPA 2D (Scala di Grigi) ---
    ax2d = axes[0, i]
    # vmin/vmax saturati al 50% per evidenziare la struttura dei lobi
    im = ax2d.imshow(Amp_2d, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], 
                     cmap='gray', vmin=0, vmax=A_max*0.5, origin='lower')
    
    ax2d.set_title(titles[i], fontweight='bold', fontsize=12)
    if i == 0: ax2d.set_ylabel("y [mm]")
    ax2d.set_xticks([]) # Rimuoviamo i tick per pulizia visiva
    
    # --- RIGA 2: PROFILO 1D (Scala Completa) ---
    ax_full = axes[1, i]
    ax_full.plot(x*1e3, Amp_1d, color='black', lw=1)
    
    ax_full.set_xlim(-L/2*1e3, L/2*1e3)
    ax_full.set_ylim(0, A_max * 1.05) 
    ax_full.grid(alpha=0.2, color='gray', linestyle=':')
    if i == 0: ax_full.set_ylabel("Ampiezza |u|")
    ax_full.set_xticks([]) 
    ax_full.set_title("Full Profile (Central Slice)", fontsize=10, style='italic')

    # --- RIGA 3: PROFILO 1D (Zoom sui Dettagli/Errori) ---
    ax_zoom = axes[2, i]
    ax_zoom.plot(x*1e3, Amp_1d, color='black', lw=1)
    
    ax_zoom.set_xlim(-L/2*1e3, L/2*1e3)
    # Zoom verticale spinto: mostriamo solo il 10% inferiore del segnale
    # Qui è dove si annidano il rumore numerico e il wrap-around
    ax_zoom.set_ylim(0, A_max * 0.1) 
    ax_zoom.grid(alpha=0.2, color='gray', linestyle=':')
    
    ax_zoom.set_xlabel("x [mm]")
    if i == 0: ax_zoom.set_ylabel("Ampiezza (Zoom 10x)")
    ax_zoom.set_title("Detail / Noise Floor", fontsize=10, style='italic')

# Titolo Globale della Figura
fig.suptitle(f"Amplitude Artifact Analysis: D={D*1e3:.1f}mm @ z={z_test*100:.0f}cm", fontsize=14)

plt.show()
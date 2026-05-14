"""
===============================================================================
 COMPARATIVE STUDY: OPTICAL PROPAGATION ALGORITHMS & SAMPLING REGIMES
===============================================================================
Author:      Marco Astarita
Reference:   K. Matsushima, "Introduction to Computer Holography", Springer (2020).
             Chapter 4 (Fresnel S-FFT) vs Chapter 6 (Angular Spectrum).

DESCRIPTION:
  Static analysis of wave field reconstruction techniques, highlighting physical
  sampling constraints and numerical artifacts.

  KEY COMPARISONS:
  1. Sampling Geometry (Field of View):
     - Single-Step Fresnel (S-FFT): Variable sampling pitch (dx2 propto z).
       Acts as a numerical "zoom", maintaining resolution in the Far Field.
     - Convolutional Methods (BLAS): Fixed sampling pitch (dx2 = dx1).
       Essential for Near Field/Diffraction preservation but limits FOV.

  2. Boundary Conditions (Artifact Suppression):
     - Angular Spectrum (AS): Suffers from circular convolution errors.
     - BLAS + Quadruple Extension: Enforces linear convolution via domain
       padding and band-limiting, ensuring artifact-free reconstruction.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from Propagators import prop_fresnel_single_step, prop_AS, prop_BLAS

# =============================================================================
# 1. SETUP FISICO E GENERAZIONE SORGENTE
# =============================================================================

# Parametri del sistema
lam = 532e-9          # Lunghezza d'onda (Green) [m]
N = 1080              # Risoluzione (punti laterali)
dx = 8e-6             # Passo di campionamento [m] (8 micron)
L = N * dx            # Dimensione totale del campo [m]

# Griglia di coordinate (Piano Sorgente, z=0)
# NOTA: Usiamo arange per avere lo zero esattamente su un pixel (FFT friendly)
x = dx * (np.arange(N) - N/2)
y = dx * (np.arange(N) - N/2)
X, Y = np.meshgrid(x, y)

# --- DEFINIZIONE APERTURA (Scegliere Quadrata o Circolare) ---
width = 100e-6        # Dimensione caratteristica (Lato o Diametro) [m]
u_in = np.zeros((N, N), dtype=complex)

# [OPZIONE 1] Fenditura Quadrata (Decommenta sotto)
mask = (np.abs(X) < width/2) & (np.abs(Y) < width/2)

# [OPZIONE 2] Fenditura Circolare (Decommenta sotto per attivare)
# mask = (X**2 + Y**2) < (width/2)**2

u_in[mask] = 1.0

# Calcolo estensione per i plot (comune ai metodi convoluzionali)
extent_in = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3] # Convertito in mm

print(f"--- PARAMETRI DI SIMULAZIONE ---")
print(f"Grid: {N}x{N}")
print(f"Pixel pitch: {dx*1e6:.1f} um")
print(f"Sorgente FOV: {L*1e3:.2f} mm")
print(f"Apertura size: {width*1e6:.1f} um")
print("-" * 40)


# =============================================================================
# FIGURA 0: IL PIANO SORGENTE (COSA STIAMO PROPAGANDO?)
# =============================================================================
plt.figure(figsize=(7, 6))
plt.imshow(np.abs(u_in)**2, extent=extent_in, cmap='gray', interpolation='nearest')
plt.title("Piano Sorgente (z = 0)", fontsize=14)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.colorbar(label="Intensità Normalizzata", fraction=0.046, pad=0.04)

# Zoommiamo per vedere l'apertura (altrimenti è troppo piccola nel FOV totale)
zoom_lim = 0.5 # mm
plt.xlim(-zoom_lim, zoom_lim)
plt.ylim(-zoom_lim, zoom_lim)
plt.tight_layout()
plt.show()


# =============================================================================
# CONFRONTO 1: IL PROBLEMA DEL CAMPIONAMENTO (S-FFT vs BLAS)
# =============================================================================
# Obiettivo: Mostrare come S-FFT cambi la "finestra" fisica di osservazione.

z1 = 0.5  # Distanza di propagazione [m]

print(f"\n[CONFRONTO 1] Propagazione a z = {z1} m")

# --- Metodo A: Single Step Fresnel (S-FFT) ---
# Questo metodo cambia il dx in uscita!
u_ss, dx2_ss, dy2_ss = prop_fresnel_single_step(u_in, lam, dx, dx, z1)

L_out_ss = N * dx2_ss  # Nuovo FOV fisico
extent_ss = [-L_out_ss/2*1e3, L_out_ss/2*1e3, -L_out_ss/2*1e3, L_out_ss/2*1e3]

print(f" -> S-FFT Output FOV: {L_out_ss*1e3:.2f} mm (Diverso dall'input!)")
print(f" -> S-FFT Pixel size: {dx2_ss*1e6:.1f} um")

# --- Metodo B: BLAS (Convoluzionale) ---
# Questo metodo mantiene dx costante
u_blas = prop_BLAS(u_in, lam, dx, dx, z1, quadruple_extension=True)
print(f" -> BLAS Output FOV:  {L*1e3:.2f} mm (Uguale all'input)")


# PLOT CONFRONTO 1
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot S-FFT
im1 = ax[0].imshow(np.abs(u_ss)**2, extent=extent_ss, cmap='inferno')
ax[0].set_title(f"Fresnel Single Step (S-FFT)\nFOV Variabile: {L_out_ss*1e3:.1f} mm", fontsize=12)
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

# Plot BLAS
im2 = ax[1].imshow(np.abs(u_blas)**2, extent=extent_in, cmap='inferno')
ax[1].set_title(f"Band-Limited AS (BLAS)\nFOV Fisso: {L*1e3:.1f} mm", fontsize=12)
ax[1].set_xlabel("x [mm]")
ax[1].set_ylabel("y [mm]")
plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

plt.suptitle(f"CONFRONTO 1: Finestre di Campionamento a z={z1*100} cm", fontsize=16)
plt.tight_layout()
plt.show()


# =============================================================================
# CONFRONTO 2: GLI ARTEFATTI DI BORDO (AS vs BLAS+Quadruple)
# =============================================================================
# Obiettivo: Mostrare l'errore di "Field Invasion" (la luce esce da dx e rientra da sx).

z2 = 0.2  # Distanza ridotta [m]

print(f"\n[CONFRONTO 2] Gestione Artefatti (Field Invasion) a z = {z2} m")

# --- Metodo A: Angular Spectrum Standard ---
# Soffre di convoluzione circolare
u_as = prop_AS(u_in, lam, dx, dx, z2)

# --- Metodo B: BLAS con Quadruple Extension ---
# Simula convoluzione lineare (zero-padding)
u_blas_quad = prop_BLAS(u_in, lam, dx, dx, z2, quadruple_extension=True)

# Usiamo scala logaritmica per evidenziare i "fantasmi" di bassa intensità ai bordi
I_as_log = np.log10(np.abs(u_as)**2 + 1e-15)
I_blas_log = np.log10(np.abs(u_blas_quad)**2 + 1e-15)


# PLOT CONFRONTO 2
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot AS Standard
im3 = ax[0].imshow(I_as_log, extent=extent_in, cmap='magma', vmin=-8, vmax=0)
ax[0].set_title("Angular Spectrum (Standard)\nErrore: Convoluzione Circolare", fontsize=12)
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
plt.colorbar(im3, ax=ax[0], fraction=0.046, pad=0.04, label="Log Intensity")

# Plot BLAS Quadruple
im4 = ax[1].imshow(I_blas_log, extent=extent_in, cmap='magma', vmin=-8, vmax=0)
ax[1].set_title("BLAS + Quadruple Extension\nCorretto: Propagazione Libera", fontsize=12)
ax[1].set_xlabel("x [mm]")
ax[1].set_ylabel("y [mm]")
plt.colorbar(im4, ax=ax[1], fraction=0.046, pad=0.04, label="Log Intensity")

plt.suptitle(f"CONFRONTO 2: Artefatti Numerici a z={z2*100} cm", fontsize=16)
plt.tight_layout()
plt.show()

print("\n--- Analisi Completata ---")
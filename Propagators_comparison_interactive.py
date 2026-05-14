"""
===============================================================================
 INTERACTIVE SIMULATION: PROPAGATION DYNAMICS AND Z-DEPENDENCE
===============================================================================
Author:      Marco Astarita
Context:     Advanced Computational Fourier Optics

DESCRIPTION:
  Interactive visualization tool designed to explore the z-dependence of
  numerical artifacts and sampling behaviors using real-time sliders.

  SIMULATION MODES:
  1. Sampling Regime Analysis (S-FFT vs. BLAS):
     Demonstrates the divergence of physical observation windows.
     - S-FFT: Window scales linearly with z (Automatic Zoom).
     - BLAS: Window remains fixed (Physical Crop).

  2. Artifact Evolution Analysis (AS vs. BLAS+Ext):
     Visualizes the onset of spatial aliasing.
     - Near Field (small z): Observation of Transfer Function aliasing noise.
     - Far Field (large z): Observation of Field Invasion (Wrap-around)
       entering the fixed computational window.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Propagators import prop_fresnel_single_step, prop_AS, prop_BLAS

# =============================================================================
# 1. SETUP FISICO (GLOBAL)
# =============================================================================

# Risoluzione (N=512 consigliato per fluidità slider live)
N = 512              
lam = 532e-9         # Lunghezza d'onda (Verde) [m]
dx = 10e-6           # Pixel pitch (10 micron) [m]
L = N * dx           # Dimensione totale campo [m]

# Apertura
width = 400e-6       # Larghezza apertura [m]

# =============================================================================
# 2. CONFIGURAZIONE SLIDER (Range di Propagazione)
# =============================================================================

# --- DEMO 1: Campionamento (Zoom vs Fisso) ---
Z_MIN_1 = 0.01       # Minima distanza [m]
Z_MAX_1 = 0.75        # Massima distanza [m]
Z_INIT_1 = 0.1       # Distanza iniziale [m]

# --- DEMO 2: Artefatti (Aliasing) ---
# Usiamo distanze brevi dove l'aliasing è più evidente o il fascio esplode
Z_MIN_2 = 0.0       # Minima distanza [m]
Z_MAX_2 = 0.2        # Massima distanza [m]
Z_INIT_2 = 0.1      # Distanza iniziale [m]


# =============================================================================
# 3. GENERAZIONE SORGENTE
# =============================================================================
# Generazione rigorosa (Zero centrato su pixel)
x = dx * (np.arange(N) - N/2)
y = dx * (np.arange(N) - N/2)
X, Y = np.meshgrid(x, y)

u_in = np.zeros((N, N), dtype=complex)

# Maschera Quadrata
mask = (np.abs(X) < width/2) & (np.abs(Y) < width/2)

# Maschera Circolare (Decommentare se serve)
# mask = (X**2 + Y**2) < (width/2)**2

u_in[mask] = 1.0

# Extent fisso per i metodi convoluzionali (in mm)
extent_fixed = [-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3]

print("--- INIZIO DEMO INTERATTIVA ---")
print(f"Grid: {N}x{N}, Wave: {lam*1e9:.0f}nm, Apertura: {width*1e6:.0f}um")
print("1. Caricamento Demo Campionamento (S-FFT vs BLAS)...")


# =============================================================================
# DEMO 1: CONFRONTO CAMPIONAMENTO (S-FFT vs BLAS)
# =============================================================================
def run_sampling_demo():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(bottom=0.20) # Spazio per lo slider

    # --- Calcolo Iniziale ---
    # S-FFT
    u_ss, dx2, _ = prop_fresnel_single_step(u_in, lam, dx, dx, Z_INIT_1)
    L_ss = N * dx2
    extent_ss = [-L_ss/2*1e3, L_ss/2*1e3, -L_ss/2*1e3, L_ss/2*1e3]
    
    # BLAS (Qui usiamo Quadruple=False per velocità massima nello slider, 
    # tanto ci interessa vedere solo la dimensione della finestra fissa)
    u_blas = prop_BLAS(u_in, lam, dx, dx, Z_INIT_1, quadruple_extension=True)

    # --- Plot Iniziali ---
    im1 = ax1.imshow(np.abs(u_ss)**2, extent=extent_ss, cmap='inferno', origin='upper')
    ax1.set_title(f"Single Step (S-FFT)\nFOV Variabile (Zoom)", fontsize=12)
    ax1.set_xlabel("x [mm]")
    
    im2 = ax2.imshow(np.abs(u_blas)**2, extent=extent_fixed, cmap='inferno', origin='upper')
    ax2.set_title(f"BLAS (Convolutional)\nFOV Fisso", fontsize=12)
    ax2.set_xlabel("x [mm]")

    fig.suptitle(f"DEMO 1: Variazione Dimensioni Fisiche (z = {Z_INIT_1:.2f} m)", fontsize=16)

    # --- SLIDER SETUP ---
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_z = Slider(
        ax=ax_slider,
        label='Distanza z [m] ',
        valmin=Z_MIN_1,
        valmax=Z_MAX_1,
        valinit=Z_INIT_1,
        color='orange'
    )

    # --- Funzione di Aggiornamento ---
    def update(val):
        z = slider_z.val
        
        # 1. Ricalcolo S-FFT
        u_ss_new, dx2_new, _ = prop_fresnel_single_step(u_in, lam, dx, dx, z)
        L_new = N * dx2_new
        ext_new = [-L_new/2*1e3, L_new/2*1e3, -L_new/2*1e3, L_new/2*1e3]
        
        im1.set_data(np.abs(u_ss_new)**2)
        im1.set_extent(ext_new) # Aggiorna l'extent dell'immagine
        
        # Aggiorna i limiti degli assi per seguire lo zoom
        ax1.set_xlim(ext_new[0], ext_new[1])
        ax1.set_ylim(ext_new[2], ext_new[3])
        
        # 2. Ricalcolo BLAS
        u_blas_new = prop_BLAS(u_in, lam, dx, dx, z, quadruple_extension=True)
        im2.set_data(np.abs(u_blas_new)**2)
        
        # 3. Refresh
        im1.autoscale() 
        im2.autoscale()
        fig.suptitle(f"DEMO 1: Variazione Dimensioni Fisiche (z = {z:.3f} m)", fontsize=16)
        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    plt.show()

# Eseguiamo la prima demo
run_sampling_demo()

print("2. Caricamento Demo Artefatti (AS vs BLAS Quad)...")


# =============================================================================
# DEMO 2: CONFRONTO ARTEFATTI (AS vs BLAS+Quadruple)
# =============================================================================
def run_artifacts_demo():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(bottom=0.20)

    # Setup iniziale
    u_as = prop_AS(u_in, lam, dx, dx, Z_INIT_2)
    u_bq = prop_BLAS(u_in, lam, dx, dx, Z_INIT_2, quadruple_extension=True)
    
    # Calcolo Logaritmico
    def get_log_intensity(u):
        return np.log10(np.abs(u)**2 + 1e-13)

    # --- Plot Iniziali (Log Scale) ---
    im1 = ax1.imshow(get_log_intensity(u_as), extent=extent_fixed, cmap='magma', 
                     vmin=-8, vmax=0, origin='upper')
    ax1.set_title("Angular Spectrum (Standard)\nAliasing / Field Invasion", fontsize=12)
    ax1.set_xlabel("x [mm]")

    im2 = ax2.imshow(get_log_intensity(u_bq), extent=extent_fixed, cmap='magma', 
                     vmin=-8, vmax=0, origin='upper')
    ax2.set_title("BLAS + Quadruple Extension\nClean Propagation", fontsize=12)
    ax2.set_xlabel("x [mm]")

    fig.suptitle(f"DEMO 2: Artefatti Numerici (z = {Z_INIT_2:.2f} m)", fontsize=16)

    # --- SLIDER SETUP ---
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_z = Slider(
        ax=ax_slider,
        label='Distanza z [m] ',
        valmin=Z_MIN_2,
        valmax=Z_MAX_2,
        valinit=Z_INIT_2,
        color='cyan'
    )

    # --- Funzione di Aggiornamento ---
    def update(val):
        z = slider_z.val
        
        # 1. Calcolo
        u_as_new = prop_AS(u_in, lam, dx, dx, z)
        u_bq_new = prop_BLAS(u_in, lam, dx, dx, z, quadruple_extension=True)
        
        # 2. Aggiorna dati
        im1.set_data(get_log_intensity(u_as_new))
        im2.set_data(get_log_intensity(u_bq_new))
        
        # 3. Refresh
        fig.suptitle(f"DEMO 2: Artefatti Numerici (z = {z:.3f} m)", fontsize=16)
        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    plt.show()

# Eseguiamo la seconda demo
run_artifacts_demo()

print("--- DEMO COMPLETATA ---")
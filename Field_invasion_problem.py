"""
===============================================================================
 NUMERICAL ARTIFACT ANALYSIS: IMPLICIT PERIODICITY & FIELD INVASION
===============================================================================
Author:      Marco Astarita
Reference:   K. Matsushima, "Introduction to Computer Holography", Springer (2020).
             Chapter 6: The Angular Spectrum Method.

DESCRIPTION:
  This script investigates the "Field Wrap-around" error (Spatial Aliasing)
  inherent in Discrete Fourier Transform (DFT) based propagation.

  VISUALIZATION PANELS:
  1. Top Panel (Domain Periodicity):
     Visualizes the mathematical implication of DFT: the source field is treated
     as an infinite periodic array (virtual replicas at m = -1, 0, +1).
     Aliasing occurs when diffracted light from virtual replicas invades the
     fundamental computational window.

  2. Bottom Panel (Convolution Regimes):
     Direct comparison between:
     - Circular Convolution (Standard FFT): Physically incorrect for free-space
       propagation due to wrap-around artifacts.
     - Linear Convolution (Padded FFT): Simulates aperiodic free-space
       propagation by extending the domain (N -> 2N), effectively isolating
       the fundamental period.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.fft import fft, ifft, fftfreq

# =============================================================================
# 1. PARAMETRI FISICI & CONFIGURAZIONE
# =============================================================================
N = 1024              # Punti griglia
lam = 532e-9         # Lunghezza d'onda [m]
dx = 10e-6           # Passo di campionamento [m]
L = N * dx           # Larghezza Dominio Fondamentale [m]

# --- CONFIGURAZIONE SLIDER ---
z_max = 0.5          # <--- Massima distanza di propagazione [m]
z_min = 0.0          # Minima distanza (Fissa a 0 come richiesto)

# Coordinate dominio fondamentale
x = dx * (np.arange(N) - N/2)

# --- SORGENTE (Shifted Aperture) ---
# Decentrata a -L/4 per forzare l'interazione asimmetrica col bordo
offset = -L/4 
width = 500e-6       # Larghezza fenditura (Parametro utente)

mask_std = (np.abs(x - offset) < width/2)
u_in_std = np.zeros(N, dtype=complex)
u_in_std[mask_std] = 1.0

# --- DOMINIO ESTESO PER VISUALIZZAZIONE (Top Plot) ---
# Usiamo 3N solo per mostrare graficamente i vicini (m=-1, 0, +1)
N_vis = N * 3
x_vis = dx * (np.arange(N_vis) - N_vis/2)

# Costruzione maschere visualizzazione
mask_c = (np.abs(x_vis - offset) < width/2)           # m = 0
mask_l = (np.abs(x_vis - (offset - L)) < width/2)     # m = -1
mask_r = (np.abs(x_vis - (offset + L)) < width/2)     # m = +1


# =============================================================================
# 2. MOTORE DI CALCOLO
# =============================================================================

def compute_fields(z):
    # Gestione caso z=0
    if z == 0:
        # Ritorna input ideale e campi nulli per le repliche
        return u_in_std, u_in_std, np.zeros(N_vis, dtype=complex), np.zeros(N_vis, dtype=complex), np.zeros(N_vis, dtype=complex)

    # --- A. LINEAR CONVOLUTION (REFERENCE) ---
    # Implementazione rigorosa "Double/Quadruple Extension"
    # Algoritmo: Pad N -> 2N, Propagate, Crop -> N
    N_pad = 2 * N
    u_padded = np.pad(u_in_std, (N//2, N//2), mode='constant') # Padding centrale
    
    fx_pad = fftfreq(N_pad, d=dx)
    # Transfer Function su 2N
    arg_sqrt_pad = 1/lam**2 - fx_pad**2
    H_pad = np.zeros_like(fx_pad, dtype=complex)
    mask_ev_pad = arg_sqrt_pad >= 0
    H_pad[mask_ev_pad] = np.exp(1j * 2 * np.pi * z * np.sqrt(arg_sqrt_pad[mask_ev_pad]))
    
    # Propagazione Lineare
    u_lin_pad = ifft(fft(u_padded) * H_pad)
    # Cropping al dominio originale
    u_linear = u_lin_pad[N//2 : N//2 + N]


    # --- B. CIRCULAR CONVOLUTION (STANDARD FFT) ---
    # Calcolo standard su N
    fx = fftfreq(N, d=dx)
    arg_sqrt = 1/lam**2 - fx**2
    H_std = np.zeros_like(fx, dtype=complex)
    mask_ev = arg_sqrt >= 0
    H_std[mask_ev] = np.exp(1j * 2 * np.pi * z * np.sqrt(arg_sqrt[mask_ev]))
    
    u_circular = ifft(fft(u_in_std) * H_std)


    # --- C. CALCOLO REPLICHE (SOLO PER VISUALIZZAZIONE TOP PLOT) ---
    # Calcoliamo le repliche su 3N per mostrare l'invasione fisica nel grafico sopra
    N_vis_calc = N_vis # Calcolo diretto su 3N
    fx_vis = fftfreq(N_vis_calc, d=dx)
    H_vis = np.zeros_like(fx_vis, dtype=complex)
    arg_sqrt_vis = 1/lam**2 - fx_vis**2
    H_vis[arg_sqrt_vis >= 0] = np.exp(1j * 2 * np.pi * z * np.sqrt(arg_sqrt_vis[arg_sqrt_vis >= 0]))
    
    def prop_mask_vis(mask_b):
        u_tmp = np.zeros(N_vis_calc, dtype=complex)
        u_tmp[mask_b] = 1.0
        return ifft(fft(u_tmp) * H_vis)

    u_m0_vis  = prop_mask_vis(mask_c)
    u_mm1_vis = prop_mask_vis(mask_l)
    u_mp1_vis = prop_mask_vis(mask_r)

    return u_linear, u_circular, u_m0_vis, u_mm1_vis, u_mp1_vis


# =============================================================================
# 3. VISUALIZZAZIONE
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
plt.subplots_adjust(bottom=0.15, hspace=0.35)

z_init = 0.0
u_lin, u_circ, u_m0, u_mm1, u_mp1 = compute_fields(z_init)

# Caso speciale z=0 per visualizzazione pulita
if z_init == 0:
    u_m0 = np.zeros(N_vis, dtype=complex); u_m0[mask_c] = 1.0
    u_mm1 = np.zeros(N_vis, dtype=complex); u_mm1[mask_l] = 1.0
    u_mp1 = np.zeros(N_vis, dtype=complex); u_mp1[mask_r] = 1.0

# --- PLOT 1: IMPLICIT PERIODICITY ---
# m=0 (Green)
l_m0, = ax1.plot(x_vis*1e3, np.abs(u_m0)**2, 'g-', lw=2, label='Fundamental (m=0)')
ax1.fill_between(x_vis*1e3, np.abs(u_m0)**2, color='green', alpha=0.15)

# m=-1, +1 (Blue)
l_mm1, = ax1.plot(x_vis*1e3, np.abs(u_mm1)**2, 'b--', alpha=0.5) 
l_mp1, = ax1.plot(x_vis*1e3, np.abs(u_mp1)**2, 'b--', lw=1.5, label='Replicas (m=$\pm$1)')
ax1.fill_between(x_vis*1e3, np.abs(u_mm1)**2, color='blue', alpha=0.05)
ax1.fill_between(x_vis*1e3, np.abs(u_mp1)**2, color='blue', alpha=0.05)

# Evidenzia dominio computazionale
ax1.axvline(-L/2*1e3, color='k', lw=1.5)
ax1.axvline(L/2*1e3, color='k', lw=1.5)
# Rettangolo giallo per evidenziare la finestra
ax1.add_patch(plt.Rectangle((-L/2*1e3, 0), L*1e3, 2, facecolor='yellow', alpha=0.1, label='Window'))

ax1.set_title("FFT IMPLICIT PERIODICITY", fontsize=12, fontweight='bold')
ax1.set_xlim(-1.5*L*1e3, 1.5*L*1e3)
ax1.set_ylim(0, 1.2)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.2)


# --- PLOT 2: COMPARISON (EFFECT) ---
# ORDINE DI PLOT: Prima l'errore (sotto), poi la corretta (sopra)
# 1. Circular (Errato) - Rosso Tratteggiato, sotto
l_circ, = ax2.plot(x*1e3, np.abs(u_circ)**2, 'r--', lw=3, alpha=0.8, label='Standard FFT (Circular)', zorder=2)

# 2. Linear (Corretto) - Nero Continuo, sopra
l_lin, = ax2.plot(x*1e3, np.abs(u_lin)**2, 'k-', lw=1.5, label='Padded FFT (Linear)', zorder=3)

ax2.set_title("SIMULATION WINDOW RESULTS", fontsize=12, fontweight='bold')
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("Intensity |u|²")
ax2.set_xlim(-L/2*1e3, L/2*1e3)
ax2.set_ylim(0, 1.2)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Box di Warning (Inizialmente invisibile)
txt_warn = ax2.text(L/4*1e3, 0.6, "WRAP-AROUND\nERROR", color='red', fontsize=12, fontweight='bold', 
                    ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', alpha=0.9))
txt_warn.set_visible(False)


# =============================================================================
# 4. SLIDER LOGIC
# =============================================================================
ax_sl = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_sl, "z [m]", z_min, z_max, valinit=z_init, color='gray') # <--- Variabile z_max inserita

def update(val):
    z = slider.val
    
    if z == 0:
        # Reset visivo a z=0
        u_l_n, u_c_n = u_in_std, u_in_std
        u_m0_n = np.zeros(N_vis); u_m0_n[mask_c] = 1.0
        u_mm1_n = np.zeros(N_vis); u_mm1_n[mask_l] = 1.0
        u_mp1_n = np.zeros(N_vis); u_mp1_n[mask_r] = 1.0
    else:
        u_l_n, u_c_n, u_m0_n, u_mm1_n, u_mp1_n = compute_fields(z)
    
    # Update Top Plot
    l_m0.set_ydata(np.abs(u_m0_n)**2)
    l_mm1.set_ydata(np.abs(u_mm1_n)**2)
    l_mp1.set_ydata(np.abs(u_mp1_n)**2)
    
    # Update Bottom Plot
    l_circ.set_ydata(np.abs(u_c_n)**2)
    l_lin.set_ydata(np.abs(u_l_n)**2)
    
    # Warning Logic: Se c'è energia significativa sul bordo destro nella circular
    # ma NON nella linear (o semplicemente se circular >> linear al bordo)
    edge_idx = -10 # Bordo destro
    err_energy = np.mean(np.abs(u_c_n[edge_idx:])**2)
    ref_energy = np.mean(np.abs(u_l_n[edge_idx:])**2)
    
    # Soglia empirica: se l'errore è alto e diverge dalla ref
    if err_energy > 0.05 and (err_energy - ref_energy) > 0.02:
        txt_warn.set_visible(True)
    else:
        txt_warn.set_visible(False)
        
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
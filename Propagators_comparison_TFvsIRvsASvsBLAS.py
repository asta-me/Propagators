"""
===============================================================================
 COMPARATIVE STUDY: OPTICAL PROPAGATION ON SLM (VOELZ VS. MATSUSHIMA)
===============================================================================
AUTHOR: Marco Astarita
DATE: February 2026
VERSION: 3.0

DESCRIPTION:
This script performs a multi-method simulation of scalar wave propagation,
rescaling Voelz's theoretical parameters to the physical dimensions of a real setup
with a Spatial Light Modulator (SLM).

1. PHYSICAL RESCALING:
   - Source: Circular aperture (D = 2mm) on a 1024x1024 grid.
   - Hardware: Pixel pitch (dx) = 10 um, Wavelength (lam) = 532 nm.
   - Critical Distance (z_c): (L * dx) / lam ≈ 19.2 cm.

2. METHODS COMPARED:
   - Transfer Function (TF): Frequency domain. Optimal for z < z_c.
   - Impulse Response (IR): Spatial domain. Optimal for z > z_c.
   - Angular Spectrum (AS): Exact solution of Helmholtz. Suffers from "Field Invasion"
     (circular convolution) at large distances.
   - BLAS (Band-Limited AS): Matsushima's method with "Quadruple Extension" to
     eliminate edge artifacts and phase aliasing.

3. WHAT TO OBSERVE (ARTIFACT ANALYSIS):
   -------------------------------------------------------------------------
   REGIME | METHOD | TECHNICAL OBSERVATION
   -------------------------------------------------------------------------
   NEAR   |  TF    | Clean reference. Well-sampled quadratic phase.
   NEAR   |  IR    | FAILURE: Lateral periodic copies (Spatial Aliasing).
   FAR    |  TF    | FAILURE: Granular "stair-step" noise (Frequency Aliasing).
   FAR    |  IR    | Stable reference. Acts as a low-pass filter.
   FAR    |  AS    | Wrap-around error (light exits right and re-enters left).
   ALL    |  BLAS  | GOLD STANDARD: Total suppression of noise and edges.
   -------------------------------------------------------------------------

REFERENCES:
- D. G. Voelz, "Computational Fourier Optics: a MATLAB tutorial" (Ch. 5).
- K. Matsushima, "Introduction to Computer Holography" (Ch. 6).
===============================================================================
"""


import numpy as np
import matplotlib.pyplot as plt
from Propagators import prop_TF, prop_IR, prop_AS, prop_BLAS

# =============================================================================
# 1. Physical Parameters (SLM + GREEN LASER)
# =============================================================================
lam = 532e-9            # 532 nm
N = 1024                
dx = 10e-6              # 10 um
L = N * dx              # 10.24 mm
z_c = (L * dx) / lam    # Critical Distance ~19.2 cm

# Source: Circular aperture (D = 2mm)
x_coords = dx * (np.arange(N) - N/2)
X, Y = np.meshgrid(x_coords, x_coords)
u_in = np.zeros((N, N), dtype=complex)
u_in[(X**2 + Y**2) < (1e-3)**2] = 1.0

# Distanze fissate
z_near = 0.05   # 5 cm (Near Field)
z_far = 5.0     # 500 cm (Far Field)
distances = [z_near, z_far]

# =============================================================================
# 2. COMPUTATION AND SINGLE FIGURE (4 COLUMNS)
# =============================================================================
fig, axes = plt.subplots(4, 4, figsize=(18, 18), sharex='row')
plt.suptitle(f"Confronto Propagatori: Rescaling Voelz + BLAS Quadruple Extension\n(z_c = {z_c*100:.1f} cm)", fontsize=18, fontweight='bold')

titles = ["Transfer Function (TF)", "Impulse Response (IR)", "Angular Spectrum (AS)", "BLAS (Quad. Ext.)"]
x_mm = x_coords * 1e3
extent = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]

for row_idx, z in enumerate(distances):
    # Compute fields (Following the signatures in Propagators.py)
    u_tf = prop_TF(u_in, dx, lam, z)
    u_ir = prop_IR(u_in, dx, lam, z)
    u_as = prop_AS(u_in, lam, dx, dx, z)
    # BLAS with quadruple extension to eliminate edge aliasing
    u_blas = prop_BLAS(u_in, lam, dx, dx, z, quadruple_extension=True)
    
    fields = [u_tf, u_ir, u_as, u_blas]
    base_row = row_idx * 2
    
    for col_idx, u in enumerate(fields):
        # --- 2D MAPS (Rows 0 and 2) ---
        ax_map = axes[base_row, col_idx]
        ax_map.imshow(np.abs(u)**2, extent=extent, cmap='magma', origin='lower')
        if base_row == 0: 
            ax_map.set_title(titles[col_idx], fontsize=13, fontweight='bold')
        ax_map.set_ylabel("y [mm]") if col_idx == 0 else None
        ax_map.set_xticks([])
        ax_prof = axes[base_row + 1, col_idx]
        ax_prof.plot(x_mm, np.abs(u[N//2, :])**2, color='black', lw=1)
 
        ax_prof.set_xlabel("x [mm]") if base_row == 2 else None
        ax_prof.set_ylabel("Intensity  [a.u.]") if col_idx == 0 else None
        ax_prof.grid(True, alpha=0.2)

   # Lateral labels for the regimes
fig.text(0.01, 0.75, f'NEAR FIELD\nz = {z_near*100:.0f} cm\n(z < z_c)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
fig.text(0.01, 0.25, f'FAR FIELD\nz = {z_far*100:.0f} cm\n(z > z_c)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
plt.show()
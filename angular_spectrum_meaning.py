"""
===============================================================================
 PHYSICAL LIMITS OF THE ANGULAR SPECTRUM METHOD
===============================================================================
Reference: K. Matsushima, "Introduction to Computer Holography", Springer.
           Chapter 6: The Angular Spectrum Method.

DESCRIPTION:
  Analysis of the propagation limits imposed by the dispersion relation.

  PANEL 1: Frequency Domain (u, v)
  Visualizes the input spatial frequencies relative to the limit circle.
  - Limit Frequency: u_lim = 1/lambda.
  - Propagating Region: u^2 + v^2 < (1/lambda)^2.

  PANEL 2: Ewald Sphere Section (k_x, k_z)
  Visualizes the conservation of the wave vector modulus |k| = 2*pi/lambda.
  - Propagating waves: k_z is Real.
  - Evanescent waves: k_z becomes Imaginary to satisfy k_x^2 + k_z^2 = k^2.

  PANEL 3: Wave Amplitude Evolution (|U| vs z)
  Visualizes the effect of the Transfer Function H.
  - Propagating: |H| = 1 (Energy is conserved).
  - Evanescent:  |H| = exp(-alpha * z) (Exponential decay).
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
# 1. PHYSICAL PARAMETERS (Matsushima Notation)
# =============================================================================
lam = 0.633         # Wavelength [microns] (HeNe Red)
u_lim = 1.0 / lam   # Limit Spatial Frequency [cycles/micron]
k_mag = 2 * np.pi / lam # Wave number magnitude |k|

# Case Studies (Spatial Frequencies u)
# We select 4 distinct values for u relative to u_lim
cases = [0.0, 0.75, 1.25, 1.5] # Fractions of the limit
labels = [
    "Axial ($u=0$)", 
    "Oblique ($0.75 u_{lim}$)", 
    "Weak Evanescent ($1.25 u_{lim}$)", 
    "Strong Evanescent ($1.5 u_{lim}$)"
]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

# Visualization Styles (to handle overlapping lines)
linestyles = ['-', '--', '-.', ':']
linewidths = [3, 2.5, 2.5, 2.5]

# Propagation Distance z [microns]
z_range = np.linspace(0, 1.0, 300) 

# =============================================================================
# 2. PLOTTING SETUP
# =============================================================================
fig = plt.figure(figsize=(16, 6))
plt.subplots_adjust(wspace=0.3, bottom=0.2, top=0.85)

# --- PANEL 1: SPATIAL FREQUENCY DOMAIN (u, v) ---
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_aspect('equal')
limit_val = 1.6 * u_lim
ax1.set_xlim(-limit_val, limit_val)
ax1.set_ylim(-limit_val, limit_val)

# Draw Limit Circle (u_lim = 1/lambda)
region_circle = Circle((0, 0), u_lim, color='green', alpha=0.1)
ax1.add_patch(region_circle)
ax1.add_patch(Circle((0, 0), u_lim, color='k', fill=False, ls='--', label=r'$u_{lim} = 1/\lambda$'))

# Plot Points (Markers 'x')
for i, factor in enumerate(cases):
    u_val = factor * u_lim
    ax1.plot(u_val, 0, color=colors[i], marker='x', markersize=12, 
             markeredgewidth=3, linestyle='None')

ax1.set_xlabel(r"Spatial Frequency $u$ [$\mu m^{-1}$]")
ax1.set_ylabel(r"Spatial Frequency $v$ [$\mu m^{-1}$]")
ax1.set_title("Frequency Domain", fontweight='bold')
ax1.text(0, -0.4*u_lim, "Propagating\nRegion", color='green', ha='center', fontweight='bold')
ax1.text(0, 1.2*u_lim, "Evanescent Region", color='red', ha='center')


# --- PANEL 2: EWALD SPHERE SECTION (k_x, k_z) ---
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_aspect('equal')
k_limit_plot = 1.6 * k_mag
ax2.set_xlim(0, k_limit_plot)
ax2.set_ylim(0, 1.2 * k_mag)

# Draw Ewald Sphere Shell (|k| = constant)
theta = np.linspace(0, np.pi/2, 100)
ax2.plot(k_mag * np.cos(theta), k_mag * np.sin(theta), 'k--', alpha=0.5, label=r'$|\mathbf{k}| = 2\pi/\lambda$')

for i, factor in enumerate(cases):
    u_val = factor * u_lim
    kx_val = 2 * np.pi * u_val
    
    if factor <= 1.0:
        # Propagating: kz is Real
        kz_val = np.sqrt(k_mag**2 - kx_val**2)
        # Vector k
        ax2.quiver(0, 0, kx_val, kz_val, angles='xy', scale_units='xy', scale=1, 
                   color=colors[i], width=0.012, alpha=0.9)
        # Projection line
        ax2.plot([kx_val, kx_val], [0, kz_val], color=colors[i], ls=':', alpha=0.5)
    else:
        # Evanescent: kz is Imaginary (Geometric violation in Real space)
        ax2.quiver(0, 0, kx_val, 0, angles='xy', scale_units='xy', scale=1, 
                   color=colors[i], width=0.012, alpha=0.5)
        ax2.text(kx_val, 0.5, "!", color=colors[i], fontweight='bold', ha='center')

ax2.set_xlabel(r"Transverse Wave Number $k_x$ [rad/$\mu m$]")
ax2.set_ylabel(r"Longitudinal Wave Number $k_z$ [rad/$\mu m$]")
ax2.set_title("Ewald Sphere Section", fontweight='bold')
ax2.grid(alpha=0.3)


# --- PANEL 3: AMPLITUDE EVOLUTION (|H| vs z) ---
ax3 = fig.add_subplot(1, 3, 3)

for i, factor in enumerate(cases):
    u_val = factor * u_lim
    kx_val = 2 * np.pi * u_val
    
    if factor <= 1.0:
        # Propagating: |H(u)| = 1
        amplitude = np.ones_like(z_range)
    else:
        # Evanescente: |H(u)| = exp(-2*pi * z * sqrt(u^2 - 1/lam^2))
        # Derivazione da Matsushima Eq. per onde evanescenti
        alpha = 2 * np.pi * np.sqrt(u_val**2 - (1/lam)**2)
        amplitude = np.exp(-alpha * z_range)
        
    ax3.plot(z_range, amplitude, color=colors[i], label=labels[i], 
             ls=linestyles[i], lw=linewidths[i])

ax3.set_xlim(0, 1.0)
ax3.set_ylim(0, 1.1)
ax3.set_xlabel(r"Propagation Distance $z$ [$\mu m$]")
ax3.set_ylabel(r"Transfer Function Magnitude $|H(u)|$")
ax3.set_title("Amplitude Evolution", fontweight='bold')
ax3.grid(alpha=0.3)
ax3.legend(loc='center right', fontsize=10)

# FIX APPLICATO QUI: rf"..." per raw-string + f-string
fig.suptitle(rf"Physics of the Angular Spectrum Method ($\lambda = {lam} \mu m$)", fontsize=16)

plt.show()
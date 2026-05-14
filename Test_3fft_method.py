"""
===============================================================================
 COMPARATIVE ANALYSIS: ANALYTICAL VS. NUMERICAL TRANSFER FUNCTIONS
===============================================================================
 Reference:  K. Matsushima, "Introduction to Computer Holography", Springer (2020).
             - Section 6.4: Band-Limited Angular Spectrum Method (BLAS).
             - Section 6.4.3: More Accurate Techniques (3-FFT / Eq. 6.59).

 OBJECTIVE:
   To compare the signal fidelity between the standard Band-Limited Angular 
   Spectrum method (using a rectangular frequency cutoff) and the 'More Accurate' 
   method (using a numerical transfer function derived from the spatial impulse response).

 OBSERVATION:
   When Quadruple Extension is applied to both methods (eliminating wrap-around),
   the results are macroscopically nearly identical. However, subtle differences 
   exist in the noise floor: the 3-FFT method (Eq. 6.59) often achieves deeper 
   minima in low-intensity regions due to the absence of the 'hard' spectral 
   cutoff (Gibbs phenomenon) inherent in the standard BLAS filter.

 METHODS:
   1. BLAS (External): Uses Analytical H(u,v) + Rectangular Band-Limiting.
   2. 3-FFT (Internal): Uses Numerical H(u,v) = FFT{ h(x,y) }.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, ifftshift, fftshift

# --- EXTERNAL MODULE IMPORT ---
try:
    from Propagators import prop_BLAS
except ImportError:
    raise ImportError("Module 'Propagators.py' not found in the current directory.")

# =============================================================================
# 1. 3-FFT IMPLEMENTATION (Matsushima Eq. 6.59)
# =============================================================================

def prop_3FFT_ext(u_in, lam, dx, dy, z):
    """
    Implementation of the 'More Accurate' method using a Numerical Transfer Function.
    Includes implicit Quadruple Extension to match the external BLAS implementation.
    
    Returns:
        u_out: Complex field at distance z (cropped to original size).
        H_num: The numerical transfer function used (for visualization).
    """
    N, M = u_in.shape
    
    # A. Quadruple Extension (2x Padding)
    ny, nx = N * 2, M * 2
    u_pad = np.zeros((ny, nx), dtype=complex)
    u_pad[N//2:N//2+N, M//2:M//2+M] = u_in
    
    # B. Extended Spatial Coordinates
    x = dx * (np.arange(nx) - nx/2)
    y = dy * (np.arange(ny) - ny/2)
    X, Y = np.meshgrid(x, y)
    
    # C. Exact Spatial Impulse Response h(x,y) (Rayleigh-Sommerfeld)
    r = np.sqrt(X**2 + Y**2 + z**2)
    k = 2 * np.pi / lam
    # Term: (1/r - ik) * exp(ikr) / r^2
    h = (z / (2*np.pi)) * (1/r - 1j*k) * (np.exp(1j*k*r) / (r**2))
    
    # D. Numerical Transfer Function (The "Third" FFT)
    # ifftshift centers the phase at index [0,0] for correct convolution
    H_num = fft2(ifftshift(h)) * (dx * dy)
    
    # E. Convolution via Spectral Domain
    U_spec = fft2(ifftshift(u_pad))
    U_out_pad = ifftshift(ifft2(U_spec * H_num))
    
    # F. Crop to original ROI
    return U_out_pad[N//2:N//2+N, M//2:M//2+M], fftshift(H_num)


def get_BLAS_TransferFunction(shape, lam, dx, dy, z):
    """
    Helper function to reconstruct the Analytical BLAS Transfer Function (with Rect filter)
    solely for visualization purposes in the comparison plot.
    """
    N, M = shape
    ny, nx = N * 2, M * 2 # Matching the Quadruple ext size
    
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)
    FX, FY = np.meshgrid(fx, fy)
    
    # Analytical H (Free space)
    root = (1/lam)**2 - FX**2 - FY**2
    H = np.exp(1j * 2 * np.pi * z * np.sqrt(root.astype(complex)))
    H[np.real(root) < 0] = 0 
    
    # Rectangular Band-Limiting (Eq. 6.44)
    u_lim = 1 / np.sqrt(lam**2 * (1 + (2*z/(nx*dx))**2))
    v_lim = 1 / np.sqrt(lam**2 * (1 + (2*z/(ny*dy))**2))
    mask = (np.abs(FX) < u_lim) & (np.abs(FY) < v_lim)
    
    return fftshift(H * mask)

# =============================================================================
# 2. SIMULATION SETUP
# =============================================================================

# Parameters
lam = 633e-9          # Wavelength (HeNe)
z_test = 0.08         # Distance [m]
D = 0.2e-3            # Aperture width [m]

# Grid
N = 1024              
dx = 2.0e-6           # 2 um pixel pitch
L = N * dx            

# Source: Square Aperture (Sharp edges maximize spectral spread along axes)
x = dx * (np.arange(N) - N/2)
y = dx * (np.arange(N) - N/2)
X, Y = np.meshgrid(x, y)

u_in = np.zeros((N, N), dtype=complex)
mask = (np.abs(X) < D/2) & (np.abs(Y) < D/2)
u_in[mask] = 1.0

print(f"--- SIMULATION STARTED ---")
print(f"Parameters: N={N}, dx={dx*1e6:.1f}um, z={z_test*100:.1f}cm")

# =============================================================================
# 3. COMPUTATION
# =============================================================================

# Method 1: BLAS (External)
print("Computing BLAS (External Module)...")
# Note: Ensure prop_BLAS signature matches (u_in, lam, dx, dy, z, quadruple_extension)
u_blas = prop_BLAS(u_in, lam, dx, dx, z_test, quadruple_extension=True)
H_blas_viz = get_BLAS_TransferFunction(u_in.shape, lam, dx, dx, z_test)

# Method 2: 3-FFT (Internal)
print("Computing 3-FFT (Eq. 6.59)...")
u_3fft, H_3fft_viz = prop_3FFT_ext(u_in, lam, dx, dx, z_test)

# =============================================================================
# 4. METRICS & ANALYSIS
# =============================================================================

# Intensity (dB scale)
I_blas = np.abs(u_blas)**2
I_3fft = np.abs(u_3fft)**2
max_val = np.max(I_3fft)

I_blas_dB = 10 * np.log10(I_blas / max_val + 1e-15)
I_3fft_dB = 10 * np.log10(I_3fft / max_val + 1e-15)

# Difference Map
diff_map = np.abs(u_blas - u_3fft)
diff_dB = 20 * np.log10(diff_map / np.max(np.abs(u_3fft)) + 1e-15)

# Cross-Sections (Central Row)
mid = N // 2
prof_blas = I_blas_dB[mid, :]
prof_3fft = I_3fft_dB[mid, :]

# Transfer Function Profiles
mid_H = H_blas_viz.shape[0] // 2
freq_axis = np.fft.fftshift(np.fft.fftfreq(H_blas_viz.shape[1], dx))
H_prof_blas = np.abs(H_blas_viz[mid_H, :])
H_prof_3fft = np.abs(H_3fft_viz[mid_H, :])

# =============================================================================
# 5. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# Plot A: Reference Reconstruction (3-FFT)
ax1 = axes[0, 0]
im1 = ax1.imshow(I_3fft_dB, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
                 cmap='inferno', vmin=-60, vmax=0)
ax1.set_title("2D Reconstruction (3-FFT Method)", fontweight='bold')
ax1.set_ylabel("y [mm]")
ax1.set_xlabel("x [mm]")
cb1 = plt.colorbar(im1, ax=ax1)
cb1.set_label("Intensity [dB]")

# Plot B: Error Distribution
ax2 = axes[0, 1]
im2 = ax2.imshow(diff_dB, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3],
                 cmap='magma', vmin=-100, vmax=-40)
ax2.set_title("Difference Map (|BLAS - 3FFT|)", fontweight='bold')
ax2.set_xlabel("x [mm]")
cb2 = plt.colorbar(im2, ax=ax2)
cb2.set_label("Difference Magnitude [dB]")

# Plot C: 1D Profile Comparison (Log Scale)
ax3 = axes[1, 0]
ax3.plot(x*1e3, prof_blas, 'r', linewidth=1.5, alpha=0.6, label='BLAS (Rect Cutoff)')
ax3.plot(x*1e3, prof_3fft, 'k--', linewidth=1.0, alpha=1.0, label='3-FFT (Soft Rolloff)')
ax3.set_title("1D Profile Analysis (Log Scale)", fontweight='bold')
ax3.set_ylabel("Intensity [dB]")
ax3.set_xlabel("x [mm]")
ax3.set_ylim(-80, 5)
ax3.set_xlim(-L/2*1e3, L/2*1e3)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Plot D: Transfer Function Analysis
ax4 = axes[1, 1]
# Zoom on low frequencies to show the cutoff shape
freq_mask = np.abs(freq_axis) < 1.0e6  # +/- 1000 mm^-1
ax4.plot(freq_axis[freq_mask]*1e-3, H_prof_blas[freq_mask], 'r', 
         linewidth=2, label='BLAS H(u) [Hard Step]')
ax4.plot(freq_axis[freq_mask]*1e-3, H_prof_3fft[freq_mask], 'k--', 
         linewidth=2, label='3-FFT H(u) [Smooth]')
ax4.set_title("Transfer Function H(u)", fontweight='bold')
ax4.set_ylabel("Magnitude |H|")
ax4.set_xlabel("Spatial Frequency [mm^-1]")
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right')

fig.suptitle("Numerical Propagation Comparison: Band-Limited AS vs. 3-FFT Method", fontsize=14)
plt.show()
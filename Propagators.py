"""
===============================================================================
 OPTICAL PROPAGATORS LIBRARY AND RELATED METHODS
===============================================================================
Author: Marco Astarita
Date:   September 2025 (Updated January 2026)
Version: 2.1

DESCRIPTION OF PROPAGATORS AND SOURCES:
---------------------------------------
1. prop_fresnel_single_step (Single-Step Fresnel):
    S-FFT implementation of the Fresnel integral. Useful for long distances.
    Source: K. Matsushima, "Introduction to Computer Holography", Ch. 4.

2. prop_TF (Transfer Function):
    Propagation in the frequency domain (approximate angular spectrum).
    Source: D. G. Voelz, "Computational Fourier Optics", Ch. 5.

3. prop_IR (Impulse Response):
    Propagation via direct convolution in the spatial domain.
    Source: D. G. Voelz, "Computational Fourier Optics", Ch. 5.

4. prop_AS (Angular Spectrum):
    Exact solution of the Helmholtz equation (no paraxial approximation).
    Source: K. Matsushima, "Introduction to Computer Holography", Ch. 6.

5. prop_BLAS (Band-Limited Angular Spectrum):
    Angular spectrum version with low-pass filter to avoid aliasing.
    Includes the "Quadruple Extension" option to eliminate edge artifacts.
    Source: K. Matsushima, "Introduction to Computer Holography", Ch. 6.4.

6. propagate_wide_field (Old & New):
    Probably Obsolete :(
    Original methods based on mapping to conjugate planes via virtual lenses.
    Allow to vary the physical extension of the arrival plane (magnification).
    Developed by: Marco Astarita and Paolo Pozzi.

===============================================================================
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# ==============================================================================
# SEZIONE 1: SINGLE STEP (FRESNEL S-FFT)
# ==============================================================================

def prop_fresnel_single_step(u_in, lam, dx1, dy1, z):
    """
    Propagazione di Fresnel tramite trasformata singola (S-FFT).
    
    DIMENSIONI FISICHE:
    Il campionamento in uscita (dx2) è legato a quello di ingresso (dx1):
    dx2 = (lam * |z|) / (N * dx1)  =>  L_out = (lam * |z|) / dx1
    """
    M, N = u_in.shape
    k = 2 * np.pi / lam
    
    x1 = dx1 * (np.arange(N) - N/2)
    y1 = dy1 * (np.arange(M) - M/2)
    X1, Y1 = np.meshgrid(x1, y1)
    
    u_lensed = u_in * np.exp(1j * k / (2 * z) * (X1**2 + Y1**2))
    U_f = fftshift(fft2(ifftshift(u_lensed)))
    
    dx2 = (lam * np.abs(z)) / (N * dx1)
    dy2 = (lam * np.abs(z)) / (M * dy1)
    x2 = dx2 * (np.arange(N) - N/2)
    y2 = dy2 * (np.arange(M) - M/2)
    X2, Y2 = np.meshgrid(x2, y2)
    
    C = (np.exp(1j * k * z) / (1j * lam * z)) * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
    u_out = C * U_f * dx1 * dy1
    
    return u_out, dx2, dy2

# ==============================================================================
# SEZIONE 2: CONVOLUTIONAL METHODS (SAMPLING COSTANTE dx2 = dx1)
# ==============================================================================

def prop_TF(u1, dx, wvl, z):
    """Transfer Function. Dimens. Fisiche: Inalterate (dx2 = dx1)."""
    M, N = u1.shape
    k = 2 * np.pi / wvl
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * wvl * z * (FX**2 + FY**2)) * np.exp(1j * k * z)
    return ifft2(fft2(u1) * H)

def prop_IR(u1, dx, wvl, z):
    """Impulse Response. Dimens. Fisiche: Inalterate (dx2 = dx1)."""
    M, N = u1.shape
    k = 2 * np.pi / wvl
    x = dx * (np.arange(N) - N/2)
    y = dx * (np.arange(M) - M/2)
    X, Y = np.meshgrid(x, y)
    h = 1/(1j * wvl * z) * np.exp(1j * k / (2 * z) * (X**2 + Y**2)) * np.exp(1j * k * z)
    H = fft2(ifftshift(h)) * dx**2
    return ifft2(fft2(u1) * H)

def prop_AS(u_in, lam, dx, dy, z):
    """Angular Spectrum (Esatto). Dimens. Fisiche: Inalterate (dx2 = dx1)."""
    M, N = u_in.shape
    u = np.fft.fftfreq(M, d=dx)
    v = np.fft.fftfreq(N, d=dy)
    U, V = np.meshgrid(u, v, indexing='ij')
    mask = (U**2 + V**2) <= (1.0 / lam**2)
    phase = np.sqrt((1.0/lam**2 - U**2 - V**2).astype(complex))
    H = np.exp(1j * 2 * np.pi * z * phase) * mask
    return ifft2(fft2(u_in) * H)

def prop_BLAS(u_in, lam, dx, dy, z, quadruple_extension=False):
    """Band-Limited Angular Spectrum con Quadruple Extension."""
    if quadruple_extension:
        M, N = u_in.shape
        u_calc = np.pad(u_in, ((M//2, M//2), (N//2, N//2)), mode='constant')
    else:
        u_calc = u_in
    
    M_p, N_p = u_calc.shape
    du, dv = 1.0 / (M_p * dx), 1.0 / (N_p * dy)
    u = np.fft.fftfreq(M_p, d=dx)
    v = np.fft.fftfreq(N_p, d=dy)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    u_lim = 1.0 / (lam * np.sqrt(1.0 + (2.0 * z * du)**2))
    v_lim = 1.0 / (lam * np.sqrt(1.0 + (2.0 * z * dv)**2))
    
    mask = (np.abs(U) <= u_lim) & (np.abs(V) <= v_lim) & ((U**2 + V**2) <= (1.0 / lam**2))
    phase = np.sqrt((1.0/lam**2 - U**2 - V**2).astype(complex))
    H = np.exp(1j * 2 * np.pi * z * phase) * mask
    
    u_out = ifft2(fft2(u_calc) * H)
    if quadruple_extension:
        M_o, N_o = u_in.shape
        u_out = u_out[M_o//2 : M_o//2 + M_o, N_o//2 : N_o//2 + N_o]
    return u_out

# ==============================================================================
# SEZIONE 3: METODI STRANI (WIDE FIELD ASTARITA-POZZI)
# ==============================================================================

def propagate_wide_field_old(input_phase, d, lam, res, z):
    """
    Versione OLD del metodo Wide Field.
    DIMENSIONI FISICHE:
    Il campo di arrivo viene riscalato di un fattore M = zconj / z.
    extent_1 = extent_0 / M
    """
    coords = np.linspace(-1.0, 1.0, res) * d * float(res) / 2.0
    x, y = np.meshgrid(coords, coords)
    zmin = (d * res) / (2 * np.tan(np.arcsin(lam / (2 * d))))
    zconj = 1 / ((1 / zmin) - (1 / z))
    magnification = zconj / z

    lens_phase = -np.pi * (x**2 + y**2) / (lam * zmin)
    phase_lensed = ((input_phase + lens_phase) % (2 * np.pi)) - np.pi
    u_0 = 1 * np.exp(1j * phase_lensed)
    
    padding = res // 2 
    u_0 = np.pad(u_0, pad_width=((padding, padding), (padding, padding)))
    L_padded = u_0.shape[0] * d
    extent_0 = [-L_padded/2, L_padded/2, -L_padded/2, L_padded/2]

    u_1 = prop_TF(u_0, d, lam, zconj)
    extent_1 = [coord / magnification for coord in extent_0]

    return u_0, u_1, extent_0, extent_1

def propagate_wide_field_new(input_phase, d, lam, res, z):
    """
    Versione NEW del metodo Wide Field (con upsampling e padding bilanciato).
    DIMENSIONI FISICHE:
    Usa la stessa logica di ingrandimento della versione OLD ma con 
    interpolazione iniziale per migliorare la risoluzione del campo lensed.
    """
    phase_upsampled = np.kron(input_phase, np.ones((2, 2)))
    d_upsampled = d / 2
    res_upsampled = res * 2
    
    coords = np.linspace(-1.0, 1.0, res_upsampled) * d * res / 2.0
    x, y = np.meshgrid(coords, coords)
    
    zmin = (d * res) / (2 * np.tan(np.arcsin(lam / (2 * d))))
    zconj = 1 / ((1 / zmin) - (1 / z))
    magnification = zconj / z

    lens_phase = -np.pi * (x**2 + y**2) / (lam * zmin)
    phase_lensed = ((phase_upsampled + lens_phase) % (2 * np.pi)) - np.pi
    u_0 = 1 * np.exp(1j * phase_lensed)
    
    padding_shape = input_phase.shape
    u_0 = np.pad(u_0, pad_width=(padding_shape, padding_shape))
    L_padded_half = res * d
    extent_0 = [-L_padded_half, L_padded_half, -L_padded_half, L_padded_half]
    
    u_1 = prop_TF(u_0, d_upsampled, lam, zconj)
    extent_1 = [coord / magnification for coord in extent_0]

    return u_0, u_1, extent_0, extent_1
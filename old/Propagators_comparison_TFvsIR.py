"""
===============================================================================
 COMPARISON: TF vs IR vs AS (THE VOELZ REGIMES)
===============================================================================
Basato su: D. Voelz, "Computational Fourier Optics", Cap. 5.
Questo script dimostra le situazioni ottimali e subottimali per i propagatori
di Fresnel (TF e IR) confrontandoli con l'Angular Spectrum (AS).
"""

import numpy as np
import matplotlib.pyplot as plt
from Propagators import prop_TF, prop_IR, prop_AS

# =============================================================================
# 1. SETUP PARAMETRI (Esempio "Square Beam" di Voelz, p. 82)
# =============================================================================
L = 0.5               # Lato della griglia [m]
M = 250               # Numero di campioni (basso per forzare l'aliasing)
dx = L / M            # Passo di campionamento (2 mm)
lam = 0.5e-6          # Lunghezza d'onda (0.5 um)
w = 0.051             # Semilarghezza dell'apertura quadrata (51 mm)

# Calcolo della DISTANZA CRITICA (Voelz Eq. 5.8)
z_c = (L * dx) / lam  # Per questi parametri z_c = 2000 m

print(f"--- ANALISI DI CAMPIONAMENTO (VOELZ) ---")
print(f"Pixel pitch (dx): {dx*1e3:.1f} mm")
print(f"Distanza Critica (z_c): {z_c:.1f} m")
print("-" * 40)

# Griglia coordinate
x = dx * (np.arange(M) - M/2)
X, Y = np.meshgrid(x, x)

# Sorgente: Apertura Quadrata
u_in = np.zeros((M, M), dtype=complex)
u_in[(np.abs(X) < w) & (np.abs(Y) < w)] = 1.0

# ================= ============================================================
# 2. DEFINIZIONE DEI CASI DI TEST
# =============================================================================
# Caso 1: z < z_c (TF Ottimale, IR Subottimale - Copie periodiche)
z_near = 1000 

# Caso 2: z > z_c (IR Ottimale, TF Subottimale - Aliasing/Stair-step)
z_far = 20000 

distances = [z_near, z_far]
labels = [f"Near Field (z={z_near}m < z_c)", f"Far Field (z={z_far}m > z_c)"]

# ================= ============================================================
# 3. ESECUZIONE E PLOT
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.3)

for row, z in enumerate(distances):
    # Calcolo Propagazioni
    # Nota: TF e IR usano l'approssimazione di Fresnel (Cap. 5)
    # AS è la soluzione esatta di Helmholtz (Cap. 4/6)
    u_tf = prop_TF(u_in, dx, lam, z)
    u_ir = prop_IR(u_in, dx, lam, z)
    u_as = prop_AS(u_in, lam, dx, dx, z)
    
    # Plotting Profili Centrali (Fetta X al centro)
    mid = M // 2
    
    # Plot TF
    axes[row, 0].plot(x, np.abs(u_tf[mid, :])**2, 'r', label='TF')
    axes[row, 0].set_title(f"{labels[row]}\nTransfer Function")
    
    # Plot IR
    axes[row, 1].plot(x, np.abs(u_ir[mid, :])**2, 'b', label='IR')
    axes[row, 1].set_title(f"{labels[row]}\nImpulse Response")
    
    # Plot AS (Riferimento Esatto)
    axes[row, 2].plot(x, np.abs(u_as[mid, :])**2, 'k', label='AS')
    axes[row, 2].set_title(f"{labels[row]}\nAngular Spectrum")

    # Formattazione assi
    for col in range(3):
        axes[row, col].set_xlabel("x [m]")
        axes[row, col].set_ylabel("Intensità")
        axes[row, col].grid(alpha=0.2)
        if row == 0: axes[row, col].set_ylim(-0.1, 2.5)

# Evidenziazione Artefatti nel testo
print("\nCOSA OSSERVARE NEI RISULTATI:")
print(f"1. A z={z_near}m (Near): IR mostra 'fantasmi' ai bordi (aliasing spaziale). TF è pulito.")
print(f"2. A z={z_far}m (Far): TF mostra oscillazioni 'stair-step' (aliasing frequenziale). IR è liscio.")
print(f"3. AS (Angular Spectrum) rimane il riferimento più solido, ma richiede più memoria.")

plt.tight_layout()
plt.show()
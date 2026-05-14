# Propagators

A comprehensive library and simulation suite for optical field propagation methods, with a focus on comparative analysis, artifact suppression, and high-fidelity modeling. Developed by Marco Astarita.

## Overview

This repository provides implementations and comparative studies of several optical propagators, including:

- **Single-Step Fresnel (S-FFT)**
- **Transfer Function (TF)**
- **Impulse Response (IR)**
- **Angular Spectrum (AS)**
- **Band-Limited Angular Spectrum (BLAS)**
- **Wide Field Propagation (Old & New)**

The codebase includes both core propagation routines and scripts for visual and quantitative comparison of their performance, especially in the context of SLM (Spatial Light Modulator) setups.

## Features

- High-resolution 2D and 1D analysis of propagation artifacts
- Comparative studies between classical and advanced methods (Voelz vs. Matsushima)
- Artifact suppression techniques (band-limiting, quadruple extension)
- Ready-to-use scripts for academic visualization and publication

## Main Scripts

- `Propagators.py`: Core library of propagation methods with references.
- `Propagators_comparison_TFvsIRvsASvsBLAS.py`: Multi-method simulation and artifact analysis for SLM setups.
- `Propagators_comparison_ASvsBLASvsBLASQ.py`: 2D amplitude analysis and artifact suppression.
- `Propagators_comparison_CONVvsONESTEP.py`, `Propagators_comparison_interactive.py`: Additional comparative and interactive studies.
- `old/`: Contains legacy and experimental propagation scripts.

## References

- D. G. Voelz, *Computational Fourier Optics: a MATLAB tutorial* (Springer)
- K. Matsushima, *Introduction to Computer Holography* (Springer)

## Getting Started

1. Clone the repository.
2. Install requirements (NumPy, Matplotlib).
3. Run any comparison script to reproduce figures and analyses.

```bash
python Propagators_comparison_TFvsIRvsASvsBLAS.py
```

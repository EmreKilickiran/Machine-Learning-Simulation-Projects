# Numerical Modeling and FDTD Simulation of Electromagnetic Wave Propagation

A complete 1D Finite-Difference Time-Domain (FDTD) electromagnetic wave propagation solver implemented in MATLAB, systematically progressing from free-space propagation to a self-designed periodic Bragg reflector. Individual course project for **Electromagnetic Waves (EE4051)**, Spring 2024.

## Overview

The solver implements the Yee algorithm to discretize Maxwell's curl equations (∂E/∂t and ∂H/∂t) into explicit update equations on a staggered spatial grid, satisfying the Courant stability condition (Δt ≤ Δx/c). The project iteratively extends the solver's boundary conditions, material models, and validation procedures at each stage.

## Simulation Scenarios

| Script | Scenario | Key Validation |
|--------|----------|----------------|
| `task1_sinusoidal.m` | Free space, sinusoidal source | Wave velocity: 2.04% error vs c₀ |
| `task1_gaussian.m` | Free space, Gaussian pulse | Pulse propagation & velocity |
| `task2_dielectric.m` | Lossless (εr=16) + lossy (σ=0.5 S/m) | Velocity: 0.87% error; attenuation vs e⁻ᵅᶻ |
| `task3_interface.m` | Single dielectric interface (εr: 1→4) | Γ = −0.333, T = 0.667 (analytical match) |
| `task4_antireflection.m` | Quarter-λ AR coating | FFT confirms reflection suppression at 1 GHz |
| `task5_bragg_reflector.m` | Periodic Bragg reflector (5 periods) | Strong reflection via constructive interference |

## Numerical Methods

- **Yee Algorithm**: Staggered grid with E and H offset by half-steps in space and time
- **Courant Condition**: Δt = Δz/(2c₀) ensures numerical stability
- **Absorbing Boundary**: First-order ABC at the right boundary to minimize spurious reflections
- **Lossy Media**: Conductivity-modified update equation with exponential decay coefficients
- **Spectral Analysis**: FFT-based frequency-domain validation of anti-reflection performance

## Repository Structure

```
.
├── README.md
├── Report.pdf                      # Full project report with figures
└── matlab/
    ├── task1_sinusoidal.m          # Free space — sinusoidal source
    ├── task1_gaussian.m            # Free space — Gaussian pulse
    ├── task2_dielectric.m          # Lossless + lossy dielectric media
    ├── task3_interface.m           # Single interface reflection/transmission
    ├── task4_antireflection.m      # Quarter-wavelength AR coating + FFT
    └── task5_bragg_reflector.m     # Periodic Bragg reflector
```

## Requirements

MATLAB (no external toolboxes required).

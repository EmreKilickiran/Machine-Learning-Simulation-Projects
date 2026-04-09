# Computational Simulation and Performance Analysis of Digital Communication Algorithms

Monte Carlo-based simulators for digital modulation, channel equalization, and convolutional coding with Viterbi decoding, implemented in MATLAB from first principles. All results validated against closed-form analytical bounds. Individual course project for **Digital Communication (EE4082)**, Spring 2024.

## Overview

Modeled digital communication channels as stochastic systems corrupted by additive white Gaussian noise (AWGN) and independently implemented all algorithms (Viterbi decoder, matrix-based equalizer, modulation/demodulation chains) to estimate error-rate performance across varying signal-to-noise conditions.

## Project Structure

```
.
├── README.md
└── matlab/
    ├── ber_analysis_qam_bpsk_dpsk_equalization.m   # BER simulation: QAM, BPSK, DPSK + equalizer
    ├── viterbi_decoding_convolutional_codes.m       # Convolutional coding, Viterbi, HDD vs SDD
    └── qam_simulink_model.slx                       # Simulink visual model for QAM system
```

## Project 1: Modulation & Equalization

| Part | Analysis | Method |
|------|----------|--------|
| A | 4/16/64-QAM BER | Monte Carlo (10 trials × 1M bits), theoretical Q-function bounds |
| B | BPSK BER vs data rate | Fixed received power, variable Rb (3–35 kbps) |
| C | BPSK vs DPSK | Coherent vs differential detection comparison |
| D | Zero-forcing equalizer | Toeplitz matrix inversion (3×3 and 5×5), residual ISI evaluation |

## Project 2: Convolutional Coding & Viterbi Decoding

| Part | Analysis | Method |
|------|----------|--------|
| A | Coding gain | Uncoded vs Rate 1/2 vs Rate 1/3 (K=3) |
| B | Constraint length sweep | K = 3, 5, 7, 9 with varying generator polynomials |
| C | Analytical bounds | Distance spectrum analysis, `bercoding()` upper bounds |
| D | Complexity benchmarking | Execution time scaling: 5s (K=3) → 21s (K=9) for 5M samples |
| E | HDD vs SDD | Hard vs 4-level soft decision Viterbi decoding (~2 dB gain) |

## Key Results

- Simulated error rates matched theoretical predictions across all configurations
- Soft decision decoding outperformed hard decision by ~2 dB at equivalent error rates
- Viterbi decoder execution time confirmed exponential scaling with state-space size
- Analytical upper bounds consistently overestimated empirical results, confirming correct implementation

## Requirements

MATLAB with Communications Toolbox (`convenc`, `vitdec`, `qammod`, `bercoding`, `distspec`).

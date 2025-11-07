# Hybrid Board–τ–Γ–Δ + FDTD (TEz) Simulator

Reference implementation for the paper **“Emergent Spacetime, Gravity, and Electrodynamics from a Layered Clock and Coherence Field (Theory of the Board)”**.

## What this does
- Simulates a causal **Board (DAG)** that feeds a path–intensity `I`.
- Evolves the **clock field τ** and curvature **Γ' = (L/τ0) ∇τ** (parabolic core).
- Evolves the **coherence field Δ** with damping by roughness `||Γ'||^2`.
- Runs a **TEz FDTD** step with constitutives:
  - `n_eff(Δ, ||Γ'||^2)` with clamp `n_eff ≥ 1` (passivity/causality)
  - `J_Γ = σ_Γ ||Γ'||^2 E` (dissipative curvature channel)

Outputs **CSV metrics**, **NPZ snapshot** of the final fields, and optional **PNG plots**.

## Quick start
```bash
python hybrid_board_tau_gamma_delta_fdtd.py --steps 200 --nx 64 --ny 64 --seed 42 --out-prefix run1 --plot
```

You will get:
- `run1_metrics.csv` — timeseries of ⟨Δ⟩, ⟨||Γ'||^2⟩, `n_eff,max`, `dt`, EM energy, etc.
- `run1_final.npz` — arrays: `tau`, `Delta`, `Ez`, `Hx`, `Hy`.
- `run1_config.json` — parameters used and wall time.
- `run1_timeseries_delta.png`, `run1_timeseries_neff.png`, `run1_timeseries_energy.png` (if `--plot`).

## Parameters (selected)
- `--a_Delta`, `--a_Gamma`, `--sigma_Gamma` — constitutive couplings
- `--D_tau`, `--D_Delta`, `--tau_Delta`, `--kappa_Gamma` — PDE/relaxation
- `--L`, `--tau0` — non-dimensionalization for Γ' = (L/τ0) ∇τ
- `--cfl` — FDTD stability factor (default 0.9)

## Notes
- `n_eff` is treated as a **refractive index** (ε_r = n_eff², μ_r = 1).
- The clamp `n_eff ≥ 1` plus `J_Γ ≥ 0` ensures **passivity**; a Yee-grid energy
  balance (see paper, Appendix D) gives monotone EM energy decay.
- Defaults match the manuscript’s executable snippet.

## License
CC BY 4.0 — see `LICENSE`.

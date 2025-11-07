#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Board–τ–Γ–Δ + FDTD (TEz) Simulator
=========================================
Reference implementation for the manuscript:
"Emergent Spacetime, Gravity, and Electrodynamics from a Layered Clock and
Coherence Field (Theory of the Board)".

Features
--------
- Board (DAG with fairness) → path intensity I
- Clock field τ and curvature Γ' = (L/τ0) ∇τ (parabolic core)
- Coherence field Δ with roughness damping
- Electrodynamics (TEz FDTD) with n_eff(Δ, ||Γ'||^2) and J_Γ = σ_Γ ||Γ'||^2 E
- Dynamic CFL, passivity clamps (n_eff ≥ 1), and basic diagnostics
- CSV logs + NPZ snapshot; optional timeseries plot

License: CC BY 4.0 (see LICENSE file)
Author: Artur do Nascimento; Lyriam Project (AI)
"""
from __future__ import annotations
import argparse, math, csv, json, time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import numpy as np

# ---------------------- Utilities ----------------------
def laplacian_periodic(X: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """5-point Laplacian with periodic BCs and proper dx, dy scaling."""
    return ((np.roll(X, -1, 0) - 2*X + np.roll(X, 1, 0)) / (dx*dx)
          + (np.roll(X, -1, 1) - 2*X + np.roll(X, 1, 1)) / (dy*dy))

def grad_norm_sq_periodic(X: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """||∇X||^2 with central differences and periodic BCs."""
    gx = (np.roll(X, -1, 0) - np.roll(X,  1, 0)) / (2*dx)
    gy = (np.roll(X, -1, 1) - np.roll(X,  1, 1)) / (2*dy)
    return gx*gx + gy*gy

# ---------------------- Board (DAG) ----------------------
def add_node_fair(parents: List[List[int]], window: int=12, max_parents: int=3, p_edge: float=0.25, rng: np.random.Generator|None=None):
    """Append a node with parents chosen from the last `window` nodes, capped by `max_parents`."""
    if rng is None:
        rng = np.random.default_rng()
    N = len(parents)
    cand = list(range(max(0, N-window), N))
    new_pars = [i for i in cand if rng.random() < p_edge]
    if len(new_pars) > max_parents:
        new_pars = rng.choice(new_pars, size=max_parents, replace=False).tolist()
    parents.append(new_pars)

def path_intensity_scalar(parents: List[List[int]]) -> int:
    """Compute a simple 'path count' intensity recursively (cheap)."""
    N = len(parents)
    pc = np.ones(N, dtype=np.int64)
    for i in range(N):
        if parents[i]:
            pc[i] = 1 + sum(pc[p] for p in parents[i])
    return int(pc[-1])

def board_to_I(Delta: np.ndarray, parents: List[List[int]], rng: np.random.Generator|None=None) -> np.ndarray:
    """Map Board state → scalar I, squash → weight, modulate by normalized Δ to get spatial I(x,y)."""
    I_scalar = path_intensity_scalar(parents)
    Iw = 1.0 / (1.0 + math.exp(-0.002*(I_scalar - 50.0)))  # logistic squash
    Dn = (Delta - Delta.min()) / (Delta.max() - Delta.min() + 1e-9)
    return np.clip(0.25 + 0.75*Iw, 0.0, 1.0) * Dn

# ---------------------- Config & Defaults ----------------------
@dataclass
class SimConfig:
    nx: int = 64
    ny: int = 64
    dx: float = 1.0
    dy: float = 1.0
    steps: int = 200
    cfl: float = 0.9
    seed: int = 42

    # τ, Δ PDE params
    D_tau: float = 0.20
    kappa_C: float = 0.0
    D_Delta: float = 0.10
    tau_Delta: float = 3.0
    g_Delta: float = 0.50
    kappa_Gamma: float = 0.30

    # Constitutives
    n0: float = 1.0
    a_Delta: float = 0.60
    a_Gamma: float = 0.20
    sigma_Gamma: float = 0.05
    tau0: float = 1.0
    L: float = 10.0
    Gamma_p_sq_max_clip: float = 4.0

    # IO
    out_prefix: str = "run"
    log_every: int = 20
    print_scales_every: int = 50
    plot: bool = False

def parse_args() -> SimConfig:
    ap = argparse.ArgumentParser(description="Hybrid Board–τ–Γ–Δ + FDTD (TEz) simulator")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cfl", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D_tau", type=float, default=0.20)
    ap.add_argument("--kappa_C", type=float, default=0.0)
    ap.add_argument("--D_Delta", type=float, default=0.10)
    ap.add_argument("--tau_Delta", type=float, default=3.0)
    ap.add_argument("--g_Delta", type=float, default=0.50)
    ap.add_argument("--kappa_Gamma", type=float, default=0.30)
    ap.add_argument("--n0", type=float, default=1.0)
    ap.add_argument("--a_Delta", type=float, default=0.60)
    ap.add_argument("--a_Gamma", type=float, default=0.20)
    ap.add_argument("--sigma_Gamma", type=float, default=0.05)
    ap.add_argument("--tau0", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--Gamma_p_sq_max_clip", type=float, default=4.0)
    ap.add_argument("--out-prefix", type=str, default="run")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--print-scales-every", type=int, default=50)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cfg = SimConfig(
        nx=args.nx, ny=args.ny, dx=args.dx, dy=args.dy, steps=args.steps, cfl=args.cfl, seed=args.seed,
        D_tau=args.D_tau, kappa_C=args.kappa_C, D_Delta=args.D_Delta, tau_Delta=args.tau_Delta,
        g_Delta=args.g_Delta, kappa_Gamma=args.kappa_Gamma, n0=args.n0, a_Delta=args.a_Delta,
        a_Gamma=args.a_Gamma, sigma_Gamma=args.sigma_Gamma, tau0=args.tau0, L=args.L,
        Gamma_p_sq_max_clip=args.Gamma_p_sq_max_clip, out_prefix=args.out_prefix,
        log_every=args.log_every, print_scales_every=args.print_scales_every, plot=args.plot
    )
    return cfg

# ---------------------- Simulation ----------------------
def run(cfg: SimConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    # Fields
    tau   = np.zeros((cfg.nx, cfg.ny))
    Delta = 0.40 + 0.05 * rng.standard_normal((cfg.nx, cfg.ny))
    Ez = np.zeros((cfg.nx, cfg.ny)); Hx = np.zeros((cfg.nx, cfg.ny)); Hy = np.zeros((cfg.nx, cfg.ny))

    # Board
    parents: List[List[int]] = [[], [0], [1]]

    # Logs
    metrics: List[Dict[str, float]] = []
    t0 = time.time()

    for t in range(cfg.steps):
        # Board growth (fairness)
        add_node_fair(parents, rng=rng)

        # τ update
        lap_tau = laplacian_periodic(tau, cfg.dx, cfg.dy)
        tau = tau + 0.02 * (cfg.D_tau * lap_tau - cfg.kappa_C * 0.0)  # source C set to 0 here

        # Γ' computation (clamped)
        Gamma_sq   = grad_norm_sq_periodic(tau, cfg.dx, cfg.dy)
        Gamma_p_sq = (cfg.L/cfg.tau0)**2 * Gamma_sq
        Gamma_p_sq = np.clip(Gamma_p_sq, 0.0, cfg.Gamma_p_sq_max_clip)

        # I from Board + Δ
        I = board_to_I(Delta, parents, rng=rng)

        # Δ update
        lap_D = laplacian_periodic(Delta, cfg.dx, cfg.dy)
        Delta = Delta + 0.02 * ( -Delta/cfg.tau_Delta + cfg.D_Delta*lap_D + cfg.g_Delta*I
                                 - cfg.kappa_Gamma*Gamma_p_sq*Delta )

        # n_eff and dt (CFL)
        n_eff = cfg.n0 * (1.0 + cfg.a_Delta*Delta - cfg.a_Gamma*Gamma_p_sq)
        n_eff = np.maximum(n_eff, 1.0)  # passivity/causality clamp
        dt_fdtd = cfg.cfl / (math.sqrt((1.0/cfg.dx**2)+(1.0/cfg.dy**2)) * (n_eff.max()) + 1e-12)

        # Curvature-induced ohmic channel
        Jg = cfg.sigma_Gamma * Gamma_p_sq

        # Yee updates (TEz)
        dE_dy = (np.roll(Ez, -1, 1) - Ez) / cfg.dy
        Hx = Hx - dt_fdtd * dE_dy

        dE_dx = (np.roll(Ez, -1, 0) - Ez) / cfg.dx
        Hy = Hy + dt_fdtd * dE_dx

        dHy_dx = (Hy - np.roll(Hy, 1, 0)) / cfg.dx
        dHx_dy = (Hx - np.roll(Hx, 1, 1)) / cfg.dy
        curlH  = dHy_dx - dHx_dy
        Ez = Ez + (dt_fdtd / (n_eff**2)) * (curlH - Jg * Ez)

        # Diagnostics
        energy_em = 0.5*((Ez**2).mean() + (Hx**2).mean() + (Hy**2).mean())
        I_scalar = path_intensity_scalar(parents)
        mrow = dict(
            t=t,
            Delta_mean=float(Delta.mean()),
            Gamma_p_sq_mean=float(Gamma_p_sq.mean()),
            n_eff_max=float(n_eff.max()),
            dt_fdtd=float(dt_fdtd),
            energy_em=float(energy_em),
            I_scalar=float(I_scalar),
        )
        metrics.append(mrow)

        if (t % cfg.log_every) == 0:
            print(f"t={t:04d}  <Δ>={mrow['Delta_mean']:.3f}  <|Γ'|^2>={mrow['Gamma_p_sq_mean']:.3f}  "
                  f"I={int(I_scalar)}  dt={dt_fdtd:.4f}  Eem={energy_em:.3e}")

        if (t % cfg.print_scales_every) == 0:
            print(f"[scales] τ0={cfg.tau0}, L={cfg.L}, max|Γ'|^2={float(Gamma_p_sq.max()):.3f}")
            print(f"[cfl] n_eff_max={float(n_eff.max()):.3f} -> dt_fdtd={dt_fdtd:.6f}")

    wall = time.time() - t0
    out = {
        "metrics": metrics,
        "final": {
            "tau": tau, "Delta": Delta, "Ez": Ez, "Hx": Hx, "Hy": Hy
        },
        "wall_time_sec": wall,
        "parents_len": len(parents)
    }
    return out

# ---------------------- IO Helpers ----------------------
def save_csv(metrics: List[Dict[str, float]], path: str):
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow(m)

def save_npz(final: Dict[str, np.ndarray], path: str):
    np.savez_compressed(path, **final)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def maybe_plot(metrics: List[Dict[str, float]], out_png: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available ({e}); skipping plot.")
        return
    t = [m["t"] for m in metrics]
    d = [m["Delta_mean"] for m in metrics]
    n = [m["n_eff_max"] for m in metrics]
    e = [m["energy_em"] for m in metrics]

    plt.figure()
    plt.plot(t, d, label="⟨Δ⟩")
    plt.xlabel("time step")
    plt.ylabel("mean Δ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_delta.png"), dpi=140)
    plt.close()

    plt.figure()
    plt.plot(t, n, label="n_eff,max")
    plt.xlabel("time step")
    plt.ylabel("n_eff,max")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_neff.png"), dpi=140)
    plt.close()

    plt.figure()
    plt.plot(t, e, label="EM energy")
    plt.xlabel("time step")
    plt.ylabel("energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_energy.png"), dpi=140)
    plt.close()

# ---------------------- Main ----------------------
def main():
    cfg = parse_args()
    print("[cfg]", asdict(cfg))
    out = run(cfg)

    base = cfg.out_prefix
    csv_path = f"{base}_metrics.csv"
    npz_path = f"{base}_final.npz"
    json_path = f"{base}_config.json"
    png_base = f"{base}_timeseries.png"

    save_csv(out["metrics"], csv_path)
    save_npz(out["final"], npz_path)
    save_json(asdict(cfg) | {"wall_time_sec": out["wall_time_sec"], "parents_len": out["parents_len"]}, json_path)

    if cfg.plot:
        maybe_plot(out["metrics"], png_base)

    print(f"[done] steps={cfg.steps} wall_time={out['wall_time_sec']:.2f}s parents={out['parents_len']}")
    print(f"[files] {csv_path}  {npz_path}  {json_path}")
    if cfg.plot:
        print(f"[plots] {png_base.replace('.png', '_delta.png')}, "
              f"{png_base.replace('.png', '_neff.png')}, "
              f"{png_base.replace('.png', '_energy.png')}")

if __name__ == "__main__":
    main()

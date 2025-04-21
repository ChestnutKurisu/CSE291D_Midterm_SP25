#!/usr/bin/env python3
"""
RK4 convergence study for the 12‑node tensegrity robot.

The script integrates the system with several time‑steps Δt, compares
each run to a much finer reference solution, and plots ‖error‖∞ vs Δt
on a log–log scale.

Usage
-----
    python convergence_study.py                # default T = 3 s
    python convergence_study.py --T 5 --save   # longer run, save PNG
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensegrity_robot import TensegrityRobot


# ----------------------------------------------------------------------
# helper to integrate for a given Δt and horizon T
# ----------------------------------------------------------------------
def simulate(dt: float, T: float) -> np.ndarray:
    """Return final 12×3 node‑position array after integrating for time T."""
    robot = TensegrityRobot()
    t = 0.0
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        robot.step(t, dt)
        t += dt
    return robot.pos.copy()


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=3.0,
                        help="time horizon in seconds (default 3)")
    parser.add_argument("--save", type=str, default="",
                        help="optional filename for the PNG figure")
    args = parser.parse_args()

    # time‑steps to test (s) – each is an integer multiple of dt_ref
    dt_list = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4])
    dt_ref = 0.5 * dt_list[-1]        # 6.25 × 10⁻⁵ s
    assert np.allclose((dt_list / dt_ref).round(), dt_list / dt_ref)

    print(f"[i] reference Δt = {dt_ref:.2e} s  ({int(args.T / dt_ref)} steps)")
    p_ref = simulate(dt_ref, args.T)

    errors: list[float] = []
    for dt in dt_list:
        p = simulate(dt, args.T)
        err = np.max(np.linalg.norm(p - p_ref, axis=1))   # ‖·‖∞ over nodes
        errors.append(err)
        print(f"   Δt = {dt:7.2e}  →  ‖error‖∞ = {err:.3e} m")

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dt_list, errors, "o-", lw=1.6, markerfacecolor="white")

    # reference slope = 4 for RK4
    slope_ref = errors[0] * (dt_list / dt_list[0]) ** 4
    ax.loglog(dt_list, slope_ref, "--", label="slope 4 ref.")

    ax.set_xlabel(r"$\Delta t$  (s)")
    ax.set_ylabel(r"$\max_i\;\|x_i(\Delta t)-x_{i,\mathrm{ref}}\|_\infty$  (m)")
    ax.set_title("RK4 convergence")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=300)
        print(f"[+] figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

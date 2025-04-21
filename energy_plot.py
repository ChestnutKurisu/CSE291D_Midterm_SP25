#!/usr/bin/env python3
"""
Plot total mechanical energy of the tensegrity robot vs. time.

Usage
-----
    python energy_plot.py           # default 10 s run
    python energy_plot.py 20        # 20 s run
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensegrity_robot import (
    TensegrityRobot,
    g, DT, K_GROUND, _plane_normal as PLANE_N,
)

def total_energy(robot: TensegrityRobot, t: float) -> float:
    """Kinetic + potential (+ penalty) energy at time *t*."""
    # kinetic
    KE = 0.5 * (robot.mass * (robot.vel ** 2).sum(axis=1)).sum()

    # gravity (y‑axis up)
    PE_g = (robot.mass * g * robot.pos[:, 1]).sum()

    # elastic energy in rods
    PE_rod = 0.0
    for r in robot.rods:
        d = robot.pos[r.j] - robot.pos[r.i]
        L  = np.linalg.norm(d)
        Lr = r.current_rest_length(t)           # works even if rods are static
        PE_rod += 0.5 * r.k * (L - Lr) ** 2

    # elastic energy in tension‑only cables
    PE_cab = 0.0
    for c in robot.cables:
        d = robot.pos[c.j] - robot.pos[c.i]
        L  = np.linalg.norm(d)
        Lr = c.current_rest_length(t)
        if L > Lr:
            PE_cab += 0.5 * c.k * (L - Lr) ** 2

    # ground‑plane penalty
    pen = robot.pos @ PLANE_N          # signed penetration (negative → inside)
    PE_pen = 0.5 * K_GROUND * (np.minimum(0.0, pen) ** 2).sum()

    return KE + PE_g + PE_rod + PE_cab + PE_pen


def main(T: float = 10.0):
    robot = TensegrityRobot()
    steps = int(T / DT)
    E = np.empty(steps)
    t = 0.0
    for k in range(steps):
        robot.step(t, DT)
        t += DT
        E[k] = total_energy(robot, t)

    plt.plot(np.linspace(0, T, steps), E)
    plt.xlabel("time (s)")
    plt.ylabel("total mechanical energy (J)")
    plt.title(f"Energy drift (RK4, Δt = {DT:.0e}s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    T_run = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    main(T_run)

#!/usr/bin/env python3
"""
Slide‑or‑stick test for the static/kinetic friction logic.

The script drops a 1 kg point mass on an inclined plane using the *exact*
same friction parameters you set in tensegrity_robot.py and plots its
penetration along the plane normal.  With a correct Coulomb switch the
curve should stay exactly at zero (no penetration) and the block should
remain stuck until the downhill component exceeds μ_s N.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from tensegrity_robot import (
    MU_STATIC, MU_KINETIC, DT, g,
    _compute_plane_normal
)

THETA_DEG = 10.0                       # inclination angle to test
N_STEPS   = int(5.0 / DT)              # five seconds
plane_n   = _compute_plane_normal(THETA_DEG)

# initial state
m   = 1.0
pos = np.zeros(3)
vel = np.zeros(3)
pen_history = np.empty(N_STEPS)

for k in range(N_STEPS):
    # gravity
    Fg = np.array([0.0, -m * g, 0.0])

    # split into normal / tangential
    Fn_mag = -(Fg @ plane_n)           # positive
    F_tan  = Fg + Fn_mag * plane_n

    v_n   = (vel @ plane_n) * plane_n
    v_tan = vel - v_n
    speed = np.linalg.norm(v_tan)

    if speed < 1e-6 and np.linalg.norm(F_tan) <= MU_STATIC * Fn_mag:
        F = np.zeros(3)                # static friction balances tangential load
    else:
        F = F_tan - MU_KINETIC * Fn_mag * v_tan / (speed + 1e-12)

    acc = F / m
    vel += acc * DT
    pos += vel * DT

    pen_history[k] = pos @ plane_n     # should stay ≥ 0

plt.plot(np.linspace(0, 5, N_STEPS), pen_history)
plt.xlabel("time (s)")
plt.ylabel("penetration along n (m)")
plt.title(f"Penetration on θ={THETA_DEG}° plane (should stay at 0)")
plt.tight_layout()
plt.show()

"""
Tensegrity Rolling Robot Simulation
-----------------------------------
This file implements a 12-node tensegrity structure (an icosahedron-like shape) using
a penalty-based collision approach, friction, damping, and an RK4 integrator.

Important Implementation Notes:
1. We use a "soft barrier" penalty for ground-plane contact (inclined or horizontal),
   instead of the direct KKT multiplier method. This means we add a large upward
   spring force if a node penetrates below the plane. High penalty stiffness can
   lead to numerical stiffness and necessitate smaller time steps.

2. The code randomly chooses 20% of cables to have time-varying rest lengths
   (simple sinusoidal or random-phase actuation). In our written report,
   we describe strut (rod) telescoping for rolling. The code is easily adaptable
   if one wishes to move the actuation to rods instead.

3. Friction is modeled via static vs. kinetic friction checks at each node.
   If the node's tangential speed is below a threshold, we attempt to apply
   static friction up to a maximum force. Otherwise, we apply kinetic friction
   with coefficient MU_KINETIC. This approach introduces non-smooth force transitions.

4. Since each node has the same mass, the global mass matrix is diagonal.
   The code lumps all ODE terms into a direct force computation, then divides
   by the mass. For large stiffness constants (K_ROD, K_CABLE, K_GROUND),
   the integrator might become marginally stable at DT=1e-3. Lower DT or an
   implicit method can improve stability and reduce energy drift.

5. GA-based shape optimization (mentioned in the PDF) is not included here.
   We only provide direct sinusoidal cable length changes for demonstration.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Attempt to import matplotlib for visualization. If unavailable, code can still run in headless mode.
try:
    _HAS_MPL = True
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm, animation
    import matplotlib.colors as mcolors
except ModuleNotFoundError:
    _HAS_MPL = False
    plt = Axes3D = animation = cm = None

Vec = np.ndarray


def icosahedron_vertices(scale: float = 1.0) -> List[Vec]:
    """
    Returns the vertices of an icosahedron, optionally scaled by a factor.

    This is a standard geometric construction for an icosahedron:
    It uses the golden ratio phi and arranges 12 vertices such that
    all the necessary symmetry properties for an icosahedron are preserved.

    :param scale: Scaling factor for the entire icosahedron.
    :return: List of 12 vertices as NumPy arrays (x, y, z).
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
    ]
    return [np.array((scale * x, scale * y, scale * z), dtype=float) for x, y, z in verts]


def icosahedron_edges() -> List[Tuple[int, int]]:
    """
    Returns a list of edges for an icosahedron.

    Each edge is represented as a tuple of (vertex_index_1, vertex_index_2).
    These indices match the ordering in `icosahedron_vertices()`.

    :return: List of edge pairs for the icosahedron.
    """
    return [
        (0, 1), (0, 4), (0, 5), (0, 8), (0, 9), (1, 6), (1, 7), (1, 8), (1, 9),
        (2, 3), (2, 4), (2, 5), (2, 10), (2, 11), (3, 6), (3, 7), (3, 10), (3, 11),
        (4, 5), (4, 8), (4, 10), (5, 9), (5, 11), (6, 7), (6, 8), (6, 10),
        (7, 9), (7, 11), (8, 9), (10, 11),
    ]


# Opposite pairs of vertices in the icosahedron, used for rods.
OPPOSITE_PAIRS = [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]

# Physical and simulation parameters
g: float = 9.81
DT: float = 1e-3
DISPLAY_INTERVAL = 33
MASS_NODE: float = 0.05
DAMP_STRUCT: float = 0.02
K_ROD: float = 1e5
K_CABLE: float = 1e4
K_GROUND: float = 1e5
MU_STATIC: float = 0.8
MU_KINETIC: float = 0.8
OMEGA_ACT: float = 2 * math.pi * 0.25
DELTA_L: float = 0.12
np.random.seed(369)
FPS: int = 60
DPI: int = 300
BITRATE: int = 12000
DISPLAY_INTERVAL = 1000 // FPS
COLOR_ROD = "#576B7E"
COLOR_CABLE_PASS = "#3498db"
COLOR_CABLE_ACTIVE = "#f39c12"
COLOR_NODE = "#f0f0f0"
PLANE_ALPHA = 0.98
PLANE_FACE_RGB = (0.16, 0.21, 0.27)
PLANE_SIZE: float = 3.5
PLANE_ANGLE_DEG: float = 2.0


def _compute_plane_normal(angle_deg: float) -> np.ndarray:
    """
    Computes and returns the unit normal vector of a plane
    inclined by angle_deg from the horizontal (y-axis).

    The plane is inclined around the x-axis, using a basic
    trigonometric relationship with tangent of the angle.
    """
    theta = math.radians(angle_deg)
    n = np.array([-math.tan(theta), 1.0, 0.0])
    return n / np.linalg.norm(n)


_plane_normal = _compute_plane_normal(PLANE_ANGLE_DEG)


def _plane_mesh():
    """
    Generates a mesh for visualization of the ground plane,
    taking into account inclination via `_plane_normal`.
    """
    size = PLANE_SIZE
    xx, zz = np.meshgrid(np.linspace(-size, size, 60), np.linspace(-size, size, 60))
    yy = -(xx * _plane_normal[0] + zz * _plane_normal[2]) / _plane_normal[1]
    return xx, yy, zz


def pair_rest_length(p: Vec, q: Vec) -> float:
    """
    Computes the initial rest length between two vertices p and q.

    :param p: First vertex position.
    :param q: Second vertex position.
    :return: Euclidean distance between the two points.
    """
    return float(np.linalg.norm(p - q))


@dataclass
class Cable:
    """
    Represents a cable in the tensegrity structure.

    Each cable has two endpoints i, j (which are node indices),
    an initial rest length L0, stiffness k, and a phase for any
    actuation if applicable.
    """
    i: int
    j: int
    L0: float
    k: float
    actuated: bool = False
    phase: float = 0.0

    def current_rest_length(self, t: float) -> float:
        """
        Computes the current rest length of this cable,
        taking into account actuation if applicable.

        :param t: Simulation time.
        :return: Adjusted rest length for the cable.
        """
        if not self.actuated:
            return self.L0
        return self.L0 * (1 - DELTA_L * 0.5 * (1 + math.sin(OMEGA_ACT * t + self.phase)))


@dataclass
class Rod:
    """
    Represents a rigid rod in the tensegrity structure.

    Rods connect opposite corners in the icosahedron structure
    and are modeled as high-stiffness elements with a fixed rest length.
    """
    i: int
    j: int
    L0: float
    k: float = K_ROD
    actuated: bool = False
    phase: float = 0.0

    def current_rest_length(self, t: float) -> float:
        if not self.actuated:
            return self.L0
        return self.L0 * (1 - DELTA_L * 0.5 * (1 + math.sin(OMEGA_ACT * t + self.phase)))


class TensegrityRobot:
    """
    Implements the tensegrity rolling robot as an icosahedron-based structure
    with rods and cables.

    The robot maintains arrays for node positions and velocities, updates forces
    and integrates positions over time. Includes methods for collisions, friction,
    and actuation of cables.
    """

    def __init__(
            self,
            *,
            scale: float = 0.05,
            drop_height: float = 0.5,
            left_offset: float = -3.0,
    ):
        """
        Initialize a tensegrity robot by setting up rods, cables, and
        initial positions in a scaled icosahedron arrangement.

        :param scale: Scale factor for the base icosahedron size.
        :param drop_height: Height above the ground plane for initial placement.
        :param left_offset: How far left to shift the entire structure along x-axis.
        """
        self.N = 12
        self.pos = np.stack(icosahedron_vertices(scale))
        self.vel = np.zeros_like(self.pos)
        self.mass = np.full(self.N, MASS_NODE)

        # Ensure robot isn't below the plane from the start
        penetration = self.pos @ _plane_normal
        min_pen = penetration.min()
        if min_pen < 0.05:
            self.pos += (0.05 - min_pen) * _plane_normal
        self.pos += drop_height * _plane_normal
        self.pos[:, 0] += left_offset

        # Initialize rods using opposite pairs of nodes
        self.rods = [Rod(i, j, pair_rest_length(self.pos[i], self.pos[j])) for i, j in OPPOSITE_PAIRS]

        # Initialize cables on all icosahedron edges, randomly actuating 20% of them
        edges = icosahedron_edges()
        act_idx = set(np.random.choice(len(edges), int(0.2 * len(edges)), replace=False))
        self.cables = [
            Cable(i, j, pair_rest_length(self.pos[i], self.pos[j]), K_CABLE, actuated=(k in act_idx),
                  phase=2 * math.pi * np.random.rand())
            for k, (i, j) in enumerate(edges)
        ]

    def _rod_and_cable_forces(self, t: float) -> np.ndarray:
        """
        Computes the forces contributed by rods and cables on each node.

        Rods always act to restore their rest length (like springs).
        Cables only exert force when stretched beyond their rest length.

        :param t: Current simulation time.
        :return: Array of shape (N, 3) with force vectors on each node.
        """
        F = np.zeros_like(self.pos)
        # Rod forces
        for r in self.rods:
            d = self.pos[r.j] - self.pos[r.i]
            dist = np.linalg.norm(d) or 1e-12
            fvec = r.k * (dist - r.L0) * d / dist
            F[r.i] += fvec
            F[r.j] -= fvec

        # Cable forces
        for c in self.cables:
            d = self.pos[c.j] - self.pos[c.i]
            dist = np.linalg.norm(d) or 1e-12
            Lr = c.current_rest_length(t)
            if dist > Lr:
                fvec = c.k * (dist - Lr) * d / dist
                F[c.i] += fvec
                F[c.j] -= fvec
        return F

    def _plane_contact(self, F_no_friction: np.ndarray) -> np.ndarray:
        """
        Computes additional contact and friction forces for nodes that contact the plane.

        :param F_no_friction: Array of forces without friction for each node.
        :return: Array of friction forces for each node (to be added to total).
        """
        dF = np.zeros_like(self.pos)
        pen = self.pos @ _plane_normal
        contact_mask = pen < 0
        if not np.any(contact_mask):
            return dF

        idx = np.where(contact_mask)[0]
        Fn_mag = -K_GROUND * pen[idx]
        # For each node in contact, compute normal and friction forces
        for node_local, i in enumerate(idx):
            dF[i] += _plane_normal * Fn_mag[node_local]
            v_rel = self.vel[i]
            vn = (v_rel @ _plane_normal) * _plane_normal
            v_tan = v_rel - vn
            speed = np.linalg.norm(v_tan)
            F_tan = F_no_friction[i] - (F_no_friction[i] @ _plane_normal) * _plane_normal
            F_tan_mag = np.linalg.norm(F_tan)

            # Static friction test
            if speed < 1e-6:
                if F_tan_mag <= MU_STATIC * Fn_mag[node_local]:
                    dF[i] -= F_tan
                else:
                    # Break static friction
                    if speed > 1e-12:
                        friction = -MU_KINETIC * Fn_mag[node_local] * (v_tan / speed)
                    else:
                        friction = -MU_KINETIC * Fn_mag[node_local] * (F_tan / (F_tan_mag + 1e-12))
                    dF[i] += friction
            else:
                # Kinetic friction
                friction = -MU_KINETIC * Fn_mag[node_local] * (v_tan / speed)
                dF[i] += friction

        return dF

    def _forces_no_friction(self, t: float) -> np.ndarray:
        """
        Computes forces on the robot, excluding friction from ground contact.

        Includes gravity, structural damping, rods, and cable tension.

        :param t: Current simulation time.
        :return: Array of shape (N, 3) representing net forces on each node (without friction).
        """
        F = np.zeros_like(self.pos)
        # Gravity
        F[:, 1] -= self.mass * g
        # Damping
        F -= DAMP_STRUCT * self.vel

        # Rod forces
        for r in self.rods:
            d = self.pos[r.j] - self.pos[r.i]
            dist = np.linalg.norm(d) or 1e-12
            L0 = r.L0
            fvec = r.k * (dist - L0) * (d / dist)
            F[r.i] += fvec
            F[r.j] -= fvec

        # Cable forces
        for c in self.cables:
            d = self.pos[c.j] - self.pos[c.i]
            dist = np.linalg.norm(d) or 1e-12
            Lc = c.current_rest_length(t)
            if dist > Lc:
                fvec = c.k * (dist - Lc) * (d / dist)
                F[c.i] += fvec
                F[c.j] -= fvec

        return F

    def _forces(self, t: float) -> np.ndarray:
        """
        Computes total forces on all nodes, including ground friction effects.

        :param t: Current simulation time.
        :return: Array of shape (N, 3) representing net forces on each node.
        """
        F0 = self._forces_no_friction(t)
        dF = self._plane_contact(F0)
        return F0 + dF

    def step(self, t: float, dt: float = DT):
        """
        Advances the simulation by one time step `dt` using a 4-stage integration (RK4).

        :param t: Current time prior to stepping.
        :param dt: Timestep size.
        """

        def accel(pos: np.ndarray, vel: np.ndarray, local_t: float):
            old_pos, old_vel = self.pos, self.vel
            self.pos, self.vel = pos, vel
            a = self._forces(local_t) / self.mass[:, None]
            self.pos, self.vel = old_pos, old_vel
            return a

        p0, v0 = self.pos.copy(), self.vel.copy()
        a0 = accel(p0, v0, t)
        p1 = p0 + 0.5 * dt * v0
        v1 = v0 + 0.5 * dt * a0
        a1 = accel(p1, v1, t + 0.5 * dt)
        p2 = p0 + 0.5 * dt * v1
        v2 = v0 + 0.5 * dt * a1
        a2 = accel(p2, v2, t + 0.5 * dt)
        p3 = p0 + dt * v2
        v3 = v0 + dt * a2
        a3 = accel(p3, v3, t + dt)
        self.pos += dt * (v0 + 2 * v1 + 2 * v2 + v3) / 6
        self.vel += dt * (a0 + 2 * a1 + 2 * a2 + a3) / 6


def create_animation(
        robot: TensegrityRobot,
        *,
        total_time: float = 10.0,
        save: Path | None = None,
        zoom: float = 1.0,
        track: bool = True,
):
    """
    Creates and displays or saves a 3D animation of the robot's motion using Matplotlib.

    :param robot: A TensegrityRobot instance.
    :param total_time: Total simulation time (in seconds).
    :param save: Optional path to save the MP4 animation.
    :param zoom: Zoom factor for the camera.
    :param track: Whether to track the center of mass in the view.
    """
    if not _HAS_MPL:
        raise RuntimeError("Matplotlib not available – run with --no‑visual or install it.")

    # Animation / figure setup
    FPS: int = 60
    DPI: int = 300
    BITRATE: int = 12000
    display_interval = 1000 // FPS

    if hasattr(animation.FuncAnimation, "_blit_draw"):
        # Potential workaround for Matplotlib bug
        animation.FuncAnimation._blit_draw = lambda self, arts: None

    fig = plt.figure(figsize=(16, 9), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_box_aspect((1, 1, 1), zoom=1.6)
    ax.set_anchor("C")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.view_init(elev=22, azim=3, roll=2)
    ax.set_box_aspect((16, 9, 9))

    bgc = "#f6f8ff"
    fig.patch.set_facecolor(bgc)
    ax.set_facecolor(bgc)

    time_text = fig.text(0.02, 0.95, "", color="black")

    xx, yy, zz = _plane_mesh()
    base = cm.get_cmap("ocean")
    plane_cmap = mcolors.ListedColormap(base(np.linspace(0.15, 0.6, 256)), name="ocean‑trunc")
    surf = ax.plot_surface(xx, zz, yy, cmap=plane_cmap, alpha=PLANE_ALPHA, shade=True, edgecolor="none", linewidth=0,
                           zorder=0)

    # Create lines for rods and cables
    rod_lines = [ax.plot([], [], [], lw=3, color=COLOR_ROD, zorder=3)[0] for _ in robot.rods]
    cable_lines = []
    for cab in robot.cables:
        col = COLOR_CABLE_ACTIVE if cab.actuated else COLOR_CABLE_PASS
        lw = 2 if cab.actuated else 1
        cable_lines.append(ax.plot([], [], [], lw=lw, color=col, zorder=4)[0])

    # Scatter for nodes and footprints
    nodes = ax.scatter([], [], [], s=15, color=COLOR_NODE, depthshade=False, zorder=5)
    footprints_x, footprints_y, footprints_z = [], [], []
    footprints_scatter = ax.scatter([], [], [], s=10, color="#ff0000", alpha=0.6, marker="o", zorder=1)

    base_span = PLANE_SIZE / max(zoom, 1e-6)
    ax.set(xlim=(-base_span, base_span), ylim=(-base_span, base_span), zlim=(-base_span * 0.2, base_span))

    n_frames = int(total_time * FPS)
    steps_per_frame = max(1, int(round((1.0 / FPS) / DT)))
    t_global = 0.0

    def _init():
        P = robot.pos
        # Update rods
        for line, rod in zip(rod_lines, robot.rods):
            i, j = rod.i, rod.j
            line.set_data(P[[i, j], 0], P[[i, j], 2])
            line.set_3d_properties(P[[i, j], 1])
        # Update cables
        for line, cab in zip(cable_lines, robot.cables):
            i, j = cab.i, cab.j
            line.set_data(P[[i, j], 0], P[[i, j], 2])
            line.set_3d_properties(P[[i, j], 1])
        nodes._offsets3d = (P[:, 0], P[:, 2], P[:, 1])
        time_text.set_text("t = 0.00 s")
        return rod_lines + cable_lines + [nodes, footprints_scatter, time_text]

    def _update(_):
        nonlocal t_global
        for __ in range(steps_per_frame):
            robot.step(t_global, DT)
            t_global += DT
        P = robot.pos

        # Update rods
        for line, rod in zip(rod_lines, robot.rods):
            i, j = rod.i, rod.j
            line.set_data(P[[i, j], 0], P[[i, j], 2])
            line.set_3d_properties(P[[i, j], 1])

        # Update cables
        for line, cab in zip(cable_lines, robot.cables):
            i, j = cab.i, cab.j
            line.set_data(P[[i, j], 0], P[[i, j], 2])
            line.set_3d_properties(P[[i, j], 1])

        # Update node positions
        nodes._offsets3d = (P[:, 0], P[:, 2], P[:, 1])

        # Track where nodes contact plane (footprints)
        contact_mask = (P @ _plane_normal) < 0
        if np.any(contact_mask):
            footprints_x.extend(P[contact_mask, 0])
            footprints_z.extend(P[contact_mask, 2])
            footprints_y.extend(P[contact_mask, 1])
            footprints_scatter._offsets3d = (footprints_x, footprints_z, footprints_y)

        # Optionally track center of mass for camera
        if track:
            com = P.mean(axis=0)
            span_x = np.max(np.abs(P[:, 0] - com[0]))
            span_z = np.max(np.abs(P[:, 2] - com[2]))
            span = max(base_span, 1.2 * max(span_x, span_z))
            ax.set_xlim(com[0] - span, com[0] + span)
            ax.set_ylim(com[2] - span, com[2] + span)

        time_text.set_text(f"t = {t_global:0.2f} s")
        return rod_lines + cable_lines + [nodes, footprints_scatter, time_text]

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=n_frames,
        interval=display_interval,
        blit=False,
    )

    # Save or show the animation
    if save:
        save = Path(save).with_suffix(".mp4")
        anim.save(
            save,
            dpi=DPI,
            fps=FPS,
            codec="libx264",
            bitrate=BITRATE,
            extra_args=["-pix_fmt", "yuv420p"],
        )
        print(f"[+] Saved high‑quality {total_time:.0f} s video to {save}")
    else:
        plt.show()


def run_headless(total_time: float):
    """
    Runs the simulation without visualization for a given total_time.
    Reports final center of mass for verification.

    :param total_time: Simulation duration in seconds.
    """
    robot = TensegrityRobot()
    for _ in range(int(total_time / DT)):
        robot.step(_, DT)
    com = robot.pos.mean(axis=0)
    print(f"[i] Simulation complete. COM after {total_time:.1f}s: {com}")


def main(argv: list[str] | None = None):
    """
    Main entry point for command-line usage of the simulation script.

    Parses command-line arguments, configures the simulation,
    and either runs headless or with visualization if matplotlib is available.
    """
    global PLANE_SIZE, PLANE_ANGLE_DEG, _plane_normal
    parser = argparse.ArgumentParser(description="Tensegrity rolling robot simulation")
    parser.add_argument("--save", type=str, help="Output MP4 filename")
    parser.add_argument("--time", type=float, default=20.0, help="Simulation duration (s)")
    parser.add_argument("--scale", type=float, default=0.2, help="Robot scale (default 0.05)")
    parser.add_argument("--drop", type=float, default=1.4, help="Drop height (m, default 0.5)")
    parser.add_argument("--left", type=float, default=-3.0, help="Metres left of centre to start (default -3)")
    parser.add_argument("--plane-size", type=float, default=PLANE_SIZE, help="Half‑edge length of drawn plane (m)")
    parser.add_argument("--angle", type=float, default=-14, help="Incline angle in degrees (default 2)")
    parser.add_argument("--zoom", type=float, default=2.5, help="Base zoom (>1 zooms‑in, default 12)")
    parser.add_argument("--no-track", action="store_true", help="Disable camera tracking / auto‑zoom")
    parser.add_argument("--no-visual", action="store_true", help="Force headless run even if Matplotlib exists")
    args = parser.parse_args(argv)

    PLANE_SIZE = args.plane_size
    PLANE_ANGLE_DEG = args.angle
    _plane_normal = _compute_plane_normal(PLANE_ANGLE_DEG)

    robot_kwargs: dict[str, float] = {
        "scale": args.scale,
        "drop_height": args.drop,
        "left_offset": args.left,
    }

    # Run with or without visualization based on Matplotlib availability and flags
    if _HAS_MPL and not args.no_visual:
        robot = TensegrityRobot(**robot_kwargs)
        save_path = Path(args.save).expanduser() if args.save else None
        create_animation(robot, total_time=args.time, save=save_path, zoom=args.zoom, track=not args.no_track)
    else:
        if not _HAS_MPL:
            print("[!] Matplotlib not found – running headless.")
        run_headless(args.time)


if __name__ == "__main__":
    main()

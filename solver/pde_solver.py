"""Explicit finite-difference solvers for the two canonical 1D linear PDEs.

* Heat / diffusion:  u_t = alpha * u_xx          (parabolic)
* Wave:              u_tt = c^2 * u_xx           (hyperbolic)

Both use an explicit scheme on a uniform space-time grid with homogeneous
Dirichlet boundaries (u = 0 at both ends) and a configurable initial profile.
Time steps are chosen automatically to satisfy the stability condition, so the
user only specifies the spatial domain, a coefficient, and a horizon.

The result carries the full ``u[t_index, x_index]`` grid, which the surface
view renders directly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from models.solution import ParsedEquation, Solution


class SolverError(RuntimeError):
    pass


def _initial_profile(name: str, x: np.ndarray) -> np.ndarray:
    """Initial spatial profile u(x, 0), normalised to the [0, L] domain."""
    L = x[-1] - x[0]
    xn = (x - x[0]) / L                       # 0..1
    if name == "gaussian":
        return np.exp(-((xn - 0.5) ** 2) / (2 * 0.05))
    if name == "square":
        u = np.zeros_like(x)
        u[(xn > 0.35) & (xn < 0.65)] = 1.0
        return u
    # default: first sine mode (clean, satisfies zero Dirichlet BC)
    return np.sin(np.pi * xn)


class PDESolver:
    """Finite-difference solver for heat and wave equations."""

    NX = 120          # spatial grid points
    NT_OUTPUT = 120   # time slices stored for the surface

    def solve(
        self,
        parsed: ParsedEquation,
        *,
        domain: tuple[float, float],
        coefficient: float = 1.0,
        t_max: float = 1.0,
        profile: str = "sine",
    ) -> Solution:
        if parsed.kind != "PDE":
            raise SolverError("PDESolver received a non-PDE equation.")
        if parsed.pde_subtype == "wave":
            return self._solve_wave(parsed, domain, coefficient, t_max, profile)
        return self._solve_heat(parsed, domain, coefficient, t_max, profile)

    # ------------------------------------------------------------------
    def _grid(self, domain, t_max, dt, store_every, x, steps):
        pass  # (helper kept for clarity; sampling handled inline)

    # ------------------------------------------------------------------
    def _solve_heat(self, parsed, domain, alpha, t_max, profile) -> Solution:
        x0, x1 = domain
        if not (x1 > x0):
            raise SolverError("Domain end must be greater than domain start.")
        alpha = abs(alpha) or 1.0

        x = np.linspace(x0, x1, self.NX)
        dx = x[1] - x[0]
        # Stability: alpha * dt / dx^2 <= 0.5. Use 0.4 for margin.
        dt = 0.4 * dx * dx / alpha
        n_steps = max(int(np.ceil(t_max / dt)), 1)
        dt = t_max / n_steps
        r = alpha * dt / (dx * dx)

        u = _initial_profile(profile, x)
        u[0] = u[-1] = 0.0

        store_every = max(n_steps // self.NT_OUTPUT, 1)
        times = [0.0]
        frames = [u.copy()]

        for step in range(1, n_steps + 1):
            lap = np.zeros_like(u)
            lap[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
            u = u + r * lap
            u[0] = u[-1] = 0.0
            if step % store_every == 0 or step == n_steps:
                times.append(step * dt)
                frames.append(u.copy())

        t = np.array(times)
        grid = np.vstack(frames)

        diagnostics = {
            "scheme": "explicit FTCS (forward-time, centred-space)",
            "alpha": alpha,
            "nx": self.NX,
            "n_time_steps": n_steps,
            "stability_r": round(r, 4),
            "dt": dt,
            "dx": dx,
        }
        return Solution(
            kind="PDE",
            x=x,
            t=t,
            grid=grid,
            method="Finite Difference (explicit)",
            labels=[parsed.dependent_var],
            message=(
                f"Solved heat equation (alpha={alpha:g}) on {self.NX} points, "
                f"{n_steps} time steps. Stability r={r:.3f}."
            ),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    def _solve_wave(self, parsed, domain, c, t_max, profile) -> Solution:
        x0, x1 = domain
        if not (x1 > x0):
            raise SolverError("Domain end must be greater than domain start.")
        c = abs(c) or 1.0

        x = np.linspace(x0, x1, self.NX)
        dx = x[1] - x[0]
        # CFL: c * dt / dx <= 1. Use 0.9 for margin.
        dt = 0.9 * dx / c
        n_steps = max(int(np.ceil(t_max / dt)), 1)
        dt = t_max / n_steps
        lam2 = (c * dt / dx) ** 2

        u_prev = _initial_profile(profile, x)
        u_prev[0] = u_prev[-1] = 0.0

        # First step using zero initial velocity: u1 = u0 + 0.5*lam2*lap(u0)
        lap = np.zeros_like(u_prev)
        lap[1:-1] = u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]
        u_curr = u_prev + 0.5 * lam2 * lap
        u_curr[0] = u_curr[-1] = 0.0

        store_every = max(n_steps // self.NT_OUTPUT, 1)
        times = [0.0]
        frames = [u_prev.copy()]

        for step in range(1, n_steps + 1):
            lap = np.zeros_like(u_curr)
            lap[1:-1] = u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]
            u_next = 2 * u_curr - u_prev + lam2 * lap
            u_next[0] = u_next[-1] = 0.0
            u_prev, u_curr = u_curr, u_next
            if step % store_every == 0 or step == n_steps:
                times.append(step * dt)
                frames.append(u_curr.copy())

        t = np.array(times)
        grid = np.vstack(frames)

        diagnostics = {
            "scheme": "explicit leapfrog (centred-time, centred-space)",
            "wave_speed_c": c,
            "nx": self.NX,
            "n_time_steps": n_steps,
            "cfl_lambda": round(float(np.sqrt(lam2)), 4),
            "dt": dt,
            "dx": dx,
        }
        return Solution(
            kind="PDE",
            x=x,
            t=t,
            grid=grid,
            method="Finite Difference (explicit)",
            labels=[parsed.dependent_var],
            message=(
                f"Solved wave equation (c={c:g}) on {self.NX} points, "
                f"{n_steps} time steps. CFL={np.sqrt(lam2):.3f}."
            ),
            diagnostics=diagnostics,
        )
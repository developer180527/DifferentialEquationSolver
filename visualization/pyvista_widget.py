"""Interactive 3D viewer built on PyVista (VTK).

Replaces matplotlib's painter's-algorithm 3D with true GPU-accelerated rendering:
smooth rotation, lighting, and large meshes. Handles three solution kinds:

* SYSTEM with 3 states  -> state-space trajectory (e.g. the Lorenz attractor),
                           drawn as a tube coloured by time.
* SYSTEM with 2 states  -> phase-plane trajectory.
* PDE                   -> the u(x, t) surface as a structured grid.
* 2nd-order ODE         -> phase portrait (y vs y').

If PyVista/pyvistaqt are unavailable the application falls back to the
matplotlib SurfaceWidget; main.py decides which to instantiate.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from models.solution import Solution

_CMAP_TRAJECTORY = "plasma"
_CMAP_SURFACE = "viridis"


class PyVistaWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)

        self._placeholder("Solve a system or PDE to see the 3D view")

    # ------------------------------------------------------------------
    def _placeholder(self, text: str):
        self.plotter.clear()
        self.plotter.add_text(text, position="upper_edge", font_size=11, color="gray")
        self.plotter.reset_camera()

    @staticmethod
    def _tube_radius(points: np.ndarray) -> float:
        span = np.ptp(points, axis=0)
        diag = float(np.linalg.norm(span)) or 1.0
        return diag * 0.004

    # ------------------------------------------------------------------
    def plot_solution(self, solution: Solution):
        self.plotter.clear()

        if solution.kind == "SYSTEM":
            n = solution.y.shape[0]
            if n >= 3:
                self._trajectory_3d(solution)
            elif n == 2:
                self._trajectory_2d(solution)
            else:
                self._placeholder("A single-variable system has no phase space.")
                return
        elif solution.kind == "PDE" and solution.grid is not None:
            self._surface(solution)
        elif solution.y.ndim > 1 and solution.y.shape[0] >= 2:
            self._phase_portrait(solution)
        else:
            self._placeholder(
                "No 3D view for a first-order ODE.\n"
                "Try a system (e.g. Lorenz), a PDE, or a 2nd-order ODE."
            )
            return

        self.plotter.reset_camera()
        self.plotter.render()

    # ------------------------------------------------------------------
    def _trajectory_3d(self, solution: Solution):
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        points = np.column_stack([x, y, z])
        poly = pv.lines_from_points(points)
        poly["time"] = solution.x
        tube = poly.tube(radius=self._tube_radius(points))
        self.plotter.add_mesh(
            tube, scalars="time", cmap=_CMAP_TRAJECTORY,
            smooth_shading=True, scalar_bar_args={"title": "t"},
        )
        self._add_endpoints(points)
        labels = solution.labels + ["?"] * 3
        self.plotter.show_grid(
            xtitle=labels[0], ytitle=labels[1], ztitle=labels[2], color="gray"
        )

    # ------------------------------------------------------------------
    def _trajectory_2d(self, solution: Solution):
        x, y = solution.y[0], solution.y[1]
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])
        poly = pv.lines_from_points(points)
        poly["time"] = solution.x
        self.plotter.add_mesh(
            poly.tube(radius=self._tube_radius(points)),
            scalars="time", cmap=_CMAP_TRAJECTORY,
            scalar_bar_args={"title": "t"},
        )
        self._add_endpoints(points)
        labels = solution.labels + ["?"] * 2
        self.plotter.show_grid(xtitle=labels[0], ytitle=labels[1], color="gray")
        self.plotter.view_xy()

    # ------------------------------------------------------------------
    def _phase_portrait(self, solution: Solution):
        y, dy = solution.y[0], solution.y[1]
        z = np.zeros_like(y)
        points = np.column_stack([y, dy, z])
        poly = pv.lines_from_points(points)
        poly["time"] = solution.x
        self.plotter.add_mesh(
            poly.tube(radius=self._tube_radius(points)),
            scalars="time", cmap=_CMAP_TRAJECTORY,
            scalar_bar_args={"title": "x"},
        )
        self._add_endpoints(points)
        self.plotter.show_grid(xtitle="y", ytitle="y'", color="gray")
        self.plotter.view_xy()

    # ------------------------------------------------------------------
    def _surface(self, solution: Solution):
        x, t, u = solution.x, solution.t, solution.grid
        X, T = np.meshgrid(x, t)
        # Scale height so the surface isn't a flat sheet or a spike.
        span_xy = max(np.ptp(x), np.ptp(t)) or 1.0
        span_z = np.ptp(u) or 1.0
        z = u * (0.4 * span_xy / span_z)
        grid = pv.StructuredGrid(X, T, z)
        grid["u"] = u.ravel(order="C")
        self.plotter.add_mesh(
            grid, scalars="u", cmap=_CMAP_SURFACE,
            smooth_shading=True, scalar_bar_args={"title": "u"},
        )
        self.plotter.show_grid(
            xtitle="x", ytitle="t",
            ztitle=solution.labels[0] if solution.labels else "u",
            color="gray",
        )

    # ------------------------------------------------------------------
    def _add_endpoints(self, points: np.ndarray):
        r = self._tube_radius(points) * 4
        self.plotter.add_mesh(pv.Sphere(radius=r, center=points[0]), color="green")
        self.plotter.add_mesh(pv.Sphere(radius=r, center=points[-1]), color="red")

    # ------------------------------------------------------------------
    def export_image(self, path: str):
        self.plotter.screenshot(path)

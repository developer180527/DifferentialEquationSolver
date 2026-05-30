"""3D surface view.

For a PDE it renders the full ``u(x, t)`` surface. For a higher-order ODE it
falls back to a phase portrait (y vs y'), which is the natural 3D-ish view of an
ODE's state space. For a plain first-order ODE there's nothing 3D to show, so it
displays an informative placeholder.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from models.solution import Solution


class SurfaceWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._placeholder("3D surface appears here for PDE solutions")

    # ------------------------------------------------------------------
    def _placeholder(self, text: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    # ------------------------------------------------------------------
    def plot_solution(self, solution: Solution):
        self.figure.clear()

        if solution.kind == "PDE" and solution.grid is not None:
            self._plot_surface(solution)
        elif solution.y.ndim > 1 and solution.y.shape[0] >= 2:
            self._plot_phase(solution)
        else:
            self._placeholder(
                "No 3D view for a first-order ODE.\n"
                "Solve a PDE (surface) or a 2nd-order ODE (phase portrait)."
            )
            return
        self.canvas.draw()

    # ------------------------------------------------------------------
    def _plot_surface(self, solution: Solution):
        ax = self.figure.add_subplot(111, projection="3d")
        X, T = np.meshgrid(solution.x, solution.t)
        surf = ax.plot_surface(
            X, T, solution.grid, cmap="viridis",
            linewidth=0, antialiased=True, rcount=80, ccount=80,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel(f"{solution.labels[0] if solution.labels else 'u'}")
        ax.view_init(elev=30, azim=-120)
        self.figure.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

    # ------------------------------------------------------------------
    def _plot_phase(self, solution: Solution):
        ax = self.figure.add_subplot(111)
        y, dy = solution.y[0], solution.y[1]
        ax.plot(y, dy, linewidth=1.5)
        ax.scatter([y[0]], [dy[0]], color="green", zorder=5, label="start")
        ax.scatter([y[-1]], [dy[-1]], color="red", zorder=5, label="end")
        ax.set_xlabel("y")
        ax.set_ylabel("y′")
        ax.set_title("Phase portrait")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    # ------------------------------------------------------------------
    def export_image(self, path: str):
        self.figure.savefig(path, dpi=150, bbox_inches="tight")

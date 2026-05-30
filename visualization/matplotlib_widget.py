"""2D line-plot view. A canvas plus the standard matplotlib navigation toolbar
(pan / zoom / save) wrapped in a single Qt widget."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from models.solution import Solution


class MatplotlibWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._show_placeholder()

    # ------------------------------------------------------------------
    def _show_placeholder(self):
        self.axes.clear()
        self.axes.text(
            0.5, 0.5, "Solve an equation to see the plot",
            ha="center", va="center", transform=self.axes.transAxes,
            color="gray", fontsize=12,
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw()

    # ------------------------------------------------------------------
    def plot_solution(self, solution: Solution):
        """Render an ODE solution (and its derivative components, if present)."""
        self.axes.clear()

        if solution.kind == "PDE":
            # Show a few time slices as a 2D family of curves.
            self._plot_pde_slices(solution)
            self.canvas.draw()
            return

        x = solution.x
        y = solution.y
        labels = solution.labels or ["y"]

        if y.ndim == 1:
            self.axes.plot(x, y, label=labels[0], linewidth=2)
        else:
            for i in range(y.shape[0]):
                lbl = labels[i] if i < len(labels) else f"component {i}"
                self.axes.plot(x, y[i], label=lbl, linewidth=2 if i == 0 else 1.2,
                               alpha=1.0 if i == 0 else 0.6)

        indep = "x"
        self.axes.set_xlabel(indep)
        self.axes.set_ylabel("value")
        self.axes.grid(True, alpha=0.3)
        self.axes.legend(loc="best", fontsize=9)
        self.canvas.draw()

    # ------------------------------------------------------------------
    def _plot_pde_slices(self, solution: Solution):
        x, t, grid = solution.x, solution.t, solution.grid
        n = grid.shape[0]
        idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
        for i in idxs:
            self.axes.plot(x, grid[i], label=f"t = {t[i]:.3g}", linewidth=1.6)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel(f"{solution.labels[0] if solution.labels else 'u'}(x, t)")
        self.axes.grid(True, alpha=0.3)
        self.axes.legend(loc="best", fontsize=9)

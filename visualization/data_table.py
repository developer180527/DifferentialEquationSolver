"""Tabular numerical view of a solution, with CSV export."""

from __future__ import annotations

import csv

import numpy as np
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from models.solution import Solution


class DataTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        self._headers: list[str] = []
        self._rows: list[list[float]] = []
        self.setAlternatingRowColors(True)

    # ------------------------------------------------------------------
    def show_solution(self, solution: Solution, max_rows: int = 500):
        if solution.kind == "PDE" and solution.grid is not None:
            self._populate_pde(solution, max_rows)
        else:
            self._populate_ode(solution, max_rows)

    # ------------------------------------------------------------------
    def _populate_ode(self, solution: Solution, max_rows: int):
        x = solution.x
        y = solution.y
        labels = solution.labels or ["y"]

        headers = [solution.independent or "x"] + labels
        components = y if y.ndim > 1 else y[np.newaxis, :]

        step = max(len(x) // max_rows, 1)
        idxs = range(0, len(x), step)

        rows = []
        for i in idxs:
            row = [float(x[i])] + [float(components[c, i]) for c in range(components.shape[0])]
            rows.append(row)

        self._fill(headers, rows)

    # ------------------------------------------------------------------
    def _populate_pde(self, solution: Solution, max_rows: int):
        # Show u(x) at a handful of evenly spaced times as columns.
        x, t, grid = solution.x, solution.t, solution.grid
        n_t = grid.shape[0]
        t_idxs = np.linspace(0, n_t - 1, min(6, n_t)).astype(int)

        headers = ["x"] + [f"t={t[ti]:.3g}" for ti in t_idxs]

        step = max(len(x) // max_rows, 1)
        rows = []
        for xi in range(0, len(x), step):
            row = [float(x[xi])] + [float(grid[ti, xi]) for ti in t_idxs]
            rows.append(row)

        self._fill(headers, rows)

    # ------------------------------------------------------------------
    def _fill(self, headers: list[str], rows: list[list[float]]):
        self._headers = headers
        self._rows = rows
        self.clear()
        self.setColumnCount(len(headers))
        self.setRowCount(len(rows))
        self.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(f"{value:.6g}")
                self.setItem(r, c, item)
        self.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def export_csv(self, path: str):
        if not self._headers:
            raise ValueError("No data to export. Solve an equation first.")
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self._headers)
            writer.writerows(self._rows)

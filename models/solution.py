"""Shared data models for parsed equations and computed solutions.

These are deliberately plain dataclasses so they can be serialised, passed
between the parser, the solvers and the visualisation widgets without any
of those layers needing to know about each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ParsedEquation:
    """Intermediate representation produced by the parser.

    The parser's job is to turn a free-form string like ``dy/dx = y - x^2``
    into this structured form. Solvers consume it; they never see raw text.
    """

    raw: str                                   # original user input
    kind: str                                  # "ODE" or "PDE"
    order: int                                 # highest derivative order
    dependent_var: str                         # e.g. "y" or "u"
    independent_vars: list[str]                # e.g. ["x"] or ["x", "t"]
    rhs_expr: object                           # sympy expr: highest deriv = rhs_expr
    is_linear: bool
    classification: str                        # human-readable summary

    # PDE-only metadata (None for ODEs)
    pde_subtype: Optional[str] = None          # "heat" | "wave"
    time_var: Optional[str] = None             # which indep var is time
    space_var: Optional[str] = None            # which indep var is space


@dataclass
class Solution:
    """Result of a solve. Covers ODEs (single equation or system) and PDEs."""

    kind: str                                  # "ODE" or "PDE"
    x: np.ndarray                              # ODE: independent samples / PDE: space grid
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: list[str] = field(default_factory=list)
    method: str = ""
    success: bool = True
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    # PDE-only
    t: Optional[np.ndarray] = None             # time grid
    grid: Optional[np.ndarray] = None          # u[t_index, x_index]

    @property
    def primary(self) -> np.ndarray:
        """The main curve to plot for an ODE (the solution itself, not its derivatives)."""
        if self.y.ndim == 1:
            return self.y
        return self.y[0]

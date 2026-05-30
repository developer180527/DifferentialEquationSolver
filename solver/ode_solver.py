"""Numerical ODE solver built on SymPy ``lambdify`` + SciPy ``solve_ivp``.

An order-*n* ODE ``y^(n) = f(x, y, y', ..., y^(n-1))`` is reduced to a
first-order system

    Y = [y, y', ..., y^(n-1)]
    Y' = [Y[1], Y[2], ..., f(x, *Y)]

and integrated with an adaptive Runge-Kutta / implicit method. The parsed RHS
is compiled once into a fast NumPy callable; user parameters (a, b, c, ...) are
substituted symbolically before compilation.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from models.solution import ParsedEquation, Solution

# UI label -> SciPy method. "Auto" is resolved at solve time.
_METHOD_MAP = {
    "RK45": "RK45",
    "BDF": "BDF",
    "Radau": "Radau",
    "Auto": "RK45",
}


class SolverError(RuntimeError):
    pass


class ODESolver:
    """Integrates a parsed ODE over a domain given initial conditions."""

    N_OUTPUT = 600  # number of dense output samples

    def solve(
        self,
        parsed: ParsedEquation,
        *,
        domain: tuple[float, float],
        initial_conditions: list[float],
        params: dict[str, float] | None = None,
        method: str = "Auto",
    ) -> Solution:
        if parsed.kind != "ODE":
            raise SolverError("ODESolver received a non-ODE equation.")

        params = params or {}
        order = parsed.order
        dep = parsed.dependent_var
        indep = sp.Symbol(parsed.independent_vars[0], real=True)

        # State symbols: y, y', ..., y^(n-1)
        state_syms = [sp.Symbol(dep, real=True)] + [
            sp.Symbol(f"{dep}__d{k}", real=True) for k in range(1, order)
        ]

        # Substitute user-supplied parameter values into the RHS.
        rhs = parsed.rhs_expr
        subs = {}
        for name, value in params.items():
            subs[sp.Symbol(name, real=True)] = value
        rhs = rhs.subs(subs)

        # Any symbol left that isn't the independent var or a state var is an
        # unspecified parameter -> fail clearly rather than silently zeroing it.
        allowed = {indep, *state_syms}
        leftover = [s for s in rhs.free_symbols if s not in allowed]
        if leftover:
            names = ", ".join(sorted(str(s) for s in leftover))
            raise SolverError(
                f"Unspecified parameter(s): {names}. "
                f"Provide values in the Parameters panel."
            )

        f = sp.lambdify((indep, *state_syms), rhs, "numpy")

        def system(t, Y):
            derivs = list(Y[1:])              # y' ... y^(n-1)
            derivs.append(float(f(t, *Y)))    # y^(n)
            return derivs

        # Initial conditions: need exactly `order` values.
        y0 = list(initial_conditions)
        if len(y0) < order:
            y0 = y0 + [0.0] * (order - len(y0))
            ic_note = (
                f"Only {len(initial_conditions)} initial condition(s) given for an "
                f"order-{order} ODE; padded missing ones with 0."
            )
        else:
            y0 = y0[:order]
            ic_note = ""

        t0, t1 = domain
        if not (t1 > t0):
            raise SolverError("Domain end must be greater than domain start.")

        scipy_method = _METHOD_MAP.get(method, "RK45")
        # Auto: prefer implicit BDF for stiff-looking linear high-order systems.
        if method == "Auto" and order >= 2 and parsed.is_linear:
            scipy_method = "RK45"  # RK45 is robust for the common oscillatory cases

        result = solve_ivp(
            system,
            (t0, t1),
            y0,
            method=scipy_method,
            dense_output=True,
            rtol=1e-7,
            atol=1e-9,
            max_step=(t1 - t0) / 50.0,
        )

        if not result.success:
            raise SolverError(f"Integration failed: {result.message}")

        xs = np.linspace(t0, t1, self.N_OUTPUT)
        ys = result.sol(xs)                   # shape (order, N)

        labels = [f"{dep}"]
        labels += [f"{dep}{chr(0x2032) * k}" for k in range(1, order)]  # y, y', y''

        diagnostics = {
            "method": scipy_method,
            "function_evaluations": int(result.nfev),
            "internal_steps": len(result.t),
            "status": result.message,
            "order": order,
            "linear": parsed.is_linear,
            "initial_conditions": y0,
        }
        if ic_note:
            diagnostics["note"] = ic_note

        message = (
            f"Solved {parsed.classification} with {scipy_method}. "
            f"{result.nfev} function evaluations."
        )

        return Solution(
            kind="ODE",
            x=xs,
            y=ys,
            labels=labels,
            method=scipy_method,
            success=True,
            message=message,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    def solve_demo(self) -> Solution:
        """A self-contained demo (dy/dx = y, y(0)=1 -> e^x). Used by the New/demo path."""
        from parser.equation_parser import parse_equation

        parsed = parse_equation("dy/dx = y")
        return self.solve(
            parsed,
            domain=(0.0, 3.0),
            initial_conditions=[1.0],
            method="RK45",
        )
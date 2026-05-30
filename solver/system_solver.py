"""Solver for systems of coupled first-order ODEs.

Given ``dx_i/dt = f_i(t, x_1, ..., x_n)`` for i = 1..n, each RHS is lambdified
once and the whole vector field is integrated with SciPy ``solve_ivp``. The
result's ``y`` is an (n_states, N) array; for a 3-variable system that array is
exactly the (x, y, z) trajectory the 3D viewer draws as the attractor.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from models.solution import ParsedSystem, Solution

_METHOD_MAP = {"RK45": "RK45", "BDF": "BDF", "Radau": "Radau", "Auto": "RK45"}


class SolverError(RuntimeError):
    pass


class SystemSolver:
    N_OUTPUT = 4000  # dense samples - chaotic trajectories need many points

    def solve(
        self,
        parsed: ParsedSystem,
        *,
        domain: tuple[float, float],
        initial_conditions: list[float],
        params: dict[str, float] | None = None,
        method: str = "Auto",
    ) -> Solution:
        params = params or {}
        indep = sp.Symbol(parsed.independent_var, real=True)
        state_syms = [sp.Symbol(v, real=True) for v in parsed.state_vars]
        n = len(state_syms)

        subs = {}
        all_syms = {s for e in parsed.rhs_exprs for s in e.free_symbols}
        for s in all_syms:
            if s.name in params:
                subs[s] = params[s.name]
        exprs = [e.subs(subs) for e in parsed.rhs_exprs]

        allowed_names = {indep.name} | {sv.name for sv in state_syms}
        leftover = sorted(
            {s.name for e in exprs for s in e.free_symbols if s.name not in allowed_names}
        )
        if leftover:
            raise SolverError(
                f"Unspecified parameter(s): {', '.join(leftover)}. "
                f"Provide values in the Parameters panel."
            )

        funcs = [sp.lambdify((indep, *state_syms), e, "numpy") for e in exprs]

        def field(t, Y):
            return [float(f(t, *Y)) for f in funcs]

        ic = list(initial_conditions)
        if len(ic) < n:
            ic = ic + [1.0] * (n - len(ic))
            ic_note = f"Padded missing initial conditions; using {ic}."
        else:
            ic = ic[:n]
            ic_note = ""

        t0, t1 = domain
        if not (t1 > t0):
            raise SolverError("Domain end must be greater than domain start.")

        scipy_method = _METHOD_MAP.get(method, "RK45")
        result = solve_ivp(
            field, (t0, t1), ic,
            method=scipy_method, dense_output=True,
            rtol=1e-8, atol=1e-10,
        )
        if not result.success:
            raise SolverError(f"Integration failed: {result.message}")

        ts = np.linspace(t0, t1, self.N_OUTPUT)
        ys = result.sol(ts)  # shape (n, N)

        diagnostics = {
            "method": scipy_method,
            "n_states": n,
            "state_vars": parsed.state_vars,
            "function_evaluations": int(result.nfev),
            "internal_steps": len(result.t),
            "linear": parsed.is_linear,
            "initial_conditions": ic,
        }
        if ic_note:
            diagnostics["note"] = ic_note

        return Solution(
            kind="SYSTEM",
            x=ts,
            y=ys,
            labels=list(parsed.state_vars),
            independent=parsed.independent_var,
            method=scipy_method,
            success=True,
            message=(
                f"Integrated {parsed.classification} with {scipy_method}. "
                f"{result.nfev} function evaluations, {n} state variables."
            ),
            diagnostics=diagnostics,
        )

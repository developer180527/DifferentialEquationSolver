"""Turn a free-form differential equation string into a :class:`ParsedEquation`.

Design
------
Parsing differential notation directly with SymPy is awkward, so this module
uses a thin regex layer to canonicalise derivative notation into ordinary
symbols, then hands the algebra to SymPy.

A derivative of order ``k`` of dependent variable ``y`` becomes the symbol
``y__d{k}`` (order 0 is just ``y``). After substitution the equation is a plain
algebraic relation that SymPy can manipulate: we build an ``Eq``, isolate the
highest-order derivative, and that explicit right-hand side is what the solver
integrates.

Supported notation
-------------------
* Leibniz:  ``dy/dx``, ``d^2y/dx^2``
* Prime:    ``y'``, ``y''``, ``y'''`` (independent variable assumed ``x``)
* Powers:   ``^`` is accepted as well as ``**``
* PDEs:     two distinct independent variables (e.g. ``t`` and ``x``) are
            detected and routed to the heat/wave finite-difference solver.

This is intentionally pragmatic rather than a full CAS frontend: it covers the
equations an educational/engineering tool is actually asked to solve.
"""

from __future__ import annotations

import re

import sympy as sp

from models.solution import ParsedEquation, ParsedSystem


class ParseError(ValueError):
    """Raised when an equation cannot be understood."""


def _deriv_symbol(dvar: str, order: int) -> str:
    return dvar if order == 0 else f"{dvar}__d{order}"


def _canonicalise(text: str) -> tuple[str, str, list[str], int]:
    """Replace derivative notation with plain symbols.

    Returns ``(canonical_text, dependent_var, independent_vars, max_order)``.
    """

    s = text.strip()
    if not s:
        raise ParseError("Equation is empty.")

    dep: str | None = None
    indep: list[str] = []
    max_order = 0

    def remember(d: str, order: int, ind: str | None = None) -> None:
        nonlocal dep, max_order
        if dep is None:
            dep = d
        elif dep != d:
            # mixed dependent variables - keep the first, ignore stray matches
            pass
        max_order = max(max_order, order)
        if ind is not None and ind not in indep:
            indep.append(ind)

    # --- Leibniz second order: d^2y/dx^2  (also d2y/dx2) -----------------
    def repl_leibniz2(m: re.Match) -> str:
        d, ind = m.group(1), m.group(2)
        remember(d, 2, ind)
        return _deriv_symbol(d, 2)

    s = re.sub(
        r"d\s*\^?\s*2\s*([a-zA-Z])\s*/\s*d\s*([a-zA-Z])\s*\^?\s*2",
        repl_leibniz2,
        s,
    )

    # --- Leibniz first order: dy/dx -------------------------------------
    def repl_leibniz1(m: re.Match) -> str:
        d, ind = m.group(1), m.group(2)
        remember(d, 1, ind)
        return _deriv_symbol(d, 1)

    s = re.sub(r"d\s*([a-zA-Z])\s*/\s*d\s*([a-zA-Z])", repl_leibniz1, s)

    # --- Prime notation: y', y'', y''' (independent var assumed x) ------
    def repl_prime(m: re.Match) -> str:
        d, primes = m.group(1), m.group(2)
        order = len(primes)
        remember(d, order)
        return _deriv_symbol(d, order)

    s = re.sub(r"([a-zA-Z])('+)", repl_prime, s)

    if dep is None:
        raise ParseError(
            "No derivative found. Use notation like dy/dx, d^2y/dx^2, y' or y''."
        )

    # If primes were used the independent variable is implicit; default to x
    # (or t if the expression mentions t but not x).
    if not indep:
        body = s.replace("**", " ")
        if re.search(r"\bt\b", body) and not re.search(r"\bx\b", body):
            indep = ["t"]
        else:
            indep = ["x"]

    s = s.replace("^", "**")
    return s, dep, indep, max_order


def _build_symbol_table(dep: str, indep: list[str], order: int) -> dict[str, sp.Symbol]:
    table: dict[str, sp.Symbol] = {}
    for v in indep:
        table[v] = sp.Symbol(v, real=True)
    table[dep] = sp.Symbol(dep, real=True)
    for k in range(1, order + 1):
        name = _deriv_symbol(dep, k)
        table[name] = sp.Symbol(name, real=True)
    # common constants users might reference
    for p in ("a", "b", "c", "alpha", "beta", "gamma", "pi"):
        table.setdefault(p, sp.Symbol(p, real=True))
    table["pi"] = sp.pi
    table["e"] = sp.E
    return table


def parse_equation(text: str) -> ParsedEquation:
    """Parse ``text`` into a :class:`ParsedEquation`. Raises :class:`ParseError`."""

    canonical, dep, indep, order = _canonicalise(text)

    if "=" not in canonical:
        raise ParseError("Equation must contain '=' (e.g. dy/dx = y - x^2).")

    lhs_raw, rhs_raw = canonical.split("=", 1)
    table = _build_symbol_table(dep, indep, order)

    try:
        lhs = sp.sympify(lhs_raw, locals=table)
        rhs = sp.sympify(rhs_raw, locals=table)
    except (sp.SympifyError, SyntaxError, TypeError) as exc:
        raise ParseError(f"Could not parse expression: {exc}") from exc

    eq_zero = sp.expand(lhs - rhs)               # F(...) = 0
    highest = table[_deriv_symbol(dep, order)]

    # Dynamical variables: the dependent var and all its derivatives.
    dyn_vars = [table[dep]] + [
        table[_deriv_symbol(dep, k)] for k in range(1, order + 1)
    ]

    # Linearity: equation is linear in the dynamical variables iff every
    # second partial derivative w.r.t. those variables vanishes.
    is_linear = True
    for i, vi in enumerate(dyn_vars):
        for vj in dyn_vars[i:]:
            if sp.simplify(sp.diff(eq_zero, vi, vj)) != 0:
                is_linear = False
                break
        if not is_linear:
            break

    # ---- PDE branch: two independent variables ------------------------
    if len(indep) >= 2:
        return _build_pde(text, dep, indep, order, rhs, is_linear)

    # ---- ODE branch: isolate the highest derivative -------------------
    try:
        solutions = sp.solve(eq_zero, highest, dict=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise ParseError(f"Could not isolate the highest derivative: {exc}") from exc

    if not solutions:
        raise ParseError(
            "Could not solve for the highest-order derivative. "
            "Make sure the equation is explicit in it (e.g. y'' = ...)."
        )
    rhs_expr = sp.simplify(solutions[0])

    linand = "linear" if is_linear else "nonlinear"
    ordinal = {1: "first", 2: "second", 3: "third"}.get(order, f"order-{order}")
    classification = f"{linand} {ordinal}-order ODE"

    return ParsedEquation(
        raw=text,
        kind="ODE",
        order=order,
        dependent_var=dep,
        independent_vars=indep,
        rhs_expr=rhs_expr,
        is_linear=is_linear,
        classification=classification,
    )


def _build_pde(text, dep, indep, order, rhs, is_linear) -> ParsedEquation:
    """Classify a PDE as heat (1st-order in time) or wave (2nd-order in time)."""

    # Identify which independent variable is time. Convention: 't' is time,
    # otherwise the second listed variable.
    time_var = "t" if "t" in indep else indep[-1]
    space_candidates = [v for v in indep if v != time_var]
    space_var = space_candidates[0] if space_candidates else "x"

    # Determine the time-derivative order from the *raw* text.
    has_2nd_time = bool(
        re.search(rf"d\s*\^?\s*2\s*{dep}\s*/\s*d\s*{time_var}\s*\^?\s*2", text)
    )
    subtype = "wave" if has_2nd_time else "heat"

    classification = (
        f"{'linear' if is_linear else 'nonlinear'} PDE "
        f"({'wave equation' if subtype == 'wave' else 'heat / diffusion equation'})"
    )

    return ParsedEquation(
        raw=text,
        kind="PDE",
        order=order,
        dependent_var=dep,
        independent_vars=indep,
        rhs_expr=rhs,
        is_linear=is_linear,
        classification=classification,
        pde_subtype=subtype,
        time_var=time_var,
        space_var=space_var,
    )


def quick_classify(text: str) -> str:
    """Best-effort one-line classification for live UI feedback (never raises)."""

    try:
        return parse_input(text).classification
    except Exception:
        return "Unknown"


# ======================================================================
# Systems of coupled first-order ODEs
# ======================================================================

def _parse_system_line(line: str):
    """Parse one ``d{var}/d{t} = rhs`` (or ``{var}' = rhs``) line.

    Returns ``(state_var, indep_var, rhs_string_canonical)``.
    """

    canonical, dep, indep, order = _canonicalise(line)
    if order != 1:
        raise ParseError(
            f"System equations must be first order; got order {order} in: {line!r}"
        )
    if "=" not in canonical:
        raise ParseError(f"Missing '=' in: {line!r}")

    lhs_raw, rhs_raw = canonical.split("=", 1)
    # The LHS must be exactly the first derivative of the state variable.
    if lhs_raw.strip() != _deriv_symbol(dep, 1):
        raise ParseError(
            f"Each system line must be explicit, like dx/dt = ...  (got {line!r})"
        )
    return dep, indep[0], rhs_raw


def parse_system(text: str) -> ParsedSystem:
    """Parse a multi-line block of coupled first-order ODEs."""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "=" in ln]
    if len(lines) < 2:
        raise ParseError("A system needs at least two equations (one per line).")

    state_vars: list[str] = []
    rhs_raw: list[str] = []
    indep_var: str | None = None

    for line in lines:
        dep, indep, rhs = _parse_system_line(line)
        if dep in state_vars:
            raise ParseError(f"State variable '{dep}' defined more than once.")
        if indep_var is None:
            indep_var = indep
        elif indep != indep_var:
            raise ParseError(
                f"Inconsistent independent variable: '{indep}' vs '{indep_var}'."
            )
        state_vars.append(dep)
        rhs_raw.append(rhs)

    # Build a symbol table covering every state variable + the independent var.
    table: dict = {sp.Symbol(v, real=True).name: sp.Symbol(v, real=True) for v in state_vars}
    table[indep_var] = sp.Symbol(indep_var, real=True)
    for p in ("a", "b", "c", "sigma", "rho", "beta", "alpha", "mu", "omega", "pi", "e"):
        table.setdefault(p, sp.Symbol(p, real=True))
    table["pi"] = sp.pi
    table["e"] = sp.E

    state_syms = [sp.Symbol(v, real=True) for v in state_vars]
    rhs_exprs = []
    parameters: set = set()
    is_linear = True

    for rhs in rhs_raw:
        try:
            expr = sp.sympify(rhs, locals=table)
        except (sp.SympifyError, SyntaxError, TypeError) as exc:
            raise ParseError(f"Could not parse '{rhs}': {exc}") from exc
        rhs_exprs.append(expr)
        # Free parameters = symbols that aren't state vars or the independent var.
        for s in expr.free_symbols:
            if s not in state_syms and s != table[indep_var]:
                parameters.add(str(s))
        # Linear iff every second partial w.r.t. the state vars vanishes.
        for i, vi in enumerate(state_syms):
            for vj in state_syms[i:]:
                if sp.simplify(sp.diff(expr, vi, vj)) != 0:
                    is_linear = False

    n = len(state_vars)
    classification = (
        f"{'linear' if is_linear else 'nonlinear'} system of "
        f"{n} coupled first-order ODEs"
    )

    return ParsedSystem(
        raw=text,
        independent_var=indep_var,
        state_vars=state_vars,
        rhs_exprs=rhs_exprs,
        parameters=sorted(parameters),
        is_linear=is_linear,
        classification=classification,
    )


def parse_input(text: str):
    """Top-level entry point: returns a ParsedSystem, or a ParsedEquation.

    Two or more equation lines (each containing '=') are treated as a system;
    otherwise the input is parsed as a single ODE/PDE.
    """

    eq_lines = [ln for ln in text.splitlines() if ln.strip() and "=" in ln]
    if len(eq_lines) >= 2:
        return parse_system(text)
    return parse_equation(text)


def free_parameters(parsed) -> list[str]:
    """Parameter names that need user-supplied values, for any parsed object."""

    if isinstance(parsed, ParsedSystem):
        return list(parsed.parameters)

    if parsed.kind == "PDE":
        return ["a"]  # diffusivity / wave speed

    # Single ODE: free symbols of the RHS minus the state + independent vars.
    dep = parsed.dependent_var
    indep = sp.Symbol(parsed.independent_vars[0], real=True)
    state = {sp.Symbol(dep, real=True)}
    state |= {sp.Symbol(f"{dep}__d{k}", real=True) for k in range(1, parsed.order)}
    params = [
        str(s) for s in parsed.rhs_expr.free_symbols
        if s != indep and s not in state
    ]
    return sorted(params)

# Differential Equation Solver

A cross-platform desktop app (PySide6) for defining, solving, and visualising
ODEs and PDEs without writing code.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Architecture
- `parser/equation_parser.py` — turns a free-form string (`dy/dx = y - x^2`,
  `y'' + y = 0`, `du/dt = d^2u/dx^2`) into a structured `ParsedEquation` using a
  regex notation layer + SymPy. Classifies order, linearity, and ODE vs PDE.
- `solver/ode_solver.py` — lambdifies the parsed RHS, reduces order-n ODEs to a
  first-order system, integrates with SciPy `solve_ivp` (RK45 / BDF / Radau).
- `solver/pde_solver.py` — explicit finite-difference schemes for the 1D heat
  (FTCS) and wave (leapfrog) equations, with automatic stable time-stepping.
- `models/` — `Solution` / `ParsedEquation` dataclasses + example library.
- `visualization/` — 2D plot, 3D surface (PDE) / phase portrait (2nd-order ODE),
  and a numerical data table with CSV export. All use the matplotlib Qt backend.
- `main.py` — wires inputs → parser → solver → views; live equation detection,
  example presets, JSON project save/load, PNG/CSV export, solver diagnostics.

## Supported notation
Leibniz (`dy/dx`, `d^2y/dx^2`), prime (`y'`, `y''`), `^` or `**` for powers.
Parameters `a`, `b`, `c` can appear in equations and are supplied in the UI.
For PDEs, parameter `a` is the diffusivity (heat) or wave speed (wave).

## Extending
The `ParsedEquation` IR decouples parsing from solving, so new backends
(finite element, finite volume, implicit PDE schemes) plug in by adding a solver
that consumes the IR — no parser or UI changes required.

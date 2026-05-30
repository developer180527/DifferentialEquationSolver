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
- `solver/system_solver.py` — integrates **systems of coupled first-order ODEs**
  (e.g. Lorenz, Rössler, Lotka–Volterra) into a state-space trajectory.
- `visualization/pyvista_widget.py` — **interactive GPU-accelerated 3D** (PyVista/VTK):
  attractor trajectories as time-coloured tubes, PDE surfaces, and phase portraits.
  Falls back to `surface_widget.py` (matplotlib 3D) if PyVista is not installed.
- `models/` — `Solution` / `ParsedEquation` dataclasses + example library.
- `visualization/` — 2D plot, 3D surface (PDE) / phase portrait (2nd-order ODE),
  and a numerical data table with CSV export. All use the matplotlib Qt backend.
- `main.py` — wires inputs → parser → solver → views; live equation detection,
  example presets, JSON project save/load, PNG/CSV export, solver diagnostics.

## Supported notation
Leibniz (`dy/dx`, `d^2y/dx^2`), prime (`y'`, `y''`), `^` or `**` for powers.
Parameters `a`, `b`, `c` can appear in equations and are supplied in the UI.
For PDEs, parameter `a` is the diffusivity (heat) or wave speed (wave).

## Systems & 3D
Enter several coupled equations, one per line, to define a system:
```
dx/dt = sigma*(y - x)
dy/dt = x*(rho - z) - y
dz/dt = x*y - beta*z
```
The Parameters panel populates **automatically** from the symbols it detects
(here sigma, rho, beta). Open the 3D tab and drag to rotate the attractor.

## Extending
The `ParsedEquation` IR decouples parsing from solving, so new backends
(finite element, finite volume, implicit PDE schemes) plug in by adding a solver
that consumes the IR — no parser or UI changes required.

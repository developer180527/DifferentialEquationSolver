"""Built-in example equations with sensible default settings.

Each entry is a dict the main window can apply directly to the input fields.
"""

EXAMPLES = [
    {
        "name": "Exponential growth  (dy/dx = y)",
        "equation": "dy/dx = y",
        "type": "ODE",
        "domain": (0.0, 4.0),
        "ics": "1",
        "params": {},
        "note": "Solution is e^x.",
    },
    {
        "name": "Logistic growth",
        "equation": "dy/dx = a*y*(1 - y/b)",
        "type": "ODE",
        "domain": (0.0, 10.0),
        "ics": "0.1",
        "params": {"a": 1.0, "b": 1.0},
        "note": "S-curve saturating at carrying capacity b.",
    },
    {
        "name": "Newton cooling  (dy/dx = -a(y - b))",
        "equation": "dy/dx = -a*(y - b)",
        "type": "ODE",
        "domain": (0.0, 8.0),
        "ics": "100",
        "params": {"a": 0.5, "b": 20.0},
        "note": "Object cooling toward ambient temperature b.",
    },
    {
        "name": "Simple harmonic motion  (y'' + y = 0)",
        "equation": "y'' + y = 0",
        "type": "ODE",
        "domain": (0.0, 12.566),
        "ics": "0, 1",
        "params": {},
        "note": "Undamped oscillator. ICs: y(0)=0, y'(0)=1 -> sin(x).",
    },
    {
        "name": "Damped oscillator  (y'' + c y' + y = 0)",
        "equation": "y'' + c*y' + y = 0",
        "type": "ODE",
        "domain": (0.0, 20.0),
        "ics": "1, 0",
        "params": {"c": 0.3},
        "note": "Damping coefficient c controls decay.",
    },
    {
        "name": "Heat / diffusion equation",
        "equation": "du/dt = d^2u/dx^2",
        "type": "PDE",
        "domain": (0.0, 1.0),
        "ics": "",
        "params": {"a": 1.0},
        "note": "Parameter a = diffusivity. Initial sine profile decays. t_max=0.2.",
        "t_max": 0.2,
    },
    {
        "name": "Wave equation",
        "equation": "d^2u/dt^2 = c^2 * d^2u/dx^2",
        "type": "PDE",
        "domain": (0.0, 1.0),
        "ics": "",
        "params": {"a": 1.0},
        "note": "Parameter a = wave speed c. Standing-wave oscillation. t_max=2.0.",
        "t_max": 2.0,
    },
]

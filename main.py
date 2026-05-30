import json
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from models.examples import EXAMPLES
from models.solution import ParsedSystem
from parser.equation_parser import (
    ParseError,
    free_parameters,
    parse_input,
    quick_classify,
)
from solver.ode_solver import ODESolver, SolverError as ODESolverError
from solver.pde_solver import PDESolver, SolverError as PDESolverError
from solver.system_solver import SystemSolver, SolverError as SysSolverError
from visualization.data_table import DataTableWidget
from visualization.matplotlib_widget import MatplotlibWidget

# Prefer the GPU-accelerated PyVista 3D viewer; fall back to matplotlib 3D.
try:
    from visualization.pyvista_widget import PyVistaWidget
    _SURFACE_CLS = PyVistaWidget
    _HAS_PYVISTA = True
except Exception:  # pragma: no cover - import/runtime GL issues
    from visualization.surface_widget import SurfaceWidget
    _SURFACE_CLS = SurfaceWidget
    _HAS_PYVISTA = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Differential Equation Solver")
        self.resize(1400, 900)

        self._last_solution = None
        self.param_fields: dict[str, QLineEdit] = {}

        self._build_toolbar()
        self._build_ui()

        self._classify_timer = QTimer(self)
        self._classify_timer.setInterval(350)
        self._classify_timer.setSingleShot(True)
        self._classify_timer.timeout.connect(self._on_equation_changed)
        self.equation_input.textChanged.connect(self._classify_timer.start)

        if not _HAS_PYVISTA:
            self.log("Note: PyVista not available — using matplotlib 3D fallback.")

    # ------------------------------------------------------------------
    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction("New", self.on_new)
        toolbar.addAction("Open", self.on_open)
        toolbar.addAction("Save", self.on_save)
        toolbar.addSeparator()

        self.examples_button = QPushButton("Examples")
        self.examples_menu = QMenu(self)
        for ex in EXAMPLES:
            act = self.examples_menu.addAction(ex["name"])
            act.triggered.connect(lambda checked=False, e=ex: self.load_example(e))
        self.examples_button.setMenu(self.examples_menu)
        toolbar.addWidget(self.examples_button)
        toolbar.addAction("Export", self.on_export)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ---------------- left panel ----------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        equation_group = QGroupBox("Equation")
        equation_layout = QVBoxLayout(equation_group)
        self.equation_input = QPlainTextEdit()
        self.equation_input.setPlaceholderText(
            "Enter an ODE, PDE, or a system (one equation per line)...\n"
            "  dy/dx = y - x^2\n"
            "  y'' + 0.3*y' + y = 0\n"
            "  du/dt = d^2u/dx^2\n"
            "System:\n"
            "  dx/dt = sigma*(y - x)\n"
            "  dy/dt = x*(rho - z) - y\n"
            "  dz/dt = x*y - beta*z"
        )
        self.equation_input.setMinimumHeight(150)
        equation_layout.addWidget(self.equation_input)
        left_layout.addWidget(equation_group)

        settings_group = QGroupBox("Solver Settings")
        settings_layout = QVBoxLayout(settings_group)
        self.equation_type = QComboBox()
        self.equation_type.addItems(["Auto-detect", "ODE", "PDE", "System"])
        self.method_box = QComboBox()
        self.method_box.addItems(["Auto", "RK45", "BDF", "Radau"])
        self.start_input = QLineEdit("0")
        self.end_input = QLineEdit("10")
        settings_layout.addWidget(QLabel("Equation Type"))
        settings_layout.addWidget(self.equation_type)
        settings_layout.addWidget(QLabel("Solver Method (ODE / System)"))
        settings_layout.addWidget(self.method_box)
        settings_layout.addWidget(QLabel("Domain Start"))
        settings_layout.addWidget(self.start_input)
        settings_layout.addWidget(QLabel("Domain End"))
        settings_layout.addWidget(self.end_input)
        left_layout.addWidget(settings_group)

        conditions_group = QGroupBox("Conditions")
        conditions_layout = QFormLayout(conditions_group)
        self.ic_input = QLineEdit("1")
        self.ic_input.setPlaceholderText("comma-separated, e.g. 1, 0  or  1, 1, 1")
        self.tmax_input = QLineEdit("1.0")
        conditions_layout.addRow("Initial conditions", self.ic_input)
        conditions_layout.addRow("PDE time horizon", self.tmax_input)
        left_layout.addWidget(conditions_group)

        # ---- dynamic parameters panel ----
        self.parameters_group = QGroupBox("Parameters")
        self.param_form = QFormLayout(self.parameters_group)
        left_layout.addWidget(self.parameters_group)
        self._sync_parameter_fields([])

        self.detected_type_label = QLabel("Detected: —")
        self.detected_type_label.setWordWrap(True)
        self.detected_type_label.setStyleSheet("color: #4a90d9; font-style: italic;")
        left_layout.addWidget(self.detected_type_label)

        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.on_solve_clicked)
        left_layout.addWidget(self.solve_button)
        left_layout.addStretch()

        # ---------------- right panel ----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        visualization_group = QGroupBox("Visualization")
        visualization_layout = QVBoxLayout(visualization_group)
        self.visualization_tabs = QTabWidget()

        self.plot_widget = MatplotlibWidget()
        self.surface_widget, surface_label = self._make_surface_widget()
        self.data_table = DataTableWidget()

        self.visualization_tabs.addTab(self.plot_widget, "Plot")
        self.visualization_tabs.addTab(self.surface_widget, surface_label)
        self.visualization_tabs.addTab(self.data_table, "Data")
        visualization_layout.addWidget(self.visualization_tabs)
        right_layout.addWidget(visualization_group)

        output_group = QGroupBox("Solver Output")
        output_layout = QVBoxLayout(output_group)
        self.output_log = QPlainTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setMaximumHeight(200)
        self.output_log.setPlaceholderText(
            "Solver diagnostics, warnings and results will appear here..."
        )
        output_layout.addWidget(self.output_log)
        right_layout.addWidget(output_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 1020])

    # ================= dynamic parameters =================
    def _make_surface_widget(self):
        """Construct the 3D viewer, falling back to matplotlib if PyVista fails."""
        if _HAS_PYVISTA:
            try:
                return _SURFACE_CLS(), "3D"
            except Exception as exc:  # GL context / driver issues at runtime
                self.log(f"3D viewer unavailable ({exc}); using matplotlib fallback.")
        from visualization.surface_widget import SurfaceWidget
        return SurfaceWidget(), "Surface"

    def _sync_parameter_fields(self, names: list[str], values: dict | None = None):
        """Rebuild the Parameters panel to show one field per detected parameter."""
        preserved = {n: w.text() for n, w in self.param_fields.items()}
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self.param_fields = {}

        if not names:
            hint = QLabel("Parameters in the equation appear here automatically.")
            hint.setStyleSheet("color: gray; font-style: italic;")
            hint.setWordWrap(True)
            self.param_form.addRow(hint)
            return

        for name in names:
            field = QLineEdit()
            if values and name in values:
                field.setText(str(values[name]))
            elif name in preserved:
                field.setText(preserved[name])
            self.param_form.addRow(name, field)
            self.param_fields[name] = field

    # ================= helpers =================
    def log(self, text: str):
        self.output_log.appendPlainText(text)

    def _collect_params(self) -> dict:
        params = {}
        for name, widget in self.param_fields.items():
            raw = widget.text().strip()
            if raw:
                try:
                    params[name] = float(raw)
                except ValueError:
                    raise ValueError(f"Parameter '{name}' is not a number: {raw!r}")
        return params

    def _collect_ics(self) -> list:
        raw = self.ic_input.text().strip()
        if not raw:
            return []
        try:
            return [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
        except ValueError:
            raise ValueError(f"Could not read initial conditions: {raw!r}")

    def _domain(self) -> tuple:
        try:
            return float(self.start_input.text()), float(self.end_input.text())
        except ValueError:
            raise ValueError("Domain start/end must be numbers.")

    def _on_equation_changed(self):
        text = self.equation_input.toPlainText().strip()
        if not text:
            self.detected_type_label.setText("Detected: —")
            self._sync_parameter_fields([])
            return
        self.detected_type_label.setText(f"Detected: {quick_classify(text)}")
        try:
            self._sync_parameter_fields(free_parameters(parse_input(text)))
        except Exception:
            pass  # leave fields as-is while the equation is mid-edit

    @staticmethod
    def _kind(parsed) -> str:
        return "SYSTEM" if isinstance(parsed, ParsedSystem) else parsed.kind

    # ================= solve =================
    def on_solve_clicked(self):
        text = self.equation_input.toPlainText().strip()
        if not text:
            self.log("Nothing to solve — enter an equation first.")
            return
        try:
            parsed = parse_input(text)
            kind = self._kind(parsed)

            forced = self.equation_type.currentText()
            forced_map = {"ODE": "ODE", "PDE": "PDE", "System": "SYSTEM"}
            if forced in forced_map and forced_map[forced] != kind:
                raise ValueError(
                    f"Equation parses as {kind} but type is forced to {forced}."
                )

            self.log(f"Parsed: {parsed.classification}")
            params = self._collect_params()

            if kind == "SYSTEM":
                solution = SystemSolver().solve(
                    parsed,
                    domain=self._domain(),
                    initial_conditions=self._collect_ics(),
                    params=params,
                    method=self.method_box.currentText(),
                )
            elif kind == "ODE":
                solution = ODESolver().solve(
                    parsed,
                    domain=self._domain(),
                    initial_conditions=self._collect_ics(),
                    params=params,
                    method=self.method_box.currentText(),
                )
            else:
                coeff = params.get("a", 1.0)
                try:
                    t_max = float(self.tmax_input.text())
                except ValueError:
                    t_max = 1.0
                solution = PDESolver().solve(
                    parsed, domain=self._domain(), coefficient=coeff, t_max=t_max,
                )

            self._render(solution)
            self.log(solution.message)
            self.log("Diagnostics: " + json.dumps(self._jsonable(solution.diagnostics)))
            self.log("-" * 40)

        except (ParseError, ValueError, ODESolverError, PDESolverError,
                SysSolverError) as exc:
            self.log(f"ERROR: {exc}")
        except Exception as exc:  # pragma: no cover
            self.log(f"UNEXPECTED ERROR: {exc}")

    def _render(self, solution):
        self._last_solution = solution
        self.plot_widget.plot_solution(solution)
        self.surface_widget.plot_solution(solution)
        self.data_table.show_solution(solution)
        # Jump to the most informative tab.
        if solution.kind in ("PDE", "SYSTEM"):
            self.visualization_tabs.setCurrentIndex(1)
        else:
            self.visualization_tabs.setCurrentIndex(0)

    @staticmethod
    def _jsonable(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                out[k] = [round(float(x), 6) if isinstance(x, float) else x for x in v]
            elif isinstance(v, float):
                out[k] = round(v, 6)
            else:
                out[k] = v
        return out

    # ================= examples / project IO =================
    def load_example(self, ex: dict):
        self.equation_input.setPlainText(ex["equation"])
        self.equation_type.setCurrentText(ex["type"])
        self.start_input.setText(str(ex["domain"][0]))
        self.end_input.setText(str(ex["domain"][1]))
        self.ic_input.setText(ex.get("ics", ""))
        self.tmax_input.setText(str(ex.get("t_max", 1.0)))
        try:
            names = free_parameters(parse_input(ex["equation"]))
        except Exception:
            names = list(ex.get("params", {}).keys())
        self._sync_parameter_fields(names, ex.get("params", {}))
        self.output_log.clear()
        self.log(f"Loaded example: {ex['name']}")
        if ex.get("note"):
            self.log(ex["note"])
        self.detected_type_label.setText(f"Detected: {quick_classify(ex['equation'])}")

    def _project_state(self) -> dict:
        return {
            "equation": self.equation_input.toPlainText(),
            "type": self.equation_type.currentText(),
            "method": self.method_box.currentText(),
            "domain_start": self.start_input.text(),
            "domain_end": self.end_input.text(),
            "ics": self.ic_input.text(),
            "t_max": self.tmax_input.text(),
            "params": {n: w.text() for n, w in self.param_fields.items()},
        }

    def on_new(self):
        self.equation_input.clear()
        self.output_log.clear()
        self.ic_input.setText("1")
        self._sync_parameter_fields([])
        self.detected_type_label.setText("Detected: —")
        self.log("New project.")

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "DE Solver Project (*.desolver *.json)"
        )
        if not path:
            return
        try:
            with open(path, "w") as fh:
                json.dump(self._project_state(), fh, indent=2)
            self.log(f"Saved project to {path}")
        except OSError as exc:
            self.log(f"ERROR saving: {exc}")

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "DE Solver Project (*.desolver *.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as fh:
                state = json.load(fh)
            self.equation_input.setPlainText(state.get("equation", ""))
            self.equation_type.setCurrentText(state.get("type", "Auto-detect"))
            self.method_box.setCurrentText(state.get("method", "Auto"))
            self.start_input.setText(state.get("domain_start", "0"))
            self.end_input.setText(state.get("domain_end", "10"))
            self.ic_input.setText(state.get("ics", "1"))
            self.tmax_input.setText(state.get("t_max", "1.0"))
            params = state.get("params", {})
            try:
                names = free_parameters(parse_input(state.get("equation", "")))
            except Exception:
                names = list(params.keys())
            self._sync_parameter_fields(names, params)
            self.log(f"Opened project from {path}")
            self.detected_type_label.setText(
                f"Detected: {quick_classify(state.get('equation', ''))}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"ERROR opening: {exc}")

    def on_export(self):
        if self._last_solution is None:
            self.log("Nothing to export — solve an equation first.")
            return
        menu = QMenu(self)
        menu.addAction("Export current view as PNG…", self._export_png)
        menu.addAction("Export data as CSV…", self._export_csv)
        menu.exec(self.cursor().pos())

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Image", "", "PNG image (*.png)")
        if not path:
            return
        use_surface = self._last_solution.kind in ("PDE", "SYSTEM")
        widget = self.surface_widget if use_surface else self.plot_widget
        try:
            widget.export_image(path)
            self.log(f"Exported image to {path}")
        except Exception as exc:
            self.log(f"ERROR exporting image: {exc}")

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV file (*.csv)")
        if not path:
            return
        try:
            self.data_table.export_csv(path)
            self.log(f"Exported data to {path}")
        except ValueError as exc:
            self.log(f"ERROR: {exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

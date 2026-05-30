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
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from models.examples import EXAMPLES
from parser.equation_parser import ParseError, parse_equation, quick_classify
from solver.ode_solver import ODESolver, SolverError as ODESolverError
from solver.pde_solver import PDESolver, SolverError as PDESolverError
from visualization.data_table import DataTableWidget
from visualization.matplotlib_widget import MatplotlibWidget
from visualization.surface_widget import SurfaceWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Differential Equation Solver")
        self.resize(1400, 900)

        self._last_solution = None

        self._build_toolbar()
        self._build_ui()

        # Debounced live classification as the user types.
        self._classify_timer = QTimer(self)
        self._classify_timer.setInterval(350)
        self._classify_timer.setSingleShot(True)
        self._classify_timer.timeout.connect(self._update_detected_type)
        self.equation_input.textChanged.connect(self._classify_timer.start)

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
            action = self.examples_menu.addAction(ex["name"])
            action.triggered.connect(lambda checked=False, e=ex: self.load_example(e))
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
            "Enter an ODE or PDE here...\n"
            "Examples:\n"
            "  dy/dx = y - x^2\n"
            "  y'' + 0.3*y' + y = 0\n"
            "  du/dt = d^2u/dx^2"
        )
        self.equation_input.setMinimumHeight(150)
        equation_layout.addWidget(self.equation_input)
        left_layout.addWidget(equation_group)

        settings_group = QGroupBox("Solver Settings")
        settings_layout = QVBoxLayout(settings_group)

        self.equation_type = QComboBox()
        self.equation_type.addItems(["Auto-detect", "ODE", "PDE"])

        self.method_box = QComboBox()
        self.method_box.addItems(["Auto", "RK45", "BDF", "Radau"])

        self.start_input = QLineEdit("0")
        self.end_input = QLineEdit("10")

        settings_layout.addWidget(QLabel("Equation Type"))
        settings_layout.addWidget(self.equation_type)
        settings_layout.addWidget(QLabel("Solver Method (ODE)"))
        settings_layout.addWidget(self.method_box)
        settings_layout.addWidget(QLabel("Domain Start"))
        settings_layout.addWidget(self.start_input)
        settings_layout.addWidget(QLabel("Domain End"))
        settings_layout.addWidget(self.end_input)
        left_layout.addWidget(settings_group)

        conditions_group = QGroupBox("Conditions")
        conditions_layout = QFormLayout(conditions_group)
        self.ic_input = QLineEdit("1")
        self.ic_input.setPlaceholderText("y(start), y'(start), ...  e.g. 0, 1")
        self.tmax_input = QLineEdit("1.0")
        conditions_layout.addRow("Initial conditions", self.ic_input)
        conditions_layout.addRow("PDE time horizon", self.tmax_input)
        left_layout.addWidget(conditions_group)

        parameters_group = QGroupBox("Parameters")
        parameters_layout = QFormLayout(parameters_group)
        self.param_a = QLineEdit()
        self.param_b = QLineEdit()
        self.param_c = QLineEdit()
        parameters_layout.addRow("a", self.param_a)
        parameters_layout.addRow("b", self.param_b)
        parameters_layout.addRow("c", self.param_c)
        left_layout.addWidget(parameters_group)

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
        self.surface_widget = SurfaceWidget()
        self.data_table = DataTableWidget()

        self.visualization_tabs.addTab(self.plot_widget, "Plot")
        self.visualization_tabs.addTab(self.surface_widget, "Surface")
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

    # ================= helpers =================
    def log(self, text: str):
        self.output_log.appendPlainText(text)

    def _collect_params(self) -> dict:
        params = {}
        for name, widget in (("a", self.param_a), ("b", self.param_b), ("c", self.param_c)):
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

    def _update_detected_type(self):
        text = self.equation_input.toPlainText().strip()
        if not text:
            self.detected_type_label.setText("Detected: —")
            return
        self.detected_type_label.setText(f"Detected: {quick_classify(text)}")

    # ================= actions =================
    def on_solve_clicked(self):
        text = self.equation_input.toPlainText().strip()
        if not text:
            self.log("Nothing to solve — enter an equation first.")
            return
        try:
            parsed = parse_equation(text)

            forced = self.equation_type.currentText()
            if forced == "ODE" and parsed.kind != "ODE":
                raise ValueError("Equation looks like a PDE but type is forced to ODE.")
            if forced == "PDE" and parsed.kind != "PDE":
                raise ValueError("Equation looks like an ODE but type is forced to PDE.")

            self.log(f"Parsed: {parsed.classification}")
            params = self._collect_params()

            if parsed.kind == "ODE":
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
                    parsed,
                    domain=self._domain(),
                    coefficient=coeff,
                    t_max=t_max,
                )

            self._render(solution)
            self.log(solution.message)
            self.log("Diagnostics: " + json.dumps(self._jsonable(solution.diagnostics)))
            self.log("-" * 40)

        except (ParseError, ValueError, ODESolverError, PDESolverError) as exc:
            self.log(f"ERROR: {exc}")
        except Exception as exc:  # pragma: no cover
            self.log(f"UNEXPECTED ERROR: {exc}")

    def _render(self, solution):
        self._last_solution = solution
        self.plot_widget.plot_solution(solution)
        self.surface_widget.plot_solution(solution)
        self.data_table.show_solution(solution)
        # Jump to the most informative tab for this solution type.
        self.visualization_tabs.setCurrentIndex(1 if solution.kind == "PDE" else 0)

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

    # ----- examples / project IO -----
    def load_example(self, ex: dict):
        self.equation_input.setPlainText(ex["equation"])
        self.equation_type.setCurrentText(ex["type"])
        self.start_input.setText(str(ex["domain"][0]))
        self.end_input.setText(str(ex["domain"][1]))
        self.ic_input.setText(ex.get("ics", ""))
        self.tmax_input.setText(str(ex.get("t_max", 1.0)))
        p = ex.get("params", {})
        self.param_a.setText(str(p.get("a", "")) if "a" in p else "")
        self.param_b.setText(str(p.get("b", "")) if "b" in p else "")
        self.param_c.setText(str(p.get("c", "")) if "c" in p else "")
        self.output_log.clear()
        self.log(f"Loaded example: {ex['name']}")
        if ex.get("note"):
            self.log(ex["note"])
        self._update_detected_type()

    def _project_state(self) -> dict:
        return {
            "equation": self.equation_input.toPlainText(),
            "type": self.equation_type.currentText(),
            "method": self.method_box.currentText(),
            "domain_start": self.start_input.text(),
            "domain_end": self.end_input.text(),
            "ics": self.ic_input.text(),
            "t_max": self.tmax_input.text(),
            "a": self.param_a.text(),
            "b": self.param_b.text(),
            "c": self.param_c.text(),
        }

    def on_new(self):
        self.equation_input.clear()
        self.output_log.clear()
        self.ic_input.setText("1")
        for w in (self.param_a, self.param_b, self.param_c):
            w.clear()
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
            self.param_a.setText(state.get("a", ""))
            self.param_b.setText(state.get("b", ""))
            self.param_c.setText(state.get("c", ""))
            self.log(f"Opened project from {path}")
            self._update_detected_type()
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"ERROR opening: {exc}")

    def on_export(self):
        if self._last_solution is None:
            self.log("Nothing to export — solve an equation first.")
            return
        menu = QMenu(self)
        menu.addAction("Export plot as PNG…", self._export_png)
        menu.addAction("Export data as CSV…", self._export_csv)
        menu.exec(self.cursor().pos())

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG image (*.png)")
        if not path:
            return
        widget = self.surface_widget if self._last_solution.kind == "PDE" else self.plot_widget
        widget.figure.savefig(path, dpi=150, bbox_inches="tight")
        self.log(f"Exported plot to {path}")

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


import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QTabWidget,
    QToolBar,
    QFormLayout,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Differential Equation Solver")
        self.resize(1400, 900)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        toolbar.addAction("New")
        toolbar.addAction("Open")
        toolbar.addAction("Save")
        toolbar.addSeparator()
        toolbar.addAction("Examples")
        toolbar.addAction("Export")

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        equation_group = QGroupBox("Equation")
        equation_layout = QVBoxLayout(equation_group)

        self.equation_input = QPlainTextEdit()
        self.equation_input.setPlaceholderText(
            "Enter an ODE or PDE here...\nExample: dy/dx = y - x^2"
        )
        self.equation_input.setMinimumHeight(260)

        equation_layout.addWidget(self.equation_input)
        left_layout.addWidget(equation_group)

        settings_group = QGroupBox("Solver Settings")
        settings_layout = QVBoxLayout(settings_group)

        self.equation_type = QComboBox()
        self.equation_type.addItems(["ODE", "PDE"])

        self.method_box = QComboBox()
        self.method_box.addItems([
            "Auto",
            "RK45",
            "BDF",
            "Finite Difference",
            "Finite Element",
        ])

        self.start_input = QLineEdit()
        self.end_input = QLineEdit()

        settings_layout.addWidget(QLabel("Equation Type"))
        settings_layout.addWidget(self.equation_type)

        settings_layout.addWidget(QLabel("Solver Method"))
        settings_layout.addWidget(self.method_box)

        settings_layout.addWidget(QLabel("Domain Start"))
        settings_layout.addWidget(self.start_input)

        settings_layout.addWidget(QLabel("Domain End"))
        settings_layout.addWidget(self.end_input)

        left_layout.addWidget(settings_group)

        parameters_group = QGroupBox("Parameters")
        parameters_layout = QFormLayout(parameters_group)

        self.param_a = QLineEdit()
        self.param_b = QLineEdit()
        self.param_c = QLineEdit()

        parameters_layout.addRow("a", self.param_a)
        parameters_layout.addRow("b", self.param_b)
        parameters_layout.addRow("c", self.param_c)

        left_layout.addWidget(parameters_group)

        self.detected_type_label = QLabel("Detected: Unknown")
        left_layout.addWidget(self.detected_type_label)

        self.solve_button = QPushButton("Solve")
        left_layout.addWidget(self.solve_button)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        visualization_group = QGroupBox("Visualization")
        visualization_layout = QVBoxLayout(visualization_group)

        self.visualization_tabs = QTabWidget()

        self.plot_placeholder = QLabel("2D Plot View")
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setStyleSheet(
            "border: 2px dashed gray; font-size: 18px;"
        )

        self.surface_placeholder = QLabel("3D Surface View")
        self.surface_placeholder.setAlignment(Qt.AlignCenter)
        self.surface_placeholder.setStyleSheet(
            "border: 2px dashed gray; font-size: 18px;"
        )

        self.data_placeholder = QLabel("Numerical Data Table")
        self.data_placeholder.setAlignment(Qt.AlignCenter)
        self.data_placeholder.setStyleSheet(
            "border: 2px dashed gray; font-size: 18px;"
        )

        self.visualization_tabs.addTab(self.plot_placeholder, "Plot")
        self.visualization_tabs.addTab(self.surface_placeholder, "Surface")
        self.visualization_tabs.addTab(self.data_placeholder, "Data")

        visualization_layout.addWidget(self.visualization_tabs)
        right_layout.addWidget(visualization_group)

        output_group = QGroupBox("Solver Output")
        output_layout = QVBoxLayout(output_group)

        self.output_log = QPlainTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setPlaceholderText(
            "Solver diagnostics, warnings and results will appear here..."
        )

        output_layout.addWidget(self.output_log)
        right_layout.addWidget(output_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
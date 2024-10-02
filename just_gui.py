import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QWidget,
    QScrollArea,
    QComboBox,
    QSizePolicy,
    QGroupBox,
    QGridLayout,
    QStatusBar,
    QCheckBox,
)
from PySide6.QtCore import Qt


class MLFrameworkCheckerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Framework and CUDA Checker (GUI Only)")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.main_layout.addWidget(self.header_widget)

        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.init_ui()
        self.init_advanced_features()

        self.set_theme("Light")

    def init_ui(self):
        theme_label = QLabel("Theme:")
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark", "Blue", "Green"])
        self.theme_selector.currentTextChanged.connect(self.set_theme)
        self.header_layout.addWidget(theme_label)
        self.header_layout.addWidget(self.theme_selector)
        self.header_layout.addStretch()

        self.add_button_and_label("Check PyTorch", lambda: None, 0, 0)
        self.add_button_and_label("Check TensorFlow", lambda: None, 1, 0)
        self.add_button_and_label("Check CUDA", lambda: None, 2, 0)

        self.add_button("Check System Specs", lambda: None, 3, 0)
        self.system_label = QTextEdit("System specifications will be displayed here")
        self.system_label.setReadOnly(True)
        self.content_layout.addWidget(self.system_label, 4, 0, 1, 2)

        self.add_button("Check Compatibility", lambda: None, 5, 0)
        self.compatibility_label = QTextEdit(
            "Compatibility information will be displayed here"
        )
        self.compatibility_label.setReadOnly(True)
        self.content_layout.addWidget(self.compatibility_label, 6, 0, 1, 2)

        self.add_button("Export Logs", lambda: None, 7, 0)
        self.add_button("Instructions", lambda: None, 7, 1)
        self.add_button("Check PyTorch Updates", lambda: None, 8, 0)
        self.add_button("Check TensorFlow Updates", lambda: None, 8, 1)
        self.add_button("Check CUDA Updates", lambda: None, 9, 0)

    def init_advanced_features(self):
        self.advanced_group = QGroupBox("Advanced Features")
        self.advanced_checkbox = QCheckBox("Enable Advanced Features")
        self.advanced_checkbox.setChecked(False)
        self.advanced_checkbox.toggled.connect(self.on_advanced_toggled)

        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(self.advanced_checkbox)

        self.persistence_mode_button = self.add_button_to_layout(
            "Enable NVIDIA Persistence Mode", lambda: None, advanced_layout
        )
        self.gpu_logging_button = self.add_button_to_layout(
            "Start GPU Logging", lambda: None, advanced_layout
        )

        self.persistence_mode_button.setVisible(False)
        self.gpu_logging_button.setVisible(False)

        self.advanced_group.setLayout(advanced_layout)
        self.content_layout.addWidget(self.advanced_group, 10, 0, 1, 2)

    def on_advanced_toggled(self, checked):
        self.persistence_mode_button.setVisible(checked)
        self.gpu_logging_button.setVisible(checked)

    def add_button_to_layout(self, text, function, layout):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(button)
        return button

    def add_button_and_label(self, text, function, row, col):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label = QLabel("Status will be displayed here")
        self.content_layout.addWidget(button, row, col)
        self.content_layout.addWidget(label, row, col + 1)
        return button, label

    def add_button(self, text, function, row, col):
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.content_layout.addWidget(button, row, col)
        return button

    def set_theme(self, theme):
        if theme == "Light":
            self.setStyleSheet(
                """
                QWidget { background-color: #f0f0f0; color: #000000; font-family: Arial; font-size: 14px; }
                QPushButton { background-color: #e0e0e0; border: 1px solid #b0b0b0; border-radius: 5px; padding: 8px; margin: 5px; }
                QPushButton:hover { background-color: #d0d0d0; }
                QLabel, QTextEdit { background-color: #ffffff; border: 1px solid #d0d0d0; border-radius: 3px; padding: 5px; }
                QGroupBox { border: 2px solid #b0b0b0; border-radius: 5px; margin-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            """
            )
        elif theme == "Dark":
            self.setStyleSheet(
                """
                QWidget { background-color: #2b2b2b; color: #ffffff; font-family: Arial; font-size: 14px; }
                QPushButton { background-color: #3b3b3b; border: 1px solid #505050; border-radius: 5px; padding: 8px; margin: 5px; }
                QPushButton:hover { background-color: #4b4b4b; }
                QLabel, QTextEdit { background-color: #3b3b3b; border: 1px solid #505050; border-radius: 3px; padding: 5px; }
                QGroupBox { border: 2px solid #505050; border-radius: 5px; margin-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            """
            )
        elif theme == "Blue":
            self.setStyleSheet(
                """
                QWidget { background-color: #e6f3ff; color: #000000; font-family: Arial; font-size: 14px; }
                QPushButton { background-color: #b3d9ff; border: 1px solid #80bfff; border-radius: 5px; padding: 8px; margin: 5px; }
                QPushButton:hover { background-color: #99ccff; }
                QLabel, QTextEdit { background-color: #ffffff; border: 1px solid #b3d9ff; border-radius: 3px; padding: 5px; }
                QGroupBox { border: 2px solid #80bfff; border-radius: 5px; margin-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            """
            )
        elif theme == "Green":
            self.setStyleSheet(
                """
                QWidget { background-color: #e6ffe6; color: #000000; font-family: Arial; font-size: 14px; }
                QPushButton { background-color: #b3ffb3; border: 1px solid #80ff80; border-radius: 5px; padding: 8px; margin: 5px; }
                QPushButton:hover { background-color: #99ff99; }
                QLabel, QTextEdit { background-color: #ffffff; border: 1px solid #b3ffb3; border-radius: 3px; padding: 5px; }
                QGroupBox { border: 2px solid #80ff80; border-radius: 5px; margin-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            """
            )


def main():
    app = QApplication(sys.argv)
    window = MLFrameworkCheckerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

import sys
import os
import logging
import webbrowser
import subprocess
import psutil
import socket
import argparse
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QWidget,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QComboBox,
    QSizePolicy,
    QGroupBox,
    QGridLayout,
    QProgressBar,
    QStatusBar,
    QCheckBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette, QColor, QFont, QIcon

# Initialize logging
log_file = f"system_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class MLFrameworkChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Framework and CUDA Checker")
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

        self.torch_installed = False
        self.tensorflow_installed = False
        self.torch = None
        self.tf = None

        # Initialize theme
        self.current_theme = "Light"
        self.set_theme(self.current_theme)

        # Check PyTorch and TensorFlow installation on startup
        self.check_pytorch()
        self.check_tensorflow()

        # Start a timer to periodically update system information
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_system_info)
        self.update_timer.start(5000)  # Update every 5 seconds

    def init_ui(self):
        # Add theme selector to header
        theme_label = QLabel("Theme:")
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark", "Blue", "Green"])
        self.theme_selector.currentTextChanged.connect(self.set_theme)
        self.header_layout.addWidget(theme_label)
        self.header_layout.addWidget(self.theme_selector)
        self.header_layout.addStretch()

        # Main content
        self.pytorch_button, self.pytorch_label = self.add_button_and_label(
            "Check PyTorch", self.check_pytorch, 0, 0
        )
        self.tensorflow_button, self.tensorflow_label = self.add_button_and_label(
            "Check TensorFlow", self.check_tensorflow, 1, 0
        )
        self.cuda_button, self.cuda_label = self.add_button_and_label(
            "Check CUDA", self.check_cuda, 2, 0
        )

        self.system_specs_button = self.add_button(
            "Check System Specs", self.check_system_specs, 3, 0
        )
        self.system_label = QTextEdit("Click 'Check System Specs' to view details")
        self.system_label.setReadOnly(True)
        self.content_layout.addWidget(self.system_label, 4, 0, 1, 2)

        self.compatibility_button = self.add_button(
            "Check Compatibility", self.check_system_compatibility, 5, 0
        )
        self.compatibility_label = QTextEdit("")
        self.compatibility_label.setReadOnly(True)
        self.content_layout.addWidget(self.compatibility_label, 6, 0, 1, 2)

        self.add_button("Export Logs", self.export_logs, 7, 0)
        self.add_button("Instructions", self.display_faq, 7, 1)
        self.add_button(
            "Check PyTorch Updates",
            lambda: self.open_webpage("https://pytorch.org/get-started/locally/"),
            8,
            0,
        )
        self.add_button(
            "Check TensorFlow Updates",
            lambda: self.open_webpage("https://www.tensorflow.org/install"),
            8,
            1,
        )
        self.add_button(
            "Check CUDA Updates",
            lambda: self.open_webpage(
                "https://developer.nvidia.com/cuda-toolkit-archive"
            ),
            9,
            0,
        )

    def init_advanced_features(self):
        self.advanced_group = QGroupBox("Advanced Features")
        self.advanced_checkbox = QCheckBox("Enable Advanced Features")
        self.advanced_checkbox.setChecked(False)
        self.advanced_checkbox.toggled.connect(self.on_advanced_toggled)

        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(self.advanced_checkbox)

        self.persistence_mode_button = self.add_button_to_layout(
            "Enable NVIDIA Persistence Mode",
            self.enable_persistence_mode,
            advanced_layout,
        )
        self.gpu_logging_button = self.add_button_to_layout(
            "Start GPU Logging", self.start_gpu_logging, advanced_layout
        )

        self.persistence_mode_button.setVisible(False)
        self.gpu_logging_button.setVisible(False)

        self.advanced_group.setLayout(advanced_layout)
        self.content_layout.addWidget(self.advanced_group, 10, 0, 1, 2)

    def on_advanced_toggled(self, checked):
        if checked:
            reply = QMessageBox.warning(
                self,
                "Advanced Features",
                "Warning: These features are for advanced users only. "
                "Improper use may affect system stability or performance. "
                "Do not use these features unless you fully understand their implications.\n\n"
                "Do you want to proceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                self.advanced_checkbox.setChecked(False)
                return

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
        label = QLabel()
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
        self.current_theme = theme
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

    def update_system_info(self):
        self.check_system_specs()

    def open_webpage(self, url):
        webbrowser.open(url, new=2)

    def export_logs(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "", "Log Files (*.log)"
        )
        if file_path:
            with open(log_file, "r") as f:
                logs = f.read()
            with open(file_path, "w") as f:
                f.write(logs)
            QMessageBox.information(
                self, "Export Successful", f"Logs exported to {file_path}"
            )

    def display_faq(self):
        QMessageBox.information(
            self,
            "Instructions and Information",
            "Welcome to the ML Framework and CUDA Checker!\n\n"
            "This application helps you check and set up your environment for machine learning tasks. "
            "Here's what you can do:\n\n"
            "1. Check PyTorch: Verifies if PyTorch is installed and offers to install it if it's not.\n"
            "2. Check TensorFlow: Verifies if TensorFlow is installed and offers to install it if it's not.\n"
            "3. Check CUDA: Checks if CUDA is available on your system and shows the version.\n"
            "4. Check System Specs: Displays detailed information about your CPU, RAM, and GPUs.\n"
            "5. Check Compatibility: Ensures your system meets the minimum requirements for ML tasks.\n"
            "6. Export Logs: Saves all the check results and actions taken to a log file.\n"
            "7. Enable NVIDIA Persistence Mode: Improves GPU performance for long-running tasks.\n"
            "8. Start GPU Logging: Begins logging detailed GPU metrics for monitoring.\n"
            "9. Toggle Theme: Switches between light and dark themes for comfort.\n"
            "10. Check for Updates: Links to the official sites for PyTorch, TensorFlow, and CUDA updates.\n\n"
            "Important Notes:\n"
            "- Always ensure you have administrator rights when installing or updating software.\n"
            "- Keep your GPU drivers up-to-date for optimal performance.\n"
            "- If you encounter any errors, check the exported logs for more details.\n"
            "- For CUDA installation issues, refer to the NVIDIA documentation.\n"
            "- Persistence mode and GPU logging require NVIDIA GPUs and appropriate drivers.\n\n"
            "If you're new to machine learning setups, take your time to understand each component. "
            "Don't hesitate to seek help from the community forums if you encounter any difficulties.",
        )

    def install_package(self, package_name):
        self.status_bar.showMessage(f"Installing {package_name}...")
        progress = QProgressBar()
        progress.setRange(0, 0)
        self.status_bar.addPermanentWidget(progress)

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            logging.info(f"{package_name} installed successfully.")
            self.status_bar.showMessage(f"{package_name} installed successfully.", 5000)
            return True
        except Exception as e:
            logging.error(f"Failed to install {package_name}: {e}")
            self.status_bar.showMessage(
                f"Failed to install {package_name}. Check logs for details.", 5000
            )
            return False
        finally:
            self.status_bar.removeWidget(progress)

    def check_pytorch(self):
        self.status_bar.showMessage("Checking PyTorch...")
        try:
            import torch

            self.torch_installed = True
            self.torch = torch
            self.pytorch_label.setText(
                f"PyTorch is available. Version: {torch.__version__}"
            )
            self.status_bar.showMessage("PyTorch check completed.", 3000)
        except ImportError:
            self.pytorch_label.setText("PyTorch is not installed.")
            if (
                QMessageBox.question(
                    self,
                    "Install PyTorch",
                    "PyTorch is not installed. Would you like to install it now?",
                )
                == QMessageBox.Yes
            ):
                if self.install_package("torch"):
                    import torch

                    self.torch_installed = True
                    self.torch = torch
                    self.pytorch_label.setText(
                        f"PyTorch has been installed. Version: {torch.__version__}"
                    )
                else:
                    self.pytorch_label.setText(
                        "Failed to install PyTorch. Please install it manually."
                    )
            self.open_webpage("https://pytorch.org/")

    def check_tensorflow(self):
        self.status_bar.showMessage("Checking TensorFlow...")
        try:
            import tensorflow as tf

            self.tensorflow_installed = True
            self.tf = tf
            self.tensorflow_label.setText(
                f"TensorFlow is available. Version: {tf.__version__}"
            )
            self.status_bar.showMessage("TensorFlow check completed.", 3000)
        except ImportError:
            self.tensorflow_label.setText("TensorFlow is not installed.")
            if (
                QMessageBox.question(
                    self,
                    "Install TensorFlow",
                    "TensorFlow is not installed. Would you like to install it now?",
                )
                == QMessageBox.Yes
            ):
                if self.install_package("tensorflow"):
                    import tensorflow as tf

                    self.tensorflow_installed = True
                    self.tf = tf
                    self.tensorflow_label.setText(
                        f"TensorFlow has been installed. Version: {tf.__version__}"
                    )
                else:
                    self.tensorflow_label.setText(
                        "Failed to install TensorFlow. Please install it manually."
                    )
            self.open_webpage("https://www.tensorflow.org/install")

    def get_cuda_version(self):
        try:
            output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
            cuda_version = output.split("release ")[-1].split(",")[0]
            return cuda_version
        except:
            return None

    def check_cuda(self):
        self.status_bar.showMessage("Checking CUDA...")
        cuda_version = self.get_cuda_version()
        if cuda_version:
            self.cuda_label.setText(f"CUDA is available. Version: {cuda_version}")
            logging.info(f"CUDA is available. Version: {cuda_version}")
        else:
            self.cuda_label.setText("CUDA is not available or not detected.")
            warning_label = QLabel(
                "Warning: Ensure the CUDA version you download is supported by PyTorch and TensorFlow."
            )
            self.content_layout.addWidget(warning_label, 2, 2)
            logging.warning("CUDA is not available or not detected.")
            self.open_webpage("https://developer.nvidia.com/cuda-downloads")
        self.status_bar.showMessage("CUDA check completed.", 3000)

    def get_gpu_info(self):
        try:
            nvidia_smi_output = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,pci.bus_id,driver_version,vbios_version,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,pstate,pcie.link.gen.current,pcie.link.gen.max",
                        "--format=csv,noheader,nounits",
                    ]
                )
                .decode("utf-8")
                .strip()
                .split("\n")
            )

            gpu_info = []
            for line in nvidia_smi_output:
                values = line.split(", ")
                gpu_info.append(
                    {
                        "index": values[0],
                        "name": values[1],
                        "pci_bus_id": values[2],
                        "driver_version": values[3],
                        "vbios_version": values[4],
                        "memory_total": values[5],
                        "memory_free": values[6],
                        "memory_used": values[7],
                        "gpu_utilization": values[8],
                        "memory_utilization": values[9],
                        "temperature": values[10],
                        "power_draw": values[11],
                        "power_limit": values[12],
                        "sm_clock": values[13],
                        "memory_clock": values[14],
                        "pstate": values[15],
                        "pcie_link_gen_current": values[16],
                        "pcie_link_gen_max": values[17],
                    }
                )
            return gpu_info
        except Exception as e:
            logging.error(f"Error getting GPU info: {e}")
            return []

    def check_system_specs(self):
        self.status_bar.showMessage("Checking system specifications...")
        try:
            cpu_info = psutil.cpu_freq().max
            cpu_name = (
                subprocess.check_output("wmic cpu get name", shell=True)
                .decode()
                .strip()
                .split("\n")[1]
            )
            ram_info = "{:.2f}".format(psutil.virtual_memory().total / (1024**3))
            gpus = self.get_gpu_info()
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            system_info = f"CPU: {cpu_name}, Speed: {cpu_info} MHz\nRAM: {ram_info} GB\nHostname: {hostname}\nIP Address: {ip_address}\n\nGPU Information:\n"
            for gpu in gpus:
                system_info += (
                    f"GPU {gpu['index']}:\n"
                    f"  Name: {gpu['name']}\n"
                    f"  PCI Bus ID: {gpu['pci_bus_id']}\n"
                    f"  Driver Version: {gpu['driver_version']}\n"
                    f"  VBIOS Version: {gpu['vbios_version']}\n"
                    f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB\n"
                    f"  GPU Utilization: {gpu['gpu_utilization']}%\n"
                    f"  Memory Utilization: {gpu['memory_utilization']}%\n"
                    f"  Temperature: {gpu['temperature']}°C\n"
                    f"  Power Draw: {gpu['power_draw']}W / {gpu['power_limit']}W\n"
                    f"  SM Clock: {gpu['sm_clock']} MHz\n"
                    f"  Memory Clock: {gpu['memory_clock']} MHz\n"
                    f"  Performance State: P{gpu['pstate']}\n"
                    f"  PCIe Link Gen (Current/Max): {gpu['pcie_link_gen_current']} / {gpu['pcie_link_gen_max']}\n\n"
                )

            self.system_label.setText(system_info)
            logging.info(f"System specs: {system_info}")
            self.status_bar.showMessage("System specifications check completed.", 3000)
        except Exception as e:
            logging.error(f"Error checking system specs: {e}")
            self.system_label.setText(
                "Error checking system specs. Please check the logs for more details."
            )
            self.status_bar.showMessage(
                "Error checking system specifications. Check logs for details.", 5000
            )

    def enable_persistence_mode(self):
        self.status_bar.showMessage("Enabling NVIDIA persistence mode...")
        try:
            subprocess.run(["nvidia-smi", "-pm", "1"], check=True)
            QMessageBox.information(self, "Success", "NVIDIA persistence mode enabled.")
            logging.info("NVIDIA persistence mode enabled.")
            self.status_bar.showMessage("NVIDIA persistence mode enabled.", 3000)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to enable NVIDIA persistence mode: {e}"
            )
            logging.error(f"Failed to enable NVIDIA persistence mode: {e}")
            self.status_bar.showMessage(
                "Failed to enable NVIDIA persistence mode. Check logs for details.",
                5000,
            )

    def start_gpu_logging(self):
        self.status_bar.showMessage("Starting GPU logging...")
        try:
            log_file = f"gpu_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                    "--format=csv",
                    "-l",
                    "5",
                    "-f",
                    log_file,
                ]
            )
            QMessageBox.information(
                self, "Success", f"GPU logging started. Log file: {log_file}"
            )
            logging.info(f"GPU logging started. Log file: {log_file}")
            self.status_bar.showMessage("GPU logging started.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start GPU logging: {e}")
            logging.error(f"Failed to start GPU logging: {e}")
            self.status_bar.showMessage(
                "Failed to start GPU logging. Check logs for details.", 5000
            )

    def check_system_compatibility(self):
        self.status_bar.showMessage("Checking system compatibility...")
        self.compatibility_label.setText("Checking System Compatibility...")
        compatible = True
        messages = []

        if psutil.cpu_count() < 4:
            messages.append(
                "Your system has less than 4 CPU cores, which may affect performance."
            )
            compatible = False
        if psutil.virtual_memory().total < 8 * (1024**3):
            messages.append(
                "Your system has less than 8 GB of RAM, which may affect performance."
            )
            compatible = False
        if not self.get_cuda_version():
            messages.append(
                "CUDA is not detected, which may limit GPU acceleration capabilities."
            )
            compatible = False

        if compatible:
            self.compatibility_label.setText("Your system is compatible.")
            logging.info("System compatibility check passed.")
        else:
            self.compatibility_label.setText("\n".join(messages))
            logging.warning(f"System compatibility issues: {', '.join(messages)}")
        self.status_bar.showMessage("System compatibility check completed.", 3000)


def main():
    app = QApplication(sys.argv)
    window = MLFrameworkChecker()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Framework and CUDA Check")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()

    if args.cli:
        print("Running in CLI mode")
        checker = MLFrameworkChecker()
        checker.check_pytorch()
        checker.check_tensorflow()
        checker.check_cuda()
        checker.check_system_specs()
        checker.check_system_compatibility()
        print(f"Logs exported to {log_file}")
    else:
        main()

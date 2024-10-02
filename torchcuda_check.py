import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sv_ttk
import webbrowser
import logging
import os
import psutil
import socket
import subprocess
import sys
import argparse
import json
from datetime import datetime

# Initialize logging
log_file = f"system_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Initialize flags for PyTorch and TensorFlow availability
torch_installed = False
tensorflow_installed = False
torch = None
tf = None

# Try to import PyTorch and TensorFlow and set the flags if available
try:
    import torch

    torch_installed = True
    logging.info(f"PyTorch is available. Version: {torch.__version__}")
except ImportError:
    logging.warning("PyTorch is not installed.")

try:
    import tensorflow as tf

    tensorflow_installed = True
    logging.info(f"TensorFlow is available. Version: {tf.__version__}")
except ImportError:
    logging.warning("TensorFlow is not installed.")


def open_webpage(url):
    """Opens the specified URL in a web browser."""
    webbrowser.open(url, new=2)


def export_logs():
    """Exports the system check logs to a text file."""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".log",
        filetypes=[("Log files", "*.log"), ("All files", "*.*")],
    )
    if file_path:
        with open(log_file, "r") as f:
            logs = f.read()
        with open(file_path, "w") as f:
            f.write(logs)
        messagebox.showinfo("Export Successful", f"Logs exported to {file_path}")


def display_faq():
    """Displays comprehensive instructions and information about the application."""
    messagebox.showinfo(
        "Instructions and Information",
        "Welcome to the ML Framework and CUDA Checker!\n\n"
        "This application helps you set up and verify your environment for machine learning tasks. "
        "Here's a detailed guide on how to use each feature:\n\n"
        "1. Check PyTorch: Verifies if PyTorch is installed. If not, it offers to install it for you.\n"
        "   - Important for deep learning tasks and neural network modeling.\n\n"
        "2. Check TensorFlow: Verifies if TensorFlow is installed. If not, it offers to install it for you.\n"
        "   - Another crucial framework for machine learning and deep learning tasks.\n\n"
        "3. Check CUDA: Checks if CUDA is available on your system and displays the version.\n"
        "   - CUDA is essential for GPU acceleration in machine learning tasks.\n\n"
        "4. Check System Specs: Displays detailed information about your CPU, RAM, and GPUs.\n"
        "   - Helps you understand if your hardware meets the requirements for ML tasks.\n\n"
        "5. Check Compatibility: Ensures your system meets the minimum requirements for ML tasks.\n"
        "   - Alerts you if your system might have performance issues.\n\n"
        "6. Export Logs: Saves all the check results and actions taken to a log file.\n"
        "   - Useful for troubleshooting or sharing system information.\n\n"
        "7. Enable NVIDIA Persistence Mode: Improves GPU performance for long-running tasks.\n"
        "   - Recommended for extended ML training sessions.\n\n"
        "8. Start GPU Logging: Begins logging detailed GPU metrics for monitoring.\n"
        "   - Helpful for tracking GPU performance during ML tasks.\n\n"
        "9. Toggle Theme: Switches between light and dark themes for visual comfort.\n\n"
        "10. Check for Updates: Provides links to official sites for PyTorch, TensorFlow, and CUDA updates.\n"
        "    - Keeping your ML frameworks and CUDA up-to-date is crucial for compatibility and performance.\n\n"
        "Important Notes:\n"
        "- Ensure you have administrator rights when installing or updating software.\n"
        "- Keep your GPU drivers up-to-date for optimal performance.\n"
        "- If you encounter any errors, check the exported logs for more details.\n"
        "- For CUDA installation issues, refer to the NVIDIA documentation.\n"
        "- Persistence mode and GPU logging features require NVIDIA GPUs and appropriate drivers.\n\n"
        "If you're new to machine learning setups, take your time to understand each component. "
        "Don't hesitate to seek help from community forums or documentation if you encounter any difficulties.",
    )


def install_package(package_name):
    """Installs a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logging.info(f"{package_name} installed successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to install {package_name}: {e}")
        return False


def check_pytorch():
    """Checks if PyTorch is installed and updates the UI and logs accordingly."""
    global torch_installed, torch
    if torch_installed:
        torch_version = torch.__version__
        pytorch_label.configure(text=f"PyTorch is available. Version: {torch_version}")
    else:
        pytorch_label.configure(text="PyTorch is not installed.")
        if messagebox.askyesno(
            "Install PyTorch",
            "PyTorch is not installed. Would you like to install it now?",
        ):
            if install_package("torch"):
                import torch

                torch_installed = True
                pytorch_label.configure(
                    text=f"PyTorch has been installed. Version: {torch.__version__}"
                )
            else:
                pytorch_label.configure(
                    text="Failed to install PyTorch. Please install it manually."
                )
        open_webpage("https://pytorch.org/")


def check_tensorflow():
    """Checks if TensorFlow is installed and updates the UI and logs accordingly."""
    global tensorflow_installed, tf
    if tensorflow_installed:
        tf_version = tf.__version__
        tensorflow_label.configure(
            text=f"TensorFlow is available. Version: {tf_version}"
        )
    else:
        tensorflow_label.configure(text="TensorFlow is not installed.")
        if messagebox.askyesno(
            "Install TensorFlow",
            "TensorFlow is not installed. Would you like to install it now?",
        ):
            if install_package("tensorflow"):
                import tensorflow as tf

                tensorflow_installed = True
                tensorflow_label.configure(
                    text=f"TensorFlow has been installed. Version: {tf.__version__}"
                )
            else:
                tensorflow_label.configure(
                    text="Failed to install TensorFlow. Please install it manually."
                )
        open_webpage("https://www.tensorflow.org/install")


def get_cuda_version():
    """Gets the real CUDA version installed on the system."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        cuda_version = output.split("release ")[-1].split(",")[0]
        return cuda_version
    except:
        return None


def check_cuda():
    """Checks if CUDA is available and updates the UI and logs accordingly."""
    cuda_version = get_cuda_version()
    if cuda_version:
        cuda_label.configure(text=f"CUDA is available. Version: {cuda_version}")
        logging.info(f"CUDA is available. Version: {cuda_version}")
    else:
        cuda_label.configure(text="CUDA is not available or not detected.")
        cuda_warning_label.configure(
            text="Warning: Ensure the CUDA version you download is supported by PyTorch and TensorFlow."
        )
        logging.warning("CUDA is not available or not detected.")
        open_webpage("https://developer.nvidia.com/cuda-downloads")


def get_gpu_info():
    """Gets detailed information about all GPUs in the system using nvidia-smi."""
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


def check_system_specs():
    """Checks the system specifications and updates the UI and logs."""
    try:
        cpu_info = psutil.cpu_freq().max
        cpu_name = (
            subprocess.check_output("wmic cpu get name", shell=True)
            .decode()
            .strip()
            .split("\n")[1]
        )
        ram_info = "{:.2f}".format(psutil.virtual_memory().total / (1024**3))
        gpus = get_gpu_info()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        system_info = f"CPU: {cpu_name}, Speed: {cpu_info} MHz\nRAM: {ram_info} GB\nHostname: {hostname}\nIP Address: {ip_address}\n\nGPU Information:\n"
        for gpu in gpus:
            system_info += f"GPU {gpu['index']}:\n"
            system_info += f"  Name: {gpu['name']}\n"
            system_info += f"  PCI Bus ID: {gpu['pci_bus_id']}\n"
            system_info += f"  Driver Version: {gpu['driver_version']}\n"
            system_info += f"  VBIOS Version: {gpu['vbios_version']}\n"
            system_info += (
                f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB\n"
            )
            system_info += f"  GPU Utilization: {gpu['gpu_utilization']}%\n"
            system_info += f"  Memory Utilization: {gpu['memory_utilization']}%\n"
            system_info += f"  Temperature: {gpu['temperature']}°C\n"
            system_info += (
                f"  Power Draw: {gpu['power_draw']}W / {gpu['power_limit']}W\n"
            )
            system_info += f"  SM Clock: {gpu['sm_clock']} MHz\n"
            system_info += f"  Memory Clock: {gpu['memory_clock']} MHz\n"
            system_info += f"  Performance State: P{gpu['pstate']}\n"
            system_info += f"  PCIe Link Gen (Current/Max): {gpu['pcie_link_gen_current']} / {gpu['pcie_link_gen_max']}\n\n"

        system_label.configure(text=system_info)
        logging.info(f"System specs: {system_info}")
    except Exception as e:
        logging.error(f"Error checking system specs: {e}")
        system_label.configure(
            text="Error checking system specs. Please check the logs for more details."
        )


def enable_persistence_mode():
    """Enables NVIDIA persistence mode."""
    try:
        subprocess.run(["nvidia-smi", "-pm", "1"], check=True)
        messagebox.showinfo("Success", "NVIDIA persistence mode enabled.")
        logging.info("NVIDIA persistence mode enabled.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to enable NVIDIA persistence mode: {e}")
        logging.error(f"Failed to enable NVIDIA persistence mode: {e}")


def start_gpu_logging():
    """Starts GPU logging."""
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
        messagebox.showinfo("Success", f"GPU logging started. Log file: {log_file}")
        logging.info(f"GPU logging started. Log file: {log_file}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start GPU logging: {e}")
        logging.error(f"Failed to start GPU logging: {e}")


def check_system_compatibility():
    """Checks if the system meets the minimum requirements."""
    compatibility_label.configure(text="Checking System Compatibility...")
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
    if not get_cuda_version():
        messages.append(
            "CUDA is not detected, which may limit GPU acceleration capabilities."
        )
        compatible = False

    if compatible:
        compatibility_label.configure(text="Your system is compatible.")
        logging.info("System compatibility check passed.")
    else:
        compatibility_label.configure(text="\n".join(messages))
        logging.warning(f"System compatibility issues: {', '.join(messages)}")


def toggle_theme():
    """Toggles between light and dark themes."""
    if sv_ttk.get_theme() == "light":
        sv_ttk.set_theme("dark")
    else:
        sv_ttk.set_theme("light")


def main(cli_mode=False):
    if cli_mode:
        print("Running in CLI mode")
        check_pytorch()
        check_tensorflow()
        check_cuda()
        check_system_specs()
        check_system_compatibility()
        print(f"Logs exported to {log_file}")
    else:
        root = tk.Tk()
        root.title("ML Framework and CUDA Check")
        root.geometry("800x800")
        root.resizable(True, True)

        sv_ttk.set_theme("light")

        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(list(range(25)), weight=1)

        global pytorch_label, tensorflow_label, cuda_label, cuda_warning_label, system_label, compatibility_label

        pytorch_label = ttk.Label(root, text="Checking PyTorch...")
        pytorch_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Button(root, text="Check PyTorch", command=check_pytorch).grid(
            row=1, column=0, padx=10, pady=5, sticky="ew"
        )

        tensorflow_label = ttk.Label(root, text="Checking TensorFlow...")
        tensorflow_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Button(root, text="Check TensorFlow", command=check_tensorflow).grid(
            row=3, column=0, padx=10, pady=5, sticky="ew"
        )

        cuda_label = ttk.Label(root, text="Checking CUDA...")
        cuda_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        ttk.Button(root, text="Check CUDA", command=check_cuda).grid(
            row=5, column=0, padx=10, pady=5, sticky="ew"
        )

        cuda_warning_label = ttk.Label(root, text="")
        cuda_warning_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")

        system_label = ttk.Label(root, text="Checking System Specs...")
        system_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        ttk.Button(root, text="Check System Specs", command=check_system_specs).grid(
            row=8, column=0, padx=10, pady=5, sticky="ew"
        )

        compatibility_label = ttk.Label(root, text="")
        compatibility_label.grid(row=9, column=0, padx=10, pady=5, sticky="w")
        ttk.Button(
            root, text="Check Compatibility", command=check_system_compatibility
        ).grid(row=10, column=0, padx=10, pady=5, sticky="ew")

        ttk.Button(root, text="Export Logs", command=export_logs).grid(
            row=11, column=0, padx=10, pady=5, sticky="ew"
        )
        ttk.Button(root, text="Instructions", command=display_faq).grid(
            row=12, column=0, padx=10, pady=5, sticky="ew"
        )
        ttk.Button(
            root,
            text="Check PyTorch Updates",
            command=lambda: open_webpage("https://pytorch.org/get-started/locally/"),
        ).grid(row=13, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(
            root,
            text="Check TensorFlow Updates",
            command=lambda: open_webpage("https://www.tensorflow.org/install"),
        ).grid(row=14, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(
            root,
            text="Check CUDA Updates",
            command=lambda: open_webpage(
                "https://developer.nvidia.com/cuda-toolkit-archive"
            ),
        ).grid(row=15, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(root, text="Toggle Theme", command=toggle_theme).grid(
            row=16, column=0, padx=10, pady=5, sticky="ew"
        )

        ttk.Button(
            root, text="Enable NVIDIA Persistence Mode", command=enable_persistence_mode
        ).grid(row=17, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(root, text="Start GPU Logging", command=start_gpu_logging).grid(
            row=18, column=0, padx=10, pady=5, sticky="ew"
        )

        root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Framework and CUDA Check")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()

    main(cli_mode=args.cli)

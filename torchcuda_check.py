import tkinter as tk
from tkinter import ttk, messagebox
import sv_ttk
import webbrowser
import logging
import os
import psutil
import socket
import subprocess
import sys

# Initialize logging
logging.basicConfig(
    filename="system_check.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Initialize flag for PyTorch availability
torch_installed = False

# Try to import PyTorch and set the flag if available
try:
    import torch

    torch_installed = True
    logging.info(f"PyTorch is available. Version: {torch.__version__}")
except ImportError:
    logging.warning("PyTorch is not installed.")


def open_webpage(url):
    """
    Opens the specified URL in a web browser.

    Args:
        url (str): The URL to open.

    Returns:
        None
    """
    webbrowser.open(url, new=2)


def export_logs():
    """
    Exports the system check logs to a text file.

    Returns:
        None
    """
    with open("system_check.log", "r") as f:
        logs = f.read()
    with open("exported_logs.txt", "w") as f:
        f.write(logs)


def display_faq():
    """
    Displays a message box with instructions on how to use the application.

    Returns:
        None
    """
    messagebox.showinfo(
        "Instructions",
        "1. Click 'Check PyTorch' to check if PyTorch is installed.\n"
        "2. Click 'Check CUDA' to check if CUDA is available.\n"
        "3. Click 'Check System Specs' to view your system specifications.\n"
        "4. Click 'Export Logs' to export the logs.\n"
        "5. Click 'Check Compatibility' to ensure your system meets the requirements.\n"
        "6. Use 'Check Updates' to find the latest versions of PyTorch and CUDA.",
    )


def install_pytorch():
    """
    Installs PyTorch using pip and updates the UI and logs accordingly.

    Returns:
        None
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        import torch

        global torch_installed
        torch_installed = True
        pytorch_label.configure(
            text=f"PyTorch has been installed. Version: {torch.__version__}"
        )
        logging.info(f"PyTorch installed successfully. Version: {torch.__version__}")
    except Exception as e:
        logging.error(f"Failed to install PyTorch: {e}")
        pytorch_label.configure(
            text="Failed to install PyTorch. Please install it manually from https://pytorch.org/"
        )


def check_pytorch():
    """
    Checks if PyTorch is installed and updates the UI and logs accordingly.
    Prompts the user to install PyTorch if it is not installed.

    Returns:
        None
    """
    if torch_installed:
        torch_version = torch.__version__
        pytorch_label.configure(text=f"PyTorch is available. Version: {torch_version}")
    else:
        pytorch_label.configure(text="PyTorch is not installed.")
        open_webpage("https://pytorch.org/")
        if messagebox.askyesno(
            "Install PyTorch",
            "PyTorch is not installed. Would you like to install it now?",
        ):
            install_pytorch()


def check_cuda():
    """
    Checks if CUDA is available and updates the UI and logs accordingly.
    Prompts the user to download the appropriate CUDA version if it is not available.

    Returns:
        None
    """
    if torch_installed:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            cuda_label.configure(text=f"CUDA is available. Version: {cuda_version}")
        else:
            cuda_label.configure(text="CUDA is not available.")
            cuda_warning_label.configure(
                text="Warning: Ensure the CUDA version you download is supported by PyTorch."
            )
            open_webpage("https://developer.nvidia.com/cuda-downloads")
    else:
        cuda_label.configure(text="Skipping CUDA check as PyTorch is not installed.")
        cuda_warning_label.configure(
            text="Warning: Ensure the CUDA version you download is supported by PyTorch."
        )
        open_webpage("https://pytorch.org/get-started/locally/")


def check_system_specs():
    """
    Checks the system specifications including CPU, RAM, GPU, hostname, and IP address.
    Updates the UI and logs the results.

    Returns:
        None
    """
    try:
        cpu_info = psutil.cpu_freq().max
        with os.popen("wmic cpu get name") as f:
            cpu_name = f.read().strip()
        ram_info = "{:.2f}".format(psutil.virtual_memory().total / (1024**3))
        with os.popen("wmic path win32_VideoController get name") as f:
            gpu_info = f.read().strip()
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        system_label.configure(
            text=f"CPU: {cpu_name}, Speed: {cpu_info} MHz\nRAM: "
            f"{ram_info} GB\nGPU: {gpu_info}\nIP Address: {ip_address}"
        )
    except Exception as e:
        logging.error(f"Error checking system specs: {e}")
        system_label.configure(
            text="Error checking system specs. Please check the logs for more details."
        )


def check_system_compatibility():
    """
    Checks if the system meets the minimum requirements for running PyTorch and CUDA.
    Updates the UI and logs the results.

    Returns:
        None
    """
    compatibility_label.configure(text="Checking System Compatibility...")
    compatible = True
    if psutil.cpu_count() < 4:
        compatibility_label.configure(
            text="Your system has less than 4 CPU cores, which may affect performance."
        )
        compatible = False
    if psutil.virtual_memory().total < 8 * (1024**3):
        compatibility_label.configure(
            text="Your system has less than 8 GB of RAM, which may affect performance."
        )
        compatible = False

    if compatible:
        compatibility_label.configure(text="Your system is compatible.")
    else:
        compatibility_label.configure(text="Your system may not be fully compatible.")


def toggle_theme():
    """
    Toggles between light and dark themes.

    Returns:
        None
    """
    if sv_ttk.get_theme() == "light":
        sv_ttk.set_theme("dark")
    else:
        sv_ttk.set_theme("light")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("TorchCUDA Check")
    root.geometry("500x800")
    root.resizable(True, True)

    # Set the initial theme to light
    sv_ttk.set_theme("light")

    # Configure the grid to make the window layout flexible
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(list(range(15)), weight=1)

    # PyTorch check UI elements
    pytorch_label = ttk.Label(root, text="Checking PyTorch...")
    pytorch_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    pytorch_button = ttk.Button(root, text="Check PyTorch", command=check_pytorch)
    pytorch_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    # CUDA check UI elements
    cuda_label = ttk.Label(root, text="Checking CUDA...")
    cuda_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
    cuda_button = ttk.Button(root, text="Check CUDA", command=check_cuda)
    cuda_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    # CUDA warning label
    cuda_warning_label = ttk.Label(root, text="")
    cuda_warning_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

    # System specs check UI elements
    system_label = ttk.Label(root, text="Checking System Specs...")
    system_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
    system_button = ttk.Button(
        root, text="Check System Specs", command=check_system_specs
    )
    system_button.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

    # System compatibility check UI elements
    compatibility_label = ttk.Label(root, text="")
    compatibility_label.grid(row=7, column=0, padx=10, pady=10, sticky="w")
    compatibility_button = ttk.Button(
        root, text="Check Compatibility", command=check_system_compatibility
    )
    compatibility_button.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

    # Export logs button
    export_logs_button = ttk.Button(root, text="Export Logs", command=export_logs)
    export_logs_button.grid(row=9, column=0, padx=10, pady=10, sticky="ew")

    # FAQ button
    faq_button = ttk.Button(root, text="Instructions", command=display_faq)
    faq_button.grid(row=10, column=0, padx=10, pady=10, sticky="ew")

    # Check for PyTorch updates button
    pytorch_update_button = ttk.Button(
        root,
        text="Check PyTorch Updates",
        command=lambda: open_webpage("https://pytorch.org/get-started/locally/"),
    )
    pytorch_update_button.grid(row=11, column=0, padx=10, pady=10, sticky="ew")

    # Check for CUDA updates button
    cuda_update_button = ttk.Button(
        root,
        text="Check CUDA Updates",
        command=lambda: open_webpage(
            "https://developer.nvidia.com/cuda-toolkit-archive"
        ),
    )
    cuda_update_button.grid(row=12, column=0, padx=10, pady=10, sticky="ew")

    # Toggle theme button
    toggle_theme_button = ttk.Button(root, text="Toggle Theme", command=toggle_theme)
    toggle_theme_button.grid(row=13, column=0, padx=10, pady=10, sticky="ew")

    root.mainloop()

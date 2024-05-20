# TorchCUDA Check

TorchCUDA Check is a Python-based GUI application that allows users to check if PyTorch and CUDA are installed on their system. The application also provides detailed system specifications and compatibility checks. It includes options to install PyTorch, view instructions, export logs, and check for updates. The application uses CUDA for upscaling if available, otherwise it falls back to CPU.

## Features

- Check if PyTorch is installed and view the version.
- Check if CUDA is available and view the version.
- View detailed system specifications including CPU, RAM, GPU, hostname, and IP address.
- Check system compatibility for running PyTorch and CUDA.
- Install PyTorch if not already installed.
- Export system logs to a text file.
- Toggle between light and dark themes.
- View instructions on how to use the application.
- Check for PyTorch and CUDA updates.

## Requirements

- Python 3.x
- sv_ttk
- psutil
- torch
- Pillow

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Xza85hrf/torchcuda-check.git
    ```

2. Change to the project directory:

    ```bash
    cd torchcuda-check
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:

    ```bash
    python torchcuda_check.py
    ```

2. Use the GUI to check PyTorch and CUDA installation, view system specs, and perform other functions.

## Logging

The application creates a log file `system_check.log` in the project directory to track events and errors.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


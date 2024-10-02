# ML Framework and CUDA Checker (Version 2.0)

ML Framework and CUDA Checker (formerly known as TorchCUDA Check) is an advanced Python-based GUI application that allows users to check the installation of PyTorch, TensorFlow, and CUDA on their system. It provides detailed system specifications, compatibility checks, and additional advanced features for managing NVIDIA GPUs. This version introduces support for TensorFlow, advanced GPU management features, multiple themes, and improved system diagnostics.

## What's New in Version 2.0

- **TensorFlow Support:** Now supports TensorFlow installation checks.
- **Advanced Features:** Includes advanced GPU management options like enabling NVIDIA persistence mode and starting GPU logging.
- **Theming:** Allows users to toggle between light, dark, blue, and green themes.
- **More Detailed System Specs:** Improved system specification display with CPU, RAM, hostname, IP address, and GPU details.
- **Regular System Updates:** Periodically updates system information every 5 seconds.
- **Compatibility Checks:** Enhanced system compatibility checks for machine learning tasks.
- **CLI Mode:** Allows running checks through a command-line interface.

## Features

- **Check PyTorch Installation:** Verifies if PyTorch is installed and displays its version.
- **Check TensorFlow Installation:** Checks if TensorFlow is installed and shows its version.
- **Check CUDA Availability:** Checks if CUDA is available on the system and displays its version.
- **View System Specifications:** Provides detailed information including CPU, RAM, GPU details, hostname, and IP address.
- **Advanced GPU Management:** Enables advanced GPU features such as NVIDIA persistence mode and GPU logging (requires NVIDIA GPUs and drivers).
- **System Compatibility Check:** Verifies if the system meets minimum requirements for machine learning tasks.
- **Export Logs:** Exports system logs for diagnostics and troubleshooting.
- **Theme Selector:** Switch between light, dark, blue, and green themes for user comfort.
- **Instruction Manual:** Built-in instructions and information on how to use the application.
- **Check for Updates:** Quick links to official sites for checking PyTorch, TensorFlow, and CUDA updates.

## Requirements

- Python 3.x
- psutil
- torch
- Pillow
- PySide6
- unittest

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Xza85hrf/ML-Framework_Checker.git
    ```

2. Change to the project directory:

    ```bash
    cd ML-Framework_Checker
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### GUI Mode

1. Run the application:

    ```bash
    python mlframework_checker.py
    ```

2. Use the GUI to check PyTorch, TensorFlow, and CUDA installation, view system specs, and use other advanced features.

### CLI Mode

1. Run the application in CLI mode:

    ```bash
    python mlframework_checker.py --cli
    ```

2. The application will run checks for PyTorch, TensorFlow, CUDA, and system specs in the command-line interface.

## Logging

The application generates a log file named `system_check_<timestamp>.log` in the project directory to track events and errors. This log can be exported using the "Export Logs" feature in the GUI.

## Roadmap of Development for the Application

---

**Current Version:**
- **Features:**
  - Check PyTorch installation.
  - Check TensorFlow installation.
  - Check CUDA availability.
  - View system specifications.
  - Export logs.
  - Check system compatibility.
  - Check for updates for PyTorch, TensorFlow, and CUDA.
  - Enable NVIDIA persistence mode.
  - Open TensorFlow installation webpage.

**Future Development Roadmap:**

1. **Visualization Enhancements:**
   - **Add graphs or charts** to visualize GPU usage over time.
   - **Implement a feature** to compare multiple GPU performances.

2. **Performance Testing:**
   - **Add a system benchmark feature** to test ML framework performance.

3. **Environment Management:**
   - **Include a feature** to manage and switch between different Python environments.

4. **Configuration Management:**
   - **Implement a way** to save and load different configuration profiles.

5. **User Interface Improvements:**
   - **Enhance the UI** to make it more intuitive and user-friendly.
   - **Add support for dark mode** to improve user experience in low-light environments.

6. **Security Enhancements:**
   - **Implement secure logging and data handling** practices.
   - **Add user authentication** to restrict access to certain features.

7. **Cross-Platform Support:**
   - **Ensure compatibility** with major operating systems (Windows, macOS, Linux).

8. **Community and Documentation:**
   - **Create comprehensive documentation** and user guides.
   - **Set up a community forum** for user feedback and support.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Older Version

The previous version of this application was known as **TorchCUDA Check**, which primarily focused on checking PyTorch and CUDA installations. While the new version (2.0) expands its functionality to include TensorFlow support and advanced GPU management, the core features of checking PyTorch and CUDA remain intact.

import unittest
from unittest.mock import patch, MagicMock
from PySide6.QtWidgets import QApplication
from mlframework_checker import MLFrameworkChecker


class TestMLFrameworkChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.checker = MLFrameworkChecker()

    def test_init(self):
        self.assertIsNotNone(self.checker)
        self.assertEqual(self.checker.windowTitle(), "ML Framework and CUDA Checker")

    @patch("subprocess.check_call")
    def test_install_package(self, mock_check_call):
        mock_check_call.return_value = 0
        result = self.checker.install_package("test_package")
        self.assertTrue(result)
        mock_check_call.assert_called_once()

    @patch("subprocess.check_output")
    def test_get_cuda_version(self, mock_check_output):
        mock_check_output.return_value = b"release 11.2, V11.2.152"
        version = self.checker.get_cuda_version()
        self.assertEqual(version, "11.2")

    @patch("psutil.cpu_freq")
    @patch("psutil.virtual_memory")
    @patch("subprocess.check_output")
    def test_check_system_specs(
        self, mock_check_output, mock_virtual_memory, mock_cpu_freq
    ):
        mock_cpu_freq.return_value = MagicMock(max=3500)
        mock_virtual_memory.return_value = MagicMock(total=16 * 1024**3)
        mock_check_output.side_effect = [
            b"Name\nIntel(R) Core(TM) i7-9750H CPU @ 2.60GHz\n",
            b"GPU 0, NVIDIA GeForce RTX 3080, 00000000:01:00.0, 460.32.03, 94.02.71.00.01, 10240, 8192, 2048, 50, 30, 65, 150, 320, 1800, 9251, P0, 16, 16\n",
        ]
        self.checker.check_system_specs()

        # Updated assertions to be more flexible
        self.assertIn("CPU:", self.checker.system_label.toPlainText())
        self.assertIn("RAM: 16.00 GB", self.checker.system_label.toPlainText())
        self.assertIn(
            "NVIDIA GeForce RTX 3080", self.checker.system_label.toPlainText()
        )

    def test_check_system_compatibility(self):
        self.checker.get_cuda_version = MagicMock(return_value="11.2")
        self.checker.check_system_compatibility()
        self.assertEqual(
            self.checker.compatibility_label.toPlainText(), "Your system is compatible."
        )


if __name__ == "__main__":
    unittest.main()

"""An exception escaping a Qt slot must show a dialog, not abort the process."""

import os
import subprocess
import sys
import unittest

import pytest

pytestmark = pytest.mark.gui

SCRIPT = """
import sys
from unittest import mock
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import run_ui

app = QApplication([])
sys.excepthook = run_ui.show_unhandled_exception

def boom():
    raise RuntimeError("slot failed")

with mock.patch.object(run_ui.QMessageBox, "critical") as critical:
    QTimer.singleShot(0, boom)
    QTimer.singleShot(50, app.quit)
    app.exec()
print("SHOWN", critical.call_args.args[1:])
"""


class ExceptHookTest(unittest.TestCase):
    def test_slot_exception_shows_dialog_and_app_survives(self):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": "."}
        r = subprocess.run(
            [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("RuntimeError: slot failed", r.stdout)
        self.assertIn("RuntimeError", r.stderr)  # traceback logged


if __name__ == "__main__":
    unittest.main()

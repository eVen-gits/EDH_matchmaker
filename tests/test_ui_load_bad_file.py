"""GUI regression test for File > Load tournament on a corrupt file (#31).

Loading a truncated/corrupt log must not crash the app (PyQt6 aborts the
process on any exception escaping a slot) and must leave the current
tournament and TournamentAction.LOGF unchanged. Qt runs offscreen.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from uuid import uuid4

import pytest

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


class LoadBadFileTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication

            import run_ui
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        self.run_ui = run_ui
        self.app = QApplication.instance() or QApplication([])
        self.addCleanup(setattr, TournamentAction, "LOGF", False)

    def test_load_corrupt_file_shows_error_and_keeps_state(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        t.add_player(["Alice", "Bob", "Carol"])
        window = self.run_ui.MainWindow(t)
        TournamentAction.LOGF = "logs/default.json"

        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as f:
                f.write('{"config": {"pod_sizes": [4')

            with (
                mock.patch.object(
                    self.run_ui.QFileDialog,
                    "getOpenFileName",
                    return_value=(bad, "*.json"),
                ),
                mock.patch.object(
                    self.run_ui.QMessageBox, "critical"
                ) as mock_critical,
            ):
                window.load_tour()

            mock_critical.assert_called_once()
            self.assertIs(window.core, t)
            self.assertEqual(TournamentAction.LOGF, "logs/default.json")

    def test_load_two_player_commander_save_is_refused(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        window = self.run_ui.MainWindow(t)
        data = t.serialize()
        data["uid"] = str(uuid4())  # not the cached tournament
        data["config"]["pod_sizes"] = [2]

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.json")
            with open(path, "w") as f:
                json.dump(data, f)
            with (
                mock.patch.object(
                    self.run_ui.QFileDialog,
                    "getOpenFileName",
                    return_value=(path, "*.json"),
                ),
                mock.patch.object(
                    self.run_ui.QMessageBox, "critical"
                ) as mock_critical,
            ):
                for attempt in range(2):
                    with self.subTest(attempt=attempt):
                        window.load_tour()
                        self.assertEqual(mock_critical.call_count, attempt + 1)
                        self.assertIn("pod_sizes", str(mock_critical.call_args))
                        self.assertIs(window.core, t)


@pytest.mark.gui
class StartupLoadBadFileTest(unittest.TestCase):
    def test_open_corrupt_file_exits_with_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            content = '{"config": {"pod_sizes": [4'
            with open(bad, "w") as f:
                f.write(content)
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=".")
            proc = subprocess.run(
                [sys.executable, "run_ui.py", "-o", bad],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            with open(bad) as f:
                self.assertEqual(f.read(), content)

        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn(f"Failed to load tournament from {bad}", proc.stderr)


if __name__ == "__main__":
    unittest.main()

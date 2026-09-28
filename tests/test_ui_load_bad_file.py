"""GUI regression test for File > Load tournament on a corrupt file (#31).

Loading a truncated/corrupt log must not crash the app (PyQt6 aborts the
process on any exception escaping a slot) and must leave the current
tournament and TournamentAction.LOGF unchanged. Qt runs offscreen.
"""

import os
import tempfile
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()

"""GUI regression test for File > Save As (#22). Qt runs offscreen."""

import json
import os
import tempfile
import unittest
from unittest import mock

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


class SaveAsTest(unittest.TestCase):
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

    def test_save_as_writes_chosen_file(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        t.add_player(["Alice", "Bob", "Carol"])
        window = self.run_ui.MainWindow(t)
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "saved")
            with mock.patch.object(
                self.run_ui.QFileDialog,
                "getSaveFileName",
                return_value=(target, "*.json"),
            ):
                window.save_as()
            self.assertEqual(TournamentAction.LOGF, target + ".json")
            with open(target + ".json") as f:
                self.assertEqual(len(json.load(f)["players"]), 3)


if __name__ == "__main__":
    unittest.main()

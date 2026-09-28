"""GUI regression test for File > Load players with blank lines (#26). Qt runs offscreen."""

import os
import tempfile
import unittest
from unittest import mock

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


class LoadPlayersTest(unittest.TestCase):
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

    def test_load_players_skips_blank_lines(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        window = self.run_ui.MainWindow(t)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "players.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Alice\nBob\n\nCarol\n\n")
            with mock.patch.object(
                self.run_ui.QFileDialog,
                "getOpenFileName",
                return_value=(path, "*.txt"),
            ):
                # Must not raise / abort the process (issue #26).
                window.load_players()
            self.assertEqual(
                sorted(p.name for p in t.players), ["Alice", "Bob", "Carol"]
            )
            window.grab().save(os.path.join(d, "load_players_ok.png"))


if __name__ == "__main__":
    unittest.main()

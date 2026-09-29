"""Startup regression for a missing --open log path."""

import os
import runpy
import sys
import tempfile
import unittest
from unittest import mock

from src.core import TournamentAction

TournamentAction.LOGF = False


class TestMissingOpen(unittest.TestCase):
    def test_missing_open_starts_with_round(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing.json")
            with mock.patch.object(sys, "argv", ["run_ui.py", "-o", path]), mock.patch.object(
                QApplication, "exec", return_value=0
            ), mock.patch.object(sys, "exit"):
                state = runpy.run_path("run_ui.py", run_name="__main__")
            core = state["core"]
            self.assertIsNotNone(core.tour_round)
            core.add_player("Alice")
            self.assertIsNotNone(core.tour_round.dropped_players)
            self.assertEqual(TournamentAction.LOGF, path)
            state["window"].close()

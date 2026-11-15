"""GUI test for the New Seed button next to Create/Reset Pods (offscreen Qt)."""

import os
import unittest
from typing import Any, cast

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


class NewSeedUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PyQt6.QtWidgets import QApplication

            import run_ui
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        cls.app = QApplication.instance() or QApplication([])
        cls.run_ui = run_ui

    def test_reset_keeps_pods_and_new_seed_changes_them(self):
        t = Tournament(TournamentConfiguration(auto_export=False, n_rounds=3))
        t.add_player([f"P{i}" for i in range(16)])
        window = self.run_ui.MainWindow(t)
        ui = cast(Any, window.ui)

        def r() -> Any:
            return t.tour_round

        def pods():
            return sorted(sorted(map(str, pod._players)) for pod in r().pods)

        self.assertTrue(ui.pb_new_seed.isEnabled())
        ui.pb_pods.click()
        first, seed = pods(), r().seed
        self.assertFalse(ui.pb_new_seed.isEnabled())

        ui.pb_reset_pods.click()
        self.assertTrue(ui.pb_new_seed.isEnabled())
        ui.pb_pods.click()
        self.assertEqual(pods(), first)
        self.assertEqual(r().seed, seed)

        ui.pb_reset_pods.click()
        ui.pb_new_seed.click()
        new_seed = window._next_seed
        self.assertIsNotNone(new_seed)
        self.assertNotEqual(new_seed, seed)
        ui.pb_pods.click()
        self.assertEqual(r().seed, new_seed)
        self.assertNotEqual(pods(), first)
        self.assertIsNone(window._next_seed)

        ui.pb_reset_pods.click()
        ui.pb_pods.click()
        self.assertEqual(r().seed, new_seed)


if __name__ == "__main__":
    unittest.main()

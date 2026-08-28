import os
import unittest

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False


class TestConfigDialogPairingRows(unittest.TestCase):
    """The config dialog's per-round pairing dropdowns behave correctly."""

    @classmethod
    def setUpClass(cls):
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PyQt6.QtWidgets import QApplication, QWidget

            import run_ui
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        cls.app = QApplication.instance() or QApplication([])
        cls.run_ui = run_ui
        cls.QWidget = QWidget

    def _dialog(self, **cfg):
        parent = self.QWidget()
        parent.core = Tournament(TournamentConfiguration(auto_export=False, **cfg))
        return self.run_ui.TournamentConfigDialog(parent), parent

    def _picks(self, dlg):
        return [c.currentData() for c in dlg._pairing_combos]

    def test_defaults_follow_adaptive_scheme(self):
        dlg, _ = self._dialog(n_rounds=3, snake_pods=True)
        self.assertEqual(
            self._picks(dlg), ["PairingRandom", "PairingSnake", "PairingDefault"]
        )

    def test_snake_off_changes_round2_default(self):
        dlg, _ = self._dialog(n_rounds=3, snake_pods=False)
        self.assertEqual(self._picks(dlg)[1], "PairingDefault")

    def test_resize_preserves_existing_picks(self):
        dlg, _ = self._dialog(n_rounds=3)
        combo = dlg._pairing_combos[2]
        combo.setCurrentIndex(combo.findData("PairingRandom"))
        dlg.ui.sb_nRounds.setValue(5)
        picks = self._picks(dlg)
        self.assertEqual(len(picks), 5)
        self.assertEqual(picks[2], "PairingRandom")  # kept
        self.assertEqual(picks[3], "PairingDefault")  # new row, adaptive

    def test_edit_mode_seeds_from_config_and_applies(self):
        dlg, parent = self._dialog(
            n_rounds=3,
            pairing_logics=["PairingSnake", "PairingDefault", "PairingRandom"],
        )
        dlg.reset = False
        self.assertEqual(
            self._picks(dlg), ["PairingSnake", "PairingDefault", "PairingRandom"]
        )
        first = dlg._pairing_combos[0]
        first.setCurrentIndex(first.findData("PairingDefault"))
        dlg.apply_choices()
        self.assertEqual(parent.core.config.pairing_logics[0], "PairingDefault")

    def test_top_cut_pairings_not_offered(self):
        dlg, _ = self._dialog(n_rounds=2)
        combo = dlg._pairing_combos[0]
        offered = [combo.itemData(i) for i in range(combo.count())]
        self.assertNotIn("PairingTop4", offered)
        self.assertIn("PairingRandom", offered)


if __name__ == "__main__":
    unittest.main()

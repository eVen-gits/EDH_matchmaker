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
        # No snake-pods checkbox any more (removed): a Commander dialog
        # always preselects Random, Snake, Default for a fresh tournament.
        dlg, _ = self._dialog(n_rounds=3)
        self.assertEqual(
            self._picks(dlg), ["PairingRandom", "PairingSnake", "PairingDefault"]
        )

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


class TestConfigDialogRuleset(unittest.TestCase):
    """The Game selector reconfigures pod sizes, scoring, top cut, byes,
    and pairing rows for the selected ruleset."""

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

    def _select_ruleset(self, dlg, name):
        dlg.ui.cb_ruleset.setCurrentIndex(dlg.ui.cb_ruleset.findData(name))

    def test_ruleset_combo_lists_both_rulesets(self):
        dlg, _ = self._dialog()
        offered = [
            dlg.ui.cb_ruleset.itemData(i) for i in range(dlg.ui.cb_ruleset.count())
        ]
        self.assertIn("CommanderRuleset", offered)
        self.assertIn("Mtg1v1Ruleset", offered)

    def test_switch_to_mtg_reconfigures_the_dialog(self):
        dlg, _ = self._dialog()
        self._select_ruleset(dlg, "Mtg1v1Ruleset")

        self.assertEqual(dlg._pod_size_editor.values(), [2])
        self.assertFalse(dlg._pod_size_editor.isEnabled())
        self.assertEqual(dlg.ui.cb_scoringLogic.currentData(), "Scoring1v1")
        top_cuts = [
            dlg.ui.cb_topCut.itemData(i) for i in range(dlg.ui.cb_topCut.count())
        ]
        self.assertEqual(
            top_cuts,
            [
                TournamentConfiguration.TopCut.NONE,
                TournamentConfiguration.TopCut.TOP_2,
                TournamentConfiguration.TopCut.TOP_4,
                TournamentConfiguration.TopCut.TOP_8,
                TournamentConfiguration.TopCut.TOP_16,
            ],
        )
        # isHidden(), not isVisible(): the dialog itself is never shown in
        # this test, so isVisible() is always False regardless of our
        # setVisible() calls (it also depends on ancestor visibility).
        self.assertTrue(dlg.ui.cb_allow_bye.isHidden())
        self.assertTrue(dlg.ui.cb_allow_bye.isChecked())
        for combo in dlg._pairing_combos:
            offered = [combo.itemData(i) for i in range(combo.count())]
            self.assertIn("Pairing1v1", offered)
            self.assertNotIn("PairingSnake", offered)  # Commander-only sizes
        scoring_offered = [
            dlg.ui.cb_scoringLogic.itemData(i)
            for i in range(dlg.ui.cb_scoringLogic.count())
        ]
        self.assertIn("Scoring1v1", scoring_offered)

    def test_commander_does_not_offer_1v1_scoring(self):
        dlg, _ = self._dialog()
        scoring_offered = [
            dlg.ui.cb_scoringLogic.itemData(i)
            for i in range(dlg.ui.cb_scoringLogic.count())
        ]
        self.assertNotIn("Scoring1v1", scoring_offered)

    def test_switch_back_to_commander_restores_defaults(self):
        dlg, _ = self._dialog()
        self._select_ruleset(dlg, "Mtg1v1Ruleset")
        self._select_ruleset(dlg, "CommanderRuleset")

        self.assertTrue(dlg._pod_size_editor.isEnabled())
        self.assertEqual(dlg.ui.cb_scoringLogic.currentData(), "ScoringDefault")
        self.assertFalse(dlg.ui.cb_allow_bye.isHidden())
        self._picks = [c.currentData() for c in dlg._pairing_combos]
        self.assertIn("PairingRandom", self._picks)
        scoring_offered = [
            dlg.ui.cb_scoringLogic.itemData(i)
            for i in range(dlg.ui.cb_scoringLogic.count())
        ]
        self.assertNotIn("Scoring1v1", scoring_offered)

    def test_mtg_playoff_ruleset_params_applied(self):
        dlg, parent = self._dialog()
        self._select_ruleset(dlg, "Mtg1v1Ruleset")
        dlg.ui.sb_nRounds.setValue(2)
        dlg.ui.cb_topCut.setCurrentIndex(
            dlg.ui.cb_topCut.findData(TournamentConfiguration.TopCut.TOP_8)
        )
        # Round 2's games_to_win -> 1 (Bo1).
        round2_form = dlg._ruleset_forms[1]
        self.assertIsNotNone(round2_form)
        round2_form._fields["games_to_win"].setValue(1)
        # The final (stage 2)'s games_to_win -> 3 (Bo5).
        final_form = dlg._playoff_forms[2]
        final_form._fields["games_to_win"].setValue(3)

        dlg.apply_choices()

        self.assertEqual(
            parent.core.config.pairing_rounds[0].get("ruleset_params", {}), {}
        )
        self.assertEqual(
            parent.core.config.pairing_rounds[1]["ruleset_params"],
            {"games_to_win": 1},
        )
        self.assertEqual(
            parent.core.config.playoff_rounds,
            {2: {"ruleset_params": {"games_to_win": 3}}},
        )

    def test_edit_mode_keeps_ruleset_and_export_fields(self):
        t = Tournament(
            TournamentConfiguration(ruleset="Mtg1v1Ruleset", auto_export=False)
        )
        parent = self.QWidget()
        parent.core = t
        dlg = self.run_ui.TournamentConfigDialog(parent, reset=False)
        self.assertEqual(dlg.ui.cb_ruleset.currentData(), "Mtg1v1Ruleset")
        dlg.apply_choices()
        self.assertEqual(t.config.ruleset, "Mtg1v1Ruleset")
        self.assertEqual(
            t.config.standings_export.fields,
            list(TournamentConfiguration(ruleset="Mtg1v1Ruleset").standings_export.fields),
        )

    def test_ruleset_combo_disabled_once_pairings_exist(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        t.new_round()
        t.add_player([f"P{i}" for i in range(8)])
        t.create_pairings()
        parent = self.QWidget()
        parent.core = t
        dlg = self.run_ui.TournamentConfigDialog(parent, reset=False)
        self.assertFalse(dlg.ui.cb_ruleset.isEnabled())


if __name__ == "__main__":
    unittest.main()

"""GUI tests for the MTG 1v1 additions to run_ui.py.

Covers MatchReportDialog, game-loss routing, error handling (UILog.with_status
catching ValueError), round_label/window title, and the OMW/GW/OGW columns in
the player list. Qt tests run offscreen - see CLAUDE.md.
"""

import os
import unittest
from unittest import mock

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


def _load_run_ui():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    import run_ui

    app = QApplication.instance() or QApplication([])
    return run_ui, app


class MtgUiTestCase(unittest.TestCase):
    """Base class: a MainWindow around a 2-player Mtg1v1Ruleset pod."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.run_ui, cls.app = _load_run_ui()
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")

    def _window(self, **cfg):
        t = Tournament(
            TournamentConfiguration(
                ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=3, **cfg
            )
        )
        t.add_player(["Alice", "Bob"])
        t.new_round()
        t.manual_pod(list(t.players))
        window = self.run_ui.MainWindow(t)
        pod = t.tour_round.pods[0]
        alice, bob = pod.players
        return window, t, pod, alice, bob

    def _playoff_final_pod(self):
        """A 4-player, single-Swiss-round, top-cut-2 tournament, already cut
        to its TOP_2 final pod."""
        t = Tournament(
            TournamentConfiguration(
                ruleset="Mtg1v1Ruleset",
                auto_export=False,
                n_rounds=1,
                top_cut=2,
            )
        )
        t.add_player(["Alice", "Bob", "Carol", "Dave"])
        t.new_round()
        t.create_pairings()
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])
        t.create_pairings()  # advances to the TOP_2 final
        window = self.run_ui.MainWindow(t)
        final_pod = t.tour_round.pods[0]
        return window, t, final_pod


class TestMatchReportDialog(MtgUiTestCase):
    def test_valid_score_enables_ok(self):
        window, t, pod, alice, bob = self._window()
        dlg = self.run_ui.MatchReportDialog(window, pod)
        dlg._spins[alice].setValue(2)
        dlg._spins[bob].setValue(1)
        self.assertTrue(dlg.pb_ok.isEnabled())
        self.assertIn("wins 2-1", dlg.lbl_status.text())

    def test_overshoot_in_bo3_disables_ok(self):
        window, t, pod, alice, bob = self._window()
        dlg = self.run_ui.MatchReportDialog(window, pod)
        dlg._spins[alice].setValue(3)  # spin box max is games_to_win (2)
        self.assertEqual(dlg._spins[alice].value(), 2)
        dlg._spins[bob].setValue(0)
        self.assertTrue(dlg.pb_ok.isEnabled())

    def test_tied_score_in_playoff_disables_ok(self):
        window, t, final_pod = self._playoff_final_pod()
        a, b = final_pod.players
        dlg = self.run_ui.MatchReportDialog(window, final_pod)
        dlg._spins[a].setValue(1)
        dlg._spins[b].setValue(1)
        self.assertFalse(dlg.pb_ok.isEnabled())
        self.assertIn("cannot end in a draw", dlg.lbl_status.text())

    def test_prefill_matches_earlier_report(self):
        window, t, pod, alice, bob = self._window()
        t.report_win(alice)  # 2-0
        dlg = self.run_ui.MatchReportDialog(window, pod)
        self.assertEqual(dlg._spins[alice].value(), 2)
        self.assertEqual(dlg._spins[bob].value(), 0)

    def test_accept_reports_the_match(self):
        # show_dialog's OK path is just app.report_match(pod, dlg._games());
        # exercise that directly rather than mocking QDialog.exec's modal loop.
        window, t, pod, alice, bob = self._window()
        dlg = self.run_ui.MatchReportDialog(window, pod)
        dlg._spins[alice].setValue(2)
        dlg._spins[bob].setValue(1)
        self.assertTrue(dlg.pb_ok.isEnabled())
        window.report_match(pod, dlg._games())
        self.assertEqual(pod.result_type.name, "WIN")
        self.assertIn(alice, pod.result)


class TestGameLossRouting(MtgUiTestCase):
    def test_seated_mtg_player_opens_score_dialog(self):
        window, t, pod, alice, bob = self._window()
        with mock.patch.object(
            self.run_ui.MatchReportDialog, "exec", return_value=0
        ) as mock_exec:
            window.game_loss([alice])
        mock_exec.assert_called_once()

    def test_commander_player_still_toggles_game_loss(self):
        t = Tournament(TournamentConfiguration(auto_export=False, n_rounds=1))
        t.add_player(["A", "B", "C", "D"])
        t.new_round()
        t.manual_pod(list(t.players))
        window = self.run_ui.MainWindow(t)
        player = next(iter(t.players))
        window.game_loss([player])
        self.assertIn(player.uid, t.tour_round._game_loss)


class TestErrorHandling(MtgUiTestCase):
    def test_report_draw_in_playoff_warns_instead_of_raising(self):
        window, t, final_pod = self._playoff_final_pod()
        a, b = final_pod.players
        with mock.patch("run_ui.QMessageBox.warning") as mock_warning:
            window.report_draw([a, b])
        mock_warning.assert_called_once()


class TestRoundLabelAndTitle(MtgUiTestCase):
    def test_round_label_swiss_and_playoff(self):
        window, t, pod, alice, bob = self._window()
        self.assertEqual(self.run_ui.round_label(t.tour_round), "Round 1")

    def test_window_title_includes_ruleset_and_round(self):
        window, t, pod, alice, bob = self._window()
        self.assertIn("Mtg1v1", window.windowTitle())
        self.assertIn("Round 1", window.windowTitle())


class TestPlayerListTiebreakerColumns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.run_ui, cls.app = _load_run_ui()
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")

    def test_mtg_player_list_shows_omw_commander_does_not(self):
        t = Tournament(
            TournamentConfiguration(
                ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=1
            )
        )
        t.add_player(["Alice", "Bob"])
        t.new_round()
        t.manual_pod(list(t.players))
        t.report_win(next(iter(t.players)))
        window = self.run_ui.MainWindow(t)
        window.ui_update_player_list()
        # item.text() is overridden (PlayerListItem.text) to recompute with
        # its own default tokens rather than return the stored display text
        # - read DisplayRole directly to see what setText() actually put there.
        texts = [
            window.ui.lv_players.item(i).data(self.run_ui.Qt.ItemDataRole.DisplayRole)
            for i in range(window.ui.lv_players.count())
        ]
        self.assertTrue(any("OMW" in text for text in texts))

        t2 = Tournament(TournamentConfiguration(auto_export=False, n_rounds=1))
        t2.add_player(["A", "B", "C", "D"])
        t2.new_round()
        t2.manual_pod(list(t2.players))
        window2 = self.run_ui.MainWindow(t2)
        window2.ui_update_player_list()
        texts2 = [
            window2.ui.lv_players.item(i).data(self.run_ui.Qt.ItemDataRole.DisplayRole)
            for i in range(window2.ui.lv_players.count())
        ]
        self.assertFalse(any("OMW" in text for text in texts2))


if __name__ == "__main__":
    unittest.main()

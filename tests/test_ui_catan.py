"""GUI tests for the Catan additions to run_ui.py (offscreen Qt, see CLAUDE.md).

Covers CatanReportDialog (routing from Report score, OK gating, per-player VP
reaching report_match), the Catan standings columns in the player list, and
the config dialog's allow_bye default on a ruleset change.
"""

import os
import unittest
from typing import Any, cast
from unittest import mock

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


class CatanUiTestCase(unittest.TestCase):
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

    def _window(self):
        t = Tournament(
            TournamentConfiguration(ruleset="CatanRuleset", auto_export=False, n_rounds=3)
        )
        t.add_player(["Ann", "Bo", "Cy", "Di"])
        t.new_round()
        t.manual_pod(list(t.players))
        window = self.run_ui.MainWindow(t)
        assert t.tour_round is not None
        return window, t, t.tour_round.pods[0]


class TestCatanReportDialog(CatanUiTestCase):
    def test_report_score_opens_the_catan_dialog(self):
        window, t, pod = self._window()
        widget = self.run_ui.PodWidget(window, pod)
        with mock.patch.object(
            self.run_ui.CatanReportDialog, "exec", return_value=0
        ) as catan, mock.patch.object(
            self.run_ui.MatchReportDialog, "exec", return_value=0
        ) as mtg:
            widget.report_score()
        catan.assert_called_once()
        mtg.assert_not_called()
        self.assertTrue(self.run_ui._vp_scored(pod))
        self.assertFalse(self.run_ui._scored(pod))

    def test_ok_disabled_until_valid(self):
        window, t, pod = self._window()
        dlg = self.run_ui.CatanReportDialog(window, pod)
        a, b, c, d = pod.players
        self.assertEqual(list(dlg._spins), list(pod.players))  # seat order
        self.assertFalse(dlg.pb_ok.isEnabled())  # all zero: a tie
        dlg._spins[a].setValue(7)  # most VP, but under the target
        self.assertFalse(dlg.pb_ok.isEnabled())
        dlg._spins[a].setValue(10)
        self.assertTrue(dlg.pb_ok.isEnabled())
        self.assertIn("wins with 10 VP", dlg.lbl_status.text())
        dlg._spins[b].setValue(10)  # a second player at the target
        self.assertFalse(dlg.pb_ok.isEnabled())

    def test_accept_reports_per_player_vp(self):
        window, t, pod = self._window()
        a, b, c, d = pod.players
        def fill_and_accept(dlg):
            for p, vp in zip((a, b, c, d), (4, 10, 6, 3)):
                dlg._spins[p].setValue(vp)
            return 1

        with mock.patch.object(
            self.run_ui.CatanReportDialog, "exec", fill_and_accept
        ), mock.patch.object(window, "report_match") as rm:
            self.run_ui.CatanReportDialog.show_dialog(window, pod)
        got_pod, games = rm.call_args.args
        self.assertIs(got_pod, pod)
        (game,) = games
        self.assertEqual(game.winners, frozenset({b.uid}))
        self.assertEqual(dict(game.scores), {a.uid: 4, b.uid: 10, c.uid: 6, d.uid: 3})

    def test_report_reaches_the_tournament(self):
        window, t, pod = self._window()
        dlg = self.run_ui.CatanReportDialog(window, pod)
        dlg._spins[pod.players[2]].setValue(11)
        games = dlg._games()
        assert games is not None
        window.report_match(pod, games)
        self.assertEqual(pod.games[-1].winners, frozenset({pod.players[2].uid}))

    def test_rereport_prefills_the_recorded_vp(self):
        window, t, pod = self._window()
        dlg = self.run_ui.CatanReportDialog(window, pod)
        dlg._spins[pod.players[0]].setValue(10)
        dlg._spins[pod.players[1]].setValue(5)
        games = dlg._games()
        assert games is not None
        window.report_match(pod, games)
        again = self.run_ui.CatanReportDialog(window, pod)
        self.assertEqual(again._spins[pod.players[0]].value(), 10)
        self.assertEqual(again._spins[pod.players[1]].value(), 5)

    def test_commander_pod_is_not_vp_scored(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        t.add_player([f"P{i}" for i in range(4)])
        t.new_round()
        t.manual_pod(list(t.players))
        assert t.tour_round is not None
        self.assertFalse(self.run_ui._vp_scored(t.tour_round.pods[0]))


class TestCatanStandingsColumns(CatanUiTestCase):
    def test_player_list_shows_catan_columns(self):
        window, t, pod = self._window()
        window.ui_update_player_list()
        assert window.ui is not None
        texts = [
            window.ui.lv_players.item(i).data(self.run_ui.Qt.ItemDataRole.DisplayRole)
            for i in range(window.ui.lv_players.count())
        ]
        self.assertTrue(texts)
        for text in texts:
            for header in ("VP ", "VP% ", "2nd ", "3rd "):
                self.assertIn(f"| {header}", text)


class TestConfigDialogAllowBye(CatanUiTestCase):
    def test_allow_bye_follows_ruleset_default(self):
        parent = self.QWidget()
        cast(Any, parent).core = Tournament(TournamentConfiguration(auto_export=False))
        dlg = self.run_ui.TournamentConfigDialog(parent)
        assert dlg.ui is not None
        combo = dlg.ui.cb_ruleset

        def pick(name):
            combo.setCurrentIndex(combo.findData(name))

        self.assertTrue(dlg.ui.cb_allow_bye.isChecked())
        pick("CatanRuleset")
        self.assertFalse(dlg.ui.cb_allow_bye.isChecked())
        pick("CommanderRuleset")
        self.assertTrue(dlg.ui.cb_allow_bye.isChecked())

"""Renaming a player: core API and the player-list context menu (issue #30)."""

import os
import unittest
from unittest import mock

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


def _tour():
    t = Tournament(TournamentConfiguration(auto_export=False))
    t.add_player(["Alice", "Bob", "Carol", "Dave"])
    t.new_round()
    t.create_pairings()
    return t


class TestRenamePlayerCore(unittest.TestCase):
    def test_rename_updates_pods_and_logs_old_name(self):
        t = _tour()
        alice = next(p for p in t.players if p.name == "Alice")
        t.rename_player(alice, "Alicia")
        self.assertEqual(alice.name, "Alicia")
        self.assertIn(
            "Alicia", [p.name for pod in t.tour_round.pods for p in pod.players]
        )
        self.assertIn("Renamed player Alice to Alicia", _last_log())

    def test_empty_names_refused(self):
        t = _tour()
        alice = next(p for p in t.players if p.name == "Alice")
        for bad in ("", "   "):
            with self.assertRaises(ValueError):
                t.rename_player(alice, bad)
        self.assertEqual(alice.name, "Alice")

    def test_rename_to_existing_name_allowed(self):
        t = _tour()
        alice = next(p for p in t.players if p.name == "Alice")
        t.rename_player(alice, "Bob")
        self.assertEqual(sum(p.name == "Bob" for p in t.players), 2)


def _last_log():
    from src.core import Log

    return str(Log.output[-1])


class TestRenamePlayerUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication

            import run_ui
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        cls.run_ui = run_ui
        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        t = _tour()
        w = self.run_ui.MainWindow(t)
        w.ui.lv_players.setCurrentRow(0)
        return w, t

    def test_rename_via_slot_updates_list(self):
        w, t = self._window()
        old = w.ui.lv_players.currentItem().data(0x0100).name
        w.lva_rename_player("Zed")
        names = [
            w.ui.lv_players.item(i).data(0x0100).name
            for i in range(w.ui.lv_players.count())
        ]
        self.assertIn("Zed", names)
        self.assertNotIn(old, names)
        out = os.environ.get("RENAME_SHOT")
        if out:
            w.ui.lv_players.grab().save(out)

    def test_empty_name_shows_message_not_raise(self):
        w, t = self._window()
        with mock.patch.object(self.run_ui.QMessageBox, "warning") as warn:
            w.lva_rename_player("  ")
        warn.assert_called_once()

    def test_menu_has_rename_entry(self):
        w, t = self._window()
        seen = []
        with mock.patch.object(
            self.run_ui.QMenu, "exec", lambda m, *a: seen.extend(x.text() for x in m.actions())
        ):
            pos = w.ui.lv_players.visualItemRect(w.ui.lv_players.item(0)).center()
            w.lv_players_rightclick_menu(pos)
        self.assertIn("Rename player...", seen)


if __name__ == "__main__":
    unittest.main()

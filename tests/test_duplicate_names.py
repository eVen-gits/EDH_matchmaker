"""Two players may share a name; the UUID is their identity."""

import csv
import io
import os
import unittest
from unittest import mock
from uuid import uuid4

import pytest

from src.core import (
    IPlayer,
    IPod,
    IRound,
    StandingsExport,
    Tournament,
    TournamentAction,
    TournamentConfiguration,
)

TournamentAction.LOGF = False  # type: ignore


def _tour():
    t = Tournament(TournamentConfiguration(pod_sizes=[4], allow_bye=False, auto_export=False))
    t.add_player(["Alex", "Alex", "Bob", "Carol"])
    return t


class TestDuplicateNames(unittest.TestCase):
    def test_same_name_players_are_added(self):
        t = _tour()
        alexes = [p for p in t.players if p.name == "Alex"]
        self.assertEqual(len(alexes), 2)
        self.assertNotEqual(alexes[0].uid, alexes[1].uid)

    def test_duplicate_uid_raises(self):
        t = _tour()
        uid = uuid4()
        t.add_player(("Dana", uid))
        with self.assertRaisesRegex(ValueError, str(uid)):
            t.add_player(("Eve", uid))

    def test_failing_batch_adds_nobody(self):
        t = _tour()
        before = len(t.players)
        existing = next(iter(t.players)).uid
        uid = uuid4()
        with self.assertRaises(ValueError):
            t.add_player([("Dana", uuid4()), ("Eve", existing)])
        with self.assertRaises(ValueError):
            t.add_player([("Dana", uid), ("Eve", uid)])
        self.assertEqual(len(t.players), before)

    def test_paired_scored_exported_distinctly(self):
        t = _tour()
        t.create_pairings()
        seated = [p for pod in t.tour_round.pods for p in pod.players]
        self.assertEqual(len(seated), 4)
        t.random_results()
        alexes = [p for p in t.players if p.name == "Alex"]
        labels = {p.display_name for p in alexes}
        self.assertEqual(len(labels), 2)
        for label in labels:
            self.assertTrue(label.startswith("Alex #"))
        self.assertEqual(next(p for p in t.players if p.name == "Bob").display_name, "Bob")
        out = t.get_standings_str(
            fields=[StandingsExport.Field.NAME], style=StandingsExport.Format.CSV
        )
        names = [row[0] for row in csv.reader(io.StringIO(out))][1:]
        self.assertEqual(sorted(names), sorted(labels | {"Bob", "Carol"}))

    def test_survive_save_and_reload(self):
        t = _tour()
        t.create_pairings()
        t.random_results()
        data = t.serialize()
        # Reload into a fresh tournament instead of the cached one.
        for cache in (Tournament.CACHE, IPlayer.CACHE, IPod.CACHE, IRound.CACHE):
            cache.clear()
        t2 = Tournament.inflate(data)
        self.assertIsNot(t2, t)
        alexes = [p for p in t2.players if p.name == "Alex"]
        self.assertEqual(len(alexes), 2)
        self.assertEqual(
            {p.uid for p in alexes}, {p.uid for p in t.players if p.name == "Alex"}
        )
        self.assertEqual(len({p.display_name for p in alexes}), 2)


@pytest.mark.gui
class TestDuplicateNamesUi(unittest.TestCase):
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

    def test_add_same_name_in_gui_lists_both(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        t.new_round()
        w = self.run_ui.MainWindow(t)
        with mock.patch.object(self.run_ui.QMessageBox, "warning"):
            w.add_player("Alex")
            w.add_player("Alex")
            w.add_player("Bob")
        texts = [w.ui.lv_players.item(i).data(0) for i in range(w.ui.lv_players.count())]
        out = os.environ.get("DUP_SHOT")
        if out:
            w.ui.lv_players.grab().save(out)
        self.assertEqual(len(texts), 3)
        self.assertEqual(sum("Alex #" in s for s in texts), 2)


if __name__ == "__main__":
    unittest.main()

import unittest
from typing import cast
from uuid import uuid4

from src.core import Pod, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult

TournamentAction.LOGF = False  # type: ignore


class TestGameScores(unittest.TestCase):
    def test_default_none_and_hash_order_independent(self):
        a, b = uuid4(), uuid4()
        self.assertIsNone(IGameResult(frozenset({a})).scores)
        g1 = IGameResult(frozenset({a}), ((a, 10), (b, 7)))
        g2 = IGameResult(frozenset({a}), {b: 7, a: 10})  # pyright: ignore
        self.assertEqual(g1, g2)
        self.assertEqual(hash(g1), hash(g2))
        self.assertNotEqual(g1, IGameResult(frozenset({a})))

    def test_round_trip_and_remove(self):
        t = Tournament(TournamentConfiguration(pod_sizes=[4], auto_export=False))
        t.add_player([f"P{i}" for i in range(4)])
        t.create_pairings()
        assert t.tour_round is not None
        pod = t.tour_round.pods[0]
        a, b, c, d = (p.uid for p in pod.players)
        pod._games = [IGameResult(frozenset({a}), ((a, 10), (b, 7), (c, 5), (d, 3)))]

        data = pod.serialize()
        self.assertEqual(len(data["games"][0]["scores"]), 4)
        self.assertNotIn("scores", Pod._serialize_game(IGameResult(frozenset({a}))))

        ser = t.serialize()
        Tournament.CACHE.clear()
        t2 = cast(Tournament, Tournament.inflate(ser))
        assert t2.tour_round is not None
        self.assertEqual(t2.tour_round.pods[0].games, pod.games)
        self.assertEqual(t2.serialize()["format_version"], "1.3")

        pod.remove_result(pod.players[1])
        self.assertEqual(pod.games[0].scores, tuple(sorted(((a, 10), (c, 5), (d, 3)))))

    def test_reads_1_2_file(self):
        t = Tournament(TournamentConfiguration(pod_sizes=[4], auto_export=False))
        t.add_player([f"P{i}" for i in range(4)])
        t.create_pairings()
        t.report_win(t.tour_round.pods[0].players[0])  # pyright: ignore
        data = t.serialize()
        data["format_version"] = "1.2"
        Tournament.CACHE.clear()
        t2 = cast(Tournament, Tournament.inflate(data))
        assert t2.tour_round is not None
        self.assertIsNone(t2.tour_round.pods[0].games[0].scores)


if __name__ == "__main__":
    unittest.main()

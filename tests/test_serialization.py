import unittest
import os

from src.core import Player, Tournament, TournamentAction, TournamentConfiguration
from src.misc import generate_player_names

TournamentAction.LOGF = False  # type: ignore


class TestSerialization(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TournamentConfiguration(
            pod_sizes=[4, 3],
            allow_bye=True,
            auto_export=False,
        )
        self.t = Tournament(self.config)
        self.n_players = 64
        self.t.add_player([f"P{i}" for i in range(self.n_players)])

    def test_tournament_serialization(self):
        self.t.create_pairings()

        assert self.t.tour_round is not None
        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

        serialized = self.t.serialize()

        old_uid = self.t.uid
        Tournament.CACHE.clear()

        t_inflated = Tournament.inflate(serialized)

        self.assertEqual(t_inflated.uid, old_uid)
        self.assertEqual(len(t_inflated.players), self.n_players)
        self.assertEqual(len(t_inflated.rounds), 1)

        orig_standings = self.t.get_standings()
        inflated_standings = t_inflated.get_standings()

        self.assertEqual(len(orig_standings), len(inflated_standings))
        for p1, p2 in zip(orig_standings, inflated_standings):
            self.assertEqual(p1.uid, p2.uid)
            self.assertEqual(
                p1.rating(self.t.tour_round), p2.rating(t_inflated.tour_round)
            )

    def test_tournament_action_store_load(self):
        import tempfile

        self.t.initialize_round()
        self.t.add_player([f"P{i}" for i in range(10)])
        self.t.create_pairings()

        assert self.t.tour_round is not None
        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

        with tempfile.TemporaryDirectory() as tmpdirname:
            log_path = os.path.join(tmpdirname, "test_log.json")
            try:
                TournamentAction.LOGF = log_path

                TournamentAction.store(self.t)

                old_uid = self.t.uid
                Tournament.CACHE.clear()

                t_loaded = TournamentAction.load(log_path)

                self.assertIsNotNone(t_loaded)
                assert t_loaded is not None
                self.assertEqual(t_loaded.uid, old_uid)
                self.assertEqual(len(t_loaded.players), len(self.t.players))
                self.assertEqual(len(t_loaded.rounds), 1)
            finally:
                TournamentAction.LOGF = False

    def test_tournament_serialization_edge_cases(self):
        self.config = TournamentConfiguration(
            pod_sizes=[4],
            allow_bye=True,
            auto_export=False,
        )
        self.t = Tournament(self.config)
        self.t.initialize_round()
        players = self.t.add_player([f"P{i}" for i in range(14)])
        self.t.create_pairings()

        assert self.t.tour_round is not None
        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])
            self.t.report_draw([pod.players[1], pod.players[2]])

        player_to_drop = self.t.tour_round.pods[0].players[3]
        self.t.drop_player(player_to_drop)

        player_for_gameloss = self.t.tour_round.pods[1].players[3]
        self.t.toggle_game_loss(player_for_gameloss)

        serialized = self.t.serialize()

        Tournament.CACHE.clear()

        t_inflated = Tournament.inflate(serialized)

        assert t_inflated is not None
        self.assertEqual(len(t_inflated.players), 14)

        inflated_drop = [p for p in t_inflated.players if p.uid == player_to_drop.uid][0]
        self.assertIn(inflated_drop, t_inflated.tour_round.dropped_players)

        inflated_gl = [p for p in t_inflated.players if p.uid == player_for_gameloss.uid][0]
        self.assertEqual(
            inflated_gl.result(t_inflated.tour_round), Player.EResult.LOSS
        )

    def test_load_real_tournament_file(self):
        log_path = "logs/tournament-state-699863dcebe4eb89e31bc50b-2026-02-25.json"
        if not os.path.exists(log_path):
            self.skipTest("Real tournament file not available")

        Tournament.CACHE.clear()
        t_loaded = TournamentAction.load(log_path)

        self.assertIsNotNone(t_loaded)
        assert t_loaded is not None
        self.assertGreater(len(t_loaded.players), 0)
        self.assertGreater(len(t_loaded.rounds), 0)
        self.assertIsNotNone(t_loaded.tour_round)

    def test_tour_round_none_safe(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        self.assertIsNone(t.tour_round)

    def test_double_load_no_collision(self):
        config = TournamentConfiguration(pod_sizes=[4], auto_export=False)
        t = Tournament(config)
        t.initialize_round()
        t.add_player([f"DL{i}" for i in range(8)])
        t.create_pairings()
        assert t.tour_round is not None
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        serialized = t.serialize()

        t1 = Tournament.inflate(serialized)
        assert t1 is not None
        self.assertEqual(len(t1.players), 8)

        t2 = Tournament.inflate(serialized)
        assert t2 is not None
        self.assertEqual(len(t2.players), 8)
        self.assertEqual(t1.uid, t2.uid)

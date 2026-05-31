import unittest
import os

from src.core import Player, Round, Tournament, TournamentAction, TournamentConfiguration
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


class TestSerializationTopCut(unittest.TestCase):
    """Serialize → inflate preserves top-cut config and state so tournaments are stateless."""

    # ------------------------------------------------------------------ helpers

    def _make_tournament(
        self,
        top_cut: TournamentConfiguration.TopCut,
        n_players: int = 16,
        n_rounds: int = 2,
    ) -> Tournament:
        cfg = TournamentConfiguration(
            pod_sizes=[4],
            allow_bye=False,
            auto_export=False,
            n_rounds=n_rounds,
            top_cut=top_cut,
        )
        t = Tournament(cfg)
        t.add_player([f"P{i}" for i in range(n_players)])
        return t

    def _run_swiss(self, t: Tournament, n: int) -> None:
        for _ in range(n):
            t.create_pairings()
            t.random_results()

    def _reload(self, t: Tournament) -> Tournament:
        """Serialize + inflate into a fresh instance (simulates loading from .json)."""
        serialized = t.serialize()
        Tournament.CACHE.clear()
        reloaded = Tournament.inflate(serialized)
        assert isinstance(reloaded, Tournament)
        return reloaded

    # ------------------------------------------------------------------ config

    def test_top_cut_config_preserved_for_all_values(self) -> None:
        """config.top_cut survives serialize/inflate for every TopCut variant."""
        for tc in TournamentConfiguration.TopCut:
            with self.subTest(top_cut=tc):
                t = self._make_tournament(tc)
                t2 = self._reload(t)
                self.assertEqual(t2.config.top_cut, tc)

    def test_n_rounds_preserved(self) -> None:
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4, n_rounds=5)
        t2 = self._reload(t)
        self.assertEqual(t2.config.n_rounds, 5)

    # ------------------------------------------------------------------ round state

    def test_round_stage_preserved_before_results(self) -> None:
        """Round.stage survives inflate when top-cut round has no results yet."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        t.create_pairings()
        self.assertEqual(t.tour_round.stage, Round.Stage.TOP_4)

        t2 = self._reload(t)
        self.assertEqual(t2.tour_round.stage, Round.Stage.TOP_4)

    def test_round_logic_name_preserved(self) -> None:
        """Round._logic (pairing logic name) survives inflate."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        t.create_pairings()
        logic_name = t.tour_round._logic

        t2 = self._reload(t)
        self.assertEqual(t2.tour_round._logic, logic_name)

    def test_disabled_players_preserved(self) -> None:
        """Non-advancing players remain disabled after reload."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        t.create_pairings()
        disabled_before = {p.uid for p in t.tour_round.disabled_players}

        t2 = self._reload(t)
        disabled_after = {p.uid for p in t2.tour_round.disabled_players}
        self.assertEqual(disabled_before, disabled_after)
        self.assertEqual(len(t2.tour_round.active_players), 4)

    def test_byes_preserved_in_top7_round(self) -> None:
        """Bye assignments in a TOP_7 round survive reload."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_7)
        self._run_swiss(t, 2)
        t.create_pairings()
        byes_before = {p.uid for p in t.tour_round.byes}
        self.assertEqual(len(byes_before), 3)

        t2 = self._reload(t)
        byes_after = {p.uid for p in t2.tour_round.byes}
        self.assertEqual(byes_before, byes_after)

    # ------------------------------------------------------------------ continuability

    def test_top4_completable_after_reload(self) -> None:
        """TOP_4 tournament reloaded mid-top-cut round can be completed."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        t.create_pairings()  # TOP_4 round, no results yet

        t2 = self._reload(t)
        self.assertEqual(len(t2.tour_round.pods), 1)
        self.assertEqual(len(t2.tour_round.pods[0].players), 4)

        t2.random_results()
        ok = t2.create_pairings()
        self.assertFalse(ok)  # tournament over

    def test_top7_reload_between_stages(self) -> None:
        """TOP_7 tournament reloaded after TOP_7 results can advance to TOP_4."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_7)
        self._run_swiss(t, 2)
        t.create_pairings()  # TOP_7 stage
        t.random_results()

        t2 = self._reload(t)
        # TOP_7 round is done; next create_pairings should open TOP_4
        ok = t2.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(t2.tour_round.stage, Round.Stage.TOP_4)
        self.assertEqual(len(t2.tour_round.active_players), 4)

        t2.random_results()
        ok = t2.create_pairings()
        self.assertFalse(ok)  # tournament over

    def test_top16_reload_between_stages(self) -> None:
        """TOP_16 tournament reloaded after TOP_16 results can advance to TOP_4."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_16)
        self._run_swiss(t, 2)
        t.create_pairings()  # TOP_16 stage
        t.random_results()

        t2 = self._reload(t)
        ok = t2.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(t2.tour_round.stage, Round.Stage.TOP_4)

    def test_tournament_complete_flag_preserved(self) -> None:
        """A fully completed tournament cannot create new rounds after reload."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        t.create_pairings()
        t.random_results()  # finish TOP_4

        t2 = self._reload(t)
        ok = t2.create_pairings()
        self.assertFalse(ok)

    # ------------------------------------------------------------------ standings

    def test_standings_order_preserved(self) -> None:
        """Player standings (order + ratings) after Swiss are identical after reload."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        final_swiss = t.last_round
        standings_before = t.get_standings(final_swiss)

        t2 = self._reload(t)
        final_swiss_2 = t2.rounds[final_swiss.seq]
        standings_after = t2.get_standings(final_swiss_2)

        self.assertEqual(len(standings_before), len(standings_after))
        for p1, p2 in zip(standings_before, standings_after):
            self.assertEqual(p1.uid, p2.uid)
            self.assertEqual(
                p1.rating(final_swiss), p2.rating(final_swiss_2)
            )

    def test_top4_active_players_match_standings_leaders(self) -> None:
        """After reload, the 4 active players in TOP_4 are still the top 4 from Swiss."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        final_swiss = t.last_round
        top4_uids = {p.uid for p in t.get_standings(final_swiss)[:4]}

        t.create_pairings()
        t2 = self._reload(t)

        active_uids = {p.uid for p in t2.tour_round.active_players}
        self.assertEqual(active_uids, top4_uids)

    # ------------------------------------------------------------------ reset + config after reload

    def test_reset_and_config_change_after_reload(self) -> None:
        """reset_pods() + config change on a reloaded tournament picks up the new config."""
        t = self._make_tournament(TournamentConfiguration.TopCut.TOP_4)
        self._run_swiss(t, 2)
        t.create_pairings()  # TOP_4 round created

        t2 = self._reload(t)
        self.assertEqual(t2.tour_round.stage, Round.Stage.TOP_4)

        # Change to TOP_7 after reload, then reset and repairq
        t2.config = TournamentConfiguration(
            pod_sizes=[4],
            allow_bye=False,
            auto_export=False,
            n_rounds=2,
            top_cut=TournamentConfiguration.TopCut.TOP_7,
        )
        t2.reset_pods()
        ok = t2.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(t2.tour_round.stage, Round.Stage.TOP_7)
        self.assertEqual(len(t2.tour_round.active_players), 7)

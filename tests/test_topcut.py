"""Tests for top-cut (playoff) tournament stages.

Setup pattern: n_rounds=2 with 16 players keeps Swiss fast.
After 2 complete Swiss rounds, each create_pairings() call advances
through the configured playoff stages.
"""
import unittest

from src.core import (
    Player,
    Round,
    Tournament,
    TournamentAction,
    TournamentConfiguration,
)
from src.misc import generate_player_names

TournamentAction.LOGF = False  # type: ignore


def _make_tournament(top_cut, n_players=16, n_rounds=2):
    cfg = TournamentConfiguration(
        pod_sizes=[4],
        allow_bye=False,
        auto_export=False,
        n_rounds=n_rounds,
        top_cut=top_cut,
    )
    t = Tournament(cfg)
    t.add_player(generate_player_names(n_players))
    return t


def _run_swiss(t, n):
    """Complete n Swiss rounds."""
    for _ in range(n):
        t.create_pairings()
        t.random_results()


class TestTopCutTop4(unittest.TestCase):
    """TOP_4: one playoff round, 4 players."""

    def setUp(self):
        self.t = _make_tournament(TournamentConfiguration.TopCut.TOP_4)
        _run_swiss(self.t, 2)

    def test_stage_transitions_to_top4(self):
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_4)

    def test_only_4_players_active(self):
        self.t.create_pairings()
        active = self.t.tour_round.active_players
        self.assertEqual(len(active), 4)

    def test_top4_pod_has_4_players(self):
        self.t.create_pairings()
        pods = self.t.tour_round.pods
        self.assertEqual(len(pods), 1)
        self.assertEqual(len(pods[0].players), 4)

    def test_top4_players_are_standings_leaders(self):
        # The 4 players in the pod must be the top 4 from final Swiss standings
        final_swiss = self.t.last_round
        swiss_standings = self.t.get_standings(final_swiss)
        top4 = set(swiss_standings[:4])

        self.t.create_pairings()
        active = set(self.t.tour_round.active_players)
        self.assertEqual(active, top4)

    def test_tournament_complete_after_top4(self):
        self.t.create_pairings()
        self.t.random_results()
        ok = self.t.create_pairings()
        self.assertFalse(ok)

    def test_advancing_players_are_winners(self):
        self.t.create_pairings()
        pod = self.t.tour_round.pods[0]
        winner = pod.players[0]
        self.t.report_win(winner)

        swiss_standings = self.t.get_standings(self.t.rounds[-2])
        advancing = self.t.tour_round.advancing_players(swiss_standings)
        self.assertIn(winner, advancing)


class TestTopCutTop7(unittest.TestCase):
    """TOP_7: two playoff stages — TOP_7 then TOP_4.
    Top 3 seeds receive byes in the TOP_7 round.
    """

    def setUp(self):
        self.t = _make_tournament(TournamentConfiguration.TopCut.TOP_7)
        _run_swiss(self.t, 2)

    def test_first_playoff_stage_is_top7(self):
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_7)

    def test_top7_has_7_active_players(self):
        self.t.create_pairings()
        self.assertEqual(len(self.t.tour_round.active_players), 7)

    def test_top3_seeds_receive_byes(self):
        # Top 3 seeds in standings must get byes
        final_swiss = self.t.rounds[-1]
        swiss_standings = self.t.get_standings(final_swiss)
        top3 = set(swiss_standings[:3])

        self.t.create_pairings()
        byes = self.t.tour_round.byes
        self.assertEqual(byes, top3)

    def test_top7_then_top4(self):
        # TOP_7 round
        self.t.create_pairings()
        self.t.random_results()
        # TOP_4 round
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_4)

    def test_top4_has_4_active_players(self):
        self.t.create_pairings()
        self.t.random_results()
        self.t.create_pairings()
        self.assertEqual(len(self.t.tour_round.active_players), 4)

    def test_tournament_complete_after_top4(self):
        self.t.create_pairings()
        self.t.random_results()
        self.t.create_pairings()
        self.t.random_results()
        ok = self.t.create_pairings()
        self.assertFalse(ok)


class TestTopCutTop16(unittest.TestCase):
    """TOP_16: no byes, 16 players play in 4 pods."""

    def setUp(self):
        self.t = _make_tournament(TournamentConfiguration.TopCut.TOP_16)
        _run_swiss(self.t, 2)

    def test_top16_stage(self):
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_16)

    def test_top16_no_byes(self):
        self.t.create_pairings()
        self.assertEqual(len(self.t.tour_round.byes), 0)

    def test_top16_all_16_active(self):
        self.t.create_pairings()
        self.assertEqual(len(self.t.tour_round.active_players), 16)

    def test_top16_then_top4(self):
        self.t.create_pairings()
        self.t.random_results()
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_4)

    def test_top16_complete_after_top4(self):
        self.t.create_pairings()
        self.t.random_results()
        self.t.create_pairings()
        self.t.random_results()
        ok = self.t.create_pairings()
        self.assertFalse(ok)


class TestTopCutTop10(unittest.TestCase):
    """TOP_10: 2 byes, 8 players in 2 pods."""

    def setUp(self):
        self.t = _make_tournament(TournamentConfiguration.TopCut.TOP_10)
        _run_swiss(self.t, 2)

    def test_top10_stage(self):
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_10)

    def test_top10_two_byes(self):
        self.t.create_pairings()
        self.assertEqual(len(self.t.tour_round.byes), 2)

    def test_top10_then_top4(self):
        self.t.create_pairings()
        self.t.random_results()
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_4)


class TestTopCutTop13(unittest.TestCase):
    """TOP_13: 1 bye, 12 players in 3 pods."""

    def setUp(self):
        self.t = _make_tournament(TournamentConfiguration.TopCut.TOP_13)
        _run_swiss(self.t, 2)

    def test_top13_stage(self):
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_13)

    def test_top13_one_bye(self):
        self.t.create_pairings()
        self.assertEqual(len(self.t.tour_round.byes), 1)

    def test_top13_then_top4(self):
        self.t.create_pairings()
        self.t.random_results()
        ok = self.t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(self.t.tour_round.stage, Round.Stage.TOP_4)


class TestTopCutNone(unittest.TestCase):
    """Verify NONE top-cut stops after n_rounds."""

    def test_no_topcut_stops_at_n_rounds(self):
        t = _make_tournament(TournamentConfiguration.TopCut.NONE)
        _run_swiss(t, 2)
        ok = t.create_pairings()
        self.assertFalse(ok)


class TestTopCutStandings(unittest.TestCase):
    """Playoff standings correctly rank advancing and eliminated players."""

    def test_playoff_standings_include_all_players(self):
        t = _make_tournament(TournamentConfiguration.TopCut.TOP_4)
        _run_swiss(t, 2)
        t.create_pairings()
        t.random_results()

        standings = t.get_standings(t.tour_round)
        self.assertEqual(len(standings), 16)

    def test_playoff_winner_ranked_first(self):
        t = _make_tournament(TournamentConfiguration.TopCut.TOP_4)
        _run_swiss(t, 2)
        t.create_pairings()
        winner = t.tour_round.pods[0].players[0]
        t.report_win(winner)

        standings = t.get_standings(t.tour_round)
        self.assertEqual(standings[0], winner)

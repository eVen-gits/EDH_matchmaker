import random
import unittest
from unittest import mock

from src.core import Pod, StandingsExport, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult
from src.logic.mtg.rules import Mtg1v1Ruleset, games_from_score

TournamentAction.LOGF = False  # type: ignore


def _games(*groups):
    """Builds a games list from UID-set groups, e.g. _games({a}, {a}, {b})."""
    return [IGameResult(frozenset(g)) for g in groups]


class Mtg1v1TestCase(unittest.TestCase):
    """Base class: a 2-player Mtg1v1Ruleset tournament with one manual pod."""

    def _tournament(self, **kwargs) -> Tournament:
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=3, **kwargs
        )
        t = Tournament(cfg)
        t.add_player(["Alice", "Bob"])
        t.new_round()
        t.manual_pod(list(t.players))
        return t

    def setUp(self) -> None:
        self.t = self._tournament()
        self.pod = self.t.tour_round.pods[0]
        self.a, self.b = (p.uid for p in self.pod.players)
        self.ruleset = self.t.ruleset
        assert isinstance(self.ruleset, Mtg1v1Ruleset)


class TestValidateReport(Mtg1v1TestCase):
    """games_to_win defaults to 2 (best of 3) for these cases."""

    def test_valid_reports(self):
        a, b = self.a, self.b
        cases = {
            "2-0": _games({a}, {a}),
            "2-1": _games({a}, {a}, {b}),
            "1-0": _games({a}),
            "1-1": _games({a}, {b}),
            "0-0-3": _games({a, b}, {a, b}, {a, b}),
            "2-0-1": _games({a}, {a}, {a, b}),
        }
        for label, games in cases.items():
            with self.subTest(label=label):
                self.ruleset.validate_report(self.pod, games)  # must not raise

    def test_empty_report_rejected(self):
        with self.assertRaises(ValueError):
            self.ruleset.validate_report(self.pod, [])

    def test_3_0_in_bo3_rejected(self):
        a = self.a
        with self.assertRaises(ValueError):
            self.ruleset.validate_report(self.pod, _games({a}, {a}, {a}))

    def test_2_2_tie_rejected(self):
        a, b = self.a, self.b
        with self.assertRaises(ValueError):
            self.ruleset.validate_report(self.pod, _games({a}, {a}, {b}, {b}))

    def test_unseated_winner_rejected(self):
        other = self.t.add_player("Carl")[0]
        with self.assertRaises(ValueError):
            self.ruleset.validate_report(self.pod, _games({other.uid}))

    def test_wrong_pod_size_rejected(self):
        cfg = TournamentConfiguration(
            ruleset="CommanderRuleset", pod_sizes=[3], auto_export=False
        )
        t = Tournament(cfg)
        t.add_player(["X", "Y", "Z"])
        t.new_round()
        t.manual_pod(list(t.players))
        pod3 = t.tour_round.pods[0]
        with self.assertRaises(ValueError):
            self.ruleset.validate_report(pod3, _games({next(iter(pod3.players)).uid}))

    def test_zero_winner_game_rejected_at_construction(self):
        # A game with nobody in it is nonsensical and rejected up front by
        # IGameResult itself, before any ruleset sees it.
        with self.assertRaises(ValueError):
            IGameResult(frozenset())


class TestMatchWinners(Mtg1v1TestCase):
    def test_table(self):
        a, b = self.a, self.b
        cases = {
            "2-0": (_games({a}, {a}), {a}),
            "2-1": (_games({a}, {a}, {b}), {a}),
            "1-0": (_games({a}), {a}),
            "1-1": (_games({a}, {b}), {a, b}),
            "0-0-3": (_games({a, b}, {a, b}, {a, b}), {a, b}),
            "2-0-1": (_games({a}, {a}, {a, b}), {a}),
        }
        for label, (games, expected) in cases.items():
            with self.subTest(label=label):
                self.pod.record_result(games)
                self.assertEqual(self.ruleset.match_winners(self.pod), expected)
                self.pod.reset_result()


class TestShorthand(Mtg1v1TestCase):
    def test_report_win_records_games_to_win_sweep(self):
        winner = self.pod.players[0]
        self.t.report_win(winner)
        self.assertEqual(len(self.pod.games), 2)  # default games_to_win
        self.assertTrue(all(g.winners == {winner.uid} for g in self.pod.games))

    def test_report_draw_records_one_drawn_game(self):
        self.t.report_draw(list(self.pod.players))
        self.assertEqual(len(self.pod.games), 1)
        self.assertEqual(self.pod.games[0].winners, {self.a, self.b})

    def test_report_replaces_previous_report(self):
        winner, loser = self.pod.players
        self.t.report_win(winner)
        self.t.report_win(loser)
        self.assertEqual(len(self.pod.games), 2)
        self.assertTrue(all(g.winners == {loser.uid} for g in self.pod.games))

    def test_games_from_score(self):
        alice, bob = self.pod.players
        games = games_from_score(self.pod, {alice: 2, bob: 1})
        self.assertEqual(
            [g.winners for g in games],
            [{alice.uid}, {alice.uid}, {bob.uid}],
        )
        self.t.report_match(self.pod, games)
        self.assertEqual(self.ruleset.match_winners(self.pod), {alice.uid})

    def test_games_from_score_draws(self):
        games = games_from_score(self.pod, {}, draws=3)
        self.assertEqual(len(games), 3)
        self.assertTrue(all(g.winners == {self.a, self.b} for g in games))
        self.t.report_match(self.pod, games)
        self.assertEqual(self.ruleset.match_winners(self.pod), {self.a, self.b})


class TestRulesetParams(Mtg1v1TestCase):
    def test_games_to_win_per_round_override(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset",
            auto_export=False,
            n_rounds=3,
            pairing_rounds=[{}, {"ruleset_params": {"games_to_win": 1}}],
        )
        t = Tournament(cfg)
        t.add_player(["Alice", "Bob"])
        ruleset = t.ruleset

        t.new_round()
        t.manual_pod(list(t.players))
        self.assertEqual(ruleset._param(t.tour_round, "games_to_win"), 2)
        t.report_win(t.tour_round.pods[0].players[0])

        t.new_round()
        t.manual_pod(list(t.players))
        self.assertEqual(ruleset._param(t.tour_round, "games_to_win"), 1)
        pod1 = t.tour_round.pods[0]
        t.report_win(pod1.players[0])
        self.assertEqual(len(pod1.games), 1)

    def test_invalid_overrides_rejected(self):
        for bad in (0, 5, "2"):
            with self.subTest(bad=bad):
                cfg = TournamentConfiguration(
                    ruleset="Mtg1v1Ruleset",
                    auto_export=False,
                    pairing_rounds=[{"ruleset_params": {"games_to_win": bad}}],
                )
                with self.assertRaises(ValueError):
                    Tournament(cfg)


class TestRandomReport(Mtg1v1TestCase):
    def test_1000_samples_always_valid(self):
        random.seed(0)
        for _ in range(1000):
            self.pod.reset_result()
            games = self.ruleset.random_report(self.pod)
            self.ruleset.validate_report(self.pod, games)  # must not raise
            self.assertGreaterEqual(len(games), 1)


class TestConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        cfg = TournamentConfiguration(ruleset="Mtg1v1Ruleset", auto_export=False)
        self.assertEqual(list(cfg.pod_sizes), [2])
        self.assertEqual(cfg.scoring_logic, "Scoring1v1")
        self.assertEqual(
            cfg.standings_export.fields,
            [
                StandingsExport.Field[f]
                for f in Mtg1v1Ruleset.DEFAULT_STANDINGS_FIELDS
            ],
        )

    def test_pod_sizes_4_rejected(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", pod_sizes=[4], auto_export=False
        )
        with self.assertRaises(ValueError):
            Tournament(cfg)


class TestEndToEnd(unittest.TestCase):
    def test_8_players_3_rounds(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=3
        )
        t = Tournament(cfg)
        t.add_player([f"P{i}" for i in range(8)])
        for _ in range(3):
            ok = t.create_pairings()
            self.assertTrue(ok)
            self.assertEqual(len(t.tour_round.pods), 4)
            for pod in t.tour_round.pods:
                self.assertEqual(len(pod.players), 2)
            t.random_results()
            for pod in t.tour_round.pods:
                self.assertTrue(pod.done)
            t.new_round()

    def test_no_seat_balancing(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=2
        )
        t = Tournament(cfg)
        t.add_player(["Alice", "Bob"])
        with mock.patch.object(
            Pod, "auto_assign_seats", autospec=True
        ) as mocked:
            t.create_pairings()
        mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()

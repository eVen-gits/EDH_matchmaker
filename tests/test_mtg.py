import random
import unittest
from fractions import Fraction
from unittest import mock

from src.core import Pod, StandingsExport, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult
from src.logic.mtg.rules import FLOOR, Mtg1v1Ruleset, games_from_score, mean_pct, mtr_stats, pct

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


class TestMtrHelpers(unittest.TestCase):
    """Spec section 8's verification cases, word for word, against the pure
    helpers (no tournament needed)."""

    def test_mw_floor_and_denominator(self):
        # 5-2-1 over 8 rounds
        self.assertAlmostEqual(float(pct(Fraction(16), 8 * 3)), 0.667, places=3)
        # 1-3-0 then drops -> 0.25 raised to the floor
        self.assertEqual(pct(Fraction(3), 4 * 3), FLOOR)
        # 3-2-0 incl. a bye, then drops
        self.assertEqual(float(pct(Fraction(9), 5 * 3)), 0.60)

    def test_gw(self):
        self.assertEqual(float(pct(Fraction(21), 3 * 10)), 0.70)
        # 1-2, 1-2, 0-2, 1-2 -> 0.27 raised to the floor
        self.assertEqual(pct(Fraction(9), 3 * 11), FLOOR)

    def test_omw_with_and_without_a_bye_opponent(self):
        def mw(pts, rounds):
            return pct(Fraction(pts), 3 * rounds)

        opps = [
            mw(12, 8), mw(21, 8), mw(4, 5), mw(10, 7),
            mw(18, 8), mw(16, 8), mw(13, 8), mw(19, 8),
        ]
        self.assertAlmostEqual(float(mean_pct(opps)), 0.62, places=2)
        # Round 1 was a bye for one opponent - dropped from the mean.
        self.assertAlmostEqual(float(mean_pct(opps[1:])), 0.63, places=2)

    def test_no_opponents_or_rounds_floors(self):
        self.assertEqual(mean_pct([]), FLOOR)
        self.assertEqual(pct(Fraction(0), 0), FLOOR)


class TestMtrStatsVectorA(unittest.TestCase):
    """Spec section 7's vector A, replayed through a real tournament.

    R1: A beats B 2-0, C beats D 2-1. R2: A beats C 2-1, B and D draw 1-1.
    Expected order: A, C, B, D (B ahead of D on OMW - both tied at 1 point).
    """

    def setUp(self) -> None:
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=2
        )
        self.t = t = Tournament(cfg)
        self.a, self.b, self.c, self.d = t.add_player(["A", "B", "C", "D"])

        t.new_round()
        t.manual_pod([self.a, self.b])
        t.manual_pod([self.c, self.d])
        pod_ab, pod_cd = t.tour_round.pods
        t.report_match(pod_ab, games_from_score(pod_ab, {self.a: 2, self.b: 0}))
        t.report_match(pod_cd, games_from_score(pod_cd, {self.c: 2, self.d: 1}))

        t.new_round()
        t.manual_pod([self.a, self.c])
        t.manual_pod([self.b, self.d])
        pod_ac, pod_bd = t.tour_round.pods
        t.report_match(pod_ac, games_from_score(pod_ac, {self.a: 2, self.c: 1}))
        t.report_match(pod_bd, games_from_score(pod_bd, {self.b: 1, self.d: 1}))

        self.r2 = t.tour_round

    def _mw_gw(self, player):
        s = mtr_stats(self.t, player, self.r2)
        mw = pct(s.match_points, 3 * s.rounds_played)
        gw = pct(s.game_points, 3 * s.games_played)
        return s, mw, gw

    def test_table(self):
        expected = {
            "a": (6, 1.0, 0.8, 0.415, 0.415),
            "c": (3, 0.5, 0.5, 0.665, 0.6),
            "b": (1, FLOOR, FLOOR, 0.665, 0.6),
            "d": (1, FLOOR, 0.4, 0.415, 0.415),
        }
        by_uid = {p.uid: p for p in [self.a, self.b, self.c, self.d]}
        by_name = {p.name.lower(): p for p in by_uid.values()}
        stats = {p.uid: mtr_stats(self.t, p, self.r2) for p in by_uid.values()}
        mw = {uid: pct(s.match_points, 3 * s.rounds_played) for uid, s in stats.items()}
        gw = {uid: pct(s.game_points, 3 * s.games_played) for uid, s in stats.items()}
        for name, (mp, exp_mw, exp_gw, exp_omw, exp_ogw) in expected.items():
            with self.subTest(player=name):
                p = by_name[name]
                s = stats[p.uid]
                self.assertEqual(s.match_points, Fraction(mp))
                self.assertEqual(mw[p.uid], exp_mw if exp_mw is FLOOR else Fraction(exp_mw))
                self.assertAlmostEqual(float(gw[p.uid]), float(exp_gw), places=3)
                omw = mean_pct([mw[o] for o in s.opponents])
                ogw = mean_pct([gw[o] for o in s.opponents])
                self.assertAlmostEqual(float(omw), exp_omw, places=3)
                self.assertAlmostEqual(float(ogw), exp_ogw, places=3)

    def test_standings_order(self):
        standings = self.t.get_standings(self.r2)
        self.assertEqual([p.name for p in standings], ["A", "C", "B", "D"])


class TestMtrDropAndBye(unittest.TestCase):
    def test_drop_denominator_uses_rounds_actually_played(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=3
        )
        t = Tournament(cfg)
        a, b, c, d = t.add_player(["A", "B", "C", "D"])

        t.new_round()
        t.manual_pod([a, b])
        t.manual_pod([c, d])
        pod_ab, pod_cd = t.tour_round.pods
        t.report_match(pod_ab, games_from_score(pod_ab, {a: 2}))
        t.report_match(pod_cd, games_from_score(pod_cd, {c: 2}))

        t.new_round()
        t.manual_pod([a, c])
        t.manual_pod([b, d])
        pod_ac, pod_bd = t.tour_round.pods
        t.report_match(pod_ac, games_from_score(pod_ac, {a: 2}))
        t.report_match(pod_bd, games_from_score(pod_bd, {b: 2}))

        t.drop_player(a)
        t.new_round()
        t.manual_pod([b, c])
        pod_bc = t.tour_round.pods[0]
        t.report_match(pod_bc, games_from_score(pod_bc, {b: 2}))
        r3 = t.tour_round

        # a played exactly 2 rounds before dropping; the denominator must
        # use that, not the tournament's configured n_rounds (3) or the
        # 3rd round they didn't play.
        s = mtr_stats(t, a, r3)
        self.assertEqual(s.rounds_played, 2)
        self.assertEqual(s.match_points, Fraction(6))
        self.assertEqual(pct(s.match_points, 3 * s.rounds_played), Fraction(1))

    def test_bye_counts_in_own_mw_gw_but_is_not_an_opponent(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=1, allow_bye=True
        )
        t = Tournament(cfg)
        a, b, c = t.add_player(["A", "B", "C"])
        t.new_round()
        t.manual_pod([a, b])
        pod_ab = t.tour_round.pods[0]
        t.report_match(pod_ab, games_from_score(pod_ab, {a: 2}))
        t.toggle_bye(c)
        r1 = t.tour_round

        s_c = mtr_stats(t, c, r1)
        self.assertEqual(s_c.rounds_played, 1)
        self.assertEqual(s_c.match_points, Fraction(3))  # bye = win_points
        self.assertEqual(s_c.games_played, 2)
        self.assertEqual(s_c.game_points, Fraction(6))
        self.assertEqual(s_c.opponents, ())  # a bye is not an opponent

        # Nobody has the byed player as an opponent either.
        s_a = mtr_stats(t, a, r1)
        self.assertNotIn(c.uid, s_a.opponents)


class TestStandingsStr(unittest.TestCase):
    def test_mtg_headers_include_omw_gw_ogw_commander_does_not(self):
        mtg_cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=1
        )
        mtg_t = Tournament(mtg_cfg)
        mtg_t.add_player(["A", "B"])
        mtg_t.create_pairings()
        mtg_t.random_results()
        mtg_str = mtg_t.get_standings_str(tour_round=mtg_t.tour_round)
        self.assertIn("OMW", mtg_str)
        self.assertIn("GW", mtg_str)
        self.assertIn("OGW", mtg_str)

        commander_cfg = TournamentConfiguration(auto_export=False, n_rounds=1)
        commander_t = Tournament(commander_cfg)
        commander_t.add_player(["A", "B", "C", "D"])
        commander_t.create_pairings()
        commander_t.random_results()
        commander_str = commander_t.get_standings_str(tour_round=commander_t.tour_round)
        self.assertNotIn("OMW", commander_str)

    def test_configured_export_fields_are_honoured(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset",
            auto_export=False,
            n_rounds=1,
            standings_export=StandingsExport(
                fields=[StandingsExport.Field.NAME, StandingsExport.Field.RATING]
            ),
        )
        t = Tournament(cfg)
        t.add_player(["A", "B"])
        t.create_pairings()
        t.random_results()
        out = t.get_standings_str(tour_round=t.tour_round)
        header = out.splitlines()[0]
        self.assertIn("name", header)
        self.assertIn("pts", header)
        self.assertNotIn("record", header)


if __name__ == "__main__":
    unittest.main()

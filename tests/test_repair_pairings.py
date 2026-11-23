"""Correcting the previous round's result, and repairing the live round."""

import unittest

from src.core import Player, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult

TournamentAction.LOGF = False  # type: ignore


def won(player: Player, games: int = 1) -> list[IGameResult]:
    return [IGameResult(frozenset({player.uid}))] * games


def drew(*players: Player) -> list[IGameResult]:
    return [IGameResult(frozenset(p.uid for p in players))]


class EventTestCase(unittest.TestCase):
    """Builds rounds from hand-made pods, so every layout is known."""

    GAMES = 1

    def event(self, names: str, **config) -> None:
        self.t = Tournament(TournamentConfiguration(auto_export=False, **config))
        self.t.add_player(list(names))
        self.p = {p.name: p for p in self.t.players}

    def play(self, *pods: str, winners: str = "", byes: str = ""):
        """Seats one round from pod strings ("AB"), then reports winners
        (one letter per pod, in pod order) when given."""
        for pod in pods:
            self.t.manual_pod([self.p[c] for c in pod])
        r = self.t.tour_round
        assert r is not None
        for c in byes:
            self.t.toggle_bye(self.p[c])
        for pod, w in zip(r.pods, winners):
            self.t.report_match(pod, won(self.p[w], self.GAMES))
        return r

    def layout(self, r) -> list[str]:
        return ["".join(p.name for p in pod.players) for pod in r.pods]

    def names(self, *letters: str) -> list[Player]:
        return [self.p[c] for c in letters]


class TestCorrection(EventTestCase):
    GAMES = 2

    def setUp(self):
        self.event("ABCDE", ruleset="Mtg1v1Ruleset", n_rounds=3)
        self.r1 = self.play("AB", "CD", winners="AC", byes="E")
        self.r2 = self.play("AC", "DE", byes="B")

    def test_correction_keeps_the_live_rounds_byes(self):
        # Regression: the report went through the live round, which dropped
        # the winner's bye there.
        self.t.report_match(self.r1.pods[0], won(self.p["B"], 2))
        self.assertIn(self.p["B"], self.r2.byes)
        self.assertEqual(self.r1.pods[0].result, {self.p["B"]})

    def test_correction_moves_the_standings(self):
        self.t.report_match(self.r1.pods[0], won(self.p["B"], 2))
        ratings = self.t.field_ratings(self.r1)
        self.assertGreater(ratings[self.p["B"].uid], ratings[self.p["A"].uid])

    def test_only_the_last_two_rounds_can_change(self):
        for pod in self.r2.pods:
            self.t.report_match(pod, won(pod.players[0], 2))
        self.play("AD", "CE")
        with self.assertRaises(ValueError):
            self.t.report_match(self.r1.pods[0], won(self.p["B"], 2))
        with self.assertRaises(ValueError):
            self.t.reset_result(self.r1.pods[0])
        self.assertEqual(self.r1.pods[0].result, {self.p["A"]})

    def test_1v1_swap_can_trade_a_bye(self):
        self.t.report_match(self.r1.pods[0], won(self.p["B"], 2))
        a, b = self.names("A", "B")
        self.t.repair_pairings([a, b], "swap", apply=True)
        self.assertIn(a, self.r2.byes)
        self.assertEqual(self.layout(self.r2), ["BC", "DE"])


class TestSwap1v1(EventTestCase):
    GAMES = 2

    def setUp(self):
        self.event("ABCDEFGH", ruleset="Mtg1v1Ruleset", n_rounds=4)
        self.play("AB", "CD", "EF", "GH", winners="ACEG")
        self.r2 = self.play("AC", "EG", "BD", "FH", winners="AEBF")
        self.r3 = self.play("AE", "CF", "GB", "DH")
        # Round 2, table 1 was reported wrong: C beat A.
        self.t.report_match(self.r2.pods[0], won(self.p["C"], 2))

    def test_swap_meets_the_higher_seed(self):
        a, c = self.names("A", "C")
        plan = self.t.repair_pairings([a, c], "swap")
        self.assertEqual(plan.moves, ((a, 1, 2), (c, 2, 1)))
        self.assertEqual(plan.tables, (1, 2))
        self.assertEqual(self.layout(self.r3), ["AE", "CF", "GB", "DH"])  # dry run

        self.assertEqual(self.t.repair_pairings([a, c], "swap", apply=True), plan)
        self.assertEqual(self.layout(self.r3), ["CE", "AF", "GB", "DH"])

    def test_reported_table_is_frozen(self):
        self.t.report_match(self.r3.pods[0], won(self.p["E"], 2))
        with self.assertRaises(ValueError):
            self.t.repair_pairings(self.names("A", "C"), "swap")

    def test_open_leaves_reported_tables(self):
        self.t.report_match(self.r3.pods[3], won(self.p["D"], 2))
        plan = self.t.repair_pairings(self.names("A", "C"), "open", apply=True)
        self.assertEqual(self.layout(self.r3)[3], "DH")
        self.assertNotIn(4, plan.tables)
        self.assertEqual(sum(len(pod) for pod in self.r3.pods), 8)


class TestSwap1v1Rematch(EventTestCase):
    GAMES = 2

    def test_swap_that_makes_a_rematch_is_skipped(self):
        self.event("ABCDEFGH", ruleset="Mtg1v1Ruleset", n_rounds=4)
        self.play("AB", "CE", "DF", "GH", winners="ACDG")
        r2 = self.play("AC", "EB", "DG", "FH", winners="AEDF")
        r3 = self.play("AE", "CB", "DH", "FG")
        self.t.report_match(r2.pods[0], won(self.p["C"], 2))
        # A swap would bring the points closer but pair C with E again.
        plan = self.t.repair_pairings(self.names("A", "C"), "swap", apply=True)
        self.assertEqual(plan.moves, ())
        self.assertEqual(self.layout(r3), ["AE", "CB", "DH", "FG"])


class TestCommander(EventTestCase):
    def setUp(self):
        self.event("ABCDEFGH", n_rounds=4, pod_sizes=[4])

    def test_swap_always_happens_with_larger_pods(self):
        r1 = self.play("ABCD", "EFGH", winners="AE")
        r2 = self.play("AECG", "BFDH")
        self.t.report_match(r1.pods[0], won(self.p["B"]))
        # B meets C again; the swap still happens.
        self.t.repair_pairings(self.names("A", "B"), "swap", apply=True)
        self.assertEqual(self.layout(r2), ["BECG", "AFDH"])

    def test_rank_gives_the_stronger_table_to_more_points(self):
        r1 = self.play("ABCD", "EFGH")
        self.t.report_match(r1.pods[0], drew(*self.names("A", "B", "C")))
        self.t.report_match(r1.pods[1], won(self.p["E"]))
        r2 = self.play("EBFG", "ACDH")
        # The draw was really A's win.
        self.t.report_match(r1.pods[0], won(self.p["A"]))
        plan = self.t.repair_pairings(self.names("A", "B", "C"), "rank", apply=True)
        self.assertEqual(self.layout(r2), ["EAFG", "BCDH"])
        self.assertEqual(plan.tables, (1, 2))


class TestCommanderPods(EventTestCase):
    def test_pods_repairs_only_the_affected_tables(self):
        self.event("ABCDEFGHIJKL", n_rounds=4, pod_sizes=[4])
        r1 = self.play("ABCD", "EFGH", "IJKL", winners="AEI")
        r2 = self.play("AEIB", "CFJD", "GKHL")
        self.assertEqual(r2.logic.name, "PairingSnake")
        self.t.report_match(r1.pods[0], drew(*self.names("B", "C")))
        affected = self.names("A", "B", "C")
        plan = self.t.repair_pairings(affected, "pods")
        self.assertEqual(self.layout(r2), ["AEIB", "CFJD", "GKHL"])  # dry run

        self.assertEqual(self.t.repair_pairings(affected, "pods", apply=True), plan)
        self.assertEqual(self.layout(r2)[2], "GKHL")
        self.assertEqual(set("".join(self.layout(r2)[:2])), set("AEIBCFJD"))
        self.assertTrue(set(plan.tables) <= {1, 2})


class TestPlayoffCorrection(unittest.TestCase):
    def setUp(self):
        cfg = TournamentConfiguration(
            ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=1, top_cut=4
        )
        self.t = Tournament(cfg)
        self.t.add_player([f"P{i}" for i in range(8)])
        self.t.create_pairings(seed=5)
        assert self.t.tour_round is not None
        self.swiss = self.t.tour_round
        for pod in self.swiss.pods:
            self.t.report_match(pod, won(pod.players[0], 2))
        self.t.create_pairings()
        assert self.t.tour_round is not None
        self.cut = self.t.tour_round

    def seated(self) -> set[Player]:
        return {p for pod in self.cut.pods for p in pod.players}

    def test_correction_repairs_an_unreported_top_cut(self):
        pod = self.swiss.pods[0]
        loser = pod.players[1]
        self.assertNotIn(loser, self.seated())
        self.t.report_match(pod, won(loser, 2))
        self.assertIn(loser, self.seated())
        self.assertNotIn(pod.players[0], self.seated())

    def test_correction_refused_once_the_top_cut_reported(self):
        cut_pod = self.cut.pods[0]
        self.t.report_match(cut_pod, won(cut_pod.players[0], 2))
        pod = self.swiss.pods[0]
        with self.assertRaises(ValueError):
            self.t.report_match(pod, won(pod.players[1], 2))
        self.assertEqual(pod.result, {pod.players[0]})

    def test_playoff_round_is_not_repaired_in_part(self):
        with self.assertRaises(ValueError):
            self.t.repair_pairings(self.cut.pods[0].players, "swap")


if __name__ == "__main__":
    unittest.main()

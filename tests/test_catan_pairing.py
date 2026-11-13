"""Catan pairing: the preliminary plan (rule 1.1.6) and the game-4
serpentine (rule 1.1.7)."""

import itertools
import unittest
from collections import Counter
from typing import cast

from src.core import Tournament, TournamentAction, TournamentConfiguration
from src.logic.catan.matching import PairingCatanPrelim, PairingCatanSerpentine
from src.logic.catan.rules import CatanRuleset
from src.logic.catan.scoring import ScoringCatan

TournamentAction.LOGF = False

_HIDDEN = (CatanRuleset, ScoringCatan, PairingCatanPrelim, PairingCatanSerpentine)


def _reset_caches():
    Tournament._ruleset_cache = {}
    Tournament._scoring_logic_cache = {}
    Tournament._pairing_logic_cache = {}


def setUpModule():
    # Hidden (IS_COMPLETE False) until the playoff slice.
    for cls in _HIDDEN:
        cls.IS_COMPLETE = True
    _reset_caches()


def tearDownModule():
    for cls in _HIDDEN:
        cls.IS_COMPLETE = False
    _reset_caches()


def _tour(n: int, **kw) -> Tournament:
    kw.setdefault("n_rounds", 3)
    t = Tournament(TournamentConfiguration(ruleset="CatanRuleset", auto_export=False, **kw))
    t.add_player([f"P{i}" for i in range(n)])
    return t


def _round(t: Tournament):
    assert t.tour_round is not None
    return t.tour_round


def _tables(t: Tournament) -> set[tuple]:
    """This round's tables, each the uids in play order."""
    return {tuple(pod._players) for pod in _round(t).pods}


def _plan(t: Tournament) -> list[set[tuple]]:
    """The plan from the current (just paired) round on, as table sets."""
    r = _round(t)
    logic = cast(PairingCatanPrelim, t.get_pairing_logic("PairingCatanPrelim"))
    field = frozenset(u for pod in r.pods for u in pod._players)
    sizes = sorted((pod.cap for pod in r.pods), reverse=True)
    return [{tuple(table) for table in game} for game in logic.plan(r, field, sizes)]


def _play(t: Tournament, games: int) -> list[set[tuple]]:
    out = []
    for _ in range(games):
        t.create_pairings()
        out.append(_tables(t))
        t.random_results()
    return out


def _repeats(games: list[set[tuple]]) -> int:
    meets = Counter(
        frozenset(pair)
        for game in games
        for table in game
        for pair in itertools.combinations(table, 2)
    )
    return sum(m - 1 for m in meets.values())


class PrelimPlanTests(unittest.TestCase):
    def test_ruleset_picks_prelim(self):
        t = _tour(16)
        self.assertEqual(t.ruleset.swiss_pairing_logic(t, 0), "PairingCatanPrelim")
        t.create_pairings()
        self.assertEqual(_round(t).logic.name, "PairingCatanPrelim")

    def test_no_repeat_opponents(self):
        for n in (16, 20):
            with self.subTest(players=n):
                games = _play(_tour(n), 3)
                self.assertEqual(_repeats(games), 0)
                seated = [u for table in games[0] for u in table]
                self.assertEqual(len(seated), n)

    def test_play_order_spread(self):
        for n in (16, 20):
            with self.subTest(players=n):
                games = _play(_tour(n), 3)
                held = {
                    u: [table.index(u) for game in games for table in game if u in table]
                    for table in games[0]
                    for u in table
                }
                for positions in held.values():
                    self.assertEqual(len(set(positions)), 3, positions)

    def test_rounds_follow_plan_ignoring_results(self):
        t = _tour(20)
        t.create_pairings()
        plan = _plan(t)
        self.assertEqual(len(plan), 3)
        self.assertEqual(_tables(t), plan[0])
        t.random_results()
        self.assertEqual(_play(t, 2), plan[1:])

    def test_same_plan_after_save_and_reload(self):
        t = _tour(16)
        _play(t, 1)
        saved = t.serialize()
        expected = _play(t, 2)
        Tournament.CACHE.clear()
        t2 = cast(Tournament, Tournament.inflate(saved))
        self.assertEqual(_play(t2, 2), expected)

    def test_drop_after_game_one_replans(self):
        t = _tour(20)
        game1 = _play(t, 1)
        dropped = sorted(t.players, key=lambda p: p.name)[0]
        t.drop_player(dropped)
        t.create_pairings()
        replan = _plan(t)
        self.assertEqual(len(replan), 2)  # games 2-3, for the 19 still in
        self.assertEqual(_tables(t), replan[0])
        self.assertEqual(sorted(len(x) for x in replan[0]), [3, 4, 4, 4, 4])
        t.random_results()
        later = _play(t, 1)
        self.assertEqual(later, replan[1:])
        for game in replan:
            self.assertNotIn(dropped.uid, {u for table in game for u in table})
        self.assertEqual(_repeats(game1 + replan), 0)

    def test_late_player_replans(self):
        t = _tour(16)
        _play(t, 1)
        t.add_player("Late")
        t.create_pairings()
        replan = _plan(t)
        self.assertEqual(len(replan), 2)
        self.assertEqual(sum(len(x) for x in replan[0]), 17)


class SerpentineTests(unittest.TestCase):
    def test_ruleset_picks_serpentine_on_elimination_round(self):
        t = _tour(
            16,
            n_rounds=4,
            pairing_rounds=[{}, {}, {}, {"ruleset_params": {"eliminate_below_wins": 1}}],
        )
        self.assertEqual(t.ruleset.swiss_pairing_logic(t, 2), "PairingCatanPrelim")
        self.assertEqual(t.ruleset.swiss_pairing_logic(t, 3), "PairingCatanSerpentine")

    def test_rule_117_45_players(self):
        t = _tour(45, n_rounds=2, pairing_logics=[None, "PairingCatanSerpentine"])
        _play(t, 1)
        ranked = sorted(t.players, key=lambda p: p.name)
        ranked.insert(0, ranked.pop(9))  # P9 is rank 1: not just name order
        t.get_standings = lambda _round=None: ranked  # type: ignore[method-assign]
        t.create_pairings()
        rank = {p.uid: i + 1 for i, p in enumerate(ranked)}
        got = {tuple(rank[u] for u in table) for table in _tables(t)}
        # Rule 1.1.7: three 3-player tables first, then snake; seat = rank.
        expected = {(1, 24, 25), (2, 23, 26), (3, 22, 27)} | {
            (k, 25 - k, 24 + k, 49 - k) for k in range(4, 13)
        }
        self.assertIn((4, 21, 28, 45), expected)
        self.assertIn((12, 13, 36, 37), expected)
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()

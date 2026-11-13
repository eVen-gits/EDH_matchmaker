"""ScoringCatan: wins, then the rule 1.4.2 tiebreaker chain."""

import unittest
from fractions import Fraction

from src.core import Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult
from src.logic.catan.rules import CatanRuleset
from src.logic.catan.scoring import ScoringCatan, dummy_score, finishing_places, vp_share

TournamentAction.LOGF = False

_HIDDEN = (CatanRuleset, ScoringCatan)


def _reset_caches():
    Tournament._ruleset_cache = {}
    Tournament._scoring_logic_cache = {}


def setUpModule():
    # Both classes are hidden (IS_COMPLETE False) until the playoff slice.
    for cls in _HIDDEN:
        cls.IS_COMPLETE = True
    _reset_caches()


def tearDownModule():
    for cls in _HIDDEN:
        cls.IS_COMPLETE = False
    _reset_caches()


def _table(own: int, total: int, n: int = 4) -> Fraction:
    """VP share of a player with `own` VP at an n-player table totalling `total`."""
    rest, others = total - own, n - 1
    scores = [rest // others + (i < rest % others) for i in range(others)]
    return vp_share([own, *scores], own)


class Rule143Tests(unittest.TestCase):
    """Worked examples from rule 1.4.3, with exact values."""

    def test_player_a(self):
        shares = [_table(10, 32), _table(8, 30), _table(10, 36), _table(10, 29)]
        self.assertEqual(
            shares,
            [Fraction(10, 32), Fraction(8, 30), Fraction(10, 36), Fraction(10, 29)],
        )
        self.assertEqual([f"{float(s):.4f}" for s in shares], ["0.3125", "0.2667", "0.2778", "0.3448"])
        self.assertEqual(sum(Fraction(f"{float(s):.4f}") for s in shares), Fraction("1.2018"))
        self.assertEqual(f"{float(sum(shares)):.4f}", "1.2018")

    def test_player_b(self):
        shares = [_table(10, 30), _table(10, 33), _table(8, 36), _table(10, 31)]
        # The rule's manual method adds the 4-decimal shares: 1.1811. The exact
        # sum (what software uses) is 1.1812 to 4 decimals.
        manual = sum(Fraction(f"{float(s):.4f}") for s in shares)
        self.assertEqual(manual, Fraction("1.1811"))
        self.assertEqual(f"{float(sum(shares)):.4f}", "1.1812")

    def test_a_ranks_above_b(self):
        a = sum([_table(10, 32), _table(8, 30), _table(10, 36), _table(10, 29)], Fraction(0))
        b = sum([_table(10, 30), _table(10, 33), _table(8, 36), _table(10, 31)], Fraction(0))
        self.assertGreater(a, b)

    def test_three_player_dummy(self):
        self.assertEqual(dummy_score([8, 10, 7]), 8)  # 25 / 3 = 8.33
        self.assertEqual(vp_share([8, 10, 7], 10), Fraction(10, 33))  # 8+10+7+8

    def test_dummy_half_up(self):
        self.assertEqual(dummy_score([5, 6]), 6)  # 5.5 rounds up; round() gives 6 only by luck
        self.assertEqual(dummy_score([6, 7]), 7)  # 6.5: banker's rounding would give 6
        self.assertEqual(dummy_score([5, 5, 5, 6]), 5)  # 5.25 rounds down
        self.assertEqual(dummy_score([4, 4, 5]), 4)  # 4.33 rounds down
        self.assertEqual(dummy_score([4, 5, 5]), 5)  # 4.67 rounds up

    def test_four_player_table_has_no_dummy(self):
        self.assertEqual(vp_share([10, 5, 4, 3], 5), Fraction(5, 22))

    def test_places_tied_share_better_place(self):
        a, b, c, d = "abcd"
        self.assertEqual(finishing_places({a: 10, b: 7, c: 7, d: 3}), {a: 1, b: 2, c: 2, d: 4})


def _round(t: Tournament):
    assert t.tour_round is not None
    return t.tour_round


def _tour(n=4, **kw) -> Tournament:
    t = Tournament(TournamentConfiguration(ruleset="CatanRuleset", auto_export=False, **kw))
    t.add_player([f"P{i}" for i in range(n)])
    return t


def _play(t: Tournament, tables: list[list[tuple[int, int]]]):
    """One round; each table is [(player index, VP)], winner = most VP."""
    players = sorted(t.players, key=lambda p: p.name)
    t.new_round()
    for table in tables:
        t.manual_pod([players[i] for i, _ in table])
    for table, pod in zip(tables, _round(t).pods):
        vp = {players[i].uid: v for i, v in table}
        winner = max(vp, key=lambda u: vp[u])
        t.report_match(pod, [IGameResult(frozenset({winner}), tuple(vp.items()))])
    return players


class ScoringCatanTests(unittest.TestCase):
    def test_vp_then_places_columns_and_ranking(self):
        t = _tour(8)
        # Round 1: P0 and P4 win. P1 and P5 take 2nd, P2/P6 3rd.
        p = _play(t, [[(0, 10), (1, 8), (2, 6), (3, 4)], [(4, 10), (5, 9), (6, 5), (7, 3)]])
        r = _round(t)
        cols = dict(t.get_scoring_logic("ScoringCatan").standings_columns(t, r))
        self.assertEqual(cols["VP"][p[1].uid], "8")
        self.assertEqual(cols["VP%"][p[0].uid], "0.3571")  # 10 / 28
        self.assertEqual(cols["2nd"][p[1].uid], "1")
        self.assertEqual(cols["3rd"][p[2].uid], "1")
        # Equal wins: more VP first (P4's 10 ties P0's 10; VP% 10/27 beats 10/28...).
        s = [x.name for x in t.get_standings(r)]
        self.assertEqual(s[:2], ["P4", "P0"])
        self.assertEqual(s[2:4], ["P5", "P1"])

    def test_vp_capped_per_game(self):
        t = _tour(4)
        p = _play(t, [[(0, 13), (1, 9), (2, 5), (3, 2)]])
        cols = dict(t.get_scoring_logic("ScoringCatan").standings_columns(t, _round(t)))
        self.assertEqual(cols["VP"][p[0].uid], "10")
        self.assertEqual(cols["VP%"][p[0].uid], f"{13 / 29:.4f}")  # VP% uses raw VP

    def test_wins_outrank_vp(self):
        t = _tour(4)
        _play(t, [[(0, 10), (1, 2), (2, 2), (3, 2)]])
        self.assertEqual(t.get_standings(_round(t))[0].name, "P0")

    def test_second_place_breaks_tie(self):
        t = _tour(4)
        # Same VP and same VP% is hard to force; check the key order directly.
        _play(t, [[(0, 10), (1, 7), (2, 7), (3, 3)]])
        r = _round(t)
        keys = t.get_scoring_logic("ScoringCatan").standings_keys(t, r, t.field_ratings(r))
        p = sorted(t.players, key=lambda x: x.name)
        self.assertEqual(keys[p[1].uid][1:5], keys[p[2].uid][1:5])  # tied through 3rd
        self.assertEqual(keys[p[1].uid][3], 1)  # shared 2nd place counts as 2nd
        self.assertEqual(keys[p[2].uid][3], 1)
        self.assertEqual(keys[p[3].uid][3:5], (0, 0))  # 4th place: neither

    def test_forced_bye_scores_bye_points_and_no_vp(self):
        t = _tour(5, allow_bye=True, scoring_params={"bye_points": 1})
        t.new_round()
        t.create_pairings()
        r = _round(t)
        assert r is not None
        bye = next(p for p in t.players if p.result(r) == p.EResult.BYE)
        self.assertEqual(t.rating(bye, r), 1)
        cols = dict(t.get_scoring_logic("ScoringCatan").standings_columns(t, r))
        self.assertEqual(cols["VP"][bye.uid], "0")
        self.assertEqual(cols["VP%"][bye.uid], "0.0000")


if __name__ == "__main__":
    unittest.main()


class CatanAutoExportTests(unittest.TestCase):
    def test_auto_export_report_and_standings(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            old = TournamentAction.LOGF
            TournamentAction.LOGF = os.path.join(d, "catan.json")
            try:
                t = _tour(4)
                t.config.auto_export = True
                t.config.standings_export.dir = os.path.join(d, "standings.txt")
                _play(t, [[(0, 10), (1, 8), (2, 6), (3, 4)]])
                out = t.get_standings_str()
            finally:
                TournamentAction.LOGF = old
            self.assertIn("P0", out)
            self.assertNotIn("avg. seat", out)
            with open(os.path.join(d, "standings.txt")) as f:
                self.assertIn("P0", f.read())

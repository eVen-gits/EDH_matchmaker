"""Catan playoffs and structure: semifinals and final (rules 1.1.10, 1.1.12),
game-4 elimination (1.1.5), playoff standings (1.4.4), the recommended
structure (1.1.3, 1.1.5) and full tournaments end to end."""

import random
import unittest

from src.core import Log, Round, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult
from src.logic.catan.rules import CatanRuleset, recommended_structure
from src.logic.catan.scoring import catan_stats

TournamentAction.LOGF = False


def _config(n: int, **kw) -> TournamentConfiguration:
    """The recommended structure for n players, unless kw overrides it."""
    rounds, top_cut, elim = recommended_structure(n)
    kw.setdefault("n_rounds", rounds)
    kw.setdefault("top_cut", top_cut)
    kw.setdefault(
        "pairing_rounds",
        [{"ruleset_params": {"eliminate_below_wins": elim[s]}} if s in elim else {}
         for s in range(kw["n_rounds"])],
    )
    return TournamentConfiguration(ruleset="CatanRuleset", auto_export=False, **kw)


def _tour(n: int, seed: int = 0, **kw) -> Tournament:
    random.seed(seed)
    t = Tournament(_config(n, **kw))
    t.add_player([f"P{i}" for i in range(n)])
    return t


def _round(t: Tournament) -> Round:
    assert t.tour_round is not None
    return t.tour_round


def _play_swiss(t: Tournament) -> None:
    for _ in range(t.config.n_rounds):
        t.create_pairings()
        t.random_results()


def _report(t: Tournament, pod, vps: list[int]) -> None:
    """Reports pod with vps by seat; the top score wins."""
    vp = {p.uid: v for p, v in zip(pod.players, vps)}
    winner = max(vp, key=lambda u: vp[u])
    t.report_match(pod, [IGameResult(frozenset({winner}), tuple(vp.items()))])


def _seats(pod) -> list:
    return list(pod.players)


class PlayoffPlanTests(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(
            CatanRuleset.PLAYOFFS,
            {
                4: ((4, "PairingTopCut"),),
                16: ((16, "PairingTopCut"), (4, "PairingTopCut")),
            },
        )

    def test_selectable(self):
        self.assertIn("CatanRuleset", Tournament.ruleset_names())
        self.assertIn("ScoringCatan", Tournament.scoring_logic_names())

    def test_catan_logics_offered_only_under_catan(self):
        """Commander and 1v1 Magic selector lists stay as they were."""
        pairing, scoring = Tournament.selectable_pairing_logics, Tournament.selectable_scoring_logics
        self.assertEqual(
            pairing([4, 3], "CommanderRuleset"),
            ["PairingDefault", "PairingRandom", "PairingSnake"],
        )
        self.assertEqual(
            scoring([4, 3], "CommanderRuleset"),
            ["ScoringDefault", "ScoringHareruya", "ScoringModifiedHareruya"],
        )
        self.assertEqual(pairing([2], "Mtg1v1Ruleset"), ["Pairing1v1", "PairingRandom"])
        self.assertEqual(scoring([2], "Mtg1v1Ruleset"), ["Scoring1v1"])
        self.assertEqual(
            pairing([4, 3], "CatanRuleset"),
            ["PairingCatanPrelim", "PairingCatanSerpentine",
             "PairingDefault", "PairingRandom", "PairingSnake"],
        )
        self.assertIn("ScoringCatan", scoring([4, 3], "CatanRuleset"))


class SemifinalTests(unittest.TestCase):
    def setUp(self):
        self.t = _tour(40)
        _play_swiss(self.t)
        self.swiss = self.t.get_standings(_round(self.t))
        self.t.create_pairings()
        self.semi = _round(self.t)

    def test_rule_1_1_10_serpentine(self):
        """Guards the reuse of Commander's PairingTopCut by name."""
        self.assertEqual(self.semi.stage, Round.Stage.TOP_16)
        self.assertEqual(self.semi.logic.name, "PairingTopCut")
        rank = {p: i + 1 for i, p in enumerate(self.swiss)}
        tables = {tuple(rank[p] for p in _seats(pod)) for pod in self.semi.pods}
        # Seat order is rank order (the pick order).
        self.assertEqual(
            tables,
            {(1, 8, 9, 16), (2, 7, 10, 15), (3, 6, 11, 14), (4, 5, 12, 13)},
        )

    def test_standings_after_semis_and_final(self):
        t = self.t
        pods = sorted(self.semi.pods, key=lambda pod: self.swiss.index(pod.players[0]))
        # By seat: table 1 ties two players on 7 for 2nd, so nobody is 3rd.
        vps = [[10, 7, 7, 3], [6, 10, 4, 8], [5, 9, 10, 2], [9, 4, 6, 10]]
        for pod, v in zip(pods, vps):
            _report(t, pod, v)
        places = [{1: [], 2: [], 3: [], 4: []} for _ in pods]
        for k, (pod, v) in enumerate(zip(pods, vps)):
            for p, s in zip(pod.players, v):
                places[k][1 + sum(x > s for x in v)].append(p)
        group = {g: {p for k in range(4) for p in places[k][g]} for g in (1, 2, 3, 4)}

        semi = t.get_standings(self.semi)
        self.assertEqual(len(semi), 40)
        self.assertEqual(set(semi[:4]), group[1])
        self.assertEqual(set(semi[4:9]), group[2])  # five 2nds: the tie
        self.assertEqual(set(semi[9:12]), group[3])
        self.assertEqual(set(semi[12:16]), group[4])
        self.assertEqual(semi[16:], self.swiss[16:])
        # Within a place: VP over prelim and semifinal games first.
        stats = catan_stats(t, t.rounds)
        for lo, hi in ((0, 4), (4, 9), (9, 12), (12, 16)):
            vp = [stats[p.uid].vp for p in semi[lo:hi]]
            self.assertEqual(vp, sorted(vp, reverse=True))

        t.create_pairings()
        final = _round(t)
        self.assertEqual(final.stage, Round.Stage.TOP_4)
        (pod,) = final.pods
        self.assertEqual(_seats(pod), semi[:4])  # pick seats by ranking
        _report(t, pod, [3, 10, 6, 8])
        standings = t.get_standings(final)
        seats = _seats(pod)
        self.assertEqual(standings[:4], [seats[1], seats[3], seats[2], seats[0]])
        self.assertEqual(standings[4:], semi[4:])

    def test_pending_semis_keep_previous_standings(self):
        _report(self.t, self.semi.pods[0], [10, 5, 4, 3])
        self.assertEqual(self.t.get_standings(self.semi), self.swiss)

    def test_dropped_semifinalist(self):
        """A semifinalist who drops leaves a 3-player table; its winner
        still advances and the dropped player ranks last of the 16."""
        t = self.t
        gone = self.semi.pods[0].players[-1]
        t.remove_player_from_pod(gone)
        t.drop_player(gone)
        self.assertEqual(len(self.semi.pods[0]), 3)
        t.random_results()
        semi = t.get_standings(self.semi)
        self.assertEqual(semi[15], gone)
        winners = {p for pod in self.semi.pods for p in pod.players
                   if p.uid in pod.games[-1].winners}
        self.assertEqual(set(semi[:4]), winners)
        t.create_pairings()
        self.assertEqual(set(_round(t).pods[0].players), winners)


class TopFourFinalTests(unittest.TestCase):
    def test_final_of_top_4(self):
        t = _tour(20)
        _play_swiss(t)
        swiss = t.get_standings(_round(t))
        t.create_pairings()
        final = _round(t)
        self.assertEqual(final.stage, Round.Stage.TOP_4)
        (pod,) = final.pods
        self.assertEqual(_seats(pod), swiss[:4])
        _report(t, pod, [7, 9, 10, 2])
        standings = t.get_standings(final)
        self.assertEqual(standings[:4], [swiss[2], swiss[1], swiss[0], swiss[3]])
        self.assertEqual(standings[4:], swiss[4:])


class EliminationTests(unittest.TestCase):
    def test_zero_win_players_sit_out_game_4_and_stay_ranked(self):
        t = _tour(24, n_rounds=4, top_cut=0, pairing_rounds=[
            {}, {}, {}, {"ruleset_params": {"eliminate_below_wins": 1}}])
        for _ in range(3):
            t.create_pairings()
            t.random_results()
        wins = {p: sum(p.result(r) == p.EResult.WIN for r in t.rounds) for p in t.players}
        out = {p for p, w in wins.items() if w == 0}
        self.assertTrue(out)
        t.create_pairings()
        game4 = _round(t)
        self.assertEqual(game4.logic.name, "PairingCatanSerpentine")
        seated = {p for pod in game4.pods for p in pod.players}
        self.assertEqual(seated, set(t.players) - out)
        self.assertTrue(out <= game4.disabled_players)
        t.random_results()
        standings = t.get_standings(game4)
        self.assertEqual(set(standings), set(t.players))
        self.assertEqual(set(standings[-len(out):]), out)

    def test_off_by_default(self):
        t = _tour(16, n_rounds=4, top_cut=0)
        for _ in range(4):
            t.create_pairings()
            t.random_results()
        self.assertEqual(_round(t).disabled_players, set())


class StructureTests(unittest.TestCase):
    def test_recommended_structure(self):
        self.assertEqual(recommended_structure(12), (3, 4, {}))
        self.assertEqual(recommended_structure(28), (3, 4, {}))  # rule 1.1.5
        self.assertEqual(recommended_structure(29), (3, 16, {}))
        self.assertEqual(recommended_structure(79), (3, 16, {}))
        self.assertEqual(recommended_structure(80), (4, 16, {3: 1}))

    def _warnings(self, t: Tournament) -> list[str]:
        Log.output.clear()
        t.create_pairings()
        return [e.msg for e in Log.output if e.level == Log.Level.WARNING]

    def test_no_warning_for_recommended_structure(self):
        self.assertEqual(self._warnings(_tour(20)), [])

    def test_warns_below_12_players(self):
        (msg,) = self._warnings(_tour(10))
        self.assertIn("1.1.3", msg)

    def test_warns_when_structure_differs(self):
        (msg,) = self._warnings(_tour(40, top_cut=4))
        self.assertIn("recommends for 40 players: 3 games, top cut 16", msg)
        self.assertIn("set to 3 games, top cut 4", msg)


class EndToEndTests(unittest.TestCase):
    def _run(self, n: int) -> Tournament:
        t = _tour(n, seed=n)
        while t.create_pairings():
            t.random_results()
        return t

    def _check(self, t: Tournament, stages: list[Round.Stage]) -> list:
        self.assertEqual([r.stage for r in t.rounds], stages)
        self.assertTrue(all(r.done for r in t.rounds))
        standings = t.get_standings()
        self.assertEqual(len(standings), len(t.players))
        self.assertEqual(set(standings), set(t.players))
        (final,) = t.rounds[-1].pods
        self.assertEqual(standings[0].uid, next(iter(final.games[-1].winners)))
        self.assertEqual(set(standings[:4]), set(final.players))
        return standings

    def test_20_players(self):
        S = Round.Stage
        self._check(self._run(20), [S.SWISS] * 3 + [S.TOP_4])

    def test_40_players(self):
        S = Round.Stage
        t = self._run(40)
        standings = self._check(t, [S.SWISS] * 3 + [S.TOP_16, S.TOP_4])
        semi = {p for pod in t.rounds[3].pods for p in pod.players}
        self.assertEqual(set(standings[:16]), semi)

    def test_90_players(self):
        S = Round.Stage
        t = self._run(90)
        standings = self._check(t, [S.SWISS] * 4 + [S.TOP_16, S.TOP_4])
        game4 = t.rounds[3]
        self.assertEqual(game4.logic.name, "PairingCatanSerpentine")
        out = {p for p in t.players
               if not any(p.result(r) == p.EResult.WIN for r in t.rounds[:3])}
        self.assertTrue(out)
        self.assertFalse(out & {p for pod in game4.pods for p in pod.players})
        self.assertEqual(sum(len(pod) for pod in game4.pods), 90 - len(out))
        self.assertEqual(set(standings[:16]), {p for pod in t.rounds[4].pods for p in pod.players})


if __name__ == "__main__":
    unittest.main()

"""PairingTopCut pairs every Commander cut exactly like the classes it replaced.

_Reference is a frozen copy of the six former per-cut classes (PairingTop4 ..
PairingTop40). Each playoff round is paired by PairingTopCut, then reset and
re-paired by the reference on the same state; pods and byes must match.
"""

import unittest

from src.core import Tournament, TournamentAction, TournamentConfiguration
from src.interface import IPlayer
from src.logic.commander.matching import CommonPairing
from src.logic.commander.rules import CommanderRuleset

TournamentAction.LOGF = False  # type: ignore

# N_BYES of the former PairingTop7/10/13/16; PairingTop4 gave no byes.
# PairingTop40 gave 16 (a bug: no pod winner could advance), so stage 40 is not
# compared; tests/test_commander_topcut_outcome.py covers it.
_OLD_BYES = {7: 3, 10: 2, 13: 1, 16: 0}


class _Reference(CommonPairing):
    def advance_topcut(self, tour_round, standings):
        for i in range(_OLD_BYES.get(tour_round.stage.value, 0)):
            standings[i].set_result(tour_round, IPlayer.EResult.BYE)

    def make_pairings(self, tour_round, players, pods):
        if tour_round.stage.value == 4:  # former PairingTop4
            prev_round = tour_round.tour.previous_round(tour_round)  # pyright: ignore[reportAttributeAccessIssue]
            standings = tour_round.tour.get_standings(prev_round)
            for p in sorted(tour_round.active_players, key=lambda x: standings.index(x)):
                pods[0].add_player(p)
            return players
        # former PairingSemiCommon.make_pairings
        standings = tour_round.tour.get_standings(tour_round)
        assignable_players = sorted(
            (tour_round.active_players - set(tour_round.byes)),
            key=lambda x: standings.index(x),
        )
        n_pods = len(pods)
        for i, p in enumerate(assignable_players):
            pass_num = i // n_pods
            pos = i % n_pods
            pod_idx = pos if pass_num % 2 == 0 else (n_pods - 1 - pos)
            pods[pod_idx].add_player(p)
        return players


def _snapshot(r):
    return (
        [[p.name for p in pod.players] for pod in r.pods],
        sorted(p.name for p in r.byes),
    )


class TestTopCutMatchesFormerClasses(unittest.TestCase):
    def setUp(self):
        Tournament.discover_pairing_logic()
        Tournament._pairing_logic_cache["_Reference"] = _Reference(name="_Reference")  # pyright: ignore[reportArgumentType]
        self.addCleanup(Tournament._pairing_logic_cache.pop, "_Reference")

    def test_every_cut_size(self):
        for top_cut in sorted(CommanderRuleset.PLAYOFFS):
            for attempt in range(3):
                with self.subTest(top_cut=top_cut, attempt=attempt):
                    self._check(top_cut)

    def _check(self, top_cut):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4], allow_bye=False, auto_export=False,
                n_rounds=2, top_cut=top_cut,
            )
        )
        t.add_player([f"P{i:02d}" for i in range(16 if top_cut <= 16 else 48)])
        for _ in range(2):
            t.create_pairings()
            t.random_results()
        stages = []
        while t.create_pairings():
            r = t.tour_round
            assert r is not None
            self.assertEqual(r.logic.name, "PairingTopCut")
            stages.append(r.stage.value)
            if r.stage.value == 40:
                t.random_results()
                continue
            new = _snapshot(r)
            t.reset_pods()
            r._logic = "_Reference"
            t.create_pairings()
            self.assertEqual(new, _snapshot(r), f"stage {r.stage.value}")
            r._logic = "PairingTopCut"
            t.random_results()
        self.assertEqual(stages, [s for s, _ in CommanderRuleset.PLAYOFFS[top_cut]])

    def test_old_names_resolve(self):
        for n in (4, 7, 10, 13, 16, 40):
            self.assertEqual(
                Tournament.get_pairing_logic(f"PairingTop{n}").name, "PairingTopCut"
            )

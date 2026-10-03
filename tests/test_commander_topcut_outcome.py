"""Every Commander cut advances exactly the bye players plus the pod winners.

Plays each cut size through every stage of CommanderRuleset.PLAYOFFS, always
letting the worst-seeded player of each pod win, so seed order alone cannot
make the assertions pass. Draws are not covered here: the engine advances the
best-seeded player of a drawn pod (Round.advancing_players).
"""

import unittest

from src.core import Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult
from src.logic.commander.rules import CommanderRuleset

TournamentAction.LOGF = False  # type: ignore

# cut size -> [(players, byes) per stage]
_EXPECTED = {
    4: [(4, 0)],
    7: [(7, 3), (4, 0)],
    10: [(10, 2), (4, 0)],
    13: [(13, 1), (4, 0)],
    16: [(16, 0), (4, 0)],
    40: [(40, 8), (16, 0), (4, 0)],
}


class TestTopCutAdvancement(unittest.TestCase):
    def test_every_cut_size(self):
        self.assertEqual(set(_EXPECTED), set(CommanderRuleset.PLAYOFFS))
        for top_cut, expected in _EXPECTED.items():
            with self.subTest(top_cut=top_cut):
                self._play(top_cut, expected)

    def _play(self, top_cut, expected):
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

        advancing = None
        for stage, (n_players, n_byes) in enumerate(expected):
            self.assertTrue(t.create_pairings())
            r = t.tour_round
            assert r is not None
            self.assertEqual(r.stage.value, CommanderRuleset.PLAYOFFS[top_cut][stage][0])
            self.assertEqual(len(r.active_players), n_players)
            self.assertEqual(len(r.byes), n_byes)
            if advancing is not None:
                self.assertEqual(r.active_players, advancing)

            standings = t.get_standings(r)
            winners = set()
            for pod in r.pods:
                worst = max(pod.players, key=standings.index)
                r.record_result(pod, [IGameResult({worst.uid})])
                winners.add(worst)
            advancing = set(r.byes) | winners
            # Next stage's size: byes + one winner per pod.
            if stage + 1 < len(expected):
                self.assertEqual(len(advancing), expected[stage + 1][0])
        self.assertFalse(t.create_pairings())


if __name__ == "__main__":
    unittest.main()

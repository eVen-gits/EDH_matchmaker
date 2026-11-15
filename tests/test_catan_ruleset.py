"""CatanRuleset report validation and random_report."""

import random
import unittest
from typing import Any
from types import SimpleNamespace
from uuid import uuid4

from src.interface import IGameResult
from src.logic.catan.rules import CatanConfiguration, CatanRuleset


def _pod(n=4, **params) -> Any:
    players = [SimpleNamespace(uid=uuid4()) for _ in range(n)]
    config = SimpleNamespace(
        ruleset_overrides=lambda _r: params, seat_win_rates=[0.26, 0.25, 0.25, 0.24]
    )
    rnd = SimpleNamespace(tour=SimpleNamespace(config=config))
    return SimpleNamespace(name="Pod 1", players=players, tour_round=rnd)


def _game(pod, winners, vps):
    return IGameResult(
        frozenset(pod.players[i].uid for i in winners),
        tuple((p.uid, v) for p, v in zip(pod.players, vps)) if vps else None,
    )


class CatanRulesetTests(unittest.TestCase):
    rules = CatanRuleset("CatanRuleset")

    def test_valid_report(self):
        pod = _pod()
        game = _game(pod, [2], [7, 5, 10, 2])
        self.rules.validate_report(pod, [game])
        pod.games = [game]
        self.assertEqual(self.rules.match_winners(pod), game.winners)

    def test_three_player_pod_valid(self):
        pod = _pod(3)
        self.rules.validate_report(pod, [_game(pod, [0], [11, 8, 3])])

    def test_winner_below_target_refused(self):
        pod = _pod()
        with self.assertRaisesRegex(ValueError, "at least 10"):
            self.rules.validate_report(pod, [_game(pod, [0], [9, 5, 4, 3])])

    def test_target_param_applies(self):
        pod = _pod(victory_points=12)
        with self.assertRaisesRegex(ValueError, "at least 12"):
            self.rules.validate_report(pod, [_game(pod, [0], [10, 5, 4, 3])])

    def test_two_winners_refused(self):
        pod = _pod()
        with self.assertRaisesRegex(ValueError, "exactly one winner"):
            self.rules.validate_report(pod, [_game(pod, [0, 1], [10, 10, 4, 3])])

    def test_other_player_at_target_refused(self):
        pod = _pod()
        with self.assertRaisesRegex(ValueError, "only the winner"):
            self.rules.validate_report(pod, [_game(pod, [0], [10, 10, 4, 3])])

    def test_missing_vp_refused(self):
        pod = _pod()
        with self.assertRaisesRegex(ValueError, "victory points"):
            self.rules.validate_report(pod, [_game(pod, [0], None)])
        partial = IGameResult(
            frozenset({pod.players[0].uid}),
            tuple((p.uid, 10) for p in pod.players[:3]),
        )
        with self.assertRaisesRegex(ValueError, "exactly the seated"):
            self.rules.validate_report(pod, [partial])

    def test_report_win_without_vp_refused(self):
        pod = _pod()
        with self.assertRaisesRegex(ValueError, "needs victory points"):
            self.rules.report_from_winners(pod, [pod.players[0].uid])

    def test_random_report_is_valid(self):
        random.seed(1)
        for n in (3, 4):
            pod = _pod(n)
            for _ in range(50):
                self.rules.validate_report(pod, self.rules.random_report(pod))

    def test_defaults_and_selectable(self):
        self.assertTrue(CatanRuleset.IS_COMPLETE)
        self.assertFalse(CatanRuleset.DEFAULT_ALLOW_BYE)
        self.assertEqual(
            CatanRuleset.DEFAULT_PARAMS,
            {"victory_points": 10, "eliminate_below_wins": 0},
        )
        self.assertEqual(
            CatanConfiguration.GAME_FIELDS["seat_win_rates"], [0.26, 0.25, 0.25, 0.24]
        )


if __name__ == "__main__":
    unittest.main()

"""Opt-in IRuleset hooks: defaults are no-ops, overrides are called and honoured."""

import unittest
from unittest import mock

from src.core import Round, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IRuleset
from src.logic.commander.rules import CommanderRuleset
from src.logic.mtg.rules import Mtg1v1Ruleset

TournamentAction.LOGF = False  # type: ignore


def _tour(**kwargs) -> Tournament:
    t = Tournament(TournamentConfiguration(auto_export=False, n_rounds=3, **kwargs))
    t.add_player([f"P{i}" for i in range(8)])
    t.create_pairings()
    t.random_results()
    return t


class RulesetHookTests(unittest.TestCase):
    def test_defaults(self):
        self.assertTrue(IRuleset.DEFAULT_ALLOW_BYE)
        t = _tour()
        rs = t.ruleset
        self.assertIsNone(rs.playoff_standings(t, t.tour_round))
        self.assertEqual(rs.swiss_eliminations(t.tour_round, t.get_standings()), set())

    def test_default_allow_bye_follows_ruleset(self):
        self.assertTrue(TournamentConfiguration(ruleset="CommanderRuleset").allow_bye)
        with mock.patch.object(CommanderRuleset, "DEFAULT_ALLOW_BYE", False):
            self.assertFalse(TournamentConfiguration(ruleset="CommanderRuleset").allow_bye)
            self.assertTrue(TournamentConfiguration(allow_bye=True).allow_bye)
        self.assertTrue(TournamentConfiguration(ruleset="Mtg1v1Ruleset").allow_bye)

    def test_swiss_eliminations_disables_but_keeps_ranked(self):
        t = _tour()
        out = t.get_standings()[-1]
        with mock.patch.object(
            CommanderRuleset, "swiss_eliminations", return_value={out}
        ) as hook:
            t.create_pairings()
        hook.assert_called_once()
        self.assertEqual(hook.call_args.args[0], t.tour_round)
        self.assertNotIn(out, t.tour_round.active_players)
        self.assertIn(out, t.get_standings())

    def test_swiss_eliminations_not_called_for_first_round(self):
        t = Tournament(TournamentConfiguration(auto_export=False))
        t.add_player([f"P{i}" for i in range(8)])
        with mock.patch.object(CommanderRuleset, "swiss_eliminations") as hook:
            t.create_pairings()
        hook.assert_not_called()

    def test_playoff_standings_override_used(self):
        t = _tour(top_cut=4)
        while t.tour_round.stage == Round.Stage.SWISS:
            t.create_pairings()
            t.random_results()
            if t.tour_round.seq < 3:
                continue
            break
        t.create_pairings()  # first playoff round
        rnd = t.tour_round
        self.assertNotEqual(rnd.stage, Round.Stage.SWISS)
        reverse = list(reversed(t.get_standings(rnd)))
        with mock.patch.object(
            CommanderRuleset, "playoff_standings", return_value=reverse
        ) as hook:
            self.assertEqual(t.get_standings(rnd), reverse)
        hook.assert_called_once_with(t, rnd)

    def test_playoff_standings_not_asked_for_swiss(self):
        t = _tour()
        with mock.patch.object(CommanderRuleset, "playoff_standings") as hook:
            t.get_standings()
        hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()

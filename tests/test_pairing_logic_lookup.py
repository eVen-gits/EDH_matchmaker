import unittest

from src.core import Round, Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


class PairingLogicLookupTest(unittest.TestCase):
    def test_commander_swiss_rounds(self):
        t = Tournament(
            TournamentConfiguration(
                ruleset="CommanderRuleset", auto_export=False, n_rounds=4, top_cut=4
            )
        )
        t.add_player([f"P{i}" for i in range(8)])
        names = [t.pairing_logic_name_for(i) for i in range(3)]
        self.assertEqual(names[0], "PairingRandom")
        self.assertIn(names[1], ("PairingSnake", "PairingDefault"))
        self.assertIn(names[2], ("PairingSnake", "PairingDefault"))

    def test_matches_created_round(self):
        t = Tournament(
            TournamentConfiguration(ruleset="CommanderRuleset", auto_export=False)
        )
        t.add_player([f"P{i}" for i in range(8)])
        expected = t.pairing_logic_name_for(0)
        t.new_round()
        self.assertEqual(t.tour_round.logic.name, expected)

    def test_top_cut_stage(self):
        t = Tournament(
            TournamentConfiguration(
                ruleset="CommanderRuleset", auto_export=False, n_rounds=2, top_cut=4
            )
        )
        t.add_player([f"P{i}" for i in range(8)])
        self.assertEqual(t.pairing_logic_name_for(2, Round.Stage.SWISS), "PairingTopCut")

    def test_no_round_after_swiss_without_cut(self):
        t = Tournament(
            TournamentConfiguration(
                ruleset="CommanderRuleset", auto_export=False, n_rounds=2
            )
        )
        self.assertIsNone(t.pairing_logic_name_for(2, Round.Stage.SWISS))

    def test_1v1(self):
        t = Tournament(
            TournamentConfiguration(
                ruleset="Mtg1v1Ruleset", auto_export=False, n_rounds=3
            )
        )
        t.add_player(["A", "B"])
        self.assertEqual(t.pairing_logic_name_for(0), "Pairing1v1")
        self.assertEqual(t.pairing_logic_name_for(2), "Pairing1v1")


if __name__ == "__main__":
    unittest.main()

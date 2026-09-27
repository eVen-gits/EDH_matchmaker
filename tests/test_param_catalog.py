import json
import unittest

from src.core import StandingsExport, TournamentAction, TournamentConfiguration
from src.param_catalog import catalog
from src.param_spec import validate_values

TournamentAction.LOGF = False


class TestConfigSidecars(unittest.TestCase):
    """Tournament config and per-game fields come from their sidecars."""

    def test_config_defaults_from_sidecar(self):
        spec = TournamentConfiguration.PARAM_SPEC
        config = TournamentConfiguration()
        for name in ("ruleset", "allow_bye", "n_rounds", "max_byes", "auto_export"):
            self.assertEqual(getattr(config, name), spec[name].default)
        self.assertEqual(config.top_cut, TournamentConfiguration.TopCut.NONE)
        self.assertEqual(spec["max_byes"].visible_when, ("allow_bye", True))
        self.assertEqual(spec["standings_export"].visible_when, ("auto_export", True))

    def test_static_choices_match_enums(self):
        spec = TournamentConfiguration.PARAM_SPEC
        self.assertEqual(
            list(spec["top_cut"].choices or ()),
            [c.value for c in TournamentConfiguration.TopCut],
        )
        self.assertEqual(
            list(spec["standings_export"].choices or ()),
            [f.name for f in StandingsExport.Field],
        )

    def test_game_fields_from_sidecar(self):
        commander = TournamentConfiguration()
        self.assertEqual(
            commander.GAME_FIELDS, {"global_wr_seats": [0.2470, 0.1928, 0.1672, 0.1458]}
        )
        mtg = TournamentConfiguration(ruleset="Mtg1v1Ruleset")
        self.assertEqual(
            mtg.GAME_FIELDS, {"match_wr_seats": [0.5, 0.5], "match_draw_rate": 0.1}
        )

    def test_validate_list_values(self):
        spec = TournamentConfiguration.PARAM_SPEC
        validate_values(spec, {"pod_sizes": [4, 3]}, "t")
        with self.assertRaises(ValueError):
            validate_values(spec, {"pod_sizes": [4, "3"]}, "t")
        with self.assertRaises(ValueError):
            validate_values(spec, {"standings_export": ["NOPE"]}, "t")


class TestParamCatalog(unittest.TestCase):
    def test_catalog_covers_everything_as_json(self):
        data = json.loads(json.dumps(catalog()))
        self.assertIn("pod_sizes", data["tournament"])
        self.assertEqual(data["tournament"]["max_byes"]["visible_when"], {"allow_bye": True})
        commander = data["games"]["commander"]
        ruleset = commander["rulesets"]["CommanderRuleset"]
        self.assertEqual(ruleset["defaults"]["pod_sizes"], [4, 3])
        self.assertIn("global_wr_seats", ruleset["config_fields"])
        self.assertIn("wager_percent", commander["scoring"]["ScoringHareruya"])
        self.assertIn("PairingDefault", commander["pairing"])
        mtg = data["games"]["mtg"]
        self.assertIn("games_to_win", mtg["rulesets"]["Mtg1v1Ruleset"]["params"])
        self.assertEqual(mtg["rulesets"]["Mtg1v1Ruleset"]["defaults"]["scoring_logic"], "Scoring1v1")
        self.assertIn("Scoring1v1", mtg["scoring"])


if __name__ == "__main__":
    unittest.main()

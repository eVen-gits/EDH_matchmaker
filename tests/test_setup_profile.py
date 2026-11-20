import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml

from src.core import Tournament, TournamentAction, TournamentConfiguration
from src.interface import IPlayer
from src.param_catalog import catalog
from src.setup_profile import load_setup_profile

TournamentAction.LOGF = False

SHIPPED = Path(__file__).parent.parent / "src" / "logic" / "catan" / "CatanRuleset.setup.yaml"


class TestShippedProfiles(unittest.TestCase):
    """The setup profiles shipped with each ruleset carry the recorded decisions."""

    def profile(self, name):
        return Tournament.get_ruleset(name).SETUP_PROFILE

    def test_every_ruleset_has_a_profile(self):
        for name in Tournament.ruleset_names():
            self.assertTrue(self.profile(name), name)

    def test_categories_and_timers(self):
        commander = self.profile("CommanderRuleset")
        mtg = self.profile("Mtg1v1Ruleset")
        catan = self.profile("CatanRuleset")
        self.assertEqual([p["category"] for p in (commander, mtg, catan)], ["mtg", "mtg", "catan"])
        self.assertEqual(commander["timer"], {"default_minutes": 75})
        self.assertEqual(mtg["timer"], {"default_minutes": 50})
        self.assertIsNone(catan["timer"]["default_minutes"])
        self.assertEqual(catan["timer"]["turn"]["default"], "judge_enforced")
        self.assertEqual(catan["timer"]["phases"]["map_study"]["seconds"], 300)

    def test_catan_profile(self):
        catan = self.profile("CatanRuleset")
        ruleset = Tournament.get_ruleset("CatanRuleset")
        self.assertEqual(catan["companion"], {"tracker": "vp", "target": ruleset.DEFAULT_PARAMS["victory_points"]})
        self.assertEqual(catan["confirmations"], "all")
        self.assertEqual(catan["seat_order"], "engine")
        self.assertEqual(catan["valid_from_players"], 12)
        self.assertEqual(sorted(ruleset.PLAYOFFS), [4, 16])  # rule 1.1.5: no top 40

    def test_points_match_scoring_defaults(self):
        for name, scoring, bye in (
            ("CommanderRuleset", "ScoringDefault", 4),  # FairPair event default, not 7
            ("Mtg1v1Ruleset", "Scoring1v1", 3),
            ("CatanRuleset", "ScoringCatan", 0),
        ):
            points = self.profile(name)["points"]
            defaults = Tournament.get_scoring_logic(scoring).DEFAULT_PARAMS
            self.assertEqual(points["win"], defaults["win_points"], name)
            self.assertEqual(points["draw"], defaults["draw_points"], name)
            self.assertEqual(points["bye"], bye, name)


class TestCatalogExport(unittest.TestCase):
    def test_catalog_exports_profiles_playoffs_and_categories(self):
        data = json.loads(json.dumps(catalog()))
        self.assertEqual(
            {c: sorted(r) for c, r in data["categories"].items()},
            {"mtg": ["CommanderRuleset", "Mtg1v1Ruleset"], "catan": ["CatanRuleset"]},
        )
        self.assertEqual(data["categories"]["mtg"]["Mtg1v1Ruleset"]["cube"], "Cube")
        catan = data["games"]["catan"]["rulesets"]["CatanRuleset"]
        self.assertEqual(catan["playoffs"], {"4": [4], "16": [16, 4]})
        self.assertEqual(catan["setup"]["report"], "scores")
        commander = data["games"]["commander"]["rulesets"]["CommanderRuleset"]
        self.assertEqual(commander["playoffs"]["40"], [40, 16, 4])


class TestLoaderValidation(unittest.TestCase):
    """A malformed sidecar fails at load, naming the file."""

    def load(self, profile, **attrs):
        with tempfile.TemporaryDirectory() as d:
            module_name = "tmp_setup_profile_mod"
            mod = types.ModuleType(module_name)
            mod.__file__ = str(Path(d) / f"{module_name}.py")
            sys.modules[module_name] = mod
            (Path(d) / "Fake.setup.yaml").write_text(yaml.safe_dump(profile))
            attrs = {"PLAYOFFS": {4: (), 16: ()}, "DEFAULT_ALLOW_BYE": False, **attrs}
            cls = type("Fake", (), {"__module__": module_name, **attrs})
            try:
                return load_setup_profile(cls)
            finally:
                del sys.modules[module_name]

    def catan(self):
        return yaml.safe_load(SHIPPED.read_text())

    def test_valid_profile_loads(self):
        self.assertEqual(self.load(self.catan())["category"], "catan")

    def test_no_sidecar_is_empty(self):
        self.assertEqual(load_setup_profile(type("NoSidecar", (), {})), {})

    def assert_rejects(self, profile, fragment, **attrs):
        with self.assertRaisesRegex(ValueError, fragment):
            self.load(profile, **attrs)

    def test_rejects_missing_and_unknown_keys(self):
        p = self.catan()
        del p["report"]
        self.assert_rejects(p, "missing keys")
        self.assert_rejects({**self.catan(), "extra": 1}, "unknown keys")

    def test_rejects_byes_disagreeing_with_ruleset(self):
        self.assert_rejects(self.catan(), "'byes'", DEFAULT_ALLOW_BYE=True)

    def test_rejects_top_cut_outside_playoffs(self):
        self.assert_rejects(self.catan(), "top_cut 16", PLAYOFFS={4: ()})

    def test_rejects_bad_values(self):
        for key, value, fragment in (
            ("confirmations", 3, "'confirmations'"),
            ("companion", {"tracker": "vp", "start": 10}, "'companion'"),
            ("points", {"win": 1, "draw": 0, "bye": 0}, "'points'"),
            ("timer", {"default_minutes": 0}, "default_minutes"),
            ("timer", {"default_minutes": None, "turn": {"default": "x", "options": {}}}, "turn.default"),
        ):
            with self.subTest(key=key, value=value):
                self.assert_rejects({**self.catan(), key: value}, fragment)


class TestUnseatableCatanField(unittest.TestCase):
    """A field Catan tables of 3-4 cannot seat is refused with a clear error."""

    def test_refused_with_clear_error(self):
        for logic in (None, "PairingRandom", "PairingDefault", "PairingSnake"):
            rounds = [{"logic": logic, "params": {}}] if logic else None
            for n in (1, 2, 5):
                with self.subTest(logic=logic, n=n):
                    self.assert_refused(n, rounds)

    def assert_refused(self, n, rounds):
        kwargs = {"pairing_rounds": rounds} if rounds else {}
        t = Tournament(TournamentConfiguration(ruleset="CatanRuleset", **kwargs))
        t.add_player([f"p{i}" for i in range(n)])
        with self.assertRaisesRegex(ValueError, f"{n} players cannot be seated"):
            t.create_pairings()
        # Refused before anyone is benched.
        assert t.tour_round is not None
        self.assertFalse(
            any(p.result(t.tour_round) == IPlayer.EResult.BYE for p in t.players)
        )

    def test_hand_seated_table_of_five_survives_pairing(self):
        # Recorded decision: a judge may hand-seat a table of 5. Pairing the
        # fully seated round must not crash (it raised "max() iterable
        # argument is empty") nor bench anyone.
        t = Tournament(TournamentConfiguration(ruleset="CatanRuleset"))
        t.add_player([f"p{i}" for i in range(5)])
        players = list(t.players)
        t.manual_pod(players[:4])
        assert t.tour_round is not None
        t.move_player_to_pod(t.tour_round.pods[0], players[4], manual=True)
        t.create_pairings()
        self.assertEqual([len(p) for p in t.tour_round.pods], [5])
        self.assertFalse(t.tour_round.byes)

    def test_seatable_fields_still_pair(self):
        for n in (3, 4, 6, 7, 9):
            with self.subTest(n=n):
                t = Tournament(TournamentConfiguration(ruleset="CatanRuleset"))
                t.add_player([f"p{i}" for i in range(n)])
                self.assertTrue(t.create_pairings())
                assert t.tour_round is not None
                self.assertEqual(sum(len(p) for p in t.tour_round.pods), n)


if __name__ == "__main__":
    unittest.main()

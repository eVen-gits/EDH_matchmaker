"""Validates the tournament-log JSON Schema against real code output and
against the checked-in conformance example files.

This is the drift detector for docs/tournament-log-spec.md: if
Tournament.serialize() ever disagrees with docs/tournament-log.schema.json,
test_real_tournament_output_matches_schema fails.
"""

import glob
import json
import os
import unittest

from jsonschema import Draft202012Validator, ValidationError

from src.core import Tournament, TournamentAction, TournamentConfiguration
from src.misc import generate_player_names

TournamentAction.LOGF = False  # type: ignore

SCHEMA_PATH = "docs/tournament-log.schema.json"
EXAMPLES_DIR = "docs/tournament-log-examples"


def _load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


class TestLogSchema(unittest.TestCase):
    def test_schema_file_is_valid_json_schema(self):
        Draft202012Validator.check_schema(_load_schema())

    def test_real_tournament_output_matches_schema(self):
        config = TournamentConfiguration(
            pod_sizes=[4, 3], allow_bye=True, auto_export=False
        )
        t = Tournament(config)
        t.new_round()
        t.add_player(generate_player_names(8))
        t.create_pairings()
        t.random_results()

        validator = Draft202012Validator(_load_schema())
        validator.validate(t.serialize())

    def test_valid_examples_pass_schema(self):
        validator = Draft202012Validator(_load_schema())
        paths = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "valid-*.json")))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path):
                with open(path) as f:
                    data = json.load(f)
                validator.validate(data)

    def test_invalid_examples_fail_schema(self):
        validator = Draft202012Validator(_load_schema())
        paths = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "invalid-*.json")))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path):
                with open(path) as f:
                    data = json.load(f)
                with self.assertRaises(ValidationError):
                    validator.validate(data)

    def test_real_captured_fixture_matches_schema(self):
        log_path = "logs/tournament-state-699863dcebe4eb89e31bc50b-2026-02-25.json"
        if not os.path.exists(log_path):
            self.skipTest("Real tournament file not available")
            
        validator = Draft202012Validator(_load_schema())
        with open(log_path) as f:
            data = json.load(f)
        # Files predating format_version have no format_version/generator/
        # created_at/updated_at keys - the schema marks all four optional,
        # so this also proves backward compatibility of the schema itself.
        validator.validate(data)


if __name__ == "__main__":
    unittest.main()

"""CLI overrides must obey the same rules as the configuration dialog."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from src.core import TournamentAction

TournamentAction.LOGF = False
ROOT = Path(__file__).resolve().parents[1]


class TestCliConfigValidation(unittest.TestCase):
    def test_invalid_pod_sizes_exit_before_gui(self):
        for sizes in (("4", "3"), ()):
            with self.subTest(sizes=sizes):
                result = subprocess.run(
                    [sys.executable, "run_ui.py", "--ruleset", "Mtg1v1Ruleset", "-s", *sizes],
                    cwd=ROOT,
                    env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pod_sizes", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_valid_overrides(self):
        from run_ui import apply_cli_config, build_arg_parser
        from src.core import Tournament, TournamentConfiguration

        tour = Tournament(TournamentConfiguration(ruleset="Mtg1v1Ruleset"))
        args = build_arg_parser().parse_args(["-s", "2", "-b", "-r", "3", "-x", "3", "1", "3"])
        apply_cli_config(tour, args)
        self.assertEqual(tour.config.pod_sizes, [2])
        self.assertEqual(tour.config.n_rounds, 3)
        self.assertTrue(tour.config.allow_bye)
        self.assertEqual(tour.config.scoring_params["win_points"], 3)

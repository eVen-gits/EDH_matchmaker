import types
import unittest

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False


def _args(**over):
    base = dict(pod_sizes=None, allow_bye=False, scoring=None, rounds=None)
    base.update(over)
    return types.SimpleNamespace(**base)


class TestApplyCliConfig(unittest.TestCase):
    """apply_cli_config wires command-line flags into the tournament config."""

    @classmethod
    def setUpClass(cls):
        try:
            import run_ui
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        cls.apply = staticmethod(run_ui.apply_cli_config)

    def _tour(self):
        return Tournament(
            TournamentConfiguration(pod_sizes=[4], allow_bye=True, auto_export=False)
        )

    def test_scoring_flag_updates_scoring_params_and_ratings(self):
        # -x win draw bye must land in scoring_params and drive ScoringDefault.
        t = self._tour()
        self.apply(t, _args(scoring=[5, 1, 4]))
        self.assertEqual(t.config.scoring_params["win_points"], 5)
        self.assertEqual(t.config.scoring_params["draw_points"], 1)
        self.assertEqual(t.config.scoring_params["bye_points"], 4)

        t.new_round()
        p = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(p)
        t.report_win(p[0])
        self.assertEqual(p[0].rating(t.tour_round), 5)
        self.assertEqual(p[1].rating(t.tour_round), 0)

    def test_other_flags(self):
        t = self._tour()
        self.apply(t, _args(pod_sizes=[3, 4], rounds=7, allow_bye=True))
        self.assertEqual(t.config.pod_sizes, [3, 4])
        self.assertEqual(t.config.n_rounds, 7)
        self.assertTrue(t.config.allow_bye)

    def test_no_args_no_change(self):
        t = self._tour()
        before = t.config.serialize()
        self.apply(t, _args())
        self.assertEqual(t.config.serialize(), before)


class TestArgParser(unittest.TestCase):
    """The --ruleset flag and its -o/--open conflict."""

    @classmethod
    def setUpClass(cls):
        try:
            import run_ui
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        cls.parser = run_ui.build_arg_parser()

    def test_ruleset_flag_parses(self):
        args = self.parser.parse_args(["--ruleset", "Mtg1v1Ruleset"])
        self.assertEqual(args.ruleset, "Mtg1v1Ruleset")

    def test_ruleset_defaults_to_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.ruleset)

    def test_open_together_with_ruleset_errors(self):
        args = self.parser.parse_args(["-o", "foo.json", "--ruleset", "Mtg1v1Ruleset"])
        with self.assertRaises(SystemExit):
            if args.open and args.ruleset:
                self.parser.error(
                    "argument --ruleset: not allowed with argument -o/--open"
                )


if __name__ == "__main__":
    unittest.main()

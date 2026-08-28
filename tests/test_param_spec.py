import sys
import tempfile
import types
import unittest
from pathlib import Path

from src.core import Tournament
from src.param_spec import ParamSpec, load_param_spec


class TestShippedParamSpecs(unittest.TestCase):
    """The sidecar YAML files shipped with each algorithm load and validate."""

    def test_default_params_derived_from_sidecar(self):
        # Derived defaults must be byte-identical to the pre-sidecar literals,
        # so no scoring behavior changes.
        logic = Tournament.get_scoring_logic("ScoringDefault")
        self.assertEqual(
            logic.DEFAULT_PARAMS,
            {"win_points": 5, "draw_points": 1, "bye_points": 4},
        )
        # Types matter: these feed StandingsExport's "{:d}" formatting.
        for value in logic.DEFAULT_PARAMS.values():
            self.assertIs(type(value), int)

    def test_hareruya_params_derived_from_sidecar(self):
        logic = Tournament.get_scoring_logic("ScoringHareruya")
        self.assertEqual(
            logic.DEFAULT_PARAMS,
            {
                "wager_percent": 0.07,
                "wagering_starting_points": 1000,
                "draw_redistribution_fraction": 1.0,
                "draw_distribution_shape": 1.0,
                "redistribute_discarded_draw_points": False,
                "draw_discard_pod_fraction": 1.0,
            },
        )
        self.assertIs(type(logic.DEFAULT_PARAMS["wagering_starting_points"]), int)
        self.assertIs(
            type(logic.DEFAULT_PARAMS["redistribute_discarded_draw_points"]), bool
        )

    def test_specs_carry_descriptions_and_metadata(self):
        spec = Tournament.get_scoring_logic("ScoringHareruya").PARAM_SPEC
        wager = spec["wager_percent"]
        self.assertIsInstance(wager, ParamSpec)
        self.assertTrue(wager.description)  # non-empty explanation
        self.assertEqual(wager.type, "float")
        self.assertEqual((wager.min, wager.max), (0, 1))
        self.assertEqual(wager.label, "Wager %")

    def test_modified_inherits_hareruya_spec(self):
        # ScoringModifiedHareruya ships no sidecar; it must inherit its parent's.
        modified = Tournament.get_scoring_logic("ScoringModifiedHareruya")
        hareruya = Tournament.get_scoring_logic("ScoringHareruya")
        self.assertEqual(modified.PARAM_SPEC, hareruya.PARAM_SPEC)
        self.assertEqual(modified.DEFAULT_PARAMS, hareruya.DEFAULT_PARAMS)

    def test_pairing_logic_without_sidecar_has_empty_spec(self):
        logic = Tournament.get_pairing_logic("PairingDefault")
        self.assertEqual(logic.PARAM_SPEC, {})
        self.assertEqual(logic.DEFAULT_PARAMS, {})


class TestLoaderValidation(unittest.TestCase):
    """load_param_spec discovers a class's sidecar and rejects malformed ones."""

    def _load(self, class_name, yaml_text):
        # Build a throwaway class whose module lives in a temp dir, drop a
        # sidecar next to it, and load - exercising discovery + validation.
        with tempfile.TemporaryDirectory() as d:
            module_name = "tmp_param_spec_mod"
            mod = types.ModuleType(module_name)
            mod.__file__ = str(Path(d) / f"{module_name}.py")
            sys.modules[module_name] = mod
            (Path(d) / f"{class_name}.params.yaml").write_text(yaml_text)
            cls = type(class_name, (), {"__module__": module_name})
            try:
                return load_param_spec(cls)
            finally:
                del sys.modules[module_name]

    def test_valid_file_loads(self):
        spec = self._load(
            "Good",
            "k:\n  default: 0.5\n  type: float\n  min: 0\n  max: 1\n"
            "  description: A fraction.\n",
        )
        self.assertEqual(spec["k"].default, 0.5)
        self.assertEqual(spec["k"].label, "k")  # defaults to the name

    def test_missing_description_raises(self):
        with self.assertRaises(ValueError):
            self._load("Bad", "k:\n  default: 1\n  type: int\n")

    def test_type_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._load(
                "Bad", "k:\n  default: 1\n  type: float\n  description: x\n"
            )

    def test_default_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self._load(
                "Bad",
                "k:\n  default: 2\n  type: int\n  min: 0\n  max: 1\n"
                "  description: x\n",
            )

    def test_no_sidecar_returns_empty(self):
        cls = type("Nope", (), {"__module__": "builtins"})
        self.assertEqual(load_param_spec(cls), {})

    def test_widget_inferred_from_type(self):
        spec = self._load(
            "Infer",
            "flag:\n  default: false\n  type: bool\n  description: x\n"
            "count:\n  default: 1\n  type: int\n  description: x\n"
            "frac:\n  default: 0.5\n  type: float\n  description: x\n",
        )
        self.assertEqual(spec["flag"].widget, "checkbox")
        self.assertEqual(spec["count"].widget, "spinbox")
        self.assertEqual(spec["frac"].widget, "doublespinbox")

    def test_choices_infer_combobox_and_validate_default(self):
        spec = self._load(
            "Ch",
            "mode:\n  default: a\n  type: str\n  choices: [a, b]\n"
            "  description: x\n",
        )
        self.assertEqual(spec["mode"].widget, "combobox")
        self.assertEqual(spec["mode"].choices, ("a", "b"))
        with self.assertRaises(ValueError):  # default not in choices
            self._load(
                "Bad",
                "mode:\n  default: z\n  type: str\n  choices: [a, b]\n"
                "  description: x\n",
            )

    def test_bad_widget_raises(self):
        with self.assertRaises(ValueError):
            self._load(
                "Bad",
                "k:\n  default: 1\n  type: int\n  widget: wobble\n"
                "  description: x\n",
            )

    def test_scale_suffix_parse(self):
        spec = self._load(
            "S",
            "pct:\n  default: 0.07\n  type: float\n  scale: 100\n"
            '  suffix: "%"\n  description: x\n',
        )
        self.assertEqual(spec["pct"].scale, 100)
        self.assertEqual(spec["pct"].suffix, "%")

    def test_visible_when_unknown_param_raises(self):
        with self.assertRaises(ValueError):
            self._load(
                "Bad",
                "k:\n  default: 1\n  type: int\n"
                "  visible_when: {ghost: true}\n  description: x\n",
            )


class TestParamForm(unittest.TestCase):
    """The GUI form generator builds the right widgets and reads them back."""

    @classmethod
    def setUpClass(cls):
        try:
            import os

            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PyQt6.QtWidgets import QApplication

            import run_ui
        except ImportError as exc:  # pragma: no cover - env without PyQt6
            raise unittest.SkipTest(f"PyQt6 unavailable: {exc}")
        cls.app = QApplication.instance() or QApplication([])
        cls.run_ui = run_ui

    def test_build_and_read_back_with_scale(self):
        spec = self.run_ui.Tournament.get_scoring_logic("ScoringHareruya").PARAM_SPEC
        form = self.run_ui.ParamForm(spec, {})
        # Wager shows scaled (0.07 -> 7.0 %) but reads back as the fraction.
        self.assertAlmostEqual(form._fields["wager_percent"].value(), 7.0)
        vals = form.values()
        self.assertAlmostEqual(vals["wager_percent"], 0.07)
        self.assertIs(vals["redistribute_discarded_draw_points"], False)
        self.assertEqual(vals["wagering_starting_points"], 1000)

    def test_visible_when_toggles(self):
        spec = self.run_ui.Tournament.get_scoring_logic("ScoringHareruya").PARAM_SPEC
        form = self.run_ui.ParamForm(spec, {})
        _, pod = form._rows["draw_discard_pod_fraction"]
        self.assertTrue(pod.isHidden())  # reclaim off by default
        form._fields["redistribute_discarded_draw_points"].setChecked(True)
        self.assertFalse(pod.isHidden())


if __name__ == "__main__":
    unittest.main()

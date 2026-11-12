"""Frozen-output (golden) tests for Commander and 1v1 MTG.

Each scenario plays a seeded tournament with a fixed clock and deterministic
UUIDs, then compares everything observable to a stored fixture in
``tests/golden/``: pods per round, standings in every export format, the pods
export, and the saved JSON. A later change that alters existing behaviour makes
these fail. Regenerate deliberately with ``GOLDEN_UPDATE=1`` (see
CONTRIBUTING.md, "Frozen-output tests").
"""

import json
import os
import random
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from uuid import UUID

from src.interface import IHashable
from src.core import StandingsExport, Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore

GOLDEN_DIR = Path(__file__).parent / "golden"
SEED = 20260101
_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]  # noqa: ARG003
        return _NOW


def _scenario(config: dict, n_players: int, n_swiss: int, cut_stages: int = 0) -> dict:
    """Plays one seeded tournament and returns all frozen outputs as plain data."""
    random.seed(SEED)
    counter = iter(range(1, 10**6))
    out: dict = {"rounds": []}
    with (
        mock.patch("src.interface.uuid4", lambda: UUID(int=next(counter))),
        mock.patch("src.core.datetime", _FixedDatetime),
        # Default object hashes are memory addresses, which change set iteration
        # order (and so seeded shuffles) between runs; hash by uid instead.
        mock.patch.object(IHashable, "__hash__", lambda self: hash(self.uid), create=True),
    ):
        t = Tournament(TournamentConfiguration(auto_export=False, **config))
        t.add_player([f"P{i:02d}" for i in range(n_players)])
        for _ in range(n_swiss + cut_stages):
            assert t.create_pairings()
            t.random_results()
            r = t.tour_round
            assert r is not None
            out["rounds"].append(
                {
                    "stage": r.stage.value,
                    "pods": [[p.name for p in pod.players] for pod in r.pods],
                    "byes": sorted(p.name for p in r.byes),
                    "pods_export": t.get_pods_str(),
                    "standings": {
                        fmt.name: t.get_standings_str(style=fmt)
                        for fmt in StandingsExport.Format
                    },
                }
            )
        out["saved"] = t.serialize()
    return out


# name -> (config, n_players, n_swiss_rounds, cut_stages)
SCENARIOS = {
    "commander_default": (dict(pod_sizes=[4], allow_bye=False, n_rounds=3), 16, 3, 0),
    "commander_byes_hareruya": (
        dict(pod_sizes=[3, 4], allow_bye=True, n_rounds=3, scoring_logic="ScoringHareruya"),
        13,
        3,
        0,
    ),
    "commander_top_cut_4": (
        dict(pod_sizes=[4], allow_bye=False, n_rounds=2, top_cut=4),
        16,
        2,
        1,
    ),
    "mtg_swiss": (dict(ruleset="Mtg1v1Ruleset", n_rounds=3), 8, 3, 0),
    "mtg_odd_field": (dict(ruleset="Mtg1v1Ruleset", n_rounds=3), 7, 3, 0),
}


class TestGolden(unittest.TestCase):
    def test_scenarios(self):
        update = os.environ.get("GOLDEN_UPDATE") == "1"
        for name, args in SCENARIOS.items():
            with self.subTest(scenario=name):
                actual = json.loads(json.dumps(_scenario(*args)))
                path = GOLDEN_DIR / f"{name}.json"
                if update:
                    GOLDEN_DIR.mkdir(exist_ok=True)
                    path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
                    continue
                self.assertEqual(actual, json.loads(path.read_text()))

    def test_scenarios_repeat(self):
        """The harness itself is deterministic: two runs give identical output."""
        args = SCENARIOS["commander_default"]
        self.assertEqual(_scenario(*args), _scenario(*args))

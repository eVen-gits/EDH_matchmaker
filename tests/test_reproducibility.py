"""A tournament is reproducible from its seed, with no test-time patching.

Each game's tournament is played in separate processes with different
PYTHONHASHSEED values, memory layouts and real uuid4; pairings, standings and the saved JSON
must match. A save and load mid-tournament must continue the same sequence.
"""

import json
import os
import random
import subprocess
import sys
import unittest
from typing import cast
from uuid import uuid4

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore

SEED = 1234
# name -> (config, n_players, n_rounds incl. playoff stages)
GAMES = {
    "commander": (dict(pod_sizes=[4, 3], allow_bye=True, n_rounds=3, top_cut=4), 13, 4),
    "mtg": (dict(ruleset="Mtg1v1Ruleset", n_rounds=3, top_cut=4), 9, 5),
    "catan": (dict(ruleset="CatanRuleset", n_rounds=4), 14, 4),
}


def _round_out(t: Tournament) -> dict:
    r = t.tour_round
    assert r is not None
    return {
        "pods": [[p.name for p in pod.players] for pod in r.pods],
        "byes": sorted(p.name for p in r.byes),
        "standings": t.get_standings_str(),
    }


def play(game: str, reload_after: int | None = None) -> dict:
    """Plays one seeded tournament; with reload_after, saves and reloads it then."""
    config, n_players, n_rounds = GAMES[game]
    random.seed(SEED)  # random_results simulates results with the global generator
    t = Tournament(TournamentConfiguration(auto_export=False, **config), seed=SEED)
    t.add_player([f"P{i:02d}" for i in range(n_players)])
    rounds = []
    for i in range(n_rounds):
        if i == reload_after:
            # Under a new uid, so it is a fresh load, not the cached tournament.
            saved = json.dumps(t.serialize()).replace(str(t.uid), str(uuid4()))
            t = cast(Tournament, Tournament.inflate(json.loads(saved)))
        assert t.create_pairings()
        t.random_results()
        rounds.append(_round_out(t))
    # The tournament uid stays random by design; everything else is seeded.
    saved = json.dumps(t.serialize()).replace(str(t.uid), "TOUR_UID")
    return {"rounds": rounds, "saved": json.loads(saved)}


def _play_in_process(game: str, hash_seed: str, padding: int) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": os.getcwd()}
    # padding shifts memory addresses, as a different session history would.
    code = (
        "import json, sys; from tests.test_reproducibility import play; "
        f"pad = [object() for _ in range({padding})]; "
        f"json.dump(play({game!r}), sys.stdout)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(proc.stdout)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_tournament_across_processes(self):
        for game in GAMES:
            with self.subTest(game=game):
                runs = [
                    _play_in_process(game, h, pad)
                    for h, pad in (("0", 0), ("1", 1001), ("random", 3333))
                ]
                # Real updated_at timestamps differ; nothing else may.
                for run in runs:
                    del run["saved"]["updated_at"]
                    del run["saved"]["created_at"]
                self.assertEqual(runs[0], runs[1])
                self.assertEqual(runs[0], runs[2])

    def test_reload_continues_same_sequence(self):
        for game in GAMES:
            with self.subTest(game=game):
                straight = play(game)
                reloaded = play(game, reload_after=2)
                for run in (straight, reloaded):
                    del run["saved"]["updated_at"]
                    del run["saved"]["created_at"]
                self.assertEqual(straight, reloaded)

    def test_reload_older_save_of_open_tournament(self):
        t = Tournament(TournamentConfiguration(auto_export=False, n_rounds=3), seed=SEED)
        t.add_player([f"P{i:02d}" for i in range(8)])
        older = t.serialize()
        t.create_pairings()
        t.random_results()
        t.add_player("Late")
        loaded = cast(Tournament, Tournament.inflate(older))
        self.assertIs(loaded, t)
        self.assertTrue(loaded.create_pairings())
        loaded.add_player("Late")
        self.assertEqual(len(loaded.players), 9)

    def test_file_without_seed_still_loads(self):
        t = Tournament(TournamentConfiguration(auto_export=False, n_rounds=2))
        t.add_player([f"P{i:02d}" for i in range(8)])
        t.create_pairings()
        t.random_results()
        saved = json.dumps(t.serialize()).replace(str(t.uid), str(uuid4()))
        data = json.loads(saved)
        del data["seed"], data["rng_counter"]
        data["format_version"] = "1.3"
        loaded = cast(Tournament, Tournament.inflate(data))
        self.assertIsInstance(loaded.seed, int)
        self.assertEqual(len(loaded.players), 8)
        self.assertTrue(loaded.create_pairings())


if __name__ == "__main__":
    unittest.main()

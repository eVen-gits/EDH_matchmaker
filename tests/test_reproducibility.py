"""A tournament is reproducible from its round seeds and inputs, with no
test-time patching.

Each game's tournament (players with given uids, a seed for the first
round) is played in separate processes with different PYTHONHASHSEED values,
memory layouts and real uuid4 for pods and rounds; pairings, standings and
the saved JSON must match. A save and load mid-tournament must continue the same sequence,
and a tournament rebuilt statelessly from its log (the way a wrapper service
rebuilds one on every call) must pair each next round the same.
"""

import json
import os
import random
import subprocess
import sys
import unittest
from typing import cast
from uuid import UUID, uuid4

from src.core import Pod, Round, Tournament, TournamentAction, TournamentConfiguration
from src.interface import IGameResult

TournamentAction.LOGF = False  # type: ignore

SEED = 1234
# name -> (config, n_players, n_rounds incl. playoff stages)
GAMES = {
    "commander": (dict(pod_sizes=[4, 3], allow_bye=True, n_rounds=3, top_cut=4), 13, 4),
    "mtg": (dict(ruleset="Mtg1v1Ruleset", n_rounds=3, top_cut=4), 9, 5),
    "catan": (dict(ruleset="CatanRuleset", n_rounds=4, top_cut=4), 14, 5),
}


def _new(config: dict, n_players: int) -> Tournament:
    """A tournament with players P00.. whose uids are given inputs."""
    t = Tournament(TournamentConfiguration(auto_export=False, **config))
    t.add_player([(f"P{i:02d}", UUID(int=i + 1)) for i in range(n_players)])
    return t


def normalized_log(t: Tournament) -> dict:
    """The saved log with the random tournament, round and pod uids named
    by position (they are not inputs, so they differ between runs)."""
    log = t.serialize()
    text = json.dumps(log).replace(str(t.uid), "TOUR")
    for k, r in enumerate(log["rounds"]):
        text = text.replace(r["uid"], f"ROUND{k}")
        for j, pod in enumerate(r["pods"]):
            text = text.replace(pod["uid"], f"POD{k}.{j}")
    return json.loads(text)


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
    t = _new(config, n_players)
    rounds = []
    for i in range(n_rounds):
        if i == reload_after:
            # Under a new uid, so it is a fresh load, not the cached tournament.
            saved = json.dumps(t.serialize()).replace(str(t.uid), str(uuid4()))
            t = cast(Tournament, Tournament.inflate(json.loads(saved)))
        # A seed for the first round only: later rounds keep the previous one.
        assert t.create_pairings(seed=SEED if i == 0 else None)
        t.random_results()
        rounds.append(_round_out(t))
    return {"rounds": rounds, "saved": normalized_log(t)}


def pair_statelessly(log: dict, seq: int) -> dict:
    """Pairs round `seq` from the log alone, as a stateless wrapper does.

    Builds a fresh tournament from the logged config, adds the players with
    their logged uids (in reverse order, so nothing can lean on insertion
    order), rebuilds rounds before `seq` directly from their pods and results
    with no pairing call (each with its logged seed, which a Catan plan reads),
    then pairs once with round `seq`'s logged seed.
    """
    config = TournamentConfiguration.inflate(log["config"])
    t = Tournament(config)
    try:
        players = {
            d["uid"]: t.add_player(name=d["name"], uid=UUID(d["uid"]))[0]
            for d in reversed(log["players"])
        }
        uids = lambda xs: {UUID(x) for x in xs}  # noqa: E731
        for d_round in log["rounds"][:seq]:
            r = Round(
                t,
                seq=len(t.rounds),
                stage=Round.Stage(d_round["stage"]),
                pairing_logic=t.get_pairing_logic(d_round["logic"]),
                byes=uids(d_round["byes"]),
                game_loss=uids(d_round["game_loss"]),
                dropped=uids(d_round["dropped"]),
                disabled=uids(d_round["disabled"]),
                seed=d_round.get("seed"),
            )
            t._rounds.append(r.uid)
            t._round = r.uid
            for i, d_pod in enumerate(d_round["pods"]):
                pod = Pod(r, table=i + 1, cap=len(d_pod["players"]))
                r._pods.append(pod.uid)
                for uid in d_pod["players"]:
                    pod.add_player(players[uid])
                pod.record_result(
                    [
                        IGameResult(
                            frozenset(uids(g["winners"])),
                            tuple((UUID(u), v) for u, v in g["scores"]) if "scores" in g else None,
                        )
                        for g in d_pod["games"]
                    ]
                )
            r.refresh_player_location_map()
        assert t.create_pairings(seed=log["rounds"][seq]["seed"])
        r = t.tour_round
        assert r is not None
        return {
            "pods": [[p.name for p in pod.players] for pod in r.pods],
            "byes": sorted(p.name for p in r.byes),
        }
    finally:
        Tournament.CACHE.pop(t.uid, None)


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

    def test_stateless_rebuild_pairs_the_same(self):
        for game in GAMES:
            stateful = play(game)
            for seq, expected in enumerate(stateful["rounds"]):
                with self.subTest(game=game, seq=seq):
                    expected = {k: expected[k] for k in ("pods", "byes")}
                    self.assertEqual(pair_statelessly(stateful["saved"], seq), expected)

    def test_repairing_a_round_gives_the_same_pods(self):
        for game, (config, n_players, _) in GAMES.items():
            with self.subTest(game=game):
                t = _new(config, n_players)
                t.create_pairings(seed=SEED)
                t.random_results()
                t.create_pairings()
                first = _round_out(t)
                assert t.reset_pods()
                t.create_pairings()
                self.assertEqual(_round_out(t), first)

    def test_undo_round_and_pair_again_gives_the_same_pods(self):
        for game, (config, n_players, _) in GAMES.items():
            with self.subTest(game=game):
                t = _new(config, n_players)
                t.create_pairings(seed=SEED)
                t.random_results()
                t.create_pairings()
                first = _round_out(t)
                assert t.tour_round is not None
                self.assertTrue(t.delete_round(t.tour_round))
                t.create_pairings()
                self.assertEqual(_round_out(t), first)

    def test_changed_seed_changes_next_round_and_keeps_past_seeds(self):
        config, n_players, _ = GAMES["commander"]

        def one_round_then_pair(new_seed: int | None) -> Tournament:
            random.seed(SEED)
            t = _new(config, n_players)
            t.create_pairings(seed=SEED)
            t.random_results()
            t.create_pairings(seed=new_seed)
            return t

        same, changed = one_round_then_pair(None), one_round_then_pair(SEED + 1)
        self.assertNotEqual(_round_out(changed)["pods"], _round_out(same)["pods"])
        log = changed.serialize()
        self.assertNotIn("seed", log)  # the seed is per round only
        self.assertEqual([r["seed"] for r in log["rounds"]], [SEED, SEED + 1])
        # Each round stays reproducible statelessly from the seed it records.
        for seq in range(2):
            stateful = changed.rounds[seq]
            self.assertEqual(
                pair_statelessly(log, seq)["pods"],
                [[p.name for p in pod.players] for pod in stateful.pods],
            )
        t = _new(config, n_players)
        with self.assertRaises(TypeError):
            t.create_pairings(seed="1")  # type: ignore[arg-type]

    def test_first_round_without_a_seed_draws_one(self):
        config, n_players, _ = GAMES["commander"]
        t = _new(config, n_players)
        t.create_pairings()
        r = t.tour_round
        assert r is not None
        self.assertIsInstance(r.seed, int)
        t.random_results()
        t.create_pairings()
        assert t.tour_round is not None
        self.assertEqual(t.tour_round.seed, r.seed)

    def test_any_input_change_gives_a_different_pairing(self):
        def round_one(n_players: int) -> list:
            t = _new(dict(n_rounds=3), n_players)
            t.create_pairings(seed=SEED)
            return _round_out(t)["pods"]

        # Adding a player: the original 16 are grouped differently.
        groups = lambda pods: {frozenset(p for p in pod if p != "P16") for pod in pods}  # noqa: E731
        self.assertNotEqual(groups(round_one(16)), groups(round_one(17)))

        # A changed result changes the round digest; identical inputs do not.
        random.seed(SEED)
        t = _new(dict(n_rounds=3), 16)
        t.create_pairings(seed=SEED)
        t.random_results()
        t.new_round()
        r = t.tour_round
        assert r is not None
        before, draw = r.input_digest(), r.rng("byes").random()
        self.assertEqual((r.input_digest(), r.rng("byes").random()), (before, draw))
        pod = t.rounds[0].pods[0]
        # A different single winner (after a draw, every player is in result).
        other = next((p for p in pod.players if p not in pod.result), pod.players[0])
        pod.record_result([IGameResult(frozenset({other.uid}))])
        self.assertNotEqual(r.input_digest(), before)
        self.assertNotEqual(r.rng("byes").random(), draw)

    def test_reload_older_save_of_open_tournament(self):
        t = _new(dict(n_rounds=3), 8)
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
        for r in data["rounds"]:
            del r["seed"]
        data["format_version"] = "1.3"
        loaded = cast(Tournament, Tournament.inflate(data))
        self.assertEqual(len(loaded.players), 8)
        self.assertIsNone(loaded.rounds[0].seed)
        self.assertTrue(loaded.create_pairings())
        assert loaded.tour_round is not None
        self.assertIsInstance(loaded.tour_round.seed, int)


if __name__ == "__main__":
    unittest.main()

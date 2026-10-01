import glob
import json
import os
import tempfile
import threading
import unittest

from src.core import Tournament, TournamentAction


def run_concurrently(*targets):
    errors = []

    def guard(fn):
        try:
            fn()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=guard, args=(fn,)) for fn in targets]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


def log_files():
    # Auto-export also writes .txt files; the action log is the .json/.tmp.
    return sorted(
        f for f in glob.glob("**/*", recursive=True) if f.endswith((".json", ".tmp"))
    )


def default_path(tour):
    return os.path.join("logs", f"tournament_{tour.uid.hex}.json")


def add_players(tour, prefix, n=40):
    return lambda: [tour.add_player([f"{prefix}{i}"]) for i in range(n)]


class TestActionLog(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, TournamentAction, "LOGF", TournamentAction.LOGF)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        cwd = os.getcwd()
        os.chdir(self.dir.name)
        self.addCleanup(os.chdir, cwd)

    def test_on_by_default_per_tournament(self):
        TournamentAction.LOGF = None
        t = Tournament()
        t.add_player(["A", "B"])
        path = default_path(t)
        self.assertEqual(TournamentAction.log_path(t), path)
        with open(path) as f:
            self.assertEqual(len(json.load(f)["players"]), 2)

    def test_per_tournament_override(self):
        TournamentAction.LOGF = None
        t = Tournament()
        t.log_path = os.path.join("custom", "mine.json")
        t.add_player(["A"])
        self.assertTrue(os.path.exists(t.log_path))

    def test_opt_out(self):
        TournamentAction.LOGF = False
        Tournament().add_player(["A", "B"])
        self.assertEqual(log_files(), [])

    def test_two_tournaments_keep_own_complete_log(self):
        TournamentAction.LOGF = None
        a, b = Tournament(), Tournament()
        self.assertEqual(run_concurrently(add_players(a, "A"), add_players(b, "B")), [])
        for tour, prefix in ((a, "A"), (b, "B")):
            with open(default_path(tour)) as f:
                names = {p["name"] for p in json.load(f)["players"]}
            self.assertEqual(names, {f"{prefix}{i}" for i in range(40)})
        self.assertEqual(  # no stray temp files
            log_files(), sorted(default_path(t) for t in (a, b))
        )

    def test_concurrent_stores_to_shared_path(self):
        TournamentAction.LOGF = "log.json"
        errors = run_concurrently(
            add_players(Tournament(), "A"), add_players(Tournament(), "B")
        )
        self.assertEqual(errors, [])
        self.assertEqual(log_files(), ["log.json"])
        with open("log.json") as f:
            json.load(f)

    def test_restored_tournament_keeps_logging_to_its_file(self):
        TournamentAction.LOGF = None
        t = Tournament()
        t.log_path = "saved.json"
        t.add_player(["A"])
        restored = TournamentAction.load("saved.json")
        assert restored is not None
        restored.add_player(["B"])
        self.assertIsNone(TournamentAction.LOGF)
        self.assertEqual(log_files(), ["saved.json"])
        with open("saved.json") as f:
            self.assertEqual(len(json.load(f)["players"]), 2)


if __name__ == "__main__":
    unittest.main()

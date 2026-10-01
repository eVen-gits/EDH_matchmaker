import json
import os
import tempfile
import threading
import unittest

from src.core import Tournament, TournamentAction


class TestActionLog(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, TournamentAction, "LOGF", TournamentAction.LOGF)

    def test_off_by_default(self):
        TournamentAction.LOGF = None
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            self.addCleanup(os.chdir, cwd)
            Tournament().add_player(["A", "B"])
            self.assertEqual(os.listdir(d), [])

    def test_concurrent_actions_on_two_tournaments(self):
        with tempfile.TemporaryDirectory() as d:
            TournamentAction.LOGF = os.path.join(d, "log.json")
            errors = []

            def work(prefix):
                try:
                    t = Tournament()
                    for i in range(40):
                        t.add_player([f"{prefix}{i}"])
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=work, args=(p,)) for p in "AB"]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(errors, [])
            self.assertEqual(os.listdir(d), ["log.json"])
            with open(TournamentAction.LOGF) as f:
                json.load(f)


if __name__ == "__main__":
    unittest.main()

"""Commander pods need 3+ players: 2-player pods are refused in the core."""

import unittest
from uuid import uuid4

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False


class TestCommanderPodSizes(unittest.TestCase):
    def test_two_player_pods_refused(self):
        uid = uuid4()
        with self.assertRaisesRegex(ValueError, "pod_sizes"):
            Tournament(TournamentConfiguration(pod_sizes=[2]), uid=uid)
        self.assertNotIn(uid, Tournament.CACHE)

    def test_two_player_pods_refused_on_load(self):
        t = Tournament(TournamentConfiguration())
        data = t.serialize()
        uid = uuid4()
        data["uid"] = str(uid)
        data["config"]["pod_sizes"] = [2]
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                with self.assertRaisesRegex(ValueError, "pod_sizes"):
                    Tournament.inflate(data)
                self.assertNotIn(uid, Tournament.CACHE)

    def test_invalid_config_refused_before_cache_reuse(self):
        t = Tournament(TournamentConfiguration())
        data = t.serialize()
        data["config"]["pod_sizes"] = [2]
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                with self.assertRaisesRegex(ValueError, "pod_sizes"):
                    Tournament.inflate(data)
                self.assertIs(Tournament.CACHE[t.uid], t)
                self.assertNotIn(2, t.config.pod_sizes)
        self.assertIs(Tournament.inflate(t.serialize()), t)

    def test_three_plus_still_allowed(self):
        for sizes in ([3], [4, 3], [5], [6]):
            Tournament(TournamentConfiguration(pod_sizes=sizes))


if __name__ == "__main__":
    unittest.main()

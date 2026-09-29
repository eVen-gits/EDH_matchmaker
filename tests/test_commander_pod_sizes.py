"""Commander pods need 3+ players: 2-player pods are refused in the core."""

import unittest
from uuid import uuid4

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False


class TestCommanderPodSizes(unittest.TestCase):
    def test_two_player_pods_refused(self):
        with self.assertRaisesRegex(ValueError, "pod_sizes"):
            Tournament(TournamentConfiguration(pod_sizes=[2]))

    def test_two_player_pods_refused_on_load(self):
        t = Tournament(TournamentConfiguration())
        data = t.serialize()
        data["uid"] = str(uuid4())  # not the cached tournament
        data["config"]["pod_sizes"] = [2]
        with self.assertRaisesRegex(ValueError, "pod_sizes"):
            Tournament.inflate(data)

    def test_three_plus_still_allowed(self):
        for sizes in ([3], [4, 3], [5], [6]):
            Tournament(TournamentConfiguration(pod_sizes=sizes))


if __name__ == "__main__":
    unittest.main()

import unittest

from src.core import Tournament, TournamentAction

TournamentAction.LOGF = False  # type: ignore


class TestPod(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament()
        self.t.new_round()
        self.t.add_player([f"Player {i}" for i in range(4)])
        self.t.tour_round.create_pods()
        self.pod = self.t.tour_round.pods[0]
        for p in self.t.players:
            self.pod.add_player(p)

    def test_reorder_players_success(self):
        original_players = list(self.pod.players)
        new_order = [3, 0, 2, 1]
        self.pod.reorder_players(new_order)

        reordered_players = self.pod.players
        self.assertEqual(reordered_players[0], original_players[3])
        self.assertEqual(reordered_players[1], original_players[0])
        self.assertEqual(reordered_players[2], original_players[2])
        self.assertEqual(reordered_players[3], original_players[1])

    def test_reorder_players_invalid_length(self):
        with self.assertRaisesRegex(ValueError, "Order must have the same length"):
            self.pod.reorder_players([0, 1, 2])

    def test_reorder_players_out_of_range(self):
        with self.assertRaisesRegex(
            ValueError, "Order must contain all integers from 0 to n-1"
        ):
            self.pod.reorder_players([0, 1, 2, 4])

    def test_reorder_players_duplicates(self):
        with self.assertRaisesRegex(
            ValueError, "Order must not contain duplicate integers"
        ):
            self.pod.reorder_players([0, 1, 2, 2])

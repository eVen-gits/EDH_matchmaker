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


class TestRoundOneSeating(unittest.TestCase):
    @staticmethod
    def _seats(seed: int) -> list[list[str]]:
        from uuid import UUID

        from src.core import TournamentConfiguration

        t = Tournament(
            TournamentConfiguration(auto_export=False, pod_sizes=[4], n_rounds=3)
        )
        t.add_player([(f"P{i:02d}", UUID(int=i + 1)) for i in range(8)])
        t.create_pairings(seed=seed)
        for pod in t.tour_round.pods:
            names = [p.name for p in pod.players]
            pod.reorder_players(sorted(range(len(names)), key=names.__getitem__))
            pod.auto_assign_seats()
        return [[p.name for p in pod.players] for pod in t.tour_round.pods]

    def test_same_seed_same_seats(self):
        self.assertEqual(self._seats(1), self._seats(1))

    def test_seats_differ_across_seeds(self):
        self.assertTrue(
            any(
                pod != sorted(pod) for s in range(1, 6) for pod in self._seats(s)
            )
        )

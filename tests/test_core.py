import unittest
from core import *
import names

class TestTournament43Nobye(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament(
            pod_sizes=[4, 3],
            allow_bye=False,
        )

    def test_correct_pod_sizing(self):
        pod_sizes = {
            0: None,
            1: None,
            2: None,
            3: [3],
            4: [4],
            5: None,
            6: [3, 3],
            7: [4, 3],
            8: [4, 4],
            9: [3, 3, 3],
            10: [4, 3, 3],
            11: [4, 4, 3],
            12: [4, 4, 4],
            13: [4, 3, 3, 3],
            14: [4, 4, 3, 3],
            15: [4, 4, 4, 3],
            16: [4, 4, 4, 4],
            17: [4, 4, 3, 3, 3],
            18: [4, 4, 4, 3, 3],
            19: [4, 4, 4, 4, 3],
            20: [4, 4, 4, 4, 4],
        }
        for n, expected in pod_sizes.items():
            with self.subTest(n=str(n).zfill(2)):
                self.assertEqual(self.t.get_pod_sizes(n), expected)

class TestTournament4Bye(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament(
            pod_sizes=[4],
            allow_bye=True,
        )

    def test_correct_pod_sizing(self):
        pod_sizes = (
            (0,  [], 0),
            (1,  [], 1),
            (2,  [], 2),
            (3,  [], 3),
            (4,  [4], 0),
            (5,  [4], 1),
            (6,  [4], 2),
            (7,  [4], 3),
            (8,  [4, 4], 0),
            (9,  [4, 4], 1),
            (10, [4, 4], 2),
            (11, [4, 4], 3),
            (12, [4, 4, 4], 0),
            (13, [4, 4, 4], 1),
            (14, [4, 4, 4], 2),
            (15, [4, 4, 4], 3),
            (16, [4, 4, 4, 4], 0),
            (17, [4, 4, 4, 4], 1),
            (18, [4, 4, 4, 4], 2),
            (19, [4, 4, 4, 4], 3),
            (20, [4, 4, 4, 4, 4], 0),
        )
        for n, expected_sizes, bench in pod_sizes:
            with self.subTest(n=str(n).zfill(2)):
                self.t.make_pods()
                sizes = [p.p_count for p in self.t.round.pods]
                self.assertListEqual(sizes, expected_sizes)
                self.assertEqual(len(self.t.round.unseated), bench)
                self.t.reset_pods()
                self.t.add_player(names.get_full_name())



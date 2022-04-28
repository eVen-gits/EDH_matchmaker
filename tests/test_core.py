import unittest
from core import *
import names

class TestPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament()

    def test_illegal_names(self):
        with open('tests/blns.txt', 'r') as f:
            name = f.readline()
            with self.subTest(name=name):
                p = Player(name, self.t)

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

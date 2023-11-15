from pydoc import plain
import unittest
from src.core import *
import names
import random


class TestPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament()

    def test_illegal_names(self):
        with open('tests/blns.txt', 'r') as f:
            name = f.readline()
            with self.subTest(name=name):
                p = Player(name, self.t)


class TestTournamentPodSizing(unittest.TestCase):

    def test_correct_pod_sizing_43_nobye(self):
        t = Tournament(
            pod_sizes=[4, 3],
            allow_bye=False,
        )

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
                self.assertEqual(t.get_pod_sizes(n), expected)

    def test_correct_pod_sizing_4_bye(self):
        t = Tournament(
            pod_sizes=[4],
            allow_bye=True,
        )
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
                t.make_pods()
                sizes = [p.p_count for p in t.round.pods]
                self.assertListEqual(sizes, expected_sizes)
                self.assertEqual(len(t.round.unseated), bench)
                t.reset_pods()
                t.add_player(names.get_full_name())

    def test_correct_pod_sizing_4_nobye(self):
        t = Tournament(
            pod_sizes=[4],
            allow_bye=False,
        )

        pod_sizes = (
            (0,  [], 0),
            (1,  [], 1),
            (2,  [], 2),
            (3,  [], 3),
            (4,  [4], 0),
            (5,  [], 5),
            (6,  [], 6),
            (7,  [], 7),
            (8,  [4, 4], 0),
            (9,  [], 9),
            (10, [], 10),
            (11, [], 11),
            (12, [4, 4, 4], 0),
            (13, [], 13),
            (14, [], 14),
            (15, [], 15),
            (16, [4, 4, 4, 4], 0),
            (17, [], 17),
            (18, [], 18),
            (19, [], 19),
            (20, [4, 4, 4, 4, 4], 0),
        )
        for n, expected_sizes, bench in pod_sizes:
            with self.subTest(n=str(n).zfill(2)):
                t.make_pods()
                sizes = [p.p_count for p in t.round.pods]
                self.assertListEqual(sizes, expected_sizes)
                self.assertEqual(len(t.round.unseated), bench)
                t.reset_pods()
                t.add_player(names.get_full_name())


class TestScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament(
            pod_sizes=[4],
            allow_bye=True,
            bye_points=4,
            win_points=4,
            draw_points=1,
        )
        Player.FORMATTING = ['-p', '-w', '-o']

    def test_bye_scoring(self):
        self.t.add_player([
            names.get_full_name()
            for _ in range(9)
        ])

        self.t.make_pods()

        benched = self.t.round.unseated[0]

        for pod in self.t.round.pods:
            self.t.report_win(pod.players[0])

        leaders = [p for p in self.t.players if p.points == 4]
        self.assertEqual(len(leaders), 3)
        self.assertEqual(benched.points, 4)
        standings = self.t.get_standings()
        self.assertEqual(standings[2], benched)

        self.t.manual_pod([benched, standings[0]])
        self.t.manual_pod([standings[1], standings[2]])
        self.t.report_game_loss(self.t.round.unseated)
        self.t.report_win([benched, standings[0]])

        new_standings = self.t.get_standings()
        self.assertEqual(new_standings[0], standings[0])
        self.assertEqual(new_standings[1], benched)

    def test_standings_constant(self):
        self.t.add_player([
            names.get_full_name()
            for _ in range(32)
        ])

        self.t.make_pods()
        for pod in self.t.round.pods:
            self.t.report_win(pod.players[0])

        orig_standings = self.t.get_standings()

        for _ in range(100):
            #shuffle players
            random.shuffle(self.t.players)

            self.assertEqual(self.t.get_standings(), orig_standings)


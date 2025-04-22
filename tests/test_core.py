import unittest
from src.core import *
import names
import random
from tqdm import tqdm
from faker import Faker
import time
fkr = Faker()

TournamentAction.LOGF = False #type: ignore


class TestPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament()

    def test_illegal_names(self):
        with open('tests/blns.txt', 'r') as f:
            name = f.readline()
            with self.subTest(name=name):
                p = Player(self.t, name)

class TestTournamentPodSizing(unittest.TestCase):

    def test_correct_pod_sizing_43_nobye(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4, 3],
                allow_bye=False,
                auto_export=False,
            )
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
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=True,
                max_byes=3,
                auto_export=False,
            )
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
                t.create_pairings()
                assert t.tour_round is not None
                sizes = [len(p) for p in t.tour_round.pods]
                self.assertListEqual(sizes, expected_sizes)
                self.assertEqual(len(t.tour_round.byes), bench)
                t.reset_pods()
                t.add_player(fkr.name())

    def test_correct_pod_sizing_4_nobye(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=False,
                auto_export=False,
            )
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
                t.create_pairings()
                assert t.tour_round is not None
                sizes = [len(p) for p in t.tour_round.pods]
                self.assertListEqual(sizes, expected_sizes)
                self.assertEqual(len(t.tour_round.byes), bench)
                t.reset_pods()
                t.add_player(fkr.name())

    def test_correct_pod_sizing_43_max_2_bye(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4, 3],
                allow_bye=True,
                max_byes=2,
                auto_export=False,
            )
        )

        pod_sizes = (
            (0,  [], 0),
            (1,  [], 1),
            (2,  [], 2),
            (3,  [3], 0),
            (4,  [4], 0),
            (5,  [4], 1),
            (6,  [4], 2),
            (7,  [4, 3], 0),
            (8,  [4, 4], 0),
            (9,  [4, 4], 1),
            (10, [4, 4], 2),
            (11, [4, 4, 3], 0),
            (12, [4, 4, 4], 0),
            (13, [4, 4, 4], 1),
            (14, [4, 4, 4], 2),
            (15, [4, 4, 4, 3], 0),
            (16, [4, 4, 4, 4], 0),
            (17, [4, 4, 4, 4], 1),
            (18, [4, 4, 4, 4], 2),
            (19, [4, 4, 4, 4, 3], 0),
            (20, [4, 4, 4, 4, 4], 0),
        )
        for n, expected_sizes, bench in pod_sizes:
            with self.subTest(n=str(n).zfill(2)):
                t.create_pairings()
                assert t.tour_round is not None
                sizes = [len(p) for p in t.tour_round.pods]
                self.assertListEqual(sizes, expected_sizes)
                self.assertEqual(len(t.tour_round.byes), bench)
                t.reset_pods()
                t.add_player(fkr.name())

class TestScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=True,
                win_points=4,
                bye_points=4,
                draw_points=1,
                auto_export=False,
            )
        )
        Player.FORMATTING = ['-p', '-w', '-o']

    def test_bye_scoring(self):
        self.t.add_player([
            fkr.name()
            for _ in range(9)
        ])

        self.t.create_pairings()

        assert self.t.tour_round is not None
        bye = self.t.tour_round.byes[0]

        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

        leaders = [p for p in self.t.players if p.rating(self.t.tour_round) == self.t.config.win_points]
        self.assertEqual(len(leaders), 3)
        self.assertEqual(bye.rating(self.t.tour_round), self.t.config.bye_points)
        standings = self.t.get_standings(self.t.tour_round)
        self.assertEqual(standings[2], bye)

        self.t.new_round()
        self.t.manual_pod([bye, standings[3]])
        self.t.manual_pod([standings[0], standings[1]])
        self.t.toggle_game_loss(self.t.tour_round.unseated)
        self.t.report_win([bye, standings[0]])

        new_standings = self.t.get_standings(self.t.tour_round)
        self.assertEqual(new_standings[0], standings[0])
        self.assertEqual(new_standings[1], bye)

    def test_standings_constant(self):
        self.t.add_player([
            fkr.name()
            for _ in range(32)
        ])

        self.t.create_pairings()
        assert self.t.tour_round is not None
        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

        orig_standings = self.t.get_standings(self.t.tour_round)

        for _ in range(100):
            #shuffle players
            random.shuffle(self.t.players)

            self.assertEqual(self.t.get_standings(self.t.tour_round), orig_standings)

    def test_random_results(self):
        self.t.add_player([
            fkr.name()
            for _ in range(128)
        ])
        for _ in range(10):
            self.t.create_pairings()
            self.t.random_results()
            self.assertTrue(self.t.tour_round.done)
            self.t.reset_pods()


class TestPerformance(unittest.TestCase):
    def test_new_round_speed(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=False,
                auto_export=False,
            )
        )
        t.add_player([
            fkr.name()
            for _ in range(128)
        ])

        n = 100
        total_time = 0
        for _ in range(n):
            time_start = time.time()
            t.new_round()
            delta_time = time.time() - time_start
            total_time += delta_time
            self.assertLess(delta_time, 0.5)
            t.create_pairings()
            t.random_results()
        self.assertLess(total_time/n, 0.5)

    def test_create_pairings_speed(self):
        tour_sizes = [16, 32, 64, 128, 256, 512, 1024]
        for n in tour_sizes:
            with self.subTest(n=str(n).zfill(2)):
                t = Tournament(
                    TournamentConfiguration(
                        pod_sizes=[4],
                        allow_bye=False,
                        auto_export=False,
                    )
                )
                t.add_player([
                    fkr.name()
                    for _ in range(n)
                ])
                total_time = 0
                n_rounds = 7
                for _ in range(n_rounds):
                    player_time_ratio = 0.05 * 1.02**(n/128)
                    time_start = time.time()
                    t.create_pairings()
                    delta_time = time.time() - time_start
                    self.assertLess(delta_time, n*player_time_ratio)
                    total_time += delta_time
                    t.random_results()
                    t.new_round()
                self.assertLess(total_time/n_rounds, n*player_time_ratio)


class TestLarge(unittest.TestCase):

    def test_many_players(self):
        tour_sizes = [
            2**i for i in range(5, 13)
        ]
        n_rounds = 5

        for n in tour_sizes:
            with self.subTest(n=str(n).zfill(2)):
                t = Tournament(
                    TournamentConfiguration(
                        pod_sizes=[4, 3],
                        allow_bye=False,
                        snake_pods=True,
                        auto_export=False,
                    )
                )
                t.add_player([
                    fkr.name()
                    for _ in range(n)
                ])
                for _ in range(n_rounds):
                    ok = t.create_pairings()
                    self.assertTrue(ok)
                    t.random_results()

    def test_many_rounds(self):
        tour_size = 256
        n_rounds = 10

        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4, 3],
                allow_bye=False,
                snake_pods=True,
                auto_export=False,
            )
        )
        t.add_player([
            fkr.name()
            for _ in range(tour_size)
        ])
        for i in range(n_rounds):
            with self.subTest(n=str(i+1).zfill(2)):
                t.create_pairings()
                t.random_results()
'''#TODO: Implement this test
class TestSeatNormalization(unittest.TestCase):
    def test_close_to_equal(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=False,
                snake_pods=True,
            )
        )
        t.add_player([
            fkr.name()
            for _ in range(16)
        ])

        for i in tqdm(range(500)):
            t.create_pairings()
            t.random_results()
            pass

        pass'''

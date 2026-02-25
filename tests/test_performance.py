import unittest
from src.core import *
import names
import random
from tqdm import tqdm
from faker import Faker
import time
from itertools import product

fkr = Faker()

TournamentAction.LOGF = False  # type: ignore


class TestPerformance(unittest.TestCase):
    def setUp(self):
        # Reset the Faker generator to ensure consistent names if needed,
        # though not strictly necessary for performance tests.
        pass

    def test_new_round_speed(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4, 3],
                max_byes=2,
                auto_export=False,
                snake_pods=True,
                allow_bye=True,
                win_points=5,
                bye_points=4,
                draw_points=1,
                n_rounds=5,
                top_cut=10,
            )
        )
        t.new_round()

        t.add_player([f"{i}:{fkr.name()}" for i in range(128)])

        n = 20
        total_time = 0
        for _ in tqdm(range(n), desc="Testing new round speed"):
            time_start = time.time()
            t.new_round()
            delta_time = time.time() - time_start
            total_time += delta_time
            self.assertLess(
                delta_time,
                1,
                msg=f"{delta_time} seconds at round {n} for {len(t.players)} players",
            )
            t.create_pairings()
            t.random_results()
        self.assertLess(total_time / n, 0.5)

    def test_create_pairings_speed(self):
        tour_sizes = [16, 32, 64, 128, 256, 512, 1024]

        for n in tour_sizes:
            total_time = 0
            player_time_ratio = 0.05 * 1.05 ** (n / 128)
            t = Tournament(
                TournamentConfiguration(
                    pod_sizes=[4],
                    allow_bye=False,
                    auto_export=False,
                    max_byes=2,
                    win_points=5,
                    bye_points=4,
                    draw_points=1,
                    snake_pods=True,
                    n_rounds=7,
                    top_cut=10,
                )
            )
            t.new_round()
            t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
            for _ in tqdm(
                range(t.config.n_rounds), desc=f"Pairing speed for {n} players"
            ):
                time_start = time.time()
                t.create_pairings()
                delta_time = time.time() - time_start
                self.assertLess(delta_time, n * player_time_ratio)
                total_time += delta_time
                t.random_results()
                t.new_round()
            with self.subTest(n=str(n).zfill(2), max_time=n * player_time_ratio):
                self.assertLess(total_time / t.config.n_rounds, n * player_time_ratio)


class TestLarge(unittest.TestCase):
    def test_many_players(self):
        tour_sizes = [2**i for i in range(5, 13)]
        n_rounds = 5

        for n in tqdm(tour_sizes, desc="Many players"):
            with self.subTest(n=str(n).zfill(2)):
                t = Tournament(
                    TournamentConfiguration(
                        pod_sizes=[4, 3],
                        allow_bye=False,
                        snake_pods=True,
                        auto_export=False,
                    )
                )
                t.new_round()

                t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
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
        t.new_round()

        t.add_player([f"{i}:{fkr.name()}" for i in range(tour_size)])
        for i in tqdm(range(n_rounds), desc="Many rounds"):
            with self.subTest(n=str(i + 1).zfill(2)):
                t.create_pairings()
                t.random_results()


class TestLarge(unittest.TestCase):
    def test_many_players(self):
        tour_sizes = [2**i for i in range(5, 13)]
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
                t.new_round()

                t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
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
        t.new_round()

        t.add_player([f"{i}:{fkr.name()}" for i in range(tour_size)])
        for i in range(n_rounds):
            with self.subTest(n=str(i + 1).zfill(2)):
                t.create_pairings()
                t.random_results()

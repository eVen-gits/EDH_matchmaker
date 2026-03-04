import unittest

from faker import Faker

from src.core import Tournament, TournamentAction, TournamentConfiguration, Pod
from src.misc import generate_player_names

fkr = Faker()
TournamentAction.LOGF = False  # type: ignore


class TestTournamentPodSizing(unittest.TestCase):
    def test_correct_pod_sizing_43_nobye(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4, 3],
                allow_bye=False,
                auto_export=False,
            )
        )
        t.new_round()

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
        t.new_round()

        pod_sizes = (
            (0, [], 0),
            (1, [], 1),
            (2, [], 2),
            (3, [], 3),
            (4, [4], 0),
            (5, [4], 1),
            (6, [4], 2),
            (7, [4], 3),
            (8, [4, 4], 0),
            (9, [4, 4], 1),
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
                t.add_player(f"{len(t.players)}:{fkr.name()}")

    def test_correct_pod_sizing_4_nobye(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=False,
                auto_export=False,
            )
        )
        t.new_round()

        pod_sizes = (
            (0, [], 0),
            (1, [], 1),
            (2, [], 2),
            (3, [], 3),
            (4, [4], 0),
            (5, [], 5),
            (6, [], 6),
            (7, [], 7),
            (8, [4, 4], 0),
            (9, [], 9),
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
        t.new_round()

        pod_sizes = (
            (0, [], 0),
            (1, [], 1),
            (2, [], 2),
            (3, [3], 0),
            (4, [4], 0),
            (5, [4], 1),
            (6, [4], 2),
            (7, [4, 3], 0),
            (8, [4, 4], 0),
            (9, [4, 4], 1),
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


class TestRoundCreation(unittest.TestCase):
    def test_modified_n_rounds(self) -> None:
        config = TournamentConfiguration(
            pod_sizes=[4],
            allow_bye=False,
            auto_export=False,
            n_rounds=2,
        )
        t = Tournament(config)
        t.add_player(generate_player_names(16))
        t.create_pairings()
        t.random_results()
        t.create_pairings()
        t.tour_round.delete()
        t.config.n_rounds = 1

        self.assertFalse(t.create_pairings())

    def test_modified_n_rounds_reset_pods(self) -> None:
        t = Tournament(config=TournamentConfiguration(
            pod_sizes=[4],
            n_rounds=2,
            allow_bye=False,
            auto_export=False,
        ))
        t.add_player(generate_player_names(16))
        t.create_pairings()
        t.random_results()
        t.create_pairings()
        t.reset_pods()
        new_config = TournamentConfiguration(
            pod_sizes=[4],
            n_rounds=1,
        )
        with self.assertRaises(ValueError):
            t.config = new_config

    def test_default_scoring_values(self):
        cfg = TournamentConfiguration()
        self.assertEqual(cfg.win_points, 5)
        self.assertEqual(cfg.bye_points, 4)
        self.assertEqual(cfg.draw_points, 1)


class TestManualPodOps(unittest.TestCase):
    """bench_players, toggle_bye, delete_pod, get_pods_str."""

    def setUp(self):
        cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=True, max_byes=2, auto_export=False)
        self.t = Tournament(cfg)
        self.t.add_player([f"P{i}" for i in range(9)])
        self.t.create_pairings()  # 2 pods of 4, 1 bye

    def test_bench_player_removes_from_pod(self):
        player = self.t.tour_round.pods[0].players[0]
        self.t.bench_players(player)
        self.assertIsNone(player.pod(self.t.tour_round))

    def test_bench_player_iterable(self):
        targets = self.t.tour_round.pods[0].players[:2]
        self.t.bench_players(targets)
        for p in targets:
            self.assertIsNone(p.pod(self.t.tour_round))

    def test_toggle_bye_adds_bye(self):
        player = self.t.tour_round.pods[0].players[0]
        self.t.toggle_bye(player)
        self.assertIn(player, self.t.tour_round.byes)

    def test_toggle_bye_removes_bye(self):
        existing_bye = next(iter(self.t.tour_round.byes))
        self.t.toggle_bye(existing_bye)
        self.assertNotIn(existing_bye, self.t.tour_round.byes)

    def test_delete_pod_removes_it(self):
        pod = self.t.tour_round.pods[0]
        n_before = len(self.t.tour_round.pods)
        self.t.delete_pod(pod)
        self.assertEqual(len(self.t.tour_round.pods), n_before - 1)
        self.assertNotIn(pod, self.t.tour_round.pods)

    def test_get_pods_str_nonempty(self):
        result = self.t.get_pods_str()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_get_pods_str_contains_player_names(self):
        result = self.t.get_pods_str()
        for pod in self.t.tour_round.pods:
            for p in pod.players:
                self.assertIn(p.name, result)

    def test_get_pods_str_no_round_returns_empty(self):
        cfg = TournamentConfiguration(auto_export=False)
        t = Tournament(cfg)
        self.assertEqual(t.get_pods_str(), "")

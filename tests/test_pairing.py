import unittest
import random
import pytest
from itertools import product

from faker import Faker

from src.core import Player, Tournament, TournamentAction, TournamentConfiguration

fkr = Faker()
TournamentAction.LOGF = False  # type: ignore


class TestMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TournamentConfiguration(
            pod_sizes=[4, 3],
            allow_bye=True,
            win_points=5,
            bye_points=4,
            draw_points=1,
            auto_export=False,
            snake_pods=True,
            max_byes=2,
        )
        self.n_rounds = 5

    @pytest.mark.slow
    def test_all_players_assigned(self):
        tour_sizes = range(16, 128)
        for n in tour_sizes:
            t = Tournament(self.config)
            t.new_round()
            t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
            for i in range(self.n_rounds):
                t.create_pairings()
                t.random_results()
                for p in t.tour_round.active_players:
                    self.assertEqual(len(p.pods(t.tour_round)), i + 1)

                self.assertEqual(len(t.tour_round.active_players), n)
                self.assertEqual(len(t.tour_round.unassigned), 0)
                t.new_round()

    @pytest.mark.slow
    def test_bye_assignment(self):
        tour_sizes = range(16, 128)
        for n in tour_sizes:
            t = Tournament(self.config)
            t.new_round()
            t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
            self.assertEqual(len(t.players), n)
            for i in range(self.n_rounds):
                with self.subTest(n=str(n).zfill(2), round=str(i + 1).zfill(2)):
                    t.create_pairings()
                    n_byes = len(t.tour_round.byes)
                    expected_byes = n % 4 if n % 4 <= 2 else 0
                    if n_byes != expected_byes:
                        t.reset_pods()
                        t.create_pairings()
                        n_byes = len(t.tour_round.byes)
                    self.assertLessEqual(n_byes, t.config.max_byes)
                    self.assertEqual(n_byes, expected_byes)
                    t.random_results()

    @pytest.mark.slow
    def test_snake_winners_not_paired(self):
        tour_sizes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 32, 64, 128]
        for n in tour_sizes:
            with self.subTest(n=n):
                t = Tournament(self.config)
                t.new_round()
                t.add_player([f"P{i}" for i in range(n)])

                t.create_pairings()

                winners_r1 = []
                for pod in t.tour_round.pods:
                    winner = pod.players[0]
                    t.report_win(winner)
                    winners_r1.append(winner)

                t.new_round()
                t.create_pairings()

                violations = 0
                for pod in t.tour_round.pods:
                    winners_in_pod = [p for p in pod.players if p in winners_r1]
                    if len(winners_in_pod) > 1:
                        violations += 1

                self.assertEqual(violations, 0)

    @pytest.mark.slow
    def test_snake_no_repeat_matching(self):
        tour_sizes = [12, 13, 14, 16, 15, 17, 18, 19, 20]
        tested = []
        for n_players in tour_sizes:
            with self.subTest(n_players=n_players):
                player_names = [f"{str(i).zfill(2)}" for i in range(n_players)]

                t = Tournament(self.config)
                pod_sizes = t.get_pod_sizes(n_players) or []
                n_pods = len(pod_sizes)
                total_capacity = sum(pod_sizes)
                if total_capacity in tested:
                    continue
                tested.append(total_capacity)

                r1_results_table = [
                    [
                        [(j >> k) & 1 == 1 for k in range(pod_sizes[i])]
                        for j in range(1, 2 ** pod_sizes[i])
                    ]
                    for i in range(n_pods)
                ]

                all_possible_outcomes = list(product(*r1_results_table))
                sample_size = min(100, len(all_possible_outcomes))
                random_sample = random.sample(all_possible_outcomes, sample_size)
                for result in random_sample:
                    t = Tournament(self.config)
                    t.new_round()
                    t.add_player(player_names)

                    t.tour_round.create_pods()
                    pod_idx = 0
                    for player in t.players:
                        if t.tour_round.pods[pod_idx].cap == len(
                            t.tour_round.pods[pod_idx].players
                        ):
                            pod_idx += 1
                            if pod_idx >= len(t.tour_round.pods):
                                raise ValueError("Pod index out of range")
                        t.tour_round.pods[pod_idx].add_player(player)

                    for i, pod in enumerate(t.tour_round.pods):
                        single_result = result[i].count(True) == 1
                        for j, player in enumerate(pod.players):
                            if result[i][j]:
                                pod.set_result(
                                    player,
                                    Player.EResult.WIN
                                    if single_result
                                    else Player.EResult.DRAW,
                                )

                    t.create_pairings()
                    repeat_pairings = t.tour_round.repeat_pairings()

                    self.assertLessEqual(
                        sum(repeat_pairings.values()), len(t.tour_round.pods)
                    )

    def test_pairing_random_all_assigned(self):
        # seq==0 always uses PairingRandom — first call to create_pairings
        cfg = TournamentConfiguration(
            pod_sizes=[4, 3],
            allow_bye=False,
            auto_export=False,
        )
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(12)])
        ok = t.create_pairings()

        self.assertTrue(ok)
        all_podded = {p for pod in t.tour_round.pods for p in pod.players}
        self.assertEqual(all_podded, set(t.players))


class TestTablePreferences(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TournamentConfiguration(
            pod_sizes=[4, 3],
            allow_bye=False,
            auto_export=False,
        )
        self.t = Tournament(self.config)
        self.t.new_round()

    def test_basic_preference(self) -> None:
        players = self.t.add_player([f"P{i}" for i in range(128)])

        p0 = players[0]
        p0.set_table_preference([1])

        for _ in range(15):
            self.t.create_pairings()
            self.assertEqual(p0.pod(self.t.tour_round).table, 1)
            self.t.reset_pods()

    def test_best_effort_satisfaction(self) -> None:
        players = self.t.add_player([f"P{i}" for i in range(128)])

        p_pref = players[:5]
        for p in p_pref:
            p.set_table_preference([1, 2, 3, 4])

        for _ in range(15):
            self.t.create_pairings()

            satisfied_players = sum(
                1
                for p in p_pref
                if p.pod(self.t.tour_round).table in p.table_preference
            )
            self.assertGreaterEqual(satisfied_players, 4)
            self.t.reset_pods()

    def test_no_preference_swap(self) -> None:
        players = self.t.add_player([f"P{i}" for i in range(128)])

        for p in players[0:4]:
            p.set_table_preference([2])
        for p in players[4:8]:
            p.set_table_preference([1])

        self.t.tour_round.create_pods()

        pod1 = self.t.tour_round.pods[0]
        pod2 = self.t.tour_round.pods[1]
        for p in players[0:4]:
            pod1.add_player(p)
        for p in players[4:8]:
            pod2.add_player(p)

        self.t.create_pairings()

        for p in players[0:4]:
            self.assertEqual(p.pod(self.t.tour_round).table, 2)
        for p in players[4:8]:
            self.assertEqual(p.pod(self.t.tour_round).table, 1)

    def test_stable_reordering(self) -> None:
        players = self.t.add_player([f"P{i}" for i in range(12)])

        round = self.t.tour_round
        round.create_pods()
        round.refresh_player_location_map()

        pods = round.pods
        p0, p1, p2 = pods[0], pods[1], pods[2]

        for i, p in enumerate(players):
            pods[i // 4].add_player(p)

        for p in p1.players:
            p.set_table_preference([1])

        round.sort_pods()

        self.assertEqual(round.pods[0].uid, p1.uid)
        self.assertEqual(round.pods[1].uid, p0.uid)
        self.assertEqual(round.pods[2].uid, p2.uid)

    def test_anonimity_serialization(self) -> None:
        p = self.t.add_player("Anon")[0]
        p.table_preference = [1, 2, 3]

        serialized = p.serialize()
        self.assertNotIn("table_preference", serialized)

        uid = p.uid
        del self.t.PLAYER_CACHE[uid]

        p_inflated = Player.inflate(self.t, serialized)
        self.assertEqual(p_inflated.table_preference, [])

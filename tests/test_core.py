import unittest
from run_ui import generate_player_names
from src.core import Tournament, Player, TournamentAction, TournamentConfiguration
from uuid import uuid4
import random
import os

# Disable tqdm in CI environment to avoid log noise
if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):

    def tqdm(iterable, *args, **kwargs):
        return iterable
else:
    from tqdm import tqdm
from faker import Faker
from itertools import product

fkr = Faker()

TournamentAction.LOGF = False  # type: ignore


class TestPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.t: Tournament = Tournament()

    def test_illegal_names(self):
        with open("tests/blns.txt", "r") as f:
            name = f.readline()
            with self.subTest(name=name):
                p = Player(self.t, name)

    def test_add_player_refactor(self):
        self.t.new_round()

        # 1. Backward compatibility: single name
        self.t.add_player("Player1")
        self.assertEqual(len(self.t.players), 1)
        p1 = list(self.t.players)[0]
        self.assertEqual(p1.name, "Player1")
        self.assertIsNone(p1.decklist)

        # 2. Backward compatibility: list of names
        self.t.add_player(["Player2", "Player3"])
        self.assertEqual(len(self.t.players), 3)

        # 3. New functionality: single tuple with UUID
        u4 = uuid4()
        self.t.add_player(("Player4", u4))
        self.assertEqual(len(self.t.players), 4)
        p4 = [p for p in self.t.players if p.name == "Player4"][0]
        self.assertEqual(p4.uid, u4)
        self.assertIsNone(p4.decklist)

        # 4. New functionality: list of tuples with UUIDs
        u6 = uuid4()
        u7 = uuid4()
        self.t.add_player([("Player6", u6), ("Player7", u7)])
        p6 = [p for p in self.t.players if p.name == "Player6"][0]
        self.assertEqual(p6.uid, u6)
        p7 = [p for p in self.t.players if p.name == "Player7"][0]
        self.assertEqual(p7.uid, u7)

        # 5. New functionality: 3-tuple with decklist link
        u8 = uuid4()
        decklist_url = "https://moxfield.com/decks/jVKoDKgc0Ey2PcapxHIB_w"
        self.t.add_player(("Player8", u8, decklist_url))
        self.assertEqual(len(self.t.players), 7)
        p8 = [p for p in self.t.players if p.name == "Player8"][0]
        self.assertEqual(p8.uid, u8)
        self.assertEqual(p8.decklist, decklist_url)

    def test_add_player_new_interface(self):
        self.t.new_round()
        u1 = uuid4()
        u2 = uuid4()

        # Test multiple positional arguments of different types
        self.t.add_player(
            "Alice",
            ("Bob", u1),
            {"name": "Charlie", "uid": u2, "decklist": "http://list.com"},
        )

        player_names = {p.name for p in self.t.players}
        self.assertTrue("Alice" in player_names)
        self.assertTrue("Bob" in player_names)
        self.assertTrue("Charlie" in player_names)

        bob = [p for p in self.t.players if p.name == "Bob"][0]
        self.assertEqual(bob.uid, u1)

        charlie = [p for p in self.t.players if p.name == "Charlie"][0]
        self.assertEqual(charlie.uid, u2)
        self.assertEqual(charlie.decklist, "http://list.com")

        # Test keyword arguments for a single player
        self.t.add_player(name="David", decklist="http://david.com")
        david = [p for p in self.t.players if p.name == "David"][0]
        self.assertEqual(david.name, "David")
        self.assertEqual(david.decklist, "http://david.com")

        # Test mixed positional and keyword
        self.t.add_player("Eve", name="Frank")
        player_names = {p.name for p in self.t.players}
        self.assertTrue("Eve" in player_names)
        self.assertTrue("Frank" in player_names)

    def test_add_player_refined_interface(self):
        self.t.new_round()

        # Test smart 2-tuple: (Name, UUID)
        u1 = uuid4()
        p = self.t.add_player(("SmartUID", u1))[0]
        self.assertEqual(p.uid, u1)
        self.assertIsNone(p.decklist)

        # Test smart 2-tuple: (Name, Decklist)
        p = self.t.add_player(("SmartDeck", "http://deck.list"))[0]
        self.assertEqual(p.decklist, "http://deck.list")

        # Test keyword merging with positional string
        p = self.t.add_player("MergeString", decklist="http://merge.str")[0]
        self.assertEqual(p.decklist, "http://merge.str")

        # Test keyword merging with positional dict
        p = self.t.add_player({"name": "MergeDict"}, decklist="http://merge.dict")[0]
        self.assertEqual(p.decklist, "http://merge.dict")

    def test_add_player_validation(self):
        self.t.new_round()
        # Test invalid type
        with self.assertRaises(ValueError):
            self.t.add_player(123)

        # Test missing name in dict
        with self.assertRaises(ValueError):
            self.t.add_player({"decklist": "..."})

    def test_player_serialization_with_decklist(self):
        """Test that decklist is properly serialized and deserialized"""
        # Add player with decklist
        self.t.new_round()

        u1 = uuid4()
        decklist_url = "https://moxfield.com/decks/jVKoDKgc0Ey2PcapxHIB_w"
        self.t.add_player(("TestPlayer", u1, decklist_url))

        p1 = [p for p in self.t.players if p.name == "TestPlayer"][0]

        # Serialize
        serialized = p1.serialize()
        self.assertEqual(serialized["name"], "TestPlayer")
        self.assertEqual(serialized["uid"], str(u1))
        self.assertEqual(serialized["decklist"], decklist_url)

        del p1.CACHE[u1]

        # Deserialize
        p2 = Player.inflate(self.t, serialized)
        self.assertEqual(p2.name, "TestPlayer")
        self.assertEqual(p2.uid, u1)
        self.assertEqual(p2.decklist, decklist_url)


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

    def test_all_players_assigned(self):
        tour_sizes = range(16, 128)
        for n in tqdm(tour_sizes, desc="Testing all players assigned"):
            t = Tournament(self.config)
            t.new_round()
            t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
            for i in range(self.n_rounds):
                # with self.subTest(n=str(n).zfill(2), round=str(i+1).zfill(2)):
                t.create_pairings()
                t.random_results()
                for p in t.tour_round.active_players:
                    self.assertEqual(len(p.pods(t.tour_round)), i + 1)

                self.assertEqual(len(t.tour_round.active_players), n)
                self.assertEqual(len(t.tour_round.unassigned), 0)
                t.new_round()

    def test_bye_assignment(self):
        tour_sizes = range(16, 128)
        for n in tour_sizes:
            t = Tournament(self.config)
            t.new_round()
            t.add_player([f"{i}:{fkr.name()}" for i in range(n)])
            self.assertEqual(len(t.players), n)
            for i in range(self.n_rounds):
                with self.subTest(n=str(n).zfill(2), round=str(i + 1).zfill(2)):
                    # with self.subTest(n=str(n).zfill(2)):
                    t.create_pairings()
                    n_byes = len(t.tour_round.byes)
                    expected_byes = n % 4 if n % 4 <= 2 else 0
                    min_score = t.get_standings()[-n_byes].rating(t.tour_round)
                    if n_byes != expected_byes:
                        t.reset_pods()
                        t.create_pairings()
                        n_byes = len(t.tour_round.byes)
                        if n_byes != expected_byes:
                            print(n_byes, expected_byes)
                    self.assertLessEqual(n_byes, t.config.max_byes)
                    self.assertEqual(n_byes, expected_byes)

                    pass
                    t.random_results()
                    pass

    def test_snake_winners_not_paired(self):
        """
        Verify that Snake pairing logic prevents winners from the previous round
        from being paired together. Since there are N pods, only N winners exist,
        so they should all be in different pods in the next round.
        """
        tour_sizes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 32, 64, 128]
        for n in tqdm(tour_sizes, desc="Snake winners not paired"):
            with self.subTest(n=n):
                t = Tournament(self.config)
                t.new_round()
                t.add_player([f"P{i}" for i in range(n)])

                # Round 1: Create pairings and determine winners
                t.create_pairings()

                # Award wins to first player in each pod (deterministic winners)
                winners_r1 = []
                for pod in t.tour_round.pods:
                    winner = pod.players[0]
                    t.report_win(winner)
                    winners_r1.append(winner)

                # Round 2: Snake pairing
                t.new_round()
                t.create_pairings()

                # Verify: No two winners from R1 should be in the same pod in R2
                violations = 0
                for pod in t.tour_round.pods:
                    winners_in_pod = [p for p in pod.players if p in winners_r1]
                    if len(winners_in_pod) > 1:
                        violations += 1
                        print(
                            f"Violation in Pod {pod.table}: {len(winners_in_pod)} winners paired together"
                        )

                self.assertEqual(
                    violations,
                    0,
                    f"Snake pairing paired {violations} groups of winners together (should be 0)",
                )

    def test_snake_no_repeat_matching(self):
        tour_sizes = [12, 13, 14, 16, 15, 17, 18, 19, 20]
        tested = []
        for n_players in tour_sizes:
            with self.subTest(n_players=n_players):
                player_names = [f"{str(i).zfill(2)}" for i in range(n_players)]

                # We can discard tests where players are awarded bye to reduce complexity
                t = Tournament(self.config)
                pod_sizes = t.get_pod_sizes(n_players) or []
                n_pods = len(pod_sizes)
                total_capacity = sum(pod_sizes)
                if total_capacity in tested:
                    continue
                tested.append(total_capacity)

                # r1_result_table = [
                #    [
                #        bool(i & (1 << (n_players - 1 - j)))
                #        for j in range(1, pod_sizes[i])
                #    ]
                #    for i in range(n_pods)
                # ]

                r1_results_table = [
                    [
                        [(j >> k) & 1 == 1 for k in range(pod_sizes[i])]
                        for j in range(1, 2 ** pod_sizes[i])
                    ]  # todo
                    for i in range(n_pods)
                ]

                # above represents possible results for each pod
                # now create a table that has all possible combinations of results for all pods from above

                # Generate all possible combinations using itertools.product

                # Each outcome is a tuple where outcome[i] is the result for pod i
                # For example, if you have 2 pods with 2 players each:
                # - Pod 0 has 3 possible results: [[True, False], [False, True], [True, True]]
                # - Pod 1 has 3 possible results: [[True, False], [False, True], [True, True]]
                # - all_possible_outcomes will have 3*3 = 9 combinations

                # Create all possible combinations of results across all pods
                all_possible_outcomes = list(product(*r1_results_table))
                # Take random 500 outcomes
                random_sample = random.sample(all_possible_outcomes, 500)
                for result in tqdm(random_sample):
                    # tests_ran += 1
                    # if tests_ran >= 100:
                    #    break

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

                    # results are bits - for each seat of 4 bits
                    # set the result of a pod based on the bits
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
                        pass
                    # Snake pods round
                    t.create_pairings()

                    repeat_pairings = t.tour_round.repeat_pairings()

                    # TODO: Actually figure out what is the minimal amount of repeat pairings
                    self.assertLessEqual(
                        sum(repeat_pairings.values()), len(t.tour_round.pods)
                    )
                    pass


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
        self.t.new_round()
        Player.FORMATTING = ["-p", "-w", "-o"]

    def test_bye_scoring(self):
        self.t.add_player([f"{i}:{fkr.name()}" for i in range(9)])

        self.t.create_pairings()

        assert self.t.tour_round is not None
        bye = next(iter(self.t.tour_round.byes))

        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

        leaders = [
            p
            for p in self.t.players
            if p.rating(self.t.tour_round) == self.t.config.win_points
        ]
        self.assertEqual(len(leaders), 3)
        self.assertEqual(bye.rating(self.t.tour_round), self.t.config.bye_points)
        standings = self.t.get_standings(self.t.tour_round)
        self.assertEqual(standings[2], bye)

        self.t.new_round()
        self.t.manual_pod([bye, standings[3]])
        self.t.manual_pod([standings[0], standings[1]])
        self.t.toggle_game_loss(self.t.tour_round.unassigned)
        self.t.report_win([bye, standings[0]])

        new_standings = self.t.get_standings(self.t.tour_round)
        self.assertEqual(new_standings[0], standings[0])
        self.assertEqual(new_standings[1], bye)

    def test_standings_constant(self):
        self.t.add_player([f"{i}:{fkr.name()}" for i in range(32)])

        self.t.create_pairings()
        assert self.t.tour_round is not None
        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

        orig_standings = self.t.get_standings(self.t.tour_round)

        for _ in range(100):
            # shuffle players by recreating the _players set in a random order
            # This changes the iteration order of the set (Python 3.7+ maintains insertion order)
            player_uids = list(self.t._players)
            random.shuffle(player_uids)
            self.t._players = set(player_uids)

            self.assertEqual(self.t.get_standings(self.t.tour_round), orig_standings)

    def test_random_results(self):
        self.t.add_player([f"{i}:{fkr.name()}" for i in range(128)])
        for _ in tqdm(range(10)):
            self.t.create_pairings()
            self.t.random_results()
            self.assertTrue(self.t.tour_round.done)

            self.t.reset_pods()


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
        """
        Test that a single player's preference is satisfied.
        """
        players = self.t.add_player([f"P{i}" for i in range(128)])

        # Lock P0 to Table with index 0 (Table 1)
        p0 = players[0]
        p0.set_table_preference([1])

        for _ in tqdm(range(50)):
            self.t.create_pairings()

            # Verify P0 is in Table 2 (index 1)
            self.assertEqual(p0.pod(self.t.tour_round).table, 1)
            self.t.reset_pods()

    def test_best_effort_satisfaction(self) -> None:
        """
        Test that the best effort satisfaction works.
        5 players want indices 1, 2, 3, 4, 5.
        At least 4 have to be satisfied.
        """
        players = self.t.add_player([f"P{i}" for i in range(128)])

        # Five players want Index 0 (Table 1).
        p_pref = players[:5]
        for p in p_pref:
            p.set_table_preference([1, 2, 3, 4])

        for _ in tqdm(range(50)):
            self.t.create_pairings()

            # at least 4 players should have preference satisfied
            satisfied_players = sum(
                [
                    1
                    for p in p_pref
                    if p.pod(self.t.tour_round).table in p.table_preference
                ]
            )
            self.assertGreaterEqual(satisfied_players, 4)
            self.t.reset_pods()

    def test_no_preference_swap(self) -> None:
        """
        Test that the no preference swap works.
        4 players want index 1, 4 players want index 0.
        Players in Pod A should swap with players in Pod B.
        """
        players = self.t.add_player([f"P{i}" for i in range(128)])

        # P0-P3 in Pod A, P4-P7 in Pod B
        # Pod A wants Index 1, Pod B wants Index 0
        for p in players[0:4]:
            p.set_table_preference([2])
        for p in players[4:8]:
            p.set_table_preference([1])

        # Manually assign
        self.t.tour_round.create_pods()

        pod1 = self.t.tour_round.pods[0]  # Index 0 / Table 1
        pod2 = self.t.tour_round.pods[1]  # Index 1 / Table 2
        for p in players[0:4]:
            pod1.add_player(p)
        for p in players[4:8]:
            pod2.add_player(p)

        self.t.create_pairings()

        # They should have swapped
        for p in players[0:4]:
            self.assertEqual(p.pod(self.t.tour_round).table, 2)
        for p in players[4:8]:
            self.assertEqual(p.pod(self.t.tour_round).table, 1)

    def test_stable_reordering(self) -> None:
        """
        Test that the stable reordering works.
        12 players = 3 pods (Pod 0, Pod 1, Pod 2).

        Pod 0: P0-P3
        Pod 1: P4-P7
        Pod 2: P8-P11

        Player in pod 1 want  Table 1 (index 0).
        Expected final order: [Pod 1, Pod 0, Pod 2]
        """
        players = self.t.add_player([f"P{i}" for i in range(12)])

        round = self.t.tour_round
        round.create_pods()
        round.refresh_player_location_map()

        pods = round.pods
        p0, p1, p2 = pods[0], pods[1], pods[2]

        # Assign players manually
        for i, p in enumerate(players):
            pods[i // 4].add_player(p)

        # Pod 1 (originally Table 2) wants Table 1 (index 0)
        for p in p1.players:
            p.set_table_preference([1])

        # pod1 is moved to index 0.
        # Remaining: pod0, pod2. They fill index 1 and index 2.
        # Expected final order: [pod1, pod0, pod2]

        round.sort_pods()

        self.assertEqual(round.pods[0].uid, p1.uid)
        self.assertEqual(round.pods[1].uid, p0.uid)
        self.assertEqual(round.pods[2].uid, p2.uid)

    def test_anonimity_serialization(self) -> None:
        p = self.t.add_player("Anon")[0]
        p.table_preference = [1, 2, 3]

        serialized = p.serialize()
        self.assertNotIn("table_preference", serialized)

        # Use the tour's cache directly to avoid collision
        uid = p.uid
        del self.t.PLAYER_CACHE[uid]

        # Re-inflate and check that it's empty
        p_inflated = Player.inflate(self.t, serialized)
        self.assertEqual(p_inflated.table_preference, [])


class TestRoundCreation(unittest.TestCase):
    def test_modified_n_rounds(self) -> None:
        """
        Test that the core does not allow to create more than n rounds even if it's modified.
        """

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
        """
        Test that the core does not allow to reset pods if n_rounds is modified.
        """

        t = Tournament(config = TournamentConfiguration(
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

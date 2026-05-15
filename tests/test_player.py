import unittest
from uuid import uuid4

from faker import Faker

from src.core import Player, SortMethod, Tournament, TournamentAction, TournamentConfiguration, TournamentContext
from src.misc import generate_player_names

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

        self.t.add_player(name="David", decklist="http://david.com")
        david = [p for p in self.t.players if p.name == "David"][0]
        self.assertEqual(david.name, "David")
        self.assertEqual(david.decklist, "http://david.com")

        self.t.add_player("Eve", name="Frank")
        player_names = {p.name for p in self.t.players}
        self.assertTrue("Eve" in player_names)
        self.assertTrue("Frank" in player_names)

    def test_add_player_refined_interface(self):
        self.t.new_round()

        u1 = uuid4()
        p = self.t.add_player(("SmartUID", u1))[0]
        self.assertEqual(p.uid, u1)
        self.assertIsNone(p.decklist)

        p = self.t.add_player(("SmartDeck", "http://deck.list"))[0]
        self.assertEqual(p.decklist, "http://deck.list")

        p = self.t.add_player("MergeString", decklist="http://merge.str")[0]
        self.assertEqual(p.decklist, "http://merge.str")

        p = self.t.add_player({"name": "MergeDict"}, decklist="http://merge.dict")[0]
        self.assertEqual(p.decklist, "http://merge.dict")

    def test_add_player_validation(self):
        self.t.new_round()
        with self.assertRaises(ValueError):
            self.t.add_player(123)

        with self.assertRaises(ValueError):
            self.t.add_player({"decklist": "..."})

    def test_player_serialization_with_decklist(self):
        self.t.new_round()

        u1 = uuid4()
        decklist_url = "https://moxfield.com/decks/jVKoDKgc0Ey2PcapxHIB_w"
        self.t.add_player(("TestPlayer", u1, decklist_url))

        p1 = [p for p in self.t.players if p.name == "TestPlayer"][0]

        serialized = p1.serialize()
        self.assertEqual(serialized["name"], "TestPlayer")
        self.assertEqual(serialized["uid"], str(u1))
        self.assertEqual(serialized["decklist"], decklist_url)

        del p1.CACHE[u1]

        p2 = Player.inflate(self.t, serialized)
        self.assertEqual(p2.name, "TestPlayer")
        self.assertEqual(p2.uid, u1)
        self.assertEqual(p2.decklist, decklist_url)

    def test_wins_count(self):
        cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=False, auto_export=False)
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(4)])
        t.create_pairings()
        r1_winner = t.tour_round.pods[0].players[0]
        r2_winner = t.tour_round.pods[0].players[1]

        t.report_win(r1_winner)

        t.new_round()
        t.create_pairings()
        t.report_win(r2_winner)

        t.new_round()
        t.create_pairings()
        t.report_draw(t.tour_round.pods[0].players)

        # One win awarded per round → 2 total across all players
        self.assertEqual(sum(p.wins(t.tour_round) for p in t.players), 2)
        self.assertEqual(r1_winner.wins(t.tour_round), 1)
        self.assertEqual(r2_winner.wins(t.tour_round), 1)
        self.assertEqual(r2_winner.wins(t.rounds[0]), 0)

    def test_losses_count(self):
        cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=False, auto_export=False)
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(4)])
        t.create_pairings()

        r1_winner = t.tour_round.pods[0].players[0]
        r2_winner = t.tour_round.pods[0].players[1]
        r1_losers = list(t.tour_round.pods[0].players[1:])
        r2_losers = list([r1_winner] + t.tour_round.pods[0].players[2:])
        t.report_win(r1_winner)

        t.new_round()
        t.create_pairings()
        t.report_win(r2_winner)

        # 3 losers per round × 2 rounds = 6 total losses
        self.assertEqual(sum(p.losses(t.tour_round) for p in t.players), 6)
        for p in r1_losers:
            self.assertGreaterEqual(p.losses(t.tour_round), 1)
        for p in r2_losers:
            self.assertGreaterEqual(p.losses(t.tour_round), 1)

    def test_draws_count(self):
        cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=False, auto_export=False)
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(4)])
        t.create_pairings()
        t.report_draw(t.tour_round.pods[0].players)

        t.new_round()
        t.create_pairings()
        t.report_draw(t.tour_round.pods[0].players)

        # Every player drew both rounds
        for p in t.players:
            self.assertEqual(p.draws(t.tour_round), 2)
            self.assertEqual(p.wins(t.tour_round), 0)
            self.assertEqual(p.losses(t.tour_round), 0)

    def test_byes_count(self):
        cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=True, max_byes=2, auto_export=False)
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(9)])
        t.create_pairings()
        r1_bye = next(iter(t.tour_round.byes))
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        t.new_round()
        t.create_pairings()
        r2_bye = next(iter(t.tour_round.byes))
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        # One bye awarded per round → 2 total across all players
        self.assertEqual(sum(p.byes(t.tour_round) for p in t.players), 2)
        self.assertGreaterEqual(r1_bye.byes(t.tour_round), 1)
        self.assertGreaterEqual(r2_bye.byes(t.tour_round), 1)

    def test_rating_after_round(self):
        cfg = TournamentConfiguration(
            pod_sizes=[4], allow_bye=False, auto_export=False, win_points=5
        )
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(8)])
        t.create_pairings()

        winner = t.tour_round.pods[0].players[0]
        t.report_win(winner)

        self.assertEqual(winner.rating(t.tour_round), 5.0)

    def test_opponent_pointrate(self):
        cfg = TournamentConfiguration(
            pod_sizes=[4], allow_bye=False, auto_export=False
        )
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(8)])
        t.create_pairings()
        t.random_results()

        for p in t.players:
            opr = p.opponent_pointrate(t.tour_round)
            self.assertIsInstance(opr, float)
            self.assertGreaterEqual(opr, 0.0)

    def test_drop_player(self):
        cfg = TournamentConfiguration(
            pod_sizes=[4, 3], allow_bye=True, max_byes=2, auto_export=False
        )
        t = Tournament(cfg)
        t.new_round()
        t.add_player(generate_player_names(8))
        t.create_pairings()

        player_to_drop = list(t.players)[0]
        t.drop_player(player_to_drop)
        t.random_results()
        t.new_round()
        t.create_pairings()

        all_podded = [p for pod in t.tour_round.pods for p in pod.players]
        self.assertNotIn(player_to_drop, all_podded)

    def test_toggle_game_loss(self):
        cfg = TournamentConfiguration(
            pod_sizes=[4], allow_bye=False, auto_export=False
        )
        t = Tournament(cfg)
        t.new_round()
        t.add_player([f"P{i}" for i in range(8)])
        t.create_pairings()

        target = t.tour_round.pods[0].players[0]
        t.toggle_game_loss(target)

        self.assertEqual(target.result(t.tour_round), Player.EResult.LOSS)


class TestPlayerComparison(unittest.TestCase):
    """__gt__ and __lt__ using SortMethod.ID, NAME, and RANK."""

    def setUp(self):
        cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=False, auto_export=False, win_points=5)
        self.t = Tournament(cfg)
        self.t.add_player([f"P{i}" for i in range(8)])
        self.t.create_pairings()
        # Give wins to deterministic winners so standings are predictable
        for pod in self.t.tour_round.pods:
            self.t.report_win(pod.players[0])

    def tearDown(self):
        # Reset class-level sort state
        Player.SORT_METHOD = SortMethod.ID

    def test_sort_by_name(self):
        Player.SORT_METHOD = SortMethod.NAME
        players = sorted(self.t.players)
        names = [p.name for p in players]
        self.assertEqual(names, sorted(names))

    def test_sort_by_id(self):
        Player.SORT_METHOD = SortMethod.ID
        players = sorted(self.t.players)
        ids = [p.uid for p in players]
        self.assertEqual(ids, sorted(ids))

    def test_sort_by_rank(self):
        Player.SORT_METHOD = SortMethod.RANK
        standings = self.t.get_standings(self.t.tour_round)
        context = TournamentContext(self.t, self.t.tour_round, standings)
        # Best player (standings[0]) should be "less than" worst (standings[-1])
        # because ascending rank order means standings[0] < standings[-1]
        best = standings[0]
        worst = standings[-1]
        self.assertTrue(best.__lt__(worst, context=context))
        self.assertTrue(worst.__gt__(best, context=context))
        self.assertFalse(best.__gt__(worst, context=context))

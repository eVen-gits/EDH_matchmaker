import unittest
import random

from faker import Faker

from src.core import Player, Tournament, TournamentAction, TournamentConfiguration

fkr = Faker()
TournamentAction.LOGF = False  # type: ignore


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
            player_uids = list(self.t._players)
            random.shuffle(player_uids)
            self.t._players = set(player_uids)

            self.assertEqual(self.t.get_standings(self.t.tour_round), orig_standings)

    def test_random_results(self):
        self.t.add_player([f"{i}:{fkr.name()}" for i in range(128)])
        for _ in range(10):
            self.t.create_pairings()
            self.t.random_results()
            self.assertTrue(self.t.tour_round.done)

            self.t.reset_pods()

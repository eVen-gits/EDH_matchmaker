import unittest

from src.core import Player, Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False  # type: ignore


def _make_wagering_tournament(**overrides) -> Tournament:
    kwargs = dict(
        pod_sizes=[4],
        allow_bye=True,
        auto_export=False,
        scoring_logic="ScoringHareruya",
        wager_percent=0.1,
        wagering_starting_points=100,
        bye_points=4,
    )
    kwargs.update(overrides)
    t = Tournament(TournamentConfiguration(**kwargs))
    t.new_round()
    return t


class TestScoringHareruya(unittest.TestCase):
    def test_round1_win_payout(self):
        t = _make_wagering_tournament()
        players = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(players)
        t.report_win(players[0])

        # wager = 10% of 100 = 10 each; pot = 40.
        self.assertEqual(players[0].rating(t.tour_round), 100 - 10 + 40)
        for loser in players[1:]:
            self.assertEqual(loser.rating(t.tour_round), 100 - 10)

    def test_draw_equal_split_ignores_wager_size(self):
        t = _make_wagering_tournament(wager_percent=0.1, wagering_starting_points=100)
        p = t.add_player([f"P{i}" for i in range(8)])

        # Round 1: two independent pods, diverging two players' stacks.
        t.manual_pod(p[0:4])
        t.report_win(p[0])
        t.manual_pod(p[4:8])
        t.report_win(p[4])
        # p[0], p[4] now have 130; the rest have 90.

        t.new_round()
        four = [p[0], p[1], p[4], p[5]]
        t.manual_pod(four)
        t.report_draw(four)  # full-pod draw, no losers

        # wagers: 13, 9, 13, 9 -> pot 44 -> equal share 11 each,
        # regardless of the underlying wager size (R=1, S=1 defaults).
        for player, current in zip(four, [130, 90, 130, 90]):
            expected = current - current * 0.1 + 11
            self.assertAlmostEqual(player.rating(t.tour_round), expected)

    def test_draw_pure_refund_at_shape_zero(self):
        t = _make_wagering_tournament(draw_distribution_shape=0.0)
        players = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(players)
        t.report_draw(players)  # full-pod draw

        # R=1 (default), S=0: each drawer gets exactly their own wager back.
        for player in players:
            self.assertAlmostEqual(player.rating(t.tour_round), 100)

    def test_draw_full_loss_at_redistribution_zero(self):
        t = _make_wagering_tournament(draw_redistribution_fraction=0.0)
        players = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(players)
        t.report_draw(players)

        # R=0: nobody gets anything back, regardless of S.
        for player in players:
            self.assertAlmostEqual(player.rating(t.tour_round), 90)

    def test_partial_draw_losers_forfeit_into_drawers_pot(self):
        t = _make_wagering_tournament()
        players = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(players)
        drawers = players[:2]
        losers = players[2:]
        t.report_draw(drawers)

        # pot = all 4 wagers (10 each) = 40; split equally among the
        # 2 drawers (R=1, S=1 defaults) = 20 each. Losers forfeit
        # their wager entirely and receive nothing.
        for player in drawers:
            self.assertAlmostEqual(player.rating(t.tour_round), 100 - 10 + 20)
        for player in losers:
            self.assertAlmostEqual(player.rating(t.tour_round), 100 - 10)

        # Total points conserved within the pod at R=1.
        total = sum(p.rating(t.tour_round) for p in players)
        self.assertAlmostEqual(total, 400)

    def test_get_standings_and_pointrate_do_not_crash(self):
        # get_standings() -> ranking() -> opponent_pointrate() ->
        # pointrate() -> pointrate_denominator() is a real call chain
        # this scoring logic must survive without dividing by zero or
        # raising, since standings are computed after every reported
        # result in the GUI.
        t = _make_wagering_tournament()
        players = t.add_player([f"P{i}" for i in range(9)])
        t.create_pairings()
        assert t.tour_round is not None
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        standings = t.get_standings(t.tour_round)
        self.assertEqual(len(standings), 9)
        # Winners must outrank losers.
        winner_ratings = {p.rating(t.tour_round) for p in standings[:2]}
        self.assertTrue(min(winner_ratings) > 100)
        for p in standings:
            rate = p.pointrate(t.tour_round)
            self.assertIsInstance(rate, (int, float))

    def test_bye_is_flat_and_untouched_by_wager_economy(self):
        t = _make_wagering_tournament()
        players = t.add_player([f"P{i}" for i in range(5)])
        t.create_pairings()

        assert t.tour_round is not None
        bye = next(iter(t.tour_round.byes))
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        self.assertAlmostEqual(bye.rating(t.tour_round), 100 + 4)

    def test_serialization_roundtrip_new_config_fields(self):
        cfg = TournamentConfiguration(
            scoring_logic="ScoringHareruya",
            wager_percent=0.05,
            wagering_starting_points=250,
            draw_redistribution_fraction=0.5,
            draw_distribution_shape=0.25,
            auto_export=False,
        )
        data = cfg.serialize()
        restored = TournamentConfiguration.inflate(data)

        self.assertEqual(restored.scoring_logic, "ScoringHareruya")
        self.assertEqual(restored.wager_percent, 0.05)
        self.assertEqual(restored.wagering_starting_points, 250)
        self.assertEqual(restored.draw_redistribution_fraction, 0.5)
        self.assertEqual(restored.draw_distribution_shape, 0.25)

    def test_old_format_data_defaults_new_fields(self):
        cfg = TournamentConfiguration(auto_export=False)
        data = cfg.serialize()
        for key in (
            "scoring_logic",
            "wager_percent",
            "wagering_starting_points",
            "draw_redistribution_fraction",
            "draw_distribution_shape",
        ):
            del data[key]

        restored = TournamentConfiguration.inflate(data)
        self.assertEqual(restored.scoring_logic, "ScoringDefault")
        self.assertEqual(restored.wager_percent, 0.07)
        self.assertEqual(restored.wagering_starting_points, 1000)
        self.assertEqual(restored.draw_redistribution_fraction, 1.0)
        self.assertEqual(restored.draw_distribution_shape, 1.0)


class TestScoringDefaultUnchanged(unittest.TestCase):
    """ScoringDefault must reproduce the pre-plugin fixed-points behavior exactly."""

    def test_default_scoring_logic_matches_legacy_behavior(self):
        t = Tournament(
            TournamentConfiguration(
                pod_sizes=[4],
                allow_bye=True,
                win_points=4,
                bye_points=4,
                draw_points=1,
                auto_export=False,
            )
        )
        t.new_round()
        players = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(players)
        t.report_win(players[0])

        self.assertEqual(t.config.scoring_logic, "ScoringDefault")
        self.assertEqual(players[0].rating(t.tour_round), 4)
        for loser in players[1:]:
            self.assertEqual(loser.rating(t.tour_round), 0)


if __name__ == "__main__":
    unittest.main()

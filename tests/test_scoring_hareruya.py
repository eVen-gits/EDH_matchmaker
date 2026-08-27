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

    def test_standings_export_does_not_crash_on_float_rating(self):
        # Regression: StandingsExport.info[Field.RATING].format was
        # "{:d}", which raises ValueError on a float rating. Under
        # Hareruya, rating() returns a float, and this is exactly the
        # path run_ui.py's export button calls.
        t = _make_wagering_tournament()
        t.add_player([f"P{i}" for i in range(9)])
        t.create_pairings()
        assert t.tour_round is not None
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        output = t.get_standings_str()
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

    def test_bye_leaves_stack_unchanged(self):
        t = _make_wagering_tournament()
        players = t.add_player([f"P{i}" for i in range(5)])
        t.create_pairings()

        assert t.tour_round is not None
        bye = next(iter(t.tour_round.byes))
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        # A bye is not a wager and not a flat bonus under Hareruya -
        # the player's stack is exactly what it started at.
        self.assertAlmostEqual(bye.rating(t.tour_round), 100)

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


class TestScoringModifiedHareruya(unittest.TestCase):
    """Modified Hareruya averages the wagering economy over every round order."""

    def _two_round_diverged(self):
        # Two players play two 1-on-1 rounds against different stakes so that
        # their records differ in order (P0: win then draw; P1: draw then win).
        t = _make_wagering_tournament(
            scoring_logic="ScoringModifiedHareruya",
            pod_sizes=[2],
        )
        p = t.add_player([f"P{i}" for i in range(4)])

        # Round 1: P0 beats P2, P1 draws P3.
        t.manual_pod([p[0], p[2]])
        t.report_win(p[0])
        t.manual_pod([p[1], p[3]])
        t.report_draw([p[1], p[3]])

        # Round 2: P0 draws P2, P1 beats P3.
        t.new_round()
        t.manual_pod([p[0], p[2]])
        t.report_draw([p[0], p[2]])
        t.manual_pod([p[1], p[3]])
        t.report_win(p[1])
        return t, p

    def test_order_invariance_win_draw_vs_draw_win(self):
        # P0 has record W,D and P1 has record D,W - same multiset of outcomes
        # against symmetric opponents, so Modified Hareruya must rate them
        # equally despite the order difference.
        t, p = self._two_round_diverged()
        r0 = p[0].rating(t.tour_round)
        r1 = p[1].rating(t.tour_round)
        self.assertAlmostEqual(r0, r1)

    def test_plain_hareruya_is_order_sensitive(self):
        # Sanity check that the two algorithms actually differ: plain Hareruya
        # rates the same W,D vs D,W records differently.
        t, p = self._two_round_diverged()
        t.config.scoring_logic = "ScoringHareruya"
        r0 = p[0].rating(t.tour_round)
        r1 = p[1].rating(t.tour_round)
        self.assertNotAlmostEqual(r0, r1)

    def test_hand_computed_average(self):
        # Ground truth, worked out by hand (start 100, wager 10%). One pod of
        # 4 plays two rounds: R1 P0 wins, R2 P1 wins.
        #
        #   order [R1, R2]: R1 -> P0=130, rest 90; R2 wagers 13/9/9/9, pot 40,
        #                   P1 wins -> P0=117, P1=121, P2=81, P3=81.
        #   order [R2, R1]: symmetric      -> P0=121, P1=117, P2=81, P3=81.
        #   average:                          P0=119, P1=119, P2=81, P3=81.
        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        p = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(p)
        t.report_win(p[0])
        t.new_round()
        t.manual_pod(p)
        t.report_win(p[1])

        self.assertAlmostEqual(p[0].rating(t.tour_round), 119)
        self.assertAlmostEqual(p[1].rating(t.tour_round), 119)
        self.assertAlmostEqual(p[2].rating(t.tour_round), 81)
        self.assertAlmostEqual(p[3].rating(t.tour_round), 81)
        # Points are conserved: no stake is created or destroyed by averaging.
        self.assertAlmostEqual(sum(pl.rating(t.tour_round) for pl in p), 400)

    def test_average_of_both_orders(self):
        # A win then a draw. Modified Hareruya must equal the mean of the two
        # single-order replays.
        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        p = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(p)
        t.report_win(p[0])
        t.new_round()
        t.manual_pod(p)
        t.report_draw(p)

        logic = t.get_scoring_logic("ScoringHareruya")
        rounds = list(logic._swiss_rounds_up_to(t, t.tour_round))
        order_a = logic._replay(t, [rounds[0], rounds[1]])
        order_b = logic._replay(t, [rounds[1], rounds[0]])
        for player in p:
            expected = (order_a[player.uid] + order_b[player.uid]) / 2
            self.assertAlmostEqual(player.rating(t.tour_round), expected)

    def test_single_round_equals_plain_hareruya(self):
        # One round: only one order, so Modified Hareruya == plain Hareruya.
        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        p = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(p)
        t.report_win(p[0])

        modified = {pl.uid: pl.rating(t.tour_round) for pl in p}
        plain = t.get_scoring_logic("ScoringHareruya").compute_ratings(t, t.tour_round)
        for pl in p:
            self.assertAlmostEqual(modified[pl.uid], plain[pl.uid])

    def test_sampling_path_is_deterministic(self):
        # Force the sampled branch with a low threshold and confirm the fixed
        # seed makes two evaluations identical (stable standings).
        from src.scoring_logic import examples

        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        p = t.add_player([f"P{i}" for i in range(4)])
        t.manual_pod(p)
        t.report_win(p[0])
        t.new_round()
        t.manual_pod(p)
        t.report_draw(p)

        logic = t.get_scoring_logic("ScoringModifiedHareruya")
        original = examples._EXACT_MAX_ROUNDS
        try:
            examples._EXACT_MAX_ROUNDS = 1  # 2 rounds -> sampled branch
            first = logic.compute_ratings(t, t.tour_round)
            second = logic.compute_ratings(t, t.tour_round)
        finally:
            examples._EXACT_MAX_ROUNDS = original
        self.assertEqual(first, second)

    def test_standings_export_does_not_crash(self):
        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        t.add_player([f"P{i}" for i in range(9)])
        t.create_pairings()
        assert t.tour_round is not None
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        output = t.get_standings_str()
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)


class TestSmallPodPot(unittest.TestCase):
    """A pod's pot is the sum of only its real seated wagers - no phantom fill."""

    def test_three_man_pod_no_phantom(self):
        t = _make_wagering_tournament(pod_sizes=[3])
        players = t.add_player([f"P{i}" for i in range(3)])
        t.manual_pod(players)
        t.report_win(players[0])

        # wager = 10% of 100 = 10 each; pot = 30 (three seats, no phantom 4th).
        # A phantom 1000-point 4th would make the pot ~130 and the winner ~220.
        self.assertAlmostEqual(players[0].rating(t.tour_round), 100 - 10 + 30)
        for loser in players[1:]:
            self.assertAlmostEqual(loser.rating(t.tour_round), 90)
        # Points conserved within the pod: exactly 3 x start, not inflated.
        total = sum(p.rating(t.tour_round) for p in players)
        self.assertAlmostEqual(total, 300)

    def test_smaller_pod_gives_smaller_reward(self):
        # The winner's net gain scales with real pod size: a 3-man win nets
        # two opponents' wagers (+20), a 4-man win nets three (+30).
        t3 = _make_wagering_tournament(pod_sizes=[3])
        p3 = t3.add_player([f"P{i}" for i in range(3)])
        t3.manual_pod(p3)
        t3.report_win(p3[0])
        gain3 = p3[0].rating(t3.tour_round) - 100

        t4 = _make_wagering_tournament(pod_sizes=[4])
        p4 = t4.add_player([f"P{i}" for i in range(4)])
        t4.manual_pod(p4)
        t4.report_win(p4[0])
        gain4 = p4[0].rating(t4.tour_round) - 100

        self.assertAlmostEqual(gain3, 20)
        self.assertAlmostEqual(gain4, 30)
        self.assertLess(gain3, gain4)


class TestStandingsRatingReuse(unittest.TestCase):
    """get_standings must compute the field's ratings once, not per player."""

    def test_compute_ratings_called_once_per_standings(self):
        # Without the pass-down, ranking() recomputes the whole field once
        # per player plus once per opponent (~players x opponents times).
        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        t.add_player([f"P{i}" for i in range(8)])
        t.create_pairings()
        assert t.tour_round is not None
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])
        t.new_round()
        t.create_pairings()
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])

        logic = t.get_scoring_logic("ScoringModifiedHareruya")
        calls = {"n": 0}
        original = logic.compute_ratings

        def counting(tour, tour_round):
            calls["n"] += 1
            return original(tour, tour_round)

        logic.compute_ratings = counting  # type: ignore[method-assign]
        try:
            standings = t.get_standings(t.tour_round)
        finally:
            del logic.compute_ratings  # restore the class method
        self.assertEqual(len(standings), 8)
        self.assertEqual(calls["n"], 1)

    def test_create_pairings_does_not_recompute_per_player(self):
        # Pairing sort keys (matching, bye_matching, snake_ranking) and the
        # pod power sort all read ratings. Each make_pairings/sort must reuse
        # one precomputed field map, not recompute it per player.
        t = _make_wagering_tournament(scoring_logic="ScoringModifiedHareruya")
        t.add_player([f"P{i}" for i in range(16)])
        t.create_pairings()
        assert t.tour_round is not None
        for pod in t.tour_round.pods:
            t.report_win(pod.players[0])
        t.new_round()

        logic = t.get_scoring_logic("ScoringModifiedHareruya")
        calls = {"n": 0}
        original = logic.compute_ratings

        def counting(tour, tour_round):
            calls["n"] += 1
            return original(tour, tour_round)

        logic.compute_ratings = counting  # type: ignore[method-assign]
        try:
            t.create_pairings()
        finally:
            del logic.compute_ratings
        # One map per sort phase (pairing + pod power), not ~players.
        self.assertLessEqual(calls["n"], 3)


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

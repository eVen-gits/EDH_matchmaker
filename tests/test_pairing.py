import unittest
import random
import pytest
from itertools import product

from faker import Faker

from src.core import (
    Log,
    Player,
    Round,
    Tournament,
    TournamentAction,
    TournamentConfiguration,
)

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


class TestResetPodsConfigChange(unittest.TestCase):
    """create_pairings() must pick up config changes regardless of when they happen."""

    def _run_swiss(self, t: Tournament, n_rounds: int) -> None:
        for _ in range(n_rounds):
            t.create_pairings()
            t.random_results()
            t.new_round()

    def _make_config(self, top_cut=TournamentConfiguration.TopCut.TOP_4) -> TournamentConfiguration:
        return TournamentConfiguration(
            pod_sizes=[4, 3],
            n_rounds=5,
            top_cut=top_cut,
            auto_export=False,
        )

    def test_config_change_before_reset(self) -> None:
        """Scenario 1: edit config → reset pods → create pairings picks up new config."""
        t = Tournament(self._make_config(TournamentConfiguration.TopCut.TOP_4))
        t.add_player([f"P{i}" for i in range(16)])
        self._run_swiss(t, 5)

        t.create_pairings()
        self.assertEqual(t.tour_round.stage, Round.Stage.TOP_4)

        t.config = self._make_config(TournamentConfiguration.TopCut.TOP_7)
        t.reset_pods()

        ok = t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(t.tour_round.stage, Round.Stage.TOP_7)

    def test_config_change_after_reset(self) -> None:
        """Scenario 2: reset pods → edit config → create pairings picks up new config."""
        t = Tournament(self._make_config(TournamentConfiguration.TopCut.TOP_4))
        t.add_player([f"P{i}" for i in range(16)])
        self._run_swiss(t, 5)

        t.create_pairings()
        self.assertEqual(t.tour_round.stage, Round.Stage.TOP_4)

        t.reset_pods()
        t.config = self._make_config(TournamentConfiguration.TopCut.TOP_7)

        ok = t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(t.tour_round.stage, Round.Stage.TOP_7)

    def test_swiss_stage_unchanged_after_reset(self) -> None:
        """reset_pods() on a Swiss round without config change keeps SWISS stage."""
        t = Tournament(self._make_config(TournamentConfiguration.TopCut.NONE))
        t.add_player([f"P{i}" for i in range(16)])
        t.create_pairings()
        self.assertEqual(t.tour_round.stage, Round.Stage.SWISS)

        t.reset_pods()

        ok = t.create_pairings()
        self.assertTrue(ok)
        self.assertEqual(t.tour_round.stage, Round.Stage.SWISS)


class TestPairingLogicsConfig(unittest.TestCase):
    """config.pairing_logics picks each Swiss round's pairing logic."""

    def _swiss_tournament(self, pairing_logics=None):
        cfg = TournamentConfiguration(
            pod_sizes=[4], n_rounds=5, top_cut=TournamentConfiguration.TopCut.TOP_4,
            allow_bye=True, auto_export=False,
            pairing_logics=pairing_logics or [],
        )
        t = Tournament(cfg)
        t.add_player([f"P{i}" for i in range(16)])
        return t

    def _logics(self, t, n):
        out = []
        for _ in range(n):
            t.create_pairings()
            out.append(t.tour_round.logic.name)
            t.random_results()
            t.new_round()
        return out

    def test_configured_logic_per_round(self):
        t = self._swiss_tournament(
            ["PairingDefault", "PairingRandom", "PairingSnake"]
        )
        self.assertEqual(
            self._logics(t, 3), ["PairingDefault", "PairingRandom", "PairingSnake"]
        )

    def test_empty_falls_back_to_adaptive(self):
        t = self._swiss_tournament([])
        self.assertEqual(
            self._logics(t, 3), ["PairingRandom", "PairingSnake", "PairingDefault"]
        )

    def test_out_of_range_seq_uses_adaptive(self):
        # Only round 0 configured; round 1 falls back to the adaptive Snake.
        t = self._swiss_tournament(["PairingDefault"])
        self.assertEqual(self._logics(t, 2), ["PairingDefault", "PairingSnake"])

    def test_top_cut_ignores_pairing_logics(self):
        t = self._swiss_tournament(["PairingRandom"] * 5)
        for _ in range(5):
            t.create_pairings()
            t.random_results()
            t.new_round()
        t.create_pairings()  # next round is the TOP_4 cut
        self.assertEqual(t.tour_round.stage, Round.Stage.TOP_4)
        self.assertEqual(t.tour_round.logic.name, "PairingTop4")

    def test_serialize_roundtrip_and_backward_compat(self):
        import json

        cfg = TournamentConfiguration(
            pairing_rounds=[
                {"logic": "PairingRandom", "params": {}},
                {"logic": "PairingDefault", "params": {
                    "rematch_penalty_exponent": 3, "small_pod_penalty": 5,
                }},
            ],
            auto_export=False,
        )
        data = cfg.serialize()
        self.assertIn("pairing_rounds", data)
        restored = TournamentConfiguration.inflate(json.loads(json.dumps(data)))
        self.assertEqual(restored.pairing_logics, ["PairingRandom", "PairingDefault"])
        self.assertEqual(
            restored.pairing_params,
            [{}, {"rematch_penalty_exponent": 3, "small_pod_penalty": 5}],
        )

        # An old file without pairing config inflates to empty (adaptive).
        del data["pairing_rounds"]
        old = TournamentConfiguration.inflate(data)
        self.assertEqual(old.pairing_rounds, [])

    def test_inflate_interim_separate_lists(self):
        # The never-released interim format stored separate logics/params lists
        # (and an empty {} for params). Inflate must zip them into pairing_rounds.
        data = TournamentConfiguration(auto_export=False).serialize()
        data.pop("pairing_rounds", None)
        data["pairing_logics"] = ["PairingRandom", "PairingDefault"]
        data["pairing_params"] = [{}, {"rematch_penalty_exponent": 4}]
        restored = TournamentConfiguration.inflate(data)
        self.assertEqual(
            restored.pairing_rounds,
            [
                {"logic": "PairingRandom", "params": {}},
                {"logic": "PairingDefault", "params": {"rematch_penalty_exponent": 4}},
            ],
        )
        # A stale empty {} for params coerces to no entries.
        data["pairing_logics"] = []
        data["pairing_params"] = {}
        self.assertEqual(TournamentConfiguration.inflate(data).pairing_rounds, [])


class TestPairingParams(unittest.TestCase):
    """config.pairing_params gives each round its own pairing overrides."""

    def _tournament(self, pairing_params=None):
        cfg = TournamentConfiguration(
            pod_sizes=[4], n_rounds=3, allow_bye=True, auto_export=False,
            pairing_params=pairing_params or [],
        )
        t = Tournament(cfg)
        t.add_player([f"P{i}" for i in range(16)])
        return t

    def _round(self, t):
        t.create_pairings()
        return t.tour_round

    def test_param_defaults_when_no_override(self):
        t = self._tournament()
        logic = Tournament.get_pairing_logic("PairingDefault")
        r = self._round(t)
        self.assertEqual(logic._param(r, "rematch_penalty_exponent"), 2)
        self.assertEqual(logic._param(r, "small_pod_penalty"), 10)

    def test_param_reads_per_round_override(self):
        # Round 0 overrides, round 1 does not.
        t = self._tournament([{"rematch_penalty_exponent": 3, "small_pod_penalty": 5}])
        logic = Tournament.get_pairing_logic("PairingDefault")
        r0 = self._round(t)
        self.assertEqual(logic._param(r0, "rematch_penalty_exponent"), 3)
        self.assertEqual(logic.params(r0), {
            "rematch_penalty_exponent": 3, "small_pod_penalty": 5,
        })
        t.random_results()
        t.new_round()
        r1 = self._round(t)  # no entry for seq 1 -> defaults
        self.assertEqual(logic._param(r1, "rematch_penalty_exponent"), 2)

    def test_override_does_not_leak_across_rounds(self):
        # An override for round 1 must not reach round 0.
        t = self._tournament([{}, {"rematch_penalty_exponent": 99}])
        logic = Tournament.get_pairing_logic("PairingDefault")
        r0 = self._round(t)
        self.assertEqual(logic._param(r0, "rematch_penalty_exponent"), 2)

    def test_pairing_completes_with_override(self):
        # Custom params must not break pod assignment: every player seated.
        t = self._tournament(
            [{"rematch_penalty_exponent": 1, "small_pod_penalty": 0}] * 3
        )
        for _ in range(3):
            t.create_pairings()
            seated = sum(len(pod.players) for pod in t.tour_round.pods)
            byes = len(t.tour_round.byes)
            self.assertEqual(seated + byes, 16)
            t.random_results()
            t.new_round()


class TestPodSizeCompatibility(unittest.TestCase):
    """Pairing logics are offered only when they support the tournament sizes."""

    def test_supports_pod_sizes(self):
        default = Tournament.get_pairing_logic("PairingDefault")
        random_ = Tournament.get_pairing_logic("PairingRandom")
        self.assertEqual(default.SUPPORTED_POD_SIZES, (3, 4, 5))
        self.assertIsNone(random_.SUPPORTED_POD_SIZES)  # any size
        # Default supports a subset of {3,4,5}, not 2.
        self.assertTrue(default.supports_pod_sizes([4, 3]))
        self.assertTrue(default.supports_pod_sizes([5, 4, 3]))
        self.assertFalse(default.supports_pod_sizes([2]))
        self.assertFalse(default.supports_pod_sizes([4, 3, 2]))
        # Random supports anything.
        self.assertTrue(random_.supports_pod_sizes([2]))
        self.assertTrue(random_.supports_pod_sizes([4, 3]))

    def test_selectable_filtered_by_pod_sizes(self):
        self.assertEqual(
            Tournament.selectable_pairing_logics([4, 3]),
            ["PairingDefault", "PairingRandom", "PairingSnake"],
        )
        self.assertEqual(
            Tournament.selectable_pairing_logics([5, 4, 3]),
            ["PairingDefault", "PairingRandom", "PairingSnake"],
        )
        # Only Random supports 2-player pods among current algorithms.
        self.assertEqual(
            Tournament.selectable_pairing_logics([2]), ["PairingRandom"]
        )
        self.assertEqual(
            Tournament.selectable_pairing_logics([4, 3, 2]), ["PairingRandom"]
        )

    def test_selectable_without_pod_sizes_returns_all(self):
        self.assertEqual(
            Tournament.selectable_pairing_logics(),
            ["PairingDefault", "PairingRandom", "PairingSnake"],
        )

    def test_supported_pod_sizes_are_valid(self):
        # Every selectable logic declares None (any) or a tuple of ints.
        for name in Tournament.selectable_pairing_logics():
            supported = Tournament.get_pairing_logic(name).SUPPORTED_POD_SIZES
            if supported is None:
                continue
            self.assertIsInstance(supported, tuple, name)
            self.assertTrue(supported, name)  # non-empty
            for size in supported:
                self.assertIsInstance(size, int, name)
                self.assertGreaterEqual(size, 2, name)

    def test_adaptive_default_respects_pod_sizes(self):
        # With 2-player pods and no explicit logics, the adaptive default must
        # never pick an incompatible logic (Snake/Default) - only Random fits.
        cfg = TournamentConfiguration(
            pod_sizes=[2], n_rounds=3, snake_pods=True, allow_bye=True,
            auto_export=False,
        )
        t = Tournament(cfg)
        t.add_player([f"P{i}" for i in range(8)])
        for _ in range(3):
            t.create_pairings()
            logic = t.tour_round.logic
            self.assertTrue(
                logic.supports_pod_sizes([2]),
                f"{logic.name} does not support pod size 2",
            )
            t.random_results()
            t.new_round()

    def test_incompatible_configured_logic_warns(self):
        # An explicit incompatible choice is honored but logged as a warning.
        Log.output.clear()
        cfg = TournamentConfiguration(
            pod_sizes=[2], n_rounds=1, allow_bye=True, auto_export=False,
            pairing_rounds=[{"logic": "PairingDefault", "params": {}}],
        )
        t = Tournament(cfg)
        t.add_player([f"P{i}" for i in range(8)])
        t.create_pairings()
        self.assertEqual(t.tour_round.logic.name, "PairingDefault")  # honored
        warnings = [
            e.msg
            for e in Log.output
            if e.level == Log.Level.WARNING and "does not support" in e.msg
        ]
        self.assertTrue(warnings, "expected a pod-size compatibility warning")

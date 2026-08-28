from __future__ import annotations
import itertools
import random
from abc import ABC
from collections.abc import Iterator
from typing import Any

from ..interface import IPlayer, IRound, IScoringLogic, ITournament

class CommonScoring(IScoringLogic, ABC):
    _SWISS = 0  # Round.Stage.SWISS / TournamentConfiguration.TopCut.NONE value

    def __init__(self, name: str):
        self.name = name

    def _swiss_rounds_up_to(
        self, tour: ITournament, tour_round: IRound
    ) -> Iterator[IRound]:
        """Yields Swiss-stage rounds in order, stopping after tour_round.

        Both concrete scoring logics only accumulate points for Swiss
        rounds - a round whose stage is not SWISS (e.g. a top-cut
        playoff round) ends the accumulation, matching the pre-existing
        Tournament.rating() behavior this replaces.
        """
        for i_tour_round in tour.rounds:
            if i_tour_round.stage.value != self._SWISS:
                break
            yield i_tour_round
            if i_tour_round == tour_round:
                break

    def rating(self, player: IPlayer, tour_round: IRound) -> float:
        # Shared fallback for scoring logics whose economy is
        # interdependent across players (e.g. ScoringHareruya), where
        # answering one player's query requires replaying everyone's.
        # ScoringDefault overrides this with a cheap, independent path.
        return self.compute_ratings(player.tour, tour_round).get(player.uid, 0)

    def params(self, tour: ITournament) -> dict[str, Any]:
        """This algorithm's params, tournament overrides on top of its own defaults.

        Cold path only (call once, not per player/opponent) - it allocates
        a merged dict. On a hot path, use _param() instead.
        """
        config = tour.config  # type: ignore[attr-defined]
        return {**self.DEFAULT_PARAMS, **config.scoring_params}

    def _param(self, tour: ITournament, key: str) -> Any:
        """Single-param lookup with no allocation - for the rating/pointrate hot path."""
        config = tour.config  # type: ignore[attr-defined]
        return config.scoring_params.get(key, self.DEFAULT_PARAMS[key])


class ScoringDefault(CommonScoring):
    """Fixed win/draw/bye point constants - the original, pre-plugin behavior."""

    IS_COMPLETE: bool = True

    # Parameters (names, defaults, descriptions) live in the sidecar
    # ScoringDefault.params.yaml; DEFAULT_PARAMS is derived from it.

    def rating(self, player: IPlayer, tour_round: IRound) -> float:
        # Independent per player - O(rounds), not O(players * rounds).
        # Deliberately NOT implemented via compute_ratings(): every
        # player's Default rating is independent of every other
        # player's, so replaying the whole tournament just to answer
        # one player's query would be pure waste on the hottest path
        # in the app (get_standings() calls this once per player).
        # _param(), not params(): this runs per player, so it must not
        # allocate a merged dict on every call.
        tour = player.tour
        win_points = self._param(tour, "win_points")
        draw_points = self._param(tour, "draw_points")
        bye_points = self._param(tour, "bye_points")
        # Kept as int, not 0.0: win/draw/bye_points are ints, and
        # StandingsExport's RATING column formats with "{:d}" - this
        # must stay byte-identical to the pre-plugin behavior.
        points: float = 0
        for i_tour_round in self._swiss_rounds_up_to(tour, tour_round):
            round_result = player.result(i_tour_round)
            if round_result == IPlayer.EResult.WIN:
                points += win_points
            elif round_result == IPlayer.EResult.DRAW:
                points += draw_points
            elif round_result == IPlayer.EResult.BYE:
                points += bye_points
        return points

    def compute_ratings(
        self, tour: ITournament, tour_round: IRound
    ) -> dict[Any, float]:
        return {p.uid: self.rating(p, tour_round) for p in tour.players}

    def pointrate_denominator(self, tour_round: IRound) -> float:
        # A win is assumed to be the maximum possible score for one
        # round, so the max total after N rounds is win_points * N.
        return self._param(tour_round.tour, "win_points") * (tour_round.seq + 1)


class ScoringHareruya(CommonScoring):
    """Percentage-of-stack wagering, winner-takes-pot, configurable draw payout.

    See docs/tournament-log-spec.md, "Scoring logic", for the formulas
    this implements.
    """

    IS_COMPLETE: bool = True

    # Parameters live in the sidecar ScoringHareruya.params.yaml.
    # ScoringModifiedHareruya inherits this spec (it ships no sidecar).

    def compute_ratings(
        self, tour: ITournament, tour_round: IRound
    ) -> dict[Any, float]:
        return self._replay(tour, list(self._swiss_rounds_up_to(tour, tour_round)))

    def _replay(
        self, tour: ITournament, ordered_rounds: list[IRound]
    ) -> dict[Any, float]:
        """Runs the wagering economy over ordered_rounds and returns final stacks.

        The round order matters: each wager is a percentage of the current
        (compounding) stack, so replaying the same rounds in a different
        order gives different totals. This is the hook ScoringModifiedHareruya
        uses to average over every round-order permutation.
        """
        # params(), not _param() x6: this runs once per compute_ratings()
        # call, not per player, so the merged-dict allocation is fine here.
        p = self.params(tour)
        index, round_ops = self._extract(tour, ordered_rounds)
        points = self._replay_indexed(
            round_ops,
            len(index),
            float(p["wagering_starting_points"]),
            p["wager_percent"],
            p["draw_redistribution_fraction"],
            p["draw_distribution_shape"],
            p["redistribute_discarded_draw_points"],
            p["draw_discard_pod_fraction"],
        )
        return {uid: points[i] for uid, i in index.items()}

    def _extract(
        self, tour: ITournament, ordered_rounds: list[IRound]
    ) -> tuple[dict[Any, int], list[list[tuple[list[int], int | None, list[int]]]]]:
        """Reduces rounds to permutation-invariant, integer-indexed pod results.

        Round order changes only the wager compounding, never who won a pod, so
        the pod results are extracted once here (the only place p.result() is
        called) and reused across every permutation. Player UUIDs are mapped to
        list indices so the replay loop needs no dict keys or UUID hashing.

        Returns (index, round_ops) where round_ops[k] aligns to
        ordered_rounds[k]. Each pod is (seated_idxs, winner_idx_or_None,
        drawer_idxs). Empty and any-PENDING pods are dropped, exactly as the
        wager loop skipped them before.
        """
        index: dict[Any, int] = {p.uid: i for i, p in enumerate(tour.players)}
        round_ops: list[list[tuple[list[int], int | None, list[int]]]] = []
        for i_tour_round in ordered_rounds:
            pods_ops: list[tuple[list[int], int | None, list[int]]] = []
            for pod in i_tour_round.pods:
                seated = pod.players
                if not seated:
                    continue
                results = [(index[p.uid], p.result(i_tour_round)) for p in seated]
                if any(r == IPlayer.EResult.PENDING for _, r in results):
                    # Pod not fully resolved yet - no wager is assessed.
                    continue
                idxs = [i for i, _ in results]
                winner = next(
                    (i for i, r in results if r == IPlayer.EResult.WIN), None
                )
                drawers = [i for i, r in results if r == IPlayer.EResult.DRAW]
                pods_ops.append((idxs, winner, drawers))
            round_ops.append(pods_ops)
        return index, round_ops

    @staticmethod
    def _replay_indexed(
        round_ops: list[list[tuple[list[int], int | None, list[int]]]],
        n: int,
        start: float,
        wager_percent: float,
        R: float,
        S: float,
        redistribute_discard: bool,
        pod_fraction: float,
    ) -> list[float]:
        """Replays the wager economy on an integer-indexed stack list.

        Pure arithmetic - no p.result(), no dict keys, no UUID hashing - so it
        is cheap to call once per permutation.
        """
        points = [start] * n
        for pod_list in round_ops:
            for idxs, winner, drawers in pod_list:
                wagers = [points[i] * wager_percent for i in idxs]
                for k, i in enumerate(idxs):
                    points[i] -= wagers[k]
                pot = sum(wagers)

                if winner is not None:
                    points[winner] += pot
                elif drawers:
                    nd = len(drawers)
                    wager_of = {i: wagers[k] for k, i in enumerate(idxs)}
                    for i in drawers:
                        points[i] += R * (S * (pot / nd) + (1 - S) * wager_of[i])

                    if redistribute_discard:
                        # (1-R)*pot is exactly what the loop above never pays
                        # out. Split it: pod_fraction back to this pod's
                        # drawers (a bonus on top of the payout above), the
                        # rest as an equal per-capita dividend to every player
                        # in the tournament, including players on a bye.
                        discarded = (1 - R) * pot
                        pod_share = pod_fraction * discarded
                        # discarded - pod_share, not a second multiply, so the
                        # two halves sum back to `discarded` exactly.
                        global_share = discarded - pod_share

                        drawer_wager_total = sum(wager_of[i] for i in drawers)
                        for i in drawers:
                            wager_share = (
                                wager_of[i] / drawer_wager_total
                                if drawer_wager_total > 0
                                else 1.0 / nd
                            )
                            points[i] += pod_share * (
                                S * (1.0 / nd) + (1 - S) * wager_share
                            )

                        if global_share:
                            per_capita = global_share / n
                            for i in range(n):
                                points[i] += per_capita
                # Any remaining seated players are plain losers: their wager
                # was already deducted above and they receive nothing further.
        return points

    def pointrate_denominator(self, tour_round: IRound) -> float:
        # No fixed per-round maximum exists under wagering (a win can
        # take an arbitrarily large pot), so this is an approximation
        # shaped like the default: starting stake x rounds played,
        # not a strict [0, 1] bound.
        # _param(): this runs per player/opponent, so it must not
        # allocate a merged dict on every call.
        starting_points = self._param(tour_round.tour, "wagering_starting_points")
        return starting_points * (tour_round.seq + 1)


class ScoringModifiedHareruya(ScoringHareruya):
    """Order-independent Hareruya: averages the wagering economy over every
    permutation of the round order.

    Plain Hareruya is order-sensitive - the record WDD scores differently
    from DDW because each wager compounds the current stack. This variant
    replays the economy for every round-order permutation and averages each
    player's final stack, so record order no longer affects the result.

    For more than _EXACT_MAX_ROUNDS rounds, exact enumeration (n!) is
    replaced by a fixed-seed random sample of _SAMPLE_COUNT orders.
    """

    IS_COMPLETE: bool = True

    # Thresholds below are algorithm internals, not per-tournament settings,
    # so kept as class constants (tests monkeypatch them) rather than
    # plumbed through TournamentConfiguration.
    _EXACT_MAX_ROUNDS = 7  # 7! = 5040 replays; above this, sample instead
    _SAMPLE_COUNT = 5000
    _SAMPLE_SEED = 0  # fixed -> deterministic average -> stable standings

    def compute_ratings(
        self, tour: ITournament, tour_round: IRound
    ) -> dict[Any, float]:
        rounds = list(self._swiss_rounds_up_to(tour, tour_round))
        n = len(rounds)
        if n <= 1:
            # Zero or one round: only one order exists, so averaging is moot.
            return self._replay(tour, rounds)

        # Extract pod results ONCE. Results are permutation-invariant, so every
        # permutation replays the same round_ops in a different order.
        # params(): this runs once per compute_ratings() call, not once per
        # permutation, so the merged-dict allocation is fine here.
        p = self.params(tour)
        start = float(p["wagering_starting_points"])
        wager_percent = p["wager_percent"]
        R = p["draw_redistribution_fraction"]
        S = p["draw_distribution_shape"]
        redistribute_discard = p["redistribute_discarded_draw_points"]
        pod_fraction = p["draw_discard_pod_fraction"]
        index, round_ops = self._extract(tour, rounds)
        n_players = len(index)

        if n <= self._EXACT_MAX_ROUNDS:
            position_orders: Iterator[Any] = itertools.permutations(range(n))
        else:
            # ponytail: n! blows up past _EXACT_MAX_ROUNDS rounds; sample
            #           fixed-seed random orders instead of enumerating.
            rng = random.Random(self._SAMPLE_SEED)
            position_orders = (
                rng.sample(range(n), n) for _ in range(self._SAMPLE_COUNT)
            )

        totals = [0.0] * n_players
        count = 0
        for perm in position_orders:
            points = self._replay_indexed(
                [round_ops[k] for k in perm],
                n_players,
                start,
                wager_percent,
                R,
                S,
                redistribute_discard,
                pod_fraction,
            )
            for i in range(n_players):
                totals[i] += points[i]
            count += 1
        return {uid: totals[i] / count for uid, i in index.items()}

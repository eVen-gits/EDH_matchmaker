from __future__ import annotations

import random
import sys
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...interface import IPairingLogic, IPlayer, IPod, IRound


class CommonPairing(IPairingLogic, ABC):
    def __init__(self, name: str):
        self.name = name

    def _round_overrides(self, tour_round: IRound) -> dict[str, Any]:
        """The pairing_params entry for this round, or an empty dict."""
        config = tour_round.tour.config  # type: ignore[attr-defined]
        params = config.pairing_params
        seq = tour_round.seq
        return params[seq] if seq < len(params) else {}

    def params(self, tour_round: IRound) -> dict[str, Any]:
        """This round's params: round overrides on top of the class defaults.

        Unlike CommonScoring.params (one flat dict for the single selected
        scoring logic), config.pairing_params is a list indexed by round, because
        pairing logic and its settings can differ per round.
        """
        return {**self.DEFAULT_PARAMS, **self._round_overrides(tour_round)}

    def _param(self, tour_round: IRound, key: str) -> Any:
        """Single-param lookup with no allocation - for a hot path."""
        overrides = self._round_overrides(tour_round)
        return overrides.get(key, self.DEFAULT_PARAMS[key])

    def field_ratings(self, tour_round: IRound) -> Mapping[Any, float]:
        """Computes the whole field's ratings once, to pass down sort keys.

        Pairing sort keys (matching, bye_matching, snake_ranking) read player
        ratings. Without this, each key call recomputes the entire field
        (once per player, plus once per opponent) - costly under wagering
        scoring. make_pairings computes this once and threads it through.
        """
        return tour_round.tour.field_ratings(tour_round)

    def evaluate_pod(self, player: IPlayer, pod: IPod, tour_round: IRound) -> int:
        score = 0
        if len(pod) == pod.cap:
            return -sys.maxsize
        exponent = self._param(tour_round, "rematch_penalty_exponent")
        for p in pod.players:
            score -= player.played(tour_round).count(p) ** exponent
        # A pod smaller than the preferred (first, highest-preference) pod size
        # is "small". Using the preferred size, not the largest, keeps the
        # 3-player pod as the small one when a larger size (for example 5) is
        # also allowed - otherwise adding 5 would make 4-player pods count as
        # small and skew pairing toward the largest pod.
        preferred_size = player.tour.config.pod_sizes[0]
        small_pod_penalty = self._param(tour_round, "small_pod_penalty")
        if pod.cap < preferred_size:
            for prev_pod in player.pods(tour_round):
                if prev_pod in IPlayer.ELocation:
                    continue
                score -= sum(
                    [
                        small_pod_penalty
                        for _ in prev_pod.players
                        if prev_pod.cap < preferred_size
                    ]
                )
        return score

    def bye_matching(
        self,
        player: IPlayer,
        tour_round: IRound,
        ratings: Mapping[Any, float] | None = None,
    ) -> tuple:
        return (
            -len(player.games(tour_round)),
            player.rating(tour_round, ratings),
            -len(player.played(tour_round)),
        )

    def assign_byes(
        self,
        tour_round: IRound,
        players: set[IPlayer],
        pods: Sequence[IPod],
        ratings: Mapping[Any, float] | None = None,
    ) -> set[IPlayer]:
        if ratings is None:
            ratings = self.field_ratings(tour_round)
        capacity = sum([pod.cap - len(pod.players) for pod in pods])
        n_byes = len(players) - capacity

        matching: Callable[[IPlayer], tuple] = lambda x: self.bye_matching(
            x, tour_round, ratings
        )
        player_matches = {p: matching(p) for p in players}
        keys: list[tuple] = sorted(set(player_matches.values()), reverse=True)

        buckets = [[p for p in players if player_matches[p] == k] for k in keys]

        byes = set()
        for b in buckets[::-1]:
            byes.update(random.sample(b, min(len(b), n_byes - len(byes))))
            if len(byes) >= n_byes:
                break
        for p in byes:
            p.set_result(tour_round, IPlayer.EResult.BYE)

        return byes

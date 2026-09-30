from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import Any

from ...interface import IPlayer, IRound, IScoringLogic, ITournament


class CommonScoring(IScoringLogic, ABC):
    _SWISS = 0  # Round.Stage.SWISS / TournamentConfiguration.TopCut.NONE value

    def __init__(self, name: str):
        self.name = name

    def _swiss_rounds_up_to(
        self, tour: ITournament, tour_round: IRound
    ) -> Iterator[IRound]:
        """Yields Swiss rounds in order, stopping after tour_round."""
        for i_tour_round in tour.rounds:
            if i_tour_round.stage.value != self._SWISS:
                break
            yield i_tour_round
            if i_tour_round == tour_round:
                break

    def rating(self, player: IPlayer, tour_round: IRound) -> float:
        return self.compute_ratings(player.tour, tour_round).get(player.uid, 0)

    def params(self, tour: ITournament) -> dict[str, Any]:
        """Tournament overrides on top of defaults; use _param on hot paths."""
        config = tour.config  # type: ignore[attr-defined]
        return {**self.DEFAULT_PARAMS, **config.scoring_params}

    def _param(self, tour: ITournament, key: str) -> Any:
        """Single-param lookup without allocation."""
        config = tour.config  # type: ignore[attr-defined]
        return config.scoring_params.get(key, self.DEFAULT_PARAMS[key])


class FixedPointsScoring(CommonScoring):
    """Shared win/draw/bye scoring, with game-specific parameters and standings."""

    def rating(self, player: IPlayer, tour_round: IRound) -> float:
        # Independent per player, O(rounds). Keep integer values for exports.
        tour = player.tour
        win_points = self._param(tour, "win_points")
        draw_points = self._param(tour, "draw_points")
        bye_points = self._param(tour, "bye_points")
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
        return self._param(tour_round.tour, "win_points") * (tour_round.seq + 1)

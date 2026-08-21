from __future__ import annotations
from abc import ABC
from collections.abc import Iterator
from typing import Any

from ..interface import IPlayer, IRound, IScoringLogic, ITournament

_SWISS = 0  # Round.Stage.SWISS / TournamentConfiguration.TopCut.NONE value


class CommonScoring(IScoringLogic, ABC):
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
            if i_tour_round.stage.value != _SWISS:
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


class ScoringDefault(CommonScoring):
    """Fixed win/draw/bye point constants - the original, pre-plugin behavior."""

    IS_COMPLETE: bool = True

    def rating(self, player: IPlayer, tour_round: IRound) -> float:
        # Independent per player - O(rounds), not O(players * rounds).
        # Deliberately NOT implemented via compute_ratings(): every
        # player's Default rating is independent of every other
        # player's, so replaying the whole tournament just to answer
        # one player's query would be pure waste on the hottest path
        # in the app (get_standings() calls this once per player).
        config = player.tour.config  # type: ignore[attr-defined]
        # Kept as int, not 0.0: win/draw/bye_points are ints, and
        # StandingsExport's RATING column formats with "{:d}" - this
        # must stay byte-identical to the pre-plugin behavior.
        points: float = 0
        for i_tour_round in self._swiss_rounds_up_to(player.tour, tour_round):
            round_result = player.result(i_tour_round)
            if round_result == IPlayer.EResult.WIN:
                points += config.win_points
            elif round_result == IPlayer.EResult.DRAW:
                points += config.draw_points
            elif round_result == IPlayer.EResult.BYE:
                points += config.bye_points
        return points

    def compute_ratings(
        self, tour: ITournament, tour_round: IRound
    ) -> dict[Any, float]:
        return {p.uid: self.rating(p, tour_round) for p in tour.players}

    def pointrate_denominator(self, tour_round: IRound) -> float:
        # A win is assumed to be the maximum possible score for one
        # round, so the max total after N rounds is win_points * N.
        config = tour_round.tour.config  # type: ignore[attr-defined]
        return config.win_points * (tour_round.seq + 1)


class ScoringHareruya(CommonScoring):
    """Percentage-of-stack wagering, winner-takes-pot, configurable draw payout.

    See docs/tournament-log-spec.md, "Scoring logic", for the formulas
    this implements.
    """

    IS_COMPLETE: bool = True

    def compute_ratings(
        self, tour: ITournament, tour_round: IRound
    ) -> dict[Any, float]:
        config = tour.config  # type: ignore[attr-defined]
        wager_percent = config.wager_percent
        R = config.draw_redistribution_fraction
        S = config.draw_distribution_shape

        points = {p.uid: float(config.wagering_starting_points) for p in tour.players}

        for i_tour_round in self._swiss_rounds_up_to(tour, tour_round):
            # A bye leaves the player's stack unchanged under Hareruya -
            # no wager, no bye_points. Nothing to do here.

            for pod in i_tour_round.pods:
                seated = pod.players
                if not seated:
                    continue
                results = {p.uid: p.result(i_tour_round) for p in seated}
                if any(r == IPlayer.EResult.PENDING for r in results.values()):
                    # Pod not fully resolved yet - no wager is assessed.
                    continue

                wagers = {uid: points[uid] * wager_percent for uid in results}
                for uid in wagers:
                    points[uid] -= wagers[uid]
                pot = sum(wagers.values())

                winners = [
                    uid for uid, r in results.items() if r == IPlayer.EResult.WIN
                ]
                drawers = [
                    uid for uid, r in results.items() if r == IPlayer.EResult.DRAW
                ]

                if winners:
                    # A pod's result set has at most one member when it is
                    # a win, so there is exactly one winner here.
                    points[winners[0]] += pot
                elif drawers:
                    n = len(drawers)
                    for uid in drawers:
                        payout = R * (S * (pot / n) + (1 - S) * wagers[uid])
                        points[uid] += payout
                # Any remaining seated players are plain losers: their
                # wager was already deducted above and they receive
                # nothing further, whether or not the pod had a winner.

        return points

    def pointrate_denominator(self, tour_round: IRound) -> float:
        # No fixed per-round maximum exists under wagering (a win can
        # take an arbitrarily large pot), so this is an approximation
        # shaped like the default: starting stake x rounds played,
        # not a strict [0, 1] bound.
        config = tour_round.tour.config  # type: ignore[attr-defined]
        return config.wagering_starting_points * (tour_round.seq + 1)

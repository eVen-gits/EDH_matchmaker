from __future__ import annotations
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from uuid import UUID

from ...interface import IPlayer, IRound, ITournament
from ..common.scoring import CommonScoring, FixedPointsScoring

# As written in the Magic Tournament Rules (3.1, Appendix C) - not 1/3.
FLOOR = Fraction(33, 100)


def pct(points: Fraction, possible: Fraction | int) -> Fraction:
    """MW/GW's percentage-with-a-floor: max(FLOOR, points / possible).

    Args:
        points: Match or game points earned.
        possible: The maximum possible (win_points * rounds_played or
            3 * games_played). FLOOR if this is 0 (no rounds/games played).
    """
    if not possible:
        return FLOOR
    return max(FLOOR, points / possible)


def mean_pct(values: list[Fraction]) -> Fraction:
    """Mean of opponents' floored percentages, or FLOOR without opponents.

    Players with only byes or unseated game losses have no opponents.
    """
    if not values:
        return FLOOR
    return sum(values, Fraction(0)) / len(values)


@dataclass(frozen=True)
class MtrStats:
    """One player's raw MTR tiebreaker inputs, as of a given round.

    `opponents` has one entry per round with a decided, seated pod - a
    forced rematch appears twice, matching the MTR's "once per round" rule
    for opponents rather than deduplicating them.
    """

    match_points: Fraction
    rounds_played: int
    game_points: Fraction
    games_played: int
    opponents: tuple[UUID, ...]


def _swiss_rounds_up_to(tour: ITournament, tour_round: IRound) -> Iterator[IRound]:
    """Yields Swiss rounds through tour_round; a playoff ends accumulation."""
    for i_tour_round in tour.rounds:
        if i_tour_round.stage.value != CommonScoring._SWISS:
            break
        yield i_tour_round
        if i_tour_round == tour_round:
            break


def mtr_stats(tour: ITournament, player: IPlayer, tour_round: IRound) -> MtrStats:
    """Computes one player's MTR tiebreaker inputs as of tour_round.

    Match points come from the configured scoring logic. Byes count as
    2-0 wins without an opponent. Unseated game losses count as a round
    without games or opponents. Pending results contribute nothing.
    """
    rounds_played = 0
    game_points = Fraction(0)
    games_played = 0
    opponents: list[UUID] = []
    for r in _swiss_rounds_up_to(tour, tour_round):
        result = player.result(r)
        if result == IPlayer.EResult.PENDING:
            continue
        rounds_played += 1
        if result == IPlayer.EResult.BYE:
            games_played += 2
            game_points += 6
            continue
        pod = player.pod(r)
        if pod is None:
            # A game-loss penalty with no seated/decided pod: no games.
            continue
        for game in pod.games:
            games_played += 1
            if player.uid in game.winners:
                game_points += Fraction(1) if len(game.winners) > 1 else Fraction(3)
        opponent = next((p for p in pod.players if p.uid != player.uid), None)
        if opponent is not None:
            opponents.append(opponent.uid)
    match_points = Fraction(tour.rating(player, tour_round)).limit_denominator(10**6)  # type: ignore[attr-defined]
    return MtrStats(match_points, rounds_played, game_points, games_played, tuple(opponents))


class Scoring1v1(FixedPointsScoring):
    """Match points and MTR tiebreakers for 1v1 Magic tournaments.

    Magic Tournament Rules Appendix C ("Match Points"): 3 points for a
    match win, 1 for a draw, 0 for a loss; a bye counts as an automatic
    2-0 win (3 points). Point values come from Scoring1v1.params.yaml.
    Equal points use OMW, GW, then OGW, with the MTR percentage floor.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)

    def standings_keys(
        self,
        tour: ITournament,
        tour_round: IRound,
        ratings: Mapping[Any, float],
    ) -> Mapping[UUID, tuple]:
        """Sort key, descending: (rating, OMW, GW, OGW, -uid) - MTR 3.1."""
        stats = {p.uid: mtr_stats(tour, p, tour_round) for p in tour.players}
        win_points = self._param(tour, "win_points")
        mw = {uid: pct(s.match_points, win_points * s.rounds_played) for uid, s in stats.items()}
        gw = {uid: pct(s.game_points, 3 * s.games_played) for uid, s in stats.items()}
        return {
            p.uid: (
                ratings.get(p.uid, 0),
                mean_pct([mw[o] for o in stats[p.uid].opponents]),
                gw[p.uid],
                mean_pct([gw[o] for o in stats[p.uid].opponents]),
                -p.uid if isinstance(p.uid, int) else -int(p.uid.int),
            )
            for p in tour.players
        }

    def standings_columns(
        self, tour: ITournament, tour_round: IRound
    ) -> list[tuple[str, Mapping[UUID, str]]]:
        stats = {p.uid: mtr_stats(tour, p, tour_round) for p in tour.players}
        win_points = self._param(tour, "win_points")
        mw = {uid: pct(s.match_points, win_points * s.rounds_played) for uid, s in stats.items()}
        gw = {uid: pct(s.game_points, 3 * s.games_played) for uid, s in stats.items()}
        omw_col: dict[UUID, str] = {}
        gw_col: dict[UUID, str] = {}
        ogw_col: dict[UUID, str] = {}
        for p in tour.players:
            opponents = stats[p.uid].opponents
            omw_col[p.uid] = f"{float(mean_pct([mw[o] for o in opponents])):.4f}"
            gw_col[p.uid] = f"{float(gw[p.uid]):.4f}"
            ogw_col[p.uid] = f"{float(mean_pct([gw[o] for o in opponents])):.4f}"
        return [("OMW", omw_col), ("GW", gw_col), ("OGW", ogw_col)]

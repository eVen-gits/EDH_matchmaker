from __future__ import annotations
from collections.abc import Mapping
from typing import Any

import numpy as np
from typing_extensions import override

from ...interface import IPlayer, IRound
from ..commander import scoring as _commander_scoring


class Scoring1v1(_commander_scoring.ScoringDefault):
    """Match points for 1v1 Magic tournaments.

    Magic Tournament Rules Appendix C ("Match Points"): 3 points for a
    match win, 1 for a draw, 0 for a loss; a bye counts as an automatic
    2-0 win (3 points). This is exactly ScoringDefault's win/draw/bye-point
    formula (src/logic/commander/scoring.py) - only the point values
    differ, MTR-correct by default instead of Commander's, set in this
    class's own Scoring1v1.params.yaml sidecar.
    """

    IS_COMPLETE = True

    # MTR Appendix C: an opponent's match-win % or game-win % contributes at
    # least this much to your own tiebreakers, regardless of how badly that
    # opponent did elsewhere - stops one blown-out opponent from tanking
    # your tiebreakers.
    _OPPONENT_FLOOR = 1 / 3

    def _game_win_pct(self, player: IPlayer, tour_round: IRound) -> float:
        """A player's own Game-Win % (MTR Appendix C): games won / games
        played, up to and including tour_round.

        A bye counts as an automatic 2-0 win (2 games won, 2 games played),
        matching this class's match-point bye convention. Pod.game_wins/
        Pod._games already track per-game results for best-of-N (see
        src/core.py) - this is the first place that sums them per player.
        """
        wins = 0
        total = 0
        for pod in player.games(tour_round):  # type: ignore[attr-defined]
            wins += pod.game_wins.get(player.uid, 0)  # type: ignore[attr-defined]
            total += len(pod._games)  # type: ignore[attr-defined]
        n_byes = 2 * player.byes(tour_round)  # type: ignore[attr-defined]
        wins += n_byes
        total += n_byes
        return wins / total if total else 0.0

    @override
    def ranking(
        self,
        x: IPlayer,
        tour_round: IRound,
        ratings: Mapping[Any, float] | None = None,
    ) -> tuple[int | float | str, ...]:
        """MTR Appendix C-faithful standings tiebreaker order.

        Match points, then Opponents' Match-Win % (floored at 1/3), then
        Game-Win %, then Opponents' Game-Win % (floored at 1/3), then a
        deterministic tiebreak. Unlike CommonScoring.ranking (Commander's
        formula, unchanged), this drops players_beaten/average_seat -
        meaningless for a 2-player pod - and adds the two Game-Win
        criteria MTR requires that Commander's formula never needed.

        Player.played() already excludes bye rounds from the opponent list
        (src/core.py), so MTR's "byes are ignored in opponents'
        percentages" rule is already satisfied by both averages below.
        """
        opponents = x.played(tour_round)  # type: ignore[attr-defined]
        if opponents:
            omw = sum(
                max(opp.pointrate(tour_round, ratings), self._OPPONENT_FLOOR)  # type: ignore[attr-defined]
                for opp in opponents
            ) / len(opponents)
            ogw = sum(
                max(self._game_win_pct(opp, tour_round), self._OPPONENT_FLOOR)
                for opp in opponents
            ) / len(opponents)
        else:
            omw = 0.0
            ogw = 0.0
        return (
            x.rating(tour_round, ratings),
            np.round(omw, 10),
            np.round(self._game_win_pct(x, tour_round), 10),
            np.round(ogw, 10),
            -x.uid if isinstance(x.uid, int) else -int(x.uid.int),
        )

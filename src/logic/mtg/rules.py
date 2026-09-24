from __future__ import annotations
import random
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID

from ...interface import IGameResult, IPlayer, IPod, IRound, IRuleset, ITournament


class Mtg1v1Ruleset(IRuleset):
    """1v1 Magic tournament rules (Magic Tournament Rules, see
    src/logic/mtg/mtr-1v1-spec.md).

    No playoff plan yet (PLAYOFFS stays the IRuleset default {}) and no MTR
    tiebreakers yet: standings_keys falls back to rating plus a stable UID
    tie-break until Phase 3 adds the OMW/GW/OGW chain.
    """

    IS_COMPLETE = True

    DEFAULT_POD_SIZES = (2,)
    ALLOWED_POD_SIZES = (2,)
    DEFAULT_SCORING_LOGIC = "Scoring1v1"
    # Spec: a coin flip or the higher seed picks play/draw (informational
    # only) - seats otherwise carry no meaning, unlike Commander's.
    SEAT_BALANCING = False

    # Round.Stage.SWISS's value, without importing core.py (see
    # CommonScoring._SWISS in src/logic/commander/scoring.py for the same
    # pattern).
    _SWISS_STAGE_VALUE = 0

    def validate_report(self, pod: IPod, games: list[IGameResult]) -> None:
        """Raises ValueError unless games is a valid 1v1 match report.

        Rules (Magic Tournament Rules 2.1): the pod seats exactly two
        players; at least one game was played; each game was won by one of
        them or drawn between both; neither player's single-game win count
        exceeds games_to_win, and they must not both reach it (a report may
        end below games_to_win when time is called).
        """
        name = pod.name  # type: ignore[attr-defined]
        seated = {p.uid for p in pod.players}
        if len(seated) != 2:
            raise ValueError(
                f"{name}: a 1v1 pod must seat exactly 2 players, got {len(seated)}."
            )
        if not games:
            raise ValueError(f"{name}: a match report needs at least one game.")

        tally: dict[UUID, int] = {uid: 0 for uid in seated}
        for game in games:
            if not game.winners <= seated:
                raise ValueError(f"{name}: game winners must be seated in the pod.")
            if len(game.winners) == 1:
                (winner,) = game.winners
                tally[winner] += 1

        g = self._param(pod.tour_round, "games_to_win")
        if any(wins > g for wins in tally.values()):
            raise ValueError(f"{name}: no player may win more than {g} games.")
        if all(wins >= g for wins in tally.values()):
            raise ValueError(f"{name}: both players cannot reach {g} game wins.")

    def match_winners(self, pod: IPod) -> frozenset[UUID]:
        tally: dict[UUID, int] = {p.uid: 0 for p in pod.players}
        for game in pod.games:
            if len(game.winners) == 1:
                (winner,) = game.winners
                tally[winner] = tally.get(winner, 0) + 1
        top = max(tally.values(), default=0)
        return frozenset(uid for uid, wins in tally.items() if wins == top)

    def report_from_winners(
        self, pod: IPod, winners: Iterable[UUID]
    ) -> list[IGameResult]:
        """report_win shorthand -> games_to_win single-winner games (a clean
        sweep); report_draw shorthand -> one drawn game (0-0-1).

        Exact scores (e.g. a 2-1 win, a time-called 1-0, an ID at 0-0-3) go
        through Tournament.report_match with games_from_score, not this
        shorthand.
        """
        winners = frozenset(winners)
        if len(winners) == 1:
            g = self._param(pod.tour_round, "games_to_win")
            return [IGameResult(winners) for _ in range(g)]
        return [IGameResult(winners)]

    def random_report(self, pod: IPod) -> list[IGameResult]:
        """Simulates games until someone reaches games_to_win: each game is
        about 5% drawn, otherwise a coin flip. In a Swiss round, there is
        also a ~5% chance after each game that time is called and the match
        stops early below games_to_win; a playoff round never stops early
        (spec: no drawn single-elimination match). Always a valid report."""
        g = self._param(pod.tour_round, "games_to_win")
        a, b = (p.uid for p in pod.players)
        tally = {a: 0, b: 0}
        is_swiss = pod.tour_round.stage.value == self._SWISS_STAGE_VALUE  # type: ignore[attr-defined]
        games: list[IGameResult] = []
        while tally[a] < g and tally[b] < g:
            roll = random.random()
            if roll < 0.05:
                games.append(IGameResult(frozenset({a, b})))
            elif roll < 0.525:
                games.append(IGameResult(frozenset({a})))
                tally[a] += 1
            else:
                games.append(IGameResult(frozenset({b})))
                tally[b] += 1
            if is_swiss and random.random() < 0.05:
                break
        return games

    def swiss_pairing_logic(self, tour: ITournament, seq: int) -> str:
        # Round 1 comes out random anyway: everyone is in one score group
        # (spec 4.1 step 1), so Pairing1v1 is always the Swiss default.
        return "Pairing1v1"

    def standings_keys(
        self,
        tour: ITournament,
        tour_round: IRound,
        ratings: Mapping[Any, float],
    ) -> Mapping[UUID, tuple]:
        return {
            p.uid: (
                ratings.get(p.uid, 0),
                -p.uid if isinstance(p.uid, int) else -int(p.uid.int),
            )
            for p in tour.players
        }


def games_from_score(
    pod: IPod, wins: Mapping[IPlayer, int], draws: int = 0
) -> list[IGameResult]:
    """Builds a whole-match report from a final score.

    Every key of `wins` must be seated in `pod`; a missing player counts as
    0. Does not check games_to_win - Tournament.report_match/
    IRuleset.validate_report does. Usage::

        t.report_match(pod, games_from_score(pod, {alice: 2, bob: 1}))  # 2-1
        t.report_match(pod, games_from_score(pod, {alice: 1}))          # time called at 1-0
        t.report_match(pod, games_from_score(pod, {}, draws=3))         # ID, 0-0-3

    Args:
        pod: The pod being reported.
        wins: Single-game wins per player, in any order (returned games
            follow the pod's seat order, not this mapping's order).
        draws: Number of drawn games to append after the single-winner ones.

    Returns:
        The games, single-winner ones in seat order, then the drawn ones.
    """
    games: list[IGameResult] = [
        IGameResult(frozenset({p.uid}))
        for p in pod.players
        for _ in range(wins.get(p, 0))
    ]
    if draws:
        seated = frozenset(p.uid for p in pod.players)
        games.extend(IGameResult(seated) for _ in range(draws))
    return games

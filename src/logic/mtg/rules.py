from __future__ import annotations
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from uuid import UUID

from ...interface import IGameResult, IPlayer, IPod, IRound, IRuleset, ITournament

# As written in the Magic Tournament Rules (3.1, Appendix C) - not 1/3.
FLOOR = Fraction(33, 100)


def pct(points: Fraction, possible: Fraction | int) -> Fraction:
    """MW/GW's percentage-with-a-floor: max(FLOOR, points / possible).

    Args:
        points: Match or game points earned.
        possible: The maximum possible (3 * rounds_played or 3 *
            games_played). FLOOR if this is 0 (no rounds/games played).
    """
    if not possible:
        return FLOOR
    return max(FLOOR, points / possible)


def mean_pct(values: list[Fraction]) -> Fraction:
    """OMW/OGW's mean of each opponent's (already-floored) MW or GW.

    FLOOR if there are no opponents (only byes and/or game-loss penalties) -
    open decision 2, Mtg1v1Ruleset only (Commander has no OMW/OGW concept).
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
    """Yields Swiss-stage rounds in order, stopping after tour_round.

    Same pattern as CommonScoring._swiss_rounds_up_to
    (src/logic/commander/scoring.py): a non-Swiss round ends accumulation.
    """
    for i_tour_round in tour.rounds:
        if i_tour_round.stage.value != Mtg1v1Ruleset._SWISS_STAGE_VALUE:  # type: ignore[attr-defined]
            break
        yield i_tour_round
        if i_tour_round == tour_round:
            break


def mtr_stats(tour: ITournament, player: IPlayer, tour_round: IRound) -> MtrStats:
    """Computes one player's MTR tiebreaker inputs as of tour_round.

    See the "Result" table in the Mtg1v1Ruleset docstring for the exact
    per-round contribution rules. match_points is read from the
    tournament's configured scoring logic (Scoring1v1 by default) rather
    than recomputed here, since the MTR's match-point values (3/1/0, or
    3/1 with a bye counted as a 2-0 win) are exactly that scoring logic's
    default win/draw/bye points.
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
    # ponytail: assumes the configured scoring logic's rating is an exact
    # integer-valued total (true for ScoringDefault-family logics, which is
    # what every practical MTG scoring logic is) - Fraction(float) is exact
    # for the float given, so this is safe for integer point values but
    # could pick up binary rounding noise for a fractional custom
    # win_points. Recompute from the scoring sidecar's own params if that
    # ever matters.
    match_points = Fraction(tour.rating(player, tour_round)).limit_denominator(10**6)  # type: ignore[attr-defined]
    return MtrStats(match_points, rounds_played, game_points, games_played, tuple(opponents))


class Mtg1v1Ruleset(IRuleset):
    """1v1 Magic tournament rules (Magic Tournament Rules, see
    src/logic/mtg/mtr-1v1-spec.md).

    Per-round contribution to the MTR tiebreaker stats (see mtr_stats),
    over Swiss rounds only, from round 0 up to and including the round in
    question:

    | Result | Match pts | Rounds | Games / game pts | Opponents |
    |---|---|---|---|---|
    | BYE | scoring logic's bye points | +1 | +2 games, +6 pts (a 2-0 win) | none |
    | WIN/DRAW/LOSS, seated in a decided pod | scoring logic's win/draw/0 | +1 | +1 game per game played; +3 if won alone, +1 if drawn, 0 if lost | the pod's other player |
    | LOSS from a game-loss penalty, no decided pod | 0 | +1 | none | none |
    | PENDING (unassigned, dropped, pending pod) | skipped entirely | | | |

    Playoff plan (spec 4): single elimination, seeded by the final Swiss
    standings - see PairingBracket (src/logic/mtg/matching.py).
    """

    IS_COMPLETE = True

    DEFAULT_POD_SIZES = (2,)
    ALLOWED_POD_SIZES = (2,)
    DEFAULT_SCORING_LOGIC = "Scoring1v1"
    # Spec: a coin flip or the higher seed picks play/draw (informational
    # only) - seats otherwise carry no meaning, unlike Commander's.
    SEAT_BALANCING = False
    # An odd player count must get a bye, never leave someone unseated.
    BYES_REQUIRED = True

    # Round.Stage.SWISS's value, without importing core.py (see
    # CommonScoring._SWISS in src/logic/commander/scoring.py for the same
    # pattern).
    _SWISS_STAGE_VALUE = 0

    PLAYOFFS = {
        2: ((2, "PairingBracket"),),
        4: ((4, "PairingBracket"), (2, "PairingBracket")),
        8: ((8, "PairingBracket"), (4, "PairingBracket"), (2, "PairingBracket")),
        16: (
            (16, "PairingBracket"),
            (8, "PairingBracket"),
            (4, "PairingBracket"),
            (2, "PairingBracket"),
        ),
    }

    def validate_report(self, pod: IPod, games: list[IGameResult]) -> None:
        """Raises ValueError unless games is a valid 1v1 match report.

        Rules (Magic Tournament Rules 2.1): the pod seats exactly two
        players; at least one game was played; each game was won by one of
        them or drawn between both; neither player's single-game win count
        exceeds games_to_win, and they must not both reach it (a report may
        end below games_to_win when time is called) - except in a playoff
        round, where a tied game-win tally (including no games decided
        either way) is never valid: a single-elimination match cannot end
        in a draw.
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

        is_playoff = pod.tour_round.stage.value != self._SWISS_STAGE_VALUE  # type: ignore[attr-defined]
        if is_playoff and len(set(tally.values())) == 1:
            raise ValueError(f"{name}: a playoff match cannot end in a draw.")

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
        """Sort key, descending: (rating, OMW, GW, OGW, -uid) - MTR 3.1."""
        stats = {p.uid: mtr_stats(tour, p, tour_round) for p in tour.players}
        mw = {uid: pct(s.match_points, 3 * s.rounds_played) for uid, s in stats.items()}
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
        mw = {uid: pct(s.match_points, 3 * s.rounds_played) for uid, s in stats.items()}
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

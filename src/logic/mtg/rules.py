from __future__ import annotations
import random
from collections.abc import Iterable, Mapping
from uuid import UUID

from ...core import TournamentConfiguration
from ...interface import IGameResult, IPlayer, IPod, IRuleset, ITournament

class Mtg1v1Configuration(TournamentConfiguration):
    """1v1 Magic's config fields, on top of the shared ones.

    Fields and defaults: the sidecar `Mtg1v1Configuration.params.yaml`.
    """

    match_wr_seats: list[float]
    match_draw_rate: float


class Mtg1v1Ruleset(IRuleset):
    """1v1 Magic tournament rules (Magic Tournament Rules, see
    src/logic/mtg/mtr-1v1-spec.md).

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
    CONFIG_CLASS = Mtg1v1Configuration

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
        """Simulates a match from config.match_draw_rate/match_wr_seats
        (see Mtg1v1Configuration). A drawn Swiss match is time called at
        k-k below games_to_win (0-0 is one drawn game); a playoff match
        never draws (spec: no drawn single-elimination match). A decided
        match goes games_to_win to 0..games_to_win-1, the winner taking the
        last game. Always a valid report."""
        config: Mtg1v1Configuration = pod.tour_round.tour.config  # type: ignore[attr-defined]
        g = self._param(pod.tour_round, "games_to_win")
        a, b = (p.uid for p in pod.players)
        is_swiss = pod.tour_round.stage.value == self._SWISS_STAGE_VALUE  # type: ignore[attr-defined]
        if is_swiss and random.random() < config.match_draw_rate:
            k = random.randrange(g)
            if k == 0:
                return [IGameResult(frozenset({a, b}))]
            return [IGameResult(frozenset({u})) for u in [a, b] * k]
        winner, loser = random.choices(
            [(a, b), (b, a)], weights=config.match_wr_seats
        )[0]
        order = [winner] * (g - 1) + [loser] * random.randrange(g)
        random.shuffle(order)
        return [IGameResult(frozenset({u})) for u in order + [winner]]

    def swiss_pairing_logic(self, tour: ITournament, seq: int) -> str:
        # Round 1 comes out random anyway: everyone is in one score group
        # (spec 4.1 step 1), so Pairing1v1 is always the Swiss default.
        return "Pairing1v1"


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

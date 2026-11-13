from __future__ import annotations
import random
from collections.abc import Iterable, Sequence
from uuid import UUID

from ...core import TournamentConfiguration
from ...interface import IGameResult, IPod, IRuleset, ITournament


class CatanConfiguration(TournamentConfiguration):
    """Catan's config fields, on top of the shared ones.

    Simulation only (they drive `random_report`). Fields and defaults: the
    sidecar `CatanConfiguration.params.yaml`.
    """

    seat_win_rates: Sequence[float]


class CatanRuleset(IRuleset):
    """Catan tournament rules (catanevents.com tournament rules, section 1).

    Every pod plays one game, ends at `victory_points` and has exactly one
    winner: there are no draws. The match report carries each player's
    victory points (`IGameResult.scores`). Save-format contracts:
    docs/tournament-log-spec.md, "`CatanRuleset`".

    Hidden from users (`IS_COMPLETE = False`) until the playoff slice lands.
    """

    IS_COMPLETE = False

    DEFAULT_POD_SIZES = (4, 3)
    ALLOWED_POD_SIZES = (3, 4)
    DEFAULT_SCORING_LOGIC = "ScoringCatan"
    # Play order is part of the pairing, not auto-balanced by core.
    SEAT_BALANCING = False
    # 3-player pods absorb remainders (rule 1.1.7). A forced bye scores
    # bye_points, never VP.
    DEFAULT_ALLOW_BYE = False
    # TODO(slice 6): PLAYOFFS for 4 and 16 (semifinals and final).
    CONFIG_CLASS = CatanConfiguration

    def validate_report(self, pod: IPod, games: Sequence[IGameResult]) -> None:
        """Raises ValueError unless games is a valid Catan report.

        One game; one winner; a VP score for every seated player and no
        others; the winner has at least `victory_points` (so the most VP);
        every other player is below `victory_points`.
        """
        name = pod.name  # type: ignore[attr-defined]
        if len(games) != 1:
            raise ValueError(
                f"{name}: Catan reports exactly one game per match, got {len(games)}."
            )
        game = games[0]
        seated = {p.uid for p in pod.players}
        if len(game.winners) != 1:
            raise ValueError(
                f"{name}: a Catan game has exactly one winner, got {len(game.winners)}."
            )
        (winner,) = game.winners
        if winner not in seated:
            raise ValueError(f"{name}: the winner must be seated in the pod.")
        if game.scores is None:
            raise ValueError(f"{name}: a Catan report needs victory points for every player.")
        vp = dict(game.scores)
        if len(vp) != len(game.scores) or set(vp) != seated:
            raise ValueError(
                f"{name}: victory points are needed for exactly the seated players."
            )
        if any(v < 0 for v in vp.values()):
            raise ValueError(f"{name}: victory points cannot be negative.")
        target = self._param(pod.tour_round, "victory_points")
        if vp[winner] < target:
            raise ValueError(
                f"{name}: the winner needs at least {target} victory points, "
                f"got {vp[winner]}."
            )
        for uid, v in vp.items():
            if uid != winner and v >= target:
                raise ValueError(
                    f"{name}: only the winner can have {target} or more victory points."
                )

    def match_winners(self, pod: IPod) -> frozenset[UUID]:
        return pod.games[-1].winners

    def report_from_winners(
        self, pod: IPod, winners: Iterable[UUID]
    ) -> list[IGameResult]:
        """Always refuses: the winner alone is not a Catan report.

        Victory points feed the tiebreakers, so report with
        Tournament.report_match and IGameResult.scores.
        """
        raise ValueError(
            f"{pod.name}: a Catan win needs victory points for every player; "  # type: ignore[attr-defined]
            "report the match with scores instead."
        )

    def random_report(self, pod: IPod) -> list[IGameResult]:
        """One game: the winner takes `victory_points`, the others get 2 up to
        one below it. Always a valid report."""
        config: CatanConfiguration = pod.tour_round.tour.config  # type: ignore[attr-defined]
        target = self._param(pod.tour_round, "victory_points")
        n = len(pod.players)
        winner = random.choices(
            pod.players, weights=list(config.seat_win_rates[:n])
        )[0]
        low = min(2, target - 1)
        scores = {
            p.uid: target if p is winner else random.randint(low, target - 1)
            for p in pod.players
        }
        return [IGameResult(frozenset({winner.uid}), tuple(scores.items()))]

    def swiss_pairing_logic(self, tour: ITournament, seq: int) -> str:
        # TODO(slice 5): PairingCatanPrelim, and PairingCatanSerpentine on the
        # round that eliminates players. PairingRandom supports any pod size.
        return "PairingRandom"

from __future__ import annotations
import random
from collections.abc import Iterable, Sequence
from uuid import UUID

import numpy as np

from ...core import TournamentConfiguration
from ...interface import IGameResult, IPod, IRuleset, ITournament


class CommanderConfiguration(TournamentConfiguration):
    """Commander's config fields, on top of the shared ones.

    Fields and defaults: the sidecar `CommanderConfiguration.params.yaml`.
    """

    global_wr_seats: Sequence[float]


class CommanderRuleset(IRuleset):
    """Commander's rules: one game per pod and the playoff plan.

    Save-format contracts: docs/tournament-log-spec.md, "Rulesets".
    No sidecar: this ruleset takes no parameters.
    """

    IS_COMPLETE = True

    DEFAULT_POD_SIZES = (4, 3)
    # Limit configured pairings to multiplayer sizes covered by ScoringDefault.
    ALLOWED_POD_SIZES = (3, 4, 5, 6)
    DEFAULT_SCORING_LOGIC = "ScoringDefault"
    # Same names, same order as today's StandingsExport.DEFAULT_FIELDS.
    DEFAULT_STANDINGS_FIELDS = (
        "STANDING",
        "NAME",
        "RATING",
        "RECORD",
        "OPP_POINTRATE",
        "OPP_BEATEN",
        "SEAT_HISTORY",
        "AVG_SEAT",
    )
    SEAT_BALANCING = True
    CONFIG_CLASS = CommanderConfiguration

    PLAYOFFS = {
        4: ((4, "PairingTop4"),),
        7: ((7, "PairingTop7"), (4, "PairingTop4")),
        10: ((10, "PairingTop10"), (4, "PairingTop4")),
        13: ((13, "PairingTop13"), (4, "PairingTop4")),
        16: ((16, "PairingTop16"), (4, "PairingTop4")),
        40: ((40, "PairingTop40"), (16, "PairingTop16"), (4, "PairingTop4")),
    }

    def validate_report(self, pod: IPod, games: list[IGameResult]) -> None:
        name = pod.name  # type: ignore[attr-defined]
        if len(games) != 1:
            raise ValueError(
                f"{name}: Commander reports exactly one game per match, "
                f"got {len(games)}."
            )
        seated = {p.uid for p in pod.players}
        if not games[0].winners <= seated:
            raise ValueError(f"{name}: game winners must be seated in the pod.")

    def match_winners(self, pod: IPod) -> frozenset[UUID]:
        return pod.games[-1].winners

    def report_from_winners(
        self, pod: IPod, winners: Iterable[UUID]
    ) -> list[IGameResult]:
        return [IGameResult(frozenset(winners))]

    def random_report(self, pod: IPod) -> list[IGameResult]:
        # Unchanged from the pre-plugin Tournament.random_results body: same
        # random.random() call, same numpy arithmetic on global_wr_seats.
        tour: ITournament = pod.tour_round.tour  # type: ignore[attr-defined]
        config: CommanderConfiguration = tour.config  # type: ignore[attr-defined]
        draw_rate = 1 - sum(config.global_wr_seats)
        result = random.random()
        rates = np.array(
            list(config.global_wr_seats[0 : len(pod.players)]) + [draw_rate]
        )
        rates = np.cumsum(rates / sum(rates))
        draw = result > rates[-2]
        if not draw:
            win = int(np.argmax([result < x for x in rates]))
            return [IGameResult(frozenset({pod.players[win].uid}))]
        return [IGameResult(frozenset(p.uid for p in pod.players))]

    def swiss_pairing_logic(self, tour: ITournament, seq: int) -> str:
        if seq == 0:
            return "PairingRandom"
        if seq == 1:
            return "PairingSnake"
        return "PairingDefault"

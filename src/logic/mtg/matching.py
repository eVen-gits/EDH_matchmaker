from __future__ import annotations
from collections.abc import Sequence
from typing_extensions import override

from ...core import Pod
from ...interface import IPlayer, IPod, IRound
from ..commander import matching as _commander_matching


class Pairing1v1(_commander_matching.PairingDefault):
    """Swiss pairing for 1v1 (2-player pod) Magic tournaments.

    Commander's PairingDefault sort key (fewest games played, fewest repeat
    opponents, rating, opponent pointrate) and greedy pod-fit scoring are
    generic Swiss-pairing logic, not multiplayer-specific, so 1v1 reuses it
    as-is - only the supported pod size differs. Ships no sidecar of its
    own, so it inherits commander/PairingDefault.params.yaml's parameters
    (including games_to_win, see commander/CommonPairing.params.yaml).
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)


# 1v1's power-of-2 single-elimination bracket, keyed by config.top_cut, each
# entry a (stage_value, n_players, pairing_logic_name) triple in play order -
# same shape as src/logic/commander/matching.py's CUT_STAGES, read by
# Tournament.__compute_stage_and_logic and Round.disable_topcut (src/core.py).
#
# Stage values are offset by 100 from the raw player count so they never
# collide with Commander's already-shipped {0, 4, 7, 10, 13, 16, 40} on the
# frozen top_cut/stage wire format (docs/tournament-log-spec.md) - a reader
# already has to branch on config.game before interpreting stage/top_cut at
# all, so this is additive, not a renumbering.
CUT_STAGES: dict[int, list[tuple[int, int, str]]] = {
    2: [(102, 2, "PairingBracket2")],
    4: [(104, 4, "PairingBracket4"), (102, 2, "PairingBracket2")],
    8: [
        (108, 8, "PairingBracket8"),
        (104, 4, "PairingBracket4"),
        (102, 2, "PairingBracket2"),
    ],
    16: [
        (116, 16, "PairingBracket16"),
        (108, 8, "PairingBracket8"),
        (104, 4, "PairingBracket4"),
        (102, 2, "PairingBracket2"),
    ],
    32: [
        (132, 32, "PairingBracket32"),
        (116, 16, "PairingBracket16"),
        (108, 8, "PairingBracket8"),
        (104, 4, "PairingBracket4"),
        (102, 2, "PairingBracket2"),
    ],
}


def _bracket_seed_order(n: int) -> list[int]:
    """Standard single-elimination bracket seed order (1-indexed) for n seeds.

    order(1) = [1]; order(2k) = for x in order(k): emit [x, 2k+1-x]. Verified
    against MTR's own worked Top-8 example: order(8) = [1, 8, 4, 5, 2, 7, 3,
    6], pairing adjacent entries gives 1v8, 4v5, 2v7, 3v6.
    """
    order = [1]
    while len(order) < n:
        k = len(order)
        order = [v for x in order for v in (x, 2 * k + 1 - x)]
    return order


class PairingBracketCommon(_commander_matching.CommonPairing):
    """Shared logic for every PairingBracketN class - a fixed power-of-2
    single-elimination bracket, MTR 10.4.

    Seeding is fixed once, at the start of the cut, from
    tour.get_standings(final_swiss_round) - results inside the cut never
    reseed it (MTR 10.4). Every round re-derives the current matchups by
    filtering that same fixed seed order down to survivors (players still in
    `players`, since Round.disable_topcut has already run and disabled
    anyone eliminated by the time make_pairings is called - see
    Round.create_pairings, src/core.py) and pairing adjacent survivors. This
    reproduces correct bracket matchups (seed 1 and 2 can only meet in the
    final) with no persisted bracket-tree data structure.

    No mid-bracket byes: advance_topcut is a no-op. A pod that ended in a
    draw would have no defined "who advances" answer - make_pairings validates
    for 2-player drawn pods and raises before creating pairings.
    """

    SELECTABLE = False  # top-cut pairing, chosen automatically by stage

    @override
    def advance_topcut(self, tour_round: IRound, standings: list[IPlayer]) -> None:
        """No-op: a bracket round awards no mid-round byes."""

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        tour = tour_round.tour
        top_cut = tour.config.top_cut
        final_swiss = tour.final_swiss_round  # type: ignore[attr-defined]
        assert final_swiss is not None, "Bracket pairing requires a completed Swiss stage."

        # Validate no 2-player pod ended in a draw (MTR §2.3 requires decisive result).
        # Only pods entirely within this round's survivors are relevant - the
        # previous round may be the full final Swiss round, which also covers
        # players who did not make the cut.
        previous_round = tour_round.tour.rounds[tour_round.seq - 1] if tour_round.seq > 0 else None
        if previous_round is not None:
            for pod in previous_round.pods:
                if (
                    isinstance(pod, Pod)
                    and pod.done
                    and pod.result_type == Pod.EResult.DRAW
                    and len(pod.players) == 2
                    and all(p in players for p in pod.players)
                ):
                    raise ValueError(
                        f"Pod {pod.table}'s match ended in a draw; MTR "
                        "requires a decisive result in single "
                        "elimination - report an additional game."
                    )

        seeds = tour.get_standings(final_swiss)[:top_cut]
        fixed_order = [seeds[s - 1] for s in _bracket_seed_order(len(seeds))]

        survivors = [p for p in fixed_order if p in players]
        assert len(survivors) == 2 * len(pods), (
            f"Expected {2 * len(pods)} survivors for {len(pods)} pods, got "
            f"{len(survivors)}."
        )
        for i, pod in enumerate(pods):
            pod.add_player(survivors[2 * i])
            pod.add_player(survivors[2 * i + 1])

        return players


class PairingBracket2(PairingBracketCommon):
    IS_COMPLETE = True


class PairingBracket4(PairingBracketCommon):
    IS_COMPLETE = True


class PairingBracket8(PairingBracketCommon):
    IS_COMPLETE = True


class PairingBracket16(PairingBracketCommon):
    IS_COMPLETE = True


class PairingBracket32(PairingBracketCommon):
    IS_COMPLETE = True

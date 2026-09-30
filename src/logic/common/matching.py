from __future__ import annotations
import random
from collections.abc import Sequence

from ...interface import IPlayer, IPod, IRound
from .pairing import CommonPairing

from typing_extensions import override


class PairingRandom(CommonPairing):
    """Random Swiss pairing - not tied to any one game, since shuffling
    players into pods ignores pod size entirely."""

    IS_COMPLETE: bool = True
    # Random shuffles players into pods of any cap, so it supports any size.
    SUPPORTED_POD_SIZES = None

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        byes = self.assign_byes(tour_round, players, pods)
        active_players = list(players - byes)
        random.shuffle(active_players)
        # PairingRandom ignores ratings for placement; assign_byes computes
        # the field ratings once internally when not passed one.

        player_index = 0
        for pod in pods:
            for _ in range(pod.cap - len(pod._players)):
                pod.add_player(active_players[player_index])
                player_index += 1

        return players

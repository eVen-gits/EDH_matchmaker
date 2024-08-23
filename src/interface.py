from __future__ import annotations
from enum import IntEnum
from typing import Sequence


class IPlayer:
    class EResult(IntEnum):
        LOSS = 0
        DRAW = 1
        WIN = 2
        BYE = 3
        PENDING = 4

    def __init__(self):
        self.ID: int = -1
        self.name: str = str()
        self.points: int|float = -1
        self.pods: list[IPod|IPlayer.EResult] = list()
        self.rounds: list[IRound] = list()

class ITournament:
    pass

class IPod:
    class EResult(IntEnum):
        LOSS = 0
        DRAW = 1
        WIN = 2

    def __init__(self):
        self.id: int = -1
        self.players: list[IPlayer] = list()
        self.cap: int = 0
        self.done: bool = False

    def add_player(self, player: IPlayer) -> bool:
        raise NotImplementedError()

    def sort(self):
        raise NotImplementedError()

class IRound:
    def __init__(self):
        self.seq:int = -1
        self.pods: list[IPod] = list()
        self.concluded: bool = False

class IPairingLogic:
    def make_pairings(self, players: Sequence[IPlayer], pods:Sequence[IPod]) -> Sequence[IPlayer]:
        raise NotImplementedError('PairingLogic.make_pairings not implemented - use subclass')
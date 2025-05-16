from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Sequence, Callable
from datetime import datetime
from uuid import UUID

class IPlayer:
    class ELocation(IntEnum):
        UNSEATED = 0
        SEATED = 1
        DROPPED = 2

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
        self.played: list[IPlayer]
        self.tour: ITournament
        self.location: IPlayer.ELocation = IPlayer.ELocation.UNSEATED
        self.result: IPlayer.EResult = IPlayer.EResult.PENDING

        self.byes: int
        self.wins: int


class ITournament:
    def __init__(self, config: ITournamentConfiguration | None = None):
        self.players: list[IPlayer] = list()
        self.rounds: list[IRound] = list()
        self.round: IRound|None = None

    def get_pod_sizes(self, n:int) -> Sequence[int]|None:
        pass

    @property
    def TC(self) -> ITournamentConfiguration:
        raise NotImplementedError()


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
        self.winner: None|IPlayer = None

    @abstractmethod
    def sort(self):
        pass

    @abstractmethod
    def add_player(self, player: IPlayer):
        pass

    @abstractmethod
    def remove_player(self, player: IPlayer):
        pass

    def __len__(self):
        return len(self.players)


class IRound:
    def __init__(self):
        self.seq:int = -1
        self.tour: ITournament
        self.players: list[IPlayer] = list()
        self.logic: IPairingLogic
        self.pods: list[IPod] = list()
        self.concluded: bool|datetime = False


class IPairingLogic:
    def make_pairings(self, players: Sequence[IPlayer], pods:Sequence[IPod]) -> Sequence[IPlayer]:
        raise NotImplementedError('PairingLogic.make_pairings not implemented - use subclass')


class ITournamentConfiguration:
    def __init__(self, **kwargs):
        self.allow_bye: bool
        self.min_pod_size: int
        self.max_pod_size: int
        self.ranking: tuple[float]
        self.matching: tuple[float]
        self.player_id: Callable[[], UUID]
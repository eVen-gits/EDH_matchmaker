from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Sequence, Callable, Any
from datetime import datetime
from uuid import UUID, uuid4

class SortMethod(IntEnum):
    ID = 0
    NAME = 1
    RANK = 2

class SortOrder(IntEnum):
    ASCENDING = 0
    DESCENDING = 1

class IHashable:
    CACHE: dict[UUID, Any] = {}

    def __init__(self):
        self.uid: UUID = uuid4()

    @classmethod
    @abstractmethod
    def get(cls, ID: UUID) -> IHashable:
        raise NotImplementedError()

class IPlayer(IHashable):
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
        super().__init__()
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

    @property
    @abstractmethod
    def pod(self) -> IPod|None:
        raise NotImplementedError()

class ITournament(IHashable):
    def __init__(self, config: ITournamentConfiguration | None = None):
        super().__init__()
        self.players: list[IPlayer] = list()
        self.rounds: list[IRound] = list()
        self.round: IRound|None = None

    def get_pod_sizes(self, n:int) -> Sequence[int]|None:
        pass

    @property
    def config(self) -> ITournamentConfiguration:
        raise NotImplementedError()

class IPod(IHashable):

    class EResult(IntEnum):
        DRAW = 0
        WIN = 1

    def __init__(self):
        super().__init__()
        self.table: int = -1
        self._players: list[UUID] = list()
        self.cap: int = 0
        self.result: set[UUID] = set()

    @property
    @abstractmethod
    def players(self) -> list[IPlayer]:
        raise NotImplementedError('Pod.players not implemented - use subclass')

    @abstractmethod
    def assign_seats(self):
        pass

    @abstractmethod
    def add_player(self, player: IPlayer):
        pass

    @abstractmethod
    def remove_player(self, player: IPlayer):
        pass

    def __len__(self):
        return len(self.players)

class IRound(IHashable):
    def __init__(self):
        super().__init__()
        self.seq:int = -1
        self.logic: IPairingLogic
        self._tour: UUID = uuid4()
        self._pods: list[UUID] = list()
        self._players: list[UUID] = list()

    @property
    @abstractmethod
    def players(self) -> list[IPlayer]:
        raise NotImplementedError('Round.players not implemented - use subclass')

    @property
    @abstractmethod
    def tour(self) -> ITournament:
        raise NotImplementedError('Round.tour not implemented - use subclass')

    @property
    @abstractmethod
    def pods(self) -> list[IPod]:
        raise NotImplementedError('Round.pods not implemented - use subclass')

class IPairingLogic:
    IS_COMPLETE=False

    def __init__(self, name: str):
        self.name = name

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
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Sequence, Callable, Any
from datetime import datetime
from uuid import UUID, uuid4

from eventsourcing.domain import Aggregate, event

class SortMethod(IntEnum):
    ID = 0
    NAME = 1
    RANK = 2

class SortOrder(IntEnum):
    ASCENDING = 0
    DESCENDING = 1

class IPlayer(Aggregate):
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

    class Registered(Aggregate.Created):
        name: str
        tour: UUID
        
    @event(Registered)
    def __init__(self, name:str, tour: UUID):
        self.name: str = name
        self.tour: UUID = tour

        self.pods: list[UUID|IPlayer.EResult] = []
        self.rounds: list[UUID] = []

        self.location: IPlayer.ELocation = IPlayer.ELocation.UNSEATED
        self.result: IPlayer.EResult = IPlayer.EResult.PENDING

        self.record: list[IPlayer.EResult] = []

    @property
    @abstractmethod
    def pod(self) -> IPod|None:
        raise NotImplementedError()

class ITournamentConfiguration(Aggregate):
    class Registered(Aggregate.Created):
        properties: dict[str, Any]

    class PropertyUpdated(Aggregate.Event):
        name: str
        value: Any

    @event(Registered)
    def __init__(self, properties: dict[str, Any]):
        self.pod_sizes: Sequence[int] = [4, 3]

        self.allow_bye: bool = True
        self.max_byes: int = 2

        self.win_points: int = 5
        self.draw_points: int = 1
        self.bye_points: int = 2

        self.n_rounds: int = 5
        self.snake_pods: bool = True

        self.auto_export: bool = False
        #standings_export: IStandingsExport

        self.global_wr_seats: Sequence[float] = [0.2553, 0.2232, 0.1847, 0.1428]

        for k, v in properties.items():
            setattr(self, k, v)

    @event(PropertyUpdated)
    def update_property(self, name: str, value: Any):
        setattr(self, name, value)

class ITournament(Aggregate):

    class Registered(Aggregate.Created):
        config: UUID

    class PlayerAdded(Aggregate.Event):
        player: UUID|list[UUID]

    class ConfigurationUpdated(Aggregate.Event):
        config: UUID

    @event(Registered)
    def __init__(self, config: UUID):

        self.config: UUID = config #type: ignore
        self.players: list[UUID] = []
        #self.rounds: list[IRound] = list()
        #self.round: IRound|None = None

    @event(PlayerAdded)
    def add_player(self, player: IPlayer|list[IPlayer]):
        if isinstance(player, IPlayer):
            self.players.append(player.id)
        elif isinstance(player, list):
            self.players.extend([p.id for p in player])

    @event(ConfigurationUpdated)
    def update_configuration(self, config: ITournamentConfiguration):
        self.config = config.id

    def get_pod_sizes(self, n:int) -> Sequence[int]|None:
        pass

    @property
    def TC(self) -> ITournamentConfiguration:
        raise NotImplementedError()

class IPod(Aggregate):

    class EResultType(IntEnum):
        DRAW = 0
        WIN = 1

    class Registered(Aggregate.Created):
        tour: UUID
        round: UUID
        table: int
        cap: int

        result: list[UUID]
        result_type: IPod.EResultType

    class PlayerAdded(Aggregate.Event):
        player: UUID

    @event(Registered)
    def __init__(self, tour:UUID, round: UUID, table: int, cap: int):

        self._tour: UUID = tour
        self._round: UUID = round

        self.table: int = table
        self.cap: int = cap

        self._players: list[UUID] = []

        self.result: set[UUID] = set()

    @property
    @abstractmethod
    def players(self) -> list[IPlayer]:
        raise NotImplementedError('Pod.players not implemented - use subclass')


    @event(PlayerAdded)
    def add_player(self, player: IPlayer):
        self._players.append(player.id)


    @abstractmethod
    def assign_seats(self):
        pass


    @abstractmethod
    def remove_player(self, player: IPlayer):
        pass

    def __len__(self):
        return len(self.players)

class IRound(Aggregate):
    def __init__(self):
        super().__init__()
        self.seq:int = -1
        self.logic: IPairingLogic
        self.concluded: bool|datetime = False
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
    def make_pairings(self, players: Sequence[IPlayer], pods:Sequence[IPod]) -> Sequence[IPlayer]:
        raise NotImplementedError('PairingLogic.make_pairings not implemented - use subclass')




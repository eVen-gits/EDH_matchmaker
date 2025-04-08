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

        self.auto_export: bool = True
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

    class RoundAdded(Aggregate.Event):
        round: UUID

    class CurrentRoundSet(Aggregate.Event):
        round: UUID

    @event(Registered)
    def __init__(self, config: UUID):

        self.config: UUID = config #type: ignore
        self.players: list[UUID] = []
        self.round: IRound|None = None
        self.rounds: list[IRound] = list()

    @event(PlayerAdded)
    def add_player(self, player: IPlayer|list[IPlayer]):
        if isinstance(player, IPlayer):
            self.players.append(player.id)
        elif isinstance(player, list):
            self.players.extend([p.id for p in player])

    @event(ConfigurationUpdated)
    def update_configuration(self, config: ITournamentConfiguration):
        self.config = config.id

    @event(RoundAdded)
    def add_round(self, round: IRound):
        if round.id not in self.rounds:
            self.rounds.append(round.id)

    @event(CurrentRoundSet)
    def set_current_round(self, round: IRound):
        self.round = round.id

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
    class Registered(Aggregate.Created):
        tour: UUID
        seq: int
        logic: IPairingLogic

    class PlayerAdded(Aggregate.Event):
        player: UUID

    class PodAdded(Aggregate.Event):
        pod: UUID

    class RoundConcluded(Aggregate.Event):
        concluded: datetime

    @event(Registered)
    def __init__(self, tour: UUID, seq: int, logic: IPairingLogic):
        self._tour: UUID = tour
        self.seq: int = seq
        self.logic: IPairingLogic = logic
        self._players: list[UUID] = []
        self._pods: list[UUID] = []
        self.concluded: bool|datetime = False

    @event(PlayerAdded)
    def add_player(self, player: IPlayer):
        if player.id not in self._players:
            self._players.append(player.id)

    @event(PodAdded)
    def add_pod(self, pod: IPod):
        if pod.id not in self._pods:
            self._pods.append(pod.id)

    @event(RoundConcluded)
    def conclude(self, concluded: datetime):
        self.concluded = concluded

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




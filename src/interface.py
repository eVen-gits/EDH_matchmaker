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
        BYE = 2
        GAMELOSS = 3
        DROPPED = 4

    class EResult(IntEnum):
        LOSS = 0
        DRAW = 1
        WIN = 2
        BYE = 3
        PENDING = 4
        DROPPED = 5

    class Registered(Aggregate.Created):
        name: str
        tour: UUID

    class SetPod(Aggregate.Event):
        pod: UUID

    class SetResult(Aggregate.Event):
        result: IPlayer.EResult


    @event(Registered)
    def __init__(self, name:str, tour: UUID):
        self.name: str = name
        self.tour: UUID = tour

        self.pods: list[UUID|IPlayer.EResult] = []
        self.rounds: list[UUID] = []

        self.location: IPlayer.ELocation = IPlayer.ELocation.UNSEATED

        self.result: IPlayer.EResult = IPlayer.EResult.PENDING

    @event(SetPod)
    def set_pod(self, pod: UUID):
        if pod not in self.pods:
            self.pods.append(pod)
        raise NotImplementedError() #TODO: Implement

    @event(SetResult)
    def set_result(self, result: IPlayer.EResult):
        self.result = result

        if self.result == IPlayer.EResult.BYE:
            self.location = IPlayer.ELocation.BYE
        elif self.result == IPlayer.EResult.DROPPED:
            self.location = IPlayer.ELocation.DROPPED
        if self.location == IPlayer.ELocation.UNSEATED:
            if result == IPlayer.EResult.LOSS:
                self.location = IPlayer.ELocation.GAMELOSS



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
        self._round: UUID|None = None
        self.rounds: list[IRound] = list()

    @event(PlayerAdded)
    def add_player(self, player: UUID|list[UUID]):
        if isinstance(player, UUID):
            self.players.append(player)
        elif isinstance(player, Sequence):
            self.players.extend([p for p in player])

    @event(ConfigurationUpdated)
    def update_configuration(self, config: ITournamentConfiguration):
        self.config = config.id

    @event(RoundAdded)
    def add_round(self, round: IRound):
        if round.id not in self.rounds:
            self.rounds.append(round.id)

    @property
    def round(self) -> UUID|None:
        return self._round

    @event(CurrentRoundSet)
    @round.setter
    def round(self, round: UUID|None):
        self._round = round

    def get_pod_sizes(self, n:int) -> Sequence[int]|None:
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

    class PlayerAdded(Aggregate.Event):
        player: UUID

    class PlayerRemoved(Aggregate.Event):
        player: UUID

    @event(Registered)
    def __init__(self, tour:UUID, round: UUID, table: int, cap: int):

        self.tour: UUID = tour
        self.round: UUID = round
        self.players: list[UUID] = []

        self.table: int = table
        self.cap: int = cap

        self.result: set[UUID] = set()


    @event(PlayerAdded)
    def add_player(self, player: UUID):
        self.players.append(player)

    @abstractmethod
    def assign_seats(self):
        pass

    @event(PlayerRemoved)
    def remove_player(self, player: UUID):
        self.players.remove(player)

    def __len__(self):
        return len(self.players)

class IRound(Aggregate):
    class Registered(Aggregate.Created):
        tour: UUID
        seq: int
        logic: str

    class PlayerAdded(Aggregate.Event):
        player: UUID

    class PodAdded(Aggregate.Event):
        pod: UUID

    class RoundConcluded(Aggregate.Event):
        concluded: datetime

    @event(Registered)
    def __init__(self, tour: UUID, seq: int, logic: str):
        self._tour: UUID = tour
        self.seq: int = seq
        self.logic: str = logic
        self._players: list[UUID] = []
        self._pods: list[UUID] = []
        self.concluded: bool|datetime = False

    @event(PlayerAdded)
    def add_player(self, player: UUID):
        if player not in self._players:
            self._players.append(player)

    @event(PodAdded)
    def add_pod(self, pod: UUID):
        if pod not in self._pods:
            self._pods.append(pod)

    @event(RoundConcluded)
    def conclude(self, concluded: datetime):
        self.concluded = concluded


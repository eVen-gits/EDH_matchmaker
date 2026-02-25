from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Sequence, Callable, Any
from datetime import datetime
from uuid import UUID, uuid4


class SortMethod(IntEnum):
    """Enum for sorting methods."""

    ID = 0
    NAME = 1
    RANK = 2


class SortOrder(IntEnum):
    """Enum for sorting order."""

    ASCENDING = 0
    DESCENDING = 1


class IHashable:
    """Interface for hashable objects with UUIDs."""

    CACHE: dict[UUID, Any] = {}

    def __init__(self, uid: UUID | None = None):
        """Initializes the IHashable object.

        Args:
            uid: The UUID of the object. If None, a new UUID is generated.

        Raises:
            ValueError: If the UUID has a collision or is of invalid type.
        """
        if uid:
            if uid in self.CACHE:
                raise ValueError("UUID collision.")
            elif not isinstance(uid, UUID):
                raise ValueError("UUID type error.")
            else:
                self.uid = uid
        else:
            self.uid: UUID = uuid4()
        self.CACHE[self.uid] = self

    @classmethod
    @abstractmethod
    def get(cls, ID: UUID) -> IHashable:
        """Retrieves an object by its UUID.

        Args:
            ID: The UUID of the object.

        Returns:
            The object with the specified UUID.
        """
        raise NotImplementedError()

class IPlayer(IHashable, ABC):
    """Interface for a player."""

    class ELocation(IntEnum):
        """Enum for player location."""

        UNASSIGNED = 0
        SEATED = 1
        GAME_LOSS = 3
        BYE = 4
        DROPPED = 5

    class EResult(IntEnum):
        """Enum for match result."""

        LOSS = 0
        DRAW = 1
        WIN = 2
        BYE = 3
        PENDING = 4

    name: str
    tour: ITournament

    @abstractmethod
    def played(self, tour_round: IRound) -> list[IPlayer]: ...

    @abstractmethod
    def location(self, tour_round: IRound) -> IPlayer.ELocation: ...

    @abstractmethod
    def pod(self, tour_round: IRound) -> IPod | None: ...

    @abstractmethod
    def set_result(
        self, tour_round: IRound, result: IPlayer.EResult
    ) -> IPlayer.EResult: ...

    @abstractmethod
    def pods(self, tour_round: IRound) -> list[IPod]: ...

    @abstractmethod
    def rating(self, tour_round: IRound) -> float: ...

    @abstractmethod
    def opponent_pointrate(self, tour_round: IRound) -> float: ...

    @abstractmethod
    def games(self, tour_round: IRound) -> list[IRound]: ...

    @abstractmethod
    def byes(self, tour_round: IRound) -> int: ...

    @abstractmethod
    def wins(self, tour_round: IRound) -> int: ...

    @abstractmethod
    def losses(self, tour_round: IRound) -> int: ...

    @abstractmethod
    def draws(self, tour_round: IRound) -> int: ...


class ITournament(IHashable, ABC):
    @property
    @abstractmethod
    def rounds(self) -> Sequence[IRound]: ...

    @property
    @abstractmethod
    def config(self) -> ITournamentConfiguration: ...

    @abstractmethod
    def get_pod_sizes(self, n: int) -> Sequence[int] | None: ...

    @abstractmethod
    def get_standings(self, tour_round: IRound) -> list[IPlayer]: ...


class IPod(IHashable, ABC):
    """Interface for a pod."""

    class EResult(IntEnum):
        """Enum for pod result."""

        DRAW = 0
        WIN = 1
        PENDING = 2

    _tour: UUID
    _round: UUID
    _players: list[UUID]
    cap: int
    _result: set[UUID]

    @property
    @abstractmethod
    def result(self) -> set[IPlayer.EResult]: ...

    @property
    @abstractmethod
    def players(self) -> list[IPlayer]: ...

    @abstractmethod
    def auto_assign_seats(self): ...

    @abstractmethod
    def add_player(self, player: IPlayer): ...

    @abstractmethod
    def remove_player(self, player: IPlayer): ...

    @property
    @abstractmethod
    def tour_round(self) -> IRound: ...

    def __len__(self):
        return len(self.players)


class IRound(IHashable, ABC):
    """Interface for a round."""
    seq: int
    logic: IPairingLogic
    _tour: UUID
    _pods: list[UUID]
    _players: list[UUID]

    @property
    @abstractmethod
    def active_players(self) -> set[IPlayer]: ...

    @property
    @abstractmethod
    def byes(self) -> set[IPlayer]: ...

    @property
    @abstractmethod
    def tour(self) -> ITournament: ...

    @property
    @abstractmethod
    def pods(self) -> list[IPod]: ...


class IPairingLogic(ABC):
    """Interface for pairing logic."""

    IS_COMPLETE: bool = False
    name: str

    @abstractmethod
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> Sequence[IPlayer]:
        """Creates pairings for a round.

        Args:
            tour_round: The current round.
            players: The set of players to pair.
            pods: The list of available pods.

        Returns:
            A sequence of players who could not be paired (if any).
        """
        ...

    @abstractmethod
    def advance_topcut(self, tour_round: IRound, standings: list[IPlayer]) -> None:
        """Advances players to the top cut.

        Args:
            tour_round: The current round.
            standings: The list of players sorted by standing.
        """
        ...


class ITournamentConfiguration(ABC):
    allow_bye: bool = False
    min_pod_size: int
    max_pod_size: int
    ranking: tuple[float, ...]
    matching: tuple[float, ...]
    player_id: Callable[[], UUID]

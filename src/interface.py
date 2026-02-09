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


class IPlayer(IHashable):
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

    def __init__(self, uid: UUID | None = None):
        """Initializes the IPlayer."""
        super().__init__(uid=uid)
        self.name: str = str()
        # self.rounds: list[IRound] = list()
        self.tour: ITournament
        # self.location: IPlayer.ELocation = IPlayer.ELocation.UNSEATED
        # self.result: IPlayer.EResult = IPlayer.EResult.PENDING

    @abstractmethod
    def played(self, tour_round: IRound) -> list[IPlayer]:
        raise NotImplementedError()

    @abstractmethod
    def location(self, tour_round: IRound) -> IPlayer.ELocation:
        raise NotImplementedError()

    @abstractmethod
    def pod(self, tour_round: IRound) -> IPod | None:
        raise NotImplementedError()

    @abstractmethod
    def set_result(
        self, tour_round: IRound, result: IPlayer.EResult
    ) -> IPlayer.EResult:
        raise NotImplementedError()

    @abstractmethod
    def pods(self, tour_round: IRound) -> list[IPod]:
        raise NotImplementedError()

    @abstractmethod
    def rating(self, tour_round: IRound) -> float:
        raise NotImplementedError()

    @abstractmethod
    def opponent_pointrate(self, tour_round: IRound) -> float:
        raise NotImplementedError()

    @abstractmethod
    def games(self, tour_round: IRound) -> list[IRound]:
        raise NotImplementedError()

    @abstractmethod
    def byes(self, tour_round: IRound) -> int:
        raise NotImplementedError()

    @abstractmethod
    def wins(self, tour_round: IRound) -> int:
        raise NotImplementedError()

    @abstractmethod
    def losses(self, tour_round: IRound) -> int:
        raise NotImplementedError()

    @abstractmethod
    def draws(self, tour_round: IRound) -> int:
        raise NotImplementedError()


class ITournament(IHashable):
    def __init__(
        self, config: ITournamentConfiguration | None = None, uid: UUID | None = None
    ):
        super().__init__(uid=uid)
        self.rounds: list[IRound] = list()
        self._round: IRound | None = None

    @abstractmethod
    def get_pod_sizes(self, n: int) -> Sequence[int] | None:
        raise NotImplementedError()

    @abstractmethod
    def get_standings(self, tour_round: IRound) -> list[IPlayer]:
        raise NotImplementedError()

    @property
    def config(self) -> ITournamentConfiguration:
        raise NotImplementedError()


class IPod(IHashable):
    """Interface for a pod."""

    class EResult(IntEnum):
        """Enum for pod result."""

        DRAW = 0
        WIN = 1
        PENDING = 2

    def __init__(self, uid: UUID | None = None):
        """Initializes the IPod."""
        super().__init__(uid=uid)
        self._tour: UUID
        self._round: UUID
        self._players: list[UUID] = list()
        self.cap: int = 0
        self._result: set[UUID] = set()

    @property
    @abstractmethod
    def result(self) -> set[IPlayer.EResult]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def players(self) -> list[IPlayer]:
        raise NotImplementedError("Pod.players not implemented - use subclass")

    @abstractmethod
    def auto_assign_seats(self):
        pass

    @abstractmethod
    def add_player(self, player: IPlayer):
        pass

    @abstractmethod
    def remove_player(self, player: IPlayer):
        pass

    @property
    @abstractmethod
    def tour_round(self) -> IRound:
        raise NotImplementedError()

    def __len__(self):
        return len(self.players)


class IRound(IHashable):
    """Interface for a round."""

    def __init__(self, uid: UUID | None = None):
        """Initializes the IRound."""
        super().__init__(uid=uid)
        self.seq: int = -1
        self.logic: IPairingLogic
        self._tour: UUID
        self._pods: list[UUID] = list()
        self._players: list[UUID] = list()

    @property
    @abstractmethod
    def active_players(self) -> set[IPlayer]:
        raise NotImplementedError("Round.players not implemented - use subclass")

    @property
    @abstractmethod
    def byes(self) -> set[IPlayer]:
        raise NotImplementedError("Round.byes not implemented - use subclass")

    @property
    @abstractmethod
    def tour(self) -> ITournament:
        raise NotImplementedError("Round.tour not implemented - use subclass")

    @property
    @abstractmethod
    def pods(self) -> list[IPod]:
        raise NotImplementedError("Round.pods not implemented - use subclass")


class IPairingLogic:
    """Interface for pairing logic."""

    IS_COMPLETE = False

    def __init__(self, name: str):
        """Initializes the IPairingLogic."""
        self.name = name

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
        raise NotImplementedError(
            "PairingLogic.make_pairings not implemented - use subclass"
        )

    def advance_topcut(self, tour_round: IRound, standings: list[IPlayer]) -> None:
        """Advances players to the top cut.

        Args:
            tour_round: The current round.
            standings: The list of players sorted by standing.
        """
        raise NotImplementedError(
            "PairingLogic.advance_topcut not implemented - use subclass"
        )


class ITournamentConfiguration:
    def __init__(self, **kwargs):
        self.allow_bye: bool
        self.min_pod_size: int
        self.max_pod_size: int
        self.ranking: tuple[float]
        self.matching: tuple[float]
        self.player_id: Callable[[], UUID]

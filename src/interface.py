from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

from .param_spec import ParamSpec, load_param_spec


@dataclass(frozen=True)
class IGameResult:
    """One game of a match: the players who did not lose it.

    One UID means that player won the game. Two or more mean the game was
    drawn among them. Every other player seated in the pod lost the game.
    """

    winners: frozenset[UUID]

    def __post_init__(self) -> None:
        winners = frozenset(self.winners)
        if not winners:
            raise ValueError("IGameResult.winners must not be empty.")
        object.__setattr__(self, "winners", winners)


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

    CACHE: dict[UUID, IHashable] = {}

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
            else:
                self.uid = uid
        else:
            self.uid: UUID = uuid4()
        self.CACHE[self.uid] = self

    @classmethod
    def get(cls, ID: UUID) -> IHashable:
        """Retrieves an object by its UUID.

        Args:
            ID: The UUID of the object.

        Returns:
            The object with the specified UUID.
        """
        return cls.CACHE[ID]


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
    def result(self, tour_round: IRound) -> IPlayer.EResult: ...

    @abstractmethod
    def pods(
        self, tour_round: IRound | None = None
    ) -> list[IPod | IPlayer.ELocation]: ...

    @abstractmethod
    def rating(self, tour_round: IRound | None = None) -> float: ...

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

    @property
    @abstractmethod
    def players(self) -> set[IPlayer]: ...

    @property
    @abstractmethod
    def final_swiss_round(self) -> IRound | None: ...

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
    _games: list[IGameResult]

    @property
    @abstractmethod
    def games(self) -> tuple[IGameResult, ...]: ...

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
    stage: Any
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

    @abstractmethod
    def remove_pod(self, pod: IPod) -> bool: ...


class IPairingLogic(ABC):
    """Interface for pairing logic."""

    IS_COMPLETE: bool = False
    # User-pickable for a Swiss round. False for top-cut pairings, which are
    # chosen automatically by stage and must not appear in the round selector.
    SELECTABLE: bool = True
    # Pod sizes this algorithm supports (its capability). None means any size.
    # The tournament's pod sizes must all be supported for this algorithm to be
    # offered - see supports_pod_sizes and Tournament.selectable_pairing_logics.
    SUPPORTED_POD_SIZES: tuple[int, ...] | None = None
    name: str

    @classmethod
    def supports_pod_sizes(cls, pod_sizes: Sequence[int]) -> bool:
        """Whether this algorithm can pair a tournament with these pod sizes.

        True if it supports any size (SUPPORTED_POD_SIZES is None) or every
        given size is in its supported set.
        """
        if cls.SUPPORTED_POD_SIZES is None:
            return True
        return set(pod_sizes).issubset(cls.SUPPORTED_POD_SIZES)
    # Loaded at class definition from the sidecar `<ClassName>.params.yaml`.
    # Per-round overrides live in config.pairing_rounds[seq]["params"].
    PARAM_SPEC: dict[str, ParamSpec] = {}
    DEFAULT_PARAMS: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.PARAM_SPEC = load_param_spec(cls)
        cls.DEFAULT_PARAMS = {n: s.default for n, s in cls.PARAM_SPEC.items()}

    @abstractmethod
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        """Creates pairings for a round.

        Args:
            tour_round: The current round.
            players: The set of players to pair.
            pods: The list of available pods.

        Returns:
            A set of players who could not be paired (if any).
        """
        ...

    def advance_topcut(self, tour_round: IRound, standings: list[IPlayer]) -> None:
        """Called once per playoff round before make_pairings.

        The base implementation is a no-op, correct for ordinary Swiss
        pairing. Top-cut pairing logics override this to give seeded byes.

        Args:
            tour_round: The current round.
            standings: The list of players sorted by standing.
        """
        return None


class IScoringLogic(ABC):
    """Interface for scoring logic (how player points are computed)."""

    IS_COMPLETE: bool = False
    name: str
    # Pod sizes this algorithm supports (its capability). None means any size.
    # The tournament's pod sizes must all be supported for this algorithm to be
    # offered - see supports_pod_sizes and Tournament.selectable_scoring_logics.
    SUPPORTED_POD_SIZES: tuple[int, ...] | None = None
    # This algorithm's parameter spec, loaded at class definition from the
    # sidecar `<ClassName>.params.yaml` (name, default, type, range,
    # description). DEFAULT_PARAMS is derived from it - the names and defaults
    # used by TournamentConfiguration.scoring_params.
    PARAM_SPEC: dict[str, ParamSpec] = {}
    DEFAULT_PARAMS: dict[str, Any] = {}

    @classmethod
    def supports_pod_sizes(cls, pod_sizes: Sequence[int]) -> bool:
        """Whether this algorithm can score a tournament with these pod sizes.

        True if it supports any size (SUPPORTED_POD_SIZES is None) or every
        given size is in its supported set.
        """
        if cls.SUPPORTED_POD_SIZES is None:
            return True
        return set(pod_sizes).issubset(cls.SUPPORTED_POD_SIZES)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.PARAM_SPEC = load_param_spec(cls)
        cls.DEFAULT_PARAMS = {n: s.default for n, s in cls.PARAM_SPEC.items()}

    @abstractmethod
    def compute_ratings(
        self, tour: ITournament, tour_round: IRound
    ) -> Mapping[UUID, float]:
        """Computes every player's point total as of a given round.

        Args:
            tour: The tournament.
            tour_round: The round up to which to compute points.

        Returns:
            A mapping of player UID to point total.
        """
        ...

    @abstractmethod
    def rating(self, player: IPlayer, tour_round: IRound) -> float:
        """Computes one player's point total as of a given round.

        Args:
            player: The player to compute the rating for.
            tour_round: The round up to which to compute points.

        Returns:
            The player's point total.
        """
        ...

    @abstractmethod
    def pointrate_denominator(self, tour_round: IRound) -> float:
        """The value a player's rating is divided by to get a 0-1 pointrate.

        Args:
            tour_round: The round the pointrate is being computed for.

        Returns:
            The denominator to divide a rating by.
        """
        ...


class IRuleset(ABC):
    """Interface for a game's rules: match reports, standings, playoffs.

    Same plugin shape as IScoringLogic/IPairingLogic (IS_COMPLETE, name,
    PARAM_SPEC/DEFAULT_PARAMS loaded from the sidecar in
    __init_subclass__). See docs/tournament-log-spec.md, "Rulesets", for
    the full contract.
    """

    IS_COMPLETE: bool = False
    name: str

    # Pod sizes of a new config, preferred first. Required.
    DEFAULT_POD_SIZES: tuple[int, ...]
    # config.pod_sizes must be a subset. None = any.
    ALLOWED_POD_SIZES: tuple[int, ...] | None = None
    # Scoring logic of a new config. Required.
    DEFAULT_SCORING_LOGIC: str
    # StandingsExport.Field names for a new config's export.
    DEFAULT_STANDINGS_FIELDS: tuple[str, ...] = (
        "STANDING",
        "NAME",
        "RATING",
        "RECORD",
    )
    # Whether core calls Pod.auto_assign_seats after Swiss pairing.
    SEAT_BALANCING: bool = True
    # Whether this ruleset requires an odd player count to get a bye rather
    # than leave someone unseated (config.allow_bye and max_byes >= 1).
    BYES_REQUIRED: bool = False
    # top_cut -> playoff rounds in play order, each (stage value, top-cut
    # pairing logic name). Its keys are the only non-zero top_cut values
    # the ruleset accepts.
    PLAYOFFS: Mapping[int, tuple[tuple[int, str], ...]] = {}

    # Loaded at class definition from the sidecar `<ClassName>.params.yaml`.
    # Per-round/playoff-stage overrides live in
    # config.pairing_rounds[seq]["ruleset_params"] /
    # config.playoff_rounds[stage]["ruleset_params"].
    PARAM_SPEC: dict[str, ParamSpec] = {}
    DEFAULT_PARAMS: dict[str, Any] = {}

    def __init__(self, name: str):
        self.name = name

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.PARAM_SPEC = load_param_spec(cls)
        cls.DEFAULT_PARAMS = {n: s.default for n, s in cls.PARAM_SPEC.items()}

    def params(self, tour_round: IRound) -> dict[str, Any]:
        """This round's ruleset params: overrides on top of the class
        defaults. Cold path only (call once, not per player/pod)."""
        config = tour_round.tour.config  # type: ignore[attr-defined]
        return {**self.DEFAULT_PARAMS, **config.ruleset_overrides(tour_round)}

    def _param(self, tour_round: IRound, key: str) -> Any:
        """Single-param lookup with no allocation - for a hot path."""
        config = tour_round.tour.config  # type: ignore[attr-defined]
        overrides = config.ruleset_overrides(tour_round)
        return overrides.get(key, self.DEFAULT_PARAMS[key])

    @abstractmethod
    def validate_report(self, pod: IPod, games: Sequence[IGameResult]) -> None:
        """Raises ValueError if games is not a valid complete report for pod
        in its round. Never mutates. The message should be fit for a
        tournament organiser to read."""
        ...

    @abstractmethod
    def match_winners(self, pod: IPod) -> frozenset[UUID]:
        """Players who did not lose the match: one = winner, several = drew.

        Called only when pod.games is non-empty. Must be total: it runs on
        stored data (old files, rosters edited after reporting) and must
        never raise.
        """
        ...

    @abstractmethod
    def report_from_winners(
        self, pod: IPod, winners: Iterable[UUID]
    ) -> list[IGameResult]:
        """Turns "A won" / "A and B drew" into a canonical report.

        Backs the report_win/report_draw shorthand.
        """
        ...

    @abstractmethod
    def random_report(self, pod: IPod) -> list[IGameResult]:
        """A plausible valid report for Tournament.random_results.

        Uses the `random` module so tests can seed it.
        """
        ...

    @abstractmethod
    def swiss_pairing_logic(self, tour: ITournament, seq: int) -> str:
        """Default pairing logic for Swiss round seq when
        config.pairing_rounds sets none."""
        ...

    @abstractmethod
    def standings_keys(
        self,
        tour: ITournament,
        tour_round: IRound,
        ratings: Mapping[Any, float],
    ) -> Mapping[UUID, tuple]:
        """One sort key per player, compared descending. Swiss rounds only.

        `ratings` is the scoring logic's field map; the first element of
        each key should be the rating.
        """
        ...

    def standings_columns(
        self, tour: ITournament, tour_round: IRound
    ) -> list[tuple[str, Mapping[UUID, str]]]:
        """Extra standings-export columns: (header, formatted cell per
        player). Default: none."""
        return []


class IStandingsExport(ABC):
    """Interface for standings export configuration."""

    dir: str

    @abstractmethod
    def serialize(self) -> Mapping[str, object]: ...


class ITournamentConfiguration(ABC):
    pod_sizes: Sequence[int] = (4, 3)
    allow_bye: bool = True
    n_rounds: int = 4
    max_byes: int = 2
    auto_export: bool = True
    standings_export: IStandingsExport
    global_wr_seats: Sequence[float] = (
        # 0.2553,
        # 0.2232,
        # 0.1847,
        # 0.1428,
        # New data: all 50+ player events since [2024-09-30;2025-05-05]
        0.2470,
        0.1928,
        0.1672,
        0.1458,
    )
    top_cut: int = 0
    scoring_logic: str = "ScoringDefault"
    # Owned by whichever class scoring_logic names - see
    # IScoringLogic.DEFAULT_PARAMS.
    scoring_params: dict[str, Any] = {}
    # Pairing config per Swiss round: one {"logic", "params"} dict per round.
    # A list (not flat like scoring_params) because pairing logic and its
    # settings can differ per round. logic None (or a short list) falls back to
    # the adaptive default - see Tournament.__compute_stage_and_logic.
    pairing_rounds: list[dict[str, Any]] = []
    # Read-only views derived from pairing_rounds (see the concrete config).
    pairing_logics: list[str | None] = []
    pairing_params: list[dict[str, Any]] = []
    # Class name of the IRuleset that owns this tournament's game rules -
    # see IRuleset. Also provides the defaults of pod_sizes, scoring_logic
    # and standings_export.fields when the caller does not pass them.
    ruleset: str = "CommanderRuleset"
    # Ruleset param overrides per playoff stage: {stage_value:
    # {"ruleset_params": {...}}}. Stages not in the current playoff plan
    # are ignored. See ruleset_overrides.
    playoff_rounds: dict[int, dict[str, Any]] = {}

    @property
    @abstractmethod
    def max_pod_size(self) -> int: ...

    @property
    @abstractmethod
    def min_pod_size(self) -> int: ...

    @abstractmethod
    def ruleset_overrides(self, tour_round: IRound) -> Mapping[str, Any]:
        """This round's ruleset param overrides (see IRuleset.params()).

        A Swiss round reads pairing_rounds[seq]["ruleset_params"]; a
        playoff round reads playoff_rounds[stage.value]["ruleset_params"].
        """
        ...

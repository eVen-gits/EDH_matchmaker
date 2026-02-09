from __future__ import annotations
from typing import List, Sequence, Union, Callable, Any, cast
from typing_extensions import override
from collections.abc import Iterable

import argparse
import math
import os
import random
from datetime import datetime
from enum import Enum

from .discord_engine import DiscordPoster
from .interface import (
    IPlayer,
    ITournament,
    IPod,
    IRound,
    IPairingLogic,
    ITournamentConfiguration,
)
from .misc import Json2Obj, generate_player_names, timeit
import numpy as np
from tqdm import tqdm  # pyright: ignore
from uuid import UUID, uuid4
import json

from dotenv import load_dotenv
import requests
import threading

import importlib
import pkgutil
from pathlib import Path
import functools

# Load configuration from .env file
load_dotenv()

# import sys
# sys.setrecursionlimit(5000)  # Increase recursion limit


class DataExport:
    """Namespace for data export constants and enums."""

    class Format(Enum):
        """Enum for export formats."""

        PLAIN = 0
        DISCORD = 1
        CSV = 2
        JSON = 3

    class Target(Enum):
        """Enum for export targets."""

        CONSOLE = 0
        FILE = 1
        WEB = 2
        DISCORD = 3


class PodsExport(DataExport):
    """Handles the export of tournament pods."""

    @classmethod
    def auto_export(cls, func):
        """Decorator to automatically export pods after a function call.

        Args:
            func: The function to decorate.

        Returns:
            The decorated function.
        """

        def auto_pods_export_wrapper(
            self: Tournament, *original_args, **original_kwargs
        ):
            try:
                tour_round = self.tour_round
            except (KeyError, ValueError):
                tour_round = None
            ret = func(self, *original_args, **original_kwargs)
            try:
                tour_round = tour_round or self.tour_round
            except (KeyError, ValueError):
                tour_round = None
            if self.config.auto_export:
                logf = TournamentAction.LOGF
                if logf and tour_round:
                    # Export pods to a file named {tournament_name}_round_{round_number}.txt
                    # And also export it into {log_directory}/pods.txt
                    context = TournamentContext(
                        self, tour_round, self.get_standings(tour_round)
                    )
                    export_str: str = "\n\n".join(
                        [pod.__repr__(context=context) for pod in tour_round.pods]
                    )
                    game_lost: list[Player] = [
                        x
                        for x in tour_round.active_players
                        if x.result == Player.EResult.LOSS
                    ]
                    byes = [
                        x
                        for x in tour_round.unassigned
                        if x.location == Player.ELocation.UNASSIGNED
                        and x.result == Player.EResult.BYE
                    ]
                    if len(game_lost) + len(byes) > 0:
                        max_len = max([len(p.name) for p in game_lost + byes])
                        if self.config.allow_bye and byes:
                            export_str += "\n\nByes:\n" + "\n".join(
                                [
                                    "\t{} | pts: {}".format(
                                        p.name.ljust(max_len),
                                        p.rating(tour_round) or "0",
                                    )
                                    for p in tour_round.unassigned
                                    if p.result == Player.EResult.BYE
                                ]
                            )
                        if game_lost:
                            export_str += "\n\nGame losses:\n" + "\n".join(
                                [
                                    "\t{} | pts: {}".format(
                                        p.name.ljust(max_len), p.rating(tour_round)
                                    )
                                    for p in game_lost
                                ]
                            )

                    path = os.path.join(
                        os.path.dirname(logf),
                        os.path.basename(logf).replace(".json", ""),
                        os.path.basename(logf).replace(
                            ".json", "_R{}.txt".format(tour_round.seq)
                        ),
                    )
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))

                    self.export_str(export_str, path, DataExport.Target.FILE)
                    if os.getenv("EXPORT_ONLINE_API_URL") and os.getenv(
                        "EXPORT_ONLINE_API_KEY"
                    ):
                        self.export_str(export_str, None, DataExport.Target.WEB)

                    path = os.path.join(os.path.dirname(logf), "pods.txt")
                    self.export_str(export_str, path, DataExport.Target.FILE)

            return ret

        return auto_pods_export_wrapper


class TournamentContext:
    """Context object holding tournament state for export operations."""

    def __init__(self, tour: Tournament, tour_round: Round, standings: list[Player]):
        """Initializes the TournamentContext.

        Args:
            tour: The tournament instance.
            tour_round: The specific round of the tournament.
            standings: The list of players in the current standings.
        """
        self.tour = tour
        self.tour_round = tour_round
        self.standings = standings


class StandingsExport(DataExport):
    class Field(Enum):
        STANDING = 0  # Standing
        ID = 1  # Player ID
        NAME = 2  # Player name
        RECORD = 3  # Record
        RATING = 4  # Number of points
        WINS = 5  # Number of wins
        OPP_BEATEN = 6  # Number of opponents beaten
        OPP_POINTRATE = 7  # Opponents' win percentage
        UNIQUE = 8  # Number of unique opponents
        POINTRATE = 9  # Winrate
        GAMES = 10  # Number of games played
        SEAT_HISTORY = 11  # Seat record
        AVG_SEAT = 12  # Average seat

    class Formatting:
        def __init__(
            self,
            label: str,
            format: str,
            denom: int | None,
            description: str,
            getter: Callable[..., Any],
        ):  # Dict of arg names to expected types
            self.name = label
            self.format = format
            self.denom = denom
            self.description = description
            self.getter = getter

        def get(self, player: Player, context: TournamentContext) -> Any:
            # Call the static method through the class
            return self.getter.__func__(player, context)

    @staticmethod
    def _get_standing(player: Player, context: TournamentContext) -> int:
        return player.standing(context.tour_round, context.standings)

    @staticmethod
    def _get_id(player: Player, context: TournamentContext) -> str:
        return player.uid.hex

    @staticmethod
    def _get_name(player: Player, context: TournamentContext) -> str:
        return player.name

    @staticmethod
    def _get_opp_winrate(player: Player, context: TournamentContext) -> float:
        return player.opponent_pointrate(context.tour_round)

    @staticmethod
    def _get_rating(player: Player, context: TournamentContext) -> float:
        return player.rating(context.tour_round)

    @staticmethod
    def _get_wins(player: Player, context: TournamentContext) -> int:
        return player.wins(context.tour_round)

    @staticmethod
    def _get_winrate(player: Player, context: TournamentContext) -> float:
        return player.pointrate(context.tour_round)

    @staticmethod
    def _get_unique_opponents(player: Player, context: TournamentContext) -> int:
        return len(player.games(context.tour_round))

    @staticmethod
    def _get_games(player: Player, context: TournamentContext) -> int:
        return len(player.games(context.tour_round))

    @staticmethod
    def _get_opponents_beaten(player: Player, context: TournamentContext) -> int:
        return len(player.players_beaten(context.tour_round))

    @staticmethod
    def _get_seat_history(player: Player, context: TournamentContext) -> str:
        return player.seat_history(context.tour_round)

    @staticmethod
    def _get_avg_seat(player: Player, context: TournamentContext) -> float:
        return player.average_seat(context.tour.rounds)

    @staticmethod
    def _get_record(player: Player, context: TournamentContext) -> str:
        return Player.fmt_record(player.record(context.tour_round))

    info = {
        Field.STANDING: Formatting(
            label="#",
            format="{:d}",
            denom=None,
            description="Player's standing in the tournament.",
            getter=_get_standing,
        ),
        Field.ID: Formatting(
            label="ID",
            format="{:s}",
            denom=None,
            description="Player ID",
            getter=_get_id,
        ),
        Field.NAME: Formatting(
            label="name",
            format="{:s}",
            denom=None,
            description="Player name",
            getter=_get_name,
        ),
        Field.OPP_POINTRATE: Formatting(
            label="opp. win %",
            format="{:.2f}%",
            denom=100,
            description="Opponents' point rate",
            getter=_get_opp_winrate,
        ),
        Field.RATING: Formatting(
            label="pts",
            format="{:d}",
            denom=None,
            description="Player rating",
            getter=_get_rating,
        ),
        Field.WINS: Formatting(
            label="# wins",
            format="{:d}",
            denom=None,
            description="Number of games won",
            getter=_get_wins,
        ),
        Field.POINTRATE: Formatting(
            label="win %",
            format="{:.2f}%",
            denom=100,
            description="Player's point rate",
            getter=_get_winrate,
        ),
        Field.UNIQUE: Formatting(
            label="uniq. opp.",
            format="{:d}",
            denom=None,
            description="Number of unique opponents",
            getter=_get_unique_opponents,
        ),
        Field.GAMES: Formatting(
            label="# games",
            format="{:d}",
            denom=None,
            description="Number of games played",
            getter=_get_games,
        ),
        Field.OPP_BEATEN: Formatting(
            label="# opp. beat",
            format="{:d}",
            denom=None,
            description="Number of opponents beaten",
            getter=_get_opponents_beaten,
        ),
        Field.SEAT_HISTORY: Formatting(
            label="seat record",
            format="{:s}",
            denom=None,
            description="Seat record",
            getter=_get_seat_history,
        ),
        Field.AVG_SEAT: Formatting(
            label="avg. seat",
            format="{:03.2f}%",
            denom=100,
            description="Average seat",
            getter=_get_avg_seat,
        ),
        Field.RECORD: Formatting(
            label="record",
            format="{:s}",
            denom=None,
            description="Player's record",
            getter=_get_record,
        ),
    }

    ext = {
        DataExport.Format.DISCORD: ".txt",
        DataExport.Format.PLAIN: ".txt",
        DataExport.Format.CSV: ".csv",
    }

    DEFAULT_FIELDS = [
        Field.STANDING,
        Field.NAME,
        Field.RATING,
        Field.RECORD,
        Field.OPP_POINTRATE,
        Field.OPP_BEATEN,
        Field.SEAT_HISTORY,
        Field.AVG_SEAT,
    ]

    def __init__(
        self,
        fields=None,
        format: DataExport.Format = DataExport.Format.PLAIN,
        dir: Union[str, None] = None,
    ):
        if fields is None:
            self.fields = self.DEFAULT_FIELDS
        else:
            self.fields = fields
        self.format = format
        if dir is None:
            self.dir = "./logs/standings" + self.ext[self.format]
        else:
            self.dir = dir

    @classmethod
    def auto_export(cls, func):
        def auto_standings_export_wrapper(
            self: Tournament, *original_args, **original_kwargs
        ):
            ret = func(self, *original_args, **original_kwargs)
            if self.config.auto_export:
                self.export_str(
                    self.get_standings_str(),
                    self.config.standings_export.dir,
                    DataExport.Target.FILE,
                )
            return ret

        return auto_standings_export_wrapper

    def serialize(self):
        """Serializes the export configuration.

        Returns:
            A dictionary containing the serialized configuration.
        """
        return {
            "fields": [f.value for f in self.fields],
            "format": self.format.value,
            "dir": self.dir,
        }

    @classmethod
    def inflate(cls, data: dict):
        """Creates a StandingsExport instance from a dictionary.

        Args:
            data: The dictionary containing the configuration.

        Returns:
            A new StandingsExport instance.
        """
        return cls(
            [StandingsExport.Field(f) for f in data["fields"]],
            StandingsExport.Format(data["format"]),
            data["dir"],
        )


class SortMethod(Enum):
    """Enum for sorting methods."""

    ID = 0
    NAME = 1
    RANK = 2


class SortOrder(Enum):
    """Enum for sorting order."""

    ASCENDING = 0
    DESCENDING = 1


class Log:
    """Handles logging messages with different severity levels."""

    class Level(Enum):
        NONE = 0
        INFO = 1
        WARNING = 2
        ERROR = 3

    class LogEntry:
        """Represents a single log entry."""

        def __init__(self, msg, level):
            """Initializes a LogEntry.

            Args:
                msg: The log message.
                level: The severity level of the log.
            """
            self.msg = msg
            self.level = level

        def short(self):
            if self.level == Log.Level.NONE:
                return ""
            if self.level == Log.Level.INFO:
                return "I"
            if self.level == Log.Level.WARNING:
                return "W"
            if self.level == Log.Level.ERROR:
                return "E"

        @override
        def __repr__(self):
            return "{}> {}".format(self.short(), self.msg)

    output = []

    PRINT = False
    DISABLE = False

    @classmethod
    def log(cls, str_log, level=Level.NONE):
        """Logs a message with a specified level.

        Args:
            str_log: The message to log.
            level: The severity level (default: Level.NONE).
        """
        if cls.DISABLE:
            return
        entry = Log.LogEntry(str_log, level)
        cls.output.append(entry)
        if cls.PRINT:
            print(entry)

    @classmethod
    def print(cls):
        """Prints all log entries to the console."""
        for entry in cls.output:
            print(entry)

    @classmethod
    def export(cls, fpath):
        """Exports the log to a file.

        Args:
            fpath: The file path to export to.
        """
        try:
            from pathlib import Path

            Path(fpath).parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w") as f:
                f.writelines([str(s) + "\n" for s in cls.output])
        except Exception as e:
            cls.log(str(e), level=cls.Level.ERROR)


class TournamentAction:
    """Serializable action that will be stored in tournament log and can be restored"""

    LOGF: bool | str | None = None
    DEFAULT_LOGF = "logs/default.json"

    @classmethod
    def action(cls, func) -> Callable:
        """Decorator to mark a function as a tournament action.

        Args:
            func: The function to decorate.

        Returns:
            The decorated function.
        """

        @StandingsExport.auto_export
        @PodsExport.auto_export
        @functools.wraps(func)
        def wrapper(self: Tournament, *original_args, **original_kwargs):
            # before = self.serialize()
            ret = func(self, *original_args, **original_kwargs)
            cls.store(self)
            return ret

        return wrapper

    @classmethod
    def store(cls, tournament: Tournament):
        """Stores the tournament state to a log file.

        Args:
            tournament: The tournament instance to store.
        """
        if cls.LOGF is None:
            cls.LOGF = cls.DEFAULT_LOGF
        if cls.LOGF:
            assert isinstance(cls.LOGF, str)
            if not os.path.exists(os.path.dirname(cls.LOGF)):
                os.makedirs(os.path.dirname(cls.LOGF))
            with open(cls.LOGF, "w") as f:
                json.dump(tournament.serialize(), f, indent=4)

    @classmethod
    def load(cls, logdir="logs/default.json") -> Tournament | None:
        """Loads the tournament state from a log file.

        Args:
            logdir: The path to the log file (default: 'logs/default.json').

        Returns:
            The loaded tournament instance, or None if the file does not exist.
        """
        if os.path.exists(logdir):
            cls.LOGF = logdir
            # try:
            with open(cls.LOGF, "r") as f:
                tour_json = json.load(f)
                tour = Tournament.inflate(tour_json)
            return tour
            # except Exception as e:
            #    Log.log(str(e), level=Log.Level.ERROR)
            #    return None
        return None

    @override
    def __repr__(self, *nargs, **kwargs):
        ret_str = ("{}{}{}").format(
            self.func_name,
            "" if not nargs else ", ".join([str(arg) for arg in nargs]),
            "" if not kwargs else ", ".join(["{}={}" for _, _ in kwargs.items()]),
        )
        return ret_str


class TournamentConfiguration(ITournamentConfiguration):
    class TopCut(Enum):
        NONE = 0
        TOP_4 = 4
        TOP_7 = 7
        TOP_10 = 10
        TOP_13 = 13
        TOP_16 = 16
        TOP_40 = 40

    def __init__(self, **kwargs):
        """Initializes the TournamentConfiguration.

        Args:
            **kwargs: Arbitrary keyword arguments using the configuration.
        """
        self.pod_sizes = kwargs.get("pod_sizes", [4, 3])
        self.allow_bye = kwargs.get("allow_bye", True)
        self.win_points = kwargs.get("win_points", 5)
        self.bye_points = kwargs.get("bye_points", 4)
        self.draw_points = kwargs.get("draw_points", 1)
        self.snake_pods = kwargs.get("snake_pods", True)
        self.n_rounds = kwargs.get("n_rounds", 5)
        # Parse int or enum for TopCut
        tc_val = kwargs.get("top_cut", TournamentConfiguration.TopCut.NONE)
        if isinstance(tc_val, TournamentConfiguration.TopCut):
            self.top_cut = tc_val
        else:
            # If it's already an int, map to Enum
            try:
                self.top_cut = TournamentConfiguration.TopCut(tc_val)
            except Exception:
                self.top_cut = TournamentConfiguration.TopCut.NONE
        self.max_byes = kwargs.get("max_byes", 2)
        self.auto_export = kwargs.get("auto_export", True)
        self.standings_export = kwargs.get("standings_export", StandingsExport())
        self.global_wr_seats = kwargs.get(
            "global_wr_seats",
            [
                # 0.2553,
                # 0.2232,
                # 0.1847,
                # 0.1428,
                # New data: all 50+ player events since [2024-09-30;2025-05-05]
                0.2470,
                0.1928,
                0.1672,
                0.1458,
            ],
        )

    @property
    def min_pod_size(self):
        """Returns the minimum pod size.

        Returns:
            The minimum pod size.
        """
        return min(self.pod_sizes)

    @property
    def max_pod_size(self):
        """Returns the maximum pod size.

        Returns:
            The maximum pod size.
        """
        return max(self.pod_sizes)

    @staticmethod
    @override
    def ranking(x: Player, tour_round: Round) -> tuple:
        """Calculates the ranking score for a player.

        Args:
            x: The player.
            tour_round: The current round.

        Returns:
            A tuple of ranking criteria.
        """
        return (
            x.rating(tour_round),
            len(x.games(tour_round)),
            np.round(x.opponent_pointrate(tour_round), 10),
            len(x.players_beaten(tour_round)),
            -x.average_seat([r for r in x.tour.rounds if r.seq <= tour_round.seq]),
            -x.uid if isinstance(x.uid, int) else -int(x.uid.int),
        )

    @override
    def __repr__(self):
        return "Tour. cfg:" + "|".join(
            ["{}:{}".format(key, val) for key, val in self.__dict__.items()]
        )

    def serialize(self):
        return {
            "pod_sizes": self.pod_sizes,
            "allow_bye": self.allow_bye,
            "win_points": self.win_points,
            "bye_points": self.bye_points,
            "draw_points": self.draw_points,
            "snake_pods": self.snake_pods,
            "n_rounds": self.n_rounds,
            "max_byes": self.max_byes,
            "auto_export": self.auto_export,
            "standings_export": self.standings_export.serialize(),
            "global_wr_seats": self.global_wr_seats,
            "top_cut": self.top_cut.value,
        }

    @classmethod
    def inflate(cls, data: dict):
        return cls(
            pod_sizes=data["pod_sizes"],
            allow_bye=data["allow_bye"],
            win_points=data["win_points"],
            bye_points=data["bye_points"],
            draw_points=data["draw_points"],
            snake_pods=data["snake_pods"],
            n_rounds=data["n_rounds"],
            max_byes=data["max_byes"],
            auto_export=data["auto_export"],
            standings_export=StandingsExport.inflate(data["standings_export"]),
            global_wr_seats=data["global_wr_seats"],
            top_cut=TournamentConfiguration.TopCut(data["top_cut"]),
        )


class Tournament(ITournament):
    """
    Represents a tournament, managing players, rounds, and pairings.

    Attributes:
        CACHE (dict[UUID, Tournament]): Global cache of tournament instances.
        _pairing_logic_cache (dict[str, type[IPairingLogic]]): Cache of discovered pairing logic implementations.
    """

    # CONFIGURATION
    # Logic: Points is primary sorting key,
    # then opponent pointrate, - CHANGE - moved this upwards and added dummy opponents with 33% pointrate
    # then number of opponents beaten,
    # then ID - this last one is to ensure deterministic sorting in case of equal values (start of tournament for example)
    CACHE: dict[UUID, Tournament] = {}

    _pairing_logic_cache: dict[str, type[IPairingLogic]] = {}

    @classmethod
    def discover_pairing_logic(cls) -> None:
        """Discover and cache all pairing logic implementations from src/pairing_logic."""
        if cls._pairing_logic_cache:
            return

        # Get the base directory of the project
        base_dir = Path(__file__).parent.parent
        pairing_logic_dir = base_dir / "src" / "pairing_logic"

        # Walk through all Python files in the pairing_logic directory
        for module_info in pkgutil.iter_modules([str(pairing_logic_dir)]):
            try:
                # Import the module
                module = importlib.import_module(
                    f"src.pairing_logic.{module_info.name}"
                )

                # Find all classes that implement IPairingLogic
                for name, obj in module.__dict__.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, IPairingLogic)
                        and obj != IPairingLogic
                        and obj.IS_COMPLETE
                    ):
                        if obj.__name__ in cls._pairing_logic_cache:
                            raise ValueError(
                                f"Pairing logic {obj.__name__} already exists"
                            )
                        cls._pairing_logic_cache[obj.__name__] = obj(
                            name=f"{obj.__name__}"
                        )
            except Exception as e:
                Log.log(
                    f"Failed to import pairing logic module {module_info.name}: {e}",
                    level=Log.Level.WARNING,
                )

    @classmethod
    def get_pairing_logic(cls, logic_name: str) -> IPairingLogic:
        """Get a pairing logic instance by name.

        Args:
            logic_name: The name of the pairing logic.

        Returns:
            The pairing logic class.
        """
        cls.discover_pairing_logic()

        if logic_name not in cls._pairing_logic_cache:
            raise ValueError(f"Unknown pairing logic: {logic_name}")

        return cls._pairing_logic_cache[logic_name]

    def __init__(
        self,
        config: Union[TournamentConfiguration, None] = None,
        uid: UUID | None = None,
    ):  # type: ignore
        """Initializes a new Tournament instance.

        Args:
            config: The configuration object for the tournament. If None, a default configuration is used.
            uid: The unique identifier for the tournament. If None, a new UUID is generated.
        """
        if config is None:
            config = TournamentConfiguration()
        super().__init__(config=config, uid=uid)
        self._config = config
        self.CACHE[self.uid] = self

        self.PLAYER_CACHE: dict[UUID, Player] = {}
        self.POD_CACHE: dict[UUID, Pod] = {}
        self.ROUND_CACHE: dict[UUID, Round] = {}

        self._rounds: list[UUID] = list()
        self._players: set[UUID] = set()
        # self._dropped: list[UUID] = list()
        # self._disabled: list[UUID] = list()  # Players disabled from top cut (but still in tournament)
        self._round: UUID | None = None

        # Direct setting - don't want to overwrite old log file
        # self.initialize_round()

    # TOURNAMENT ACTIONS
    # IMPORTANT: No nested tournament actions

    @classmethod
    def get(cls, uid: UUID) -> Tournament:
        """Retrieves a tournament by its UUID.

        Args:
            uid: The tournament UUID.

        Returns:
            The tournament instance.
        """
        return cls.CACHE[uid]

    @property
    def players(self) -> set[Player]:
        return {Player.get(self, x) for x in self._players}

    @property
    def tour_round(self) -> Round:
        return Round.get(self, self._round)  # type: ignore

    @tour_round.setter
    def tour_round(self, tour_round: Round):
        """Sets the current tournament round.

        Args:
            tour_round: The new round.
        """
        self._round = tour_round.uid

    @property
    def last_round(self) -> Round | None:
        """
        Returns the last round of the tournament.

        Returns:
            Round|None:
                - The last round if it has been played.
                - None if no rounds have been played.
        """
        return self.rounds[-1] if self.rounds else None

    def previous_round(self, tour_round: Round | None = None) -> Round | None:
        """
        Returns the previous round of the tournament.

        Args:
            tour_round: The current round.

        Returns:
            Round|None:
                - The previous round if it exists.
                - None if the current round is the first round.
        """
        if tour_round is None:
            tour_round = self.tour_round
        return (
            self.rounds[self.rounds.index(tour_round) - 1]
            if self.rounds.index(tour_round) > 0
            else None
        )

    @property
    def final_swiss_round(self) -> Round | None:
        """
        Returns the final Swiss round of the tournament.

        Returns:
            Round|None:
                - The final Swiss round if it has been played.
                - None if the final round has not been played yet.
        """
        if len(self.rounds) >= self.config.n_rounds:
            last_round = self.rounds[self.config.n_rounds - 1]
            if last_round.stage != Round.Stage.SWISS:
                raise ValueError("Last round is not a Swiss round.")
            return last_round
        return None

    @property
    def pods(self) -> list[Pod] | None:
        """
        Returns the list of pods in the current round.

        Returns:
            list[Pod]|None:
                - The list of pods if the current round has been set.
                - None if the current round has not been set.
        """
        if not self.tour_round:
            return None
        return self.tour_round.pods

    @property
    def rounds(self) -> list[Round]:
        """
        Returns the list of rounds in the tournament.

        Returns:
            list[Round]:
                - The list of rounds.
        """
        return [Round.get(self, x) for x in self._rounds]

    @rounds.setter
    def rounds(self, rounds: list[Round]):
        """
        Sets the list of rounds in the tournament.

        Args:
            rounds: The list of rounds.
        """
        self._rounds = [r.uid for r in rounds]

    @property
    def swiss_rounds(self):
        """
        Returns the list of swiss rounds in the tournament.

        Returns:
            list[Round]:
                - The list of swiss rounds.
        """
        return [r for r in self.rounds if r.stage == Round.Stage.SWISS]

    @property
    def ended_rounds(self):
        """
        Returns the list of completed rounds in the tournament.

        Returns:
            list[Round]:
                - The list of completed rounds.
        """
        return [r for r in self.rounds if r.done]

    @property
    def draw_rate(self) -> float:
        """
        Calculates the draw rate for the tournament.

        Returns:
            float:
                - The draw rate as a float.
        """
        draws = 0
        matches = 0
        for tour_round in self.rounds:
            for pod in tour_round.pods:
                if pod.done:
                    matches += 1
                    if pod.result_type == Pod.EResult.DRAW:
                        draws += len(pod._result)
        return draws / matches

    @property
    def config(self) -> TournamentConfiguration:
        """
        Returns the tournament configuration.

        Returns:
            TournamentConfiguration:
                - The tournament configuration.
        """
        return self._config

    @config.setter
    @TournamentAction.action
    def config(self, config: TournamentConfiguration):
        """
        Sets the tournament configuration.

        Args:
            config: The tournament configuration.
        """
        self._config = config

    @TournamentAction.action
    def add_player(self, *specs: Any, **player_attrs) -> list[Player]:
        """Adds players to the tournament.

        This method supports flexible input formats for defining players.

        Args:
            *specs: Variable length argument list. Each argument can be:
                - A Player object.
                - A tuple/list of (name,), (name, uid/decklist), or (name, uid, decklist).
                - A dictionary containing 'name', and optionally 'uid' and 'decklist'.
                - A string representing the player's name.
            **player_attrs: arbitrary keyword arguments to be applied to all new players
                            (e.g., decklist="link", uid=UUID(...)).
                            If only one positional argument is provided and it's a string or dict,
                            these attributes are merged with it.

        Returns:
            list[Player]: A list of the newly created and added Player objects.

        Raises:
            ValueError: If the player specification is invalid or incomplete.
        """

        # Handle keyword arguments merging with a single positional spec
        if player_attrs and len(specs) == 1 and "name" not in player_attrs:
            spec = specs[0]
            if isinstance(spec, str):
                data = [{"name": spec, **player_attrs}]
            elif isinstance(spec, dict):
                data = [{**spec, **player_attrs}]
            else:
                data = list(specs) + [player_attrs]
        else:
            data = list(specs)
            if player_attrs:
                data.append(player_attrs)

        # Handle backward compatibility: single positional list
        if len(data) == 1 and isinstance(data[0], list):
            data = data[0]

        new_players = []
        existing_names = set([p.name for p in self.players])
        existing_uids = set([p.uid for p in self.players])

        for entry in data:
            if entry is None:
                continue

            name, uid, decklist = None, None, None

            if isinstance(entry, (tuple, list)):
                # Handle 1-tuple (name), 2-tuple (smart: name, uid/decklist), or 3-tuple (name, uid, decklist)
                if len(entry) == 1:
                    name = entry[0]
                elif len(entry) == 2:
                    name, second = entry
                    if isinstance(second, UUID):
                        uid = second
                    elif isinstance(second, (str, type(None))):
                        decklist = second
                    else:
                        raise ValueError(
                            f"Unknown type for second element in player tuple: {type(second)}"
                        )
                elif len(entry) == 3:
                    name, uid, decklist = entry
                else:
                    raise ValueError(
                        f"Player tuple/list must have 1-3 elements, got {len(entry)}"
                    )
            elif isinstance(entry, dict):
                # Handle dictionary specification
                name = entry.get("name")
                uid = entry.get("uid")
                decklist = entry.get("decklist")
            elif isinstance(entry, str):
                # Handle single string as name
                name = entry
            else:
                raise ValueError(
                    f"Invalid player specification type: {type(entry)}. Expected str, dict, tuple, or list."
                )

            if not name or not isinstance(name, str):
                raise ValueError(
                    f"Player name must be a non-empty string, got {type(name)}: {name}"
                )

            if name in existing_names:
                Log.log(
                    "\tPlayer {} already enlisted.".format(name),
                    level=Log.Level.WARNING,
                )
                continue
            if uid and uid in existing_uids:
                Log.log(
                    "\tPlayer with UID {} already enlisted.".format(uid),
                    level=Log.Level.WARNING,
                )
                continue

            # Create and register the player
            p = Player(self, name, uid, decklist)
            self._players.add(p.uid)
            if self._round and p.uid not in self.tour_round._players:
                self.tour_round._players.append(p.uid)
            new_players.append(p)
            existing_names.add(name)
            existing_uids.add(p.uid)
            Log.log("\tAdded player {}".format(p.name), level=Log.Level.INFO)
        return new_players

    @TournamentAction.action
    def drop_player(self, players: list[Player] | Player) -> bool:
        """Drops a player or list of players from the tournament.

        Dropped players are removed from future pairings but their history remains.
        If a player is dropped during an active round, they might need to be resolved in the current pod first.

        Args:
            players: The player or list of players to drop.

        Returns:
            bool: True if the drop was successful, False otherwise.
        """
        if not isinstance(players, list):
            players = [players]
        for p in players:
            if self.tour_round and p.seated(self.tour_round):
                if self.tour_round.done and self.tour_round != self.last_round:
                    # Log.log('Can\'t drop {} during an active tour_round.\nComplete the tour_round or remove player from pod first.'.format(
                    #    p.name), level=Log.Level.WARNING)
                    return False

            # If player has not played yet, it can safely be deleted without being saved
            if p.played(self.tour_round):
                self.tour_round.drop_player(p)
            else:
                self._players.remove(p.uid)
                self.tour_round._players.remove(p.uid)
            # Remove from disabled set if they were disabled
            self.tour_round.disable_player(p, set_disabled=False)
        return True
        # Log.log('\tRemoved player {}'.format(p.name), level=Log.Level.INFO)

    @TournamentAction.action
    def disable_player(
        self, players: list[Player] | Player, set_disabled: bool = True
    ) -> bool:
        """Disables or enables players for top cut participation.

        Disabled players remain in the tournament structure but are excluded from top cut calculations and pairings.

        Args:
            players: The player or list of players to disable/enable.
            set_disabled: If True, disables the player. If False, re-enables them.

        Returns:
            bool: Always returns True.
        """
        if not isinstance(players, list):
            players = [players]
        for p in players:
            self.tour_round.disable_player(p, set_disabled=set_disabled)
        return True

    @TournamentAction.action
    def rename_player(self, player, new_name):
        """Renames a player in the tournament.

        This updates the player's name across all historical records in the tournament (pods, rounds).

        Args:
            player: The player object to rename.
            new_name: The new name for the player.
        """
        if player.name == new_name:
            return
        if new_name in [p.name for p in self.active_players]:
            Log.log(
                "\tPlayer {} already enlisted.".format(new_name),
                level=Log.Level.WARNING,
            )
            return
        if new_name:
            player.name = new_name
            for tour_round in self.rounds:
                for pod in tour_round.pods:
                    for p in pod.players:
                        if p.name == player.name:
                            p.name = new_name
            Log.log(
                "\tRenamed player {} to {}".format(player.name, new_name),
                level=Log.Level.INFO,
            )

    def get_pod_sizes(self, n) -> list[int] | None:
        """Determines possible pod sizes for a given number of players based on configuration.

        Args:
            n: The number of players.

        Returns:
            A list of integers representing the sizes of the pods, or None if no valid combination is found.
        """
        # Stack to store (remaining_players, current_pod_size_index, current_solution)
        stack = [(n, 0, [])]

        while stack:
            remaining, pod_size_idx, current_solution = stack.pop()

            # If we've processed all pod sizes, continue to next iteration
            if pod_size_idx >= len(self.config.pod_sizes):
                continue

            pod_size = self.config.pod_sizes[pod_size_idx]
            rem = remaining - pod_size

            # Skip if this pod size would exceed remaining players
            if rem < 0:
                stack.append((remaining, pod_size_idx + 1, current_solution))
                continue

            # If this pod size exactly matches remaining players, we found a solution
            if rem == 0:
                return current_solution + [pod_size]

            # Handle case where remaining players is less than minimum pod size
            if rem < self.config.min_pod_size:
                if self.config.allow_bye and rem <= self.config.max_byes:
                    return current_solution + [pod_size]
                elif pod_size == self.config.pod_sizes[-1]:
                    continue
                else:
                    stack.append((remaining, pod_size_idx + 1, current_solution))
                    continue

            # If remaining players is valid, try this pod size and continue with remaining players
            if rem >= self.config.min_pod_size:
                stack.append((remaining, pod_size_idx + 1, current_solution))
                stack.append((rem, 0, current_solution + [pod_size]))

        return None

    def initialize_round(self) -> bool:
        """Initializes a new round in the tournament.

        This method determines the appropriate stage (Swiss, Top Cut) and pairing logic based on the
        tournament configuration and current progress. It does not create pairings, only sets up the round structure.

        Returns:
            bool: True if a new ro  und was successfully initialized. False if a round is already in progress,
                  the maximum number of rounds has been reached, or the tournament is completed.
        """
        if self._round is not None and not self.tour_round.done:
            return False
        seq = len(self.rounds)
        stage = Round.Stage.SWISS
        logic = None
        if seq >= self.config.n_rounds and self.last_round:
            if self.config.top_cut == TournamentConfiguration.TopCut.NONE:
                Log.log("Maximum number of rounds reached.", level=Log.Level.WARNING)
                return False
            if self.config.top_cut == TournamentConfiguration.TopCut.TOP_4:
                if self.last_round.stage == Round.Stage.SWISS:
                    logic = self.get_pairing_logic("PairingTop4")
                    stage = Round.Stage.TOP_4
                else:
                    Log.log("Tournament completed.")
                    return False
            elif self.config.top_cut == TournamentConfiguration.TopCut.TOP_7:
                if self.last_round.stage == Round.Stage.SWISS:
                    stage = Round.Stage.TOP_7
                    logic = self.get_pairing_logic("PairingTop7")
                elif self.last_round.stage == Round.Stage.TOP_7:
                    stage = Round.Stage.TOP_4
                    logic = self.get_pairing_logic("PairingTop4")
                else:
                    Log.log("Tournament completed.")
                    return False
            elif self.config.top_cut == TournamentConfiguration.TopCut.TOP_10:
                if self.last_round.stage == Round.Stage.SWISS:
                    stage = Round.Stage.TOP_10
                    logic = self.get_pairing_logic("PairingTop10")
                elif self.last_round.stage == Round.Stage.TOP_10:
                    stage = Round.Stage.TOP_4
                    logic = self.get_pairing_logic("PairingTop4")
                else:
                    Log.log("Tournament completed.")
                    return False
            elif self.config.top_cut == TournamentConfiguration.TopCut.TOP_13:
                if self.last_round.stage == Round.Stage.SWISS:
                    stage = Round.Stage.TOP_13
                    logic = self.get_pairing_logic("PairingTop13")
                elif self.last_round.stage == Round.Stage.TOP_13:
                    stage = Round.Stage.TOP_4
                    logic = self.get_pairing_logic("PairingTop4")
                else:
                    Log.log("Tournament completed.")
                    return False
            elif self.config.top_cut == TournamentConfiguration.TopCut.TOP_16:
                if self.last_round.stage == Round.Stage.SWISS:
                    stage = Round.Stage.TOP_16
                    logic = self.get_pairing_logic("PairingTop16")
                elif self.last_round.stage == Round.Stage.TOP_16:
                    stage = Round.Stage.TOP_4
                    logic = self.get_pairing_logic("PairingTop4")
                else:
                    Log.log("Tournament completed.")
                    return False
            elif self.config.top_cut == TournamentConfiguration.TopCut.TOP_40:
                if self.last_round.stage == Round.Stage.SWISS:
                    stage = Round.Stage.TOP_40
                    logic = self.get_pairing_logic("PairingTop40")
                elif self.last_round.stage == Round.Stage.TOP_40:
                    stage = Round.Stage.TOP_16
                    logic = self.get_pairing_logic("PairingTop16")
                elif self.last_round.stage == Round.Stage.TOP_16:
                    stage = Round.Stage.TOP_4
                    logic = self.get_pairing_logic("PairingTop4")
                else:
                    Log.log("Tournament completed.")
                    return False
            else:
                raise ValueError(f"Unknown top cut: {self.config.top_cut}")
        else:
            if seq == 0:
                logic = self.get_pairing_logic("PairingRandom")
            elif seq == 1 and self.config.snake_pods:
                logic = self.get_pairing_logic("PairingSnake")
            else:
                logic = self.get_pairing_logic("PairingDefault")

        if not logic:
            Log.log("No pairing logic found.", level=Log.Level.ERROR)
            return False
        elif not stage:
            Log.log("No stage found.", level=Log.Level.ERROR)
            return False
        new_round = Round(
            self,
            len(self.rounds),
            stage,
            logic,
            dropped=self.tour_round._dropped if self._round else set(),
            disabled=self.tour_round._disabled if self._round else set(),
        )
        self._rounds.append(new_round.uid)
        self.tour_round = new_round
        return True

    @TournamentAction.action
    def create_pairings(self) -> bool:
        """Creates pairings for the current round.

        If the round has not been initialized or previous rounds are not complete, this method attempts
        to handle those states.

        Returns:
            bool: True if pairings were successfully created (or were already created). False if
                  pairings could not be created (e.g., due to initialization failure).
        """
        if self.last_round is None or self.last_round.done:
            ok = self.initialize_round()
            if not ok:
                return False
        # self.last_round._byes.clear()
        assert self.last_round is not None
        if not self.last_round.all_players_assigned:
            self.last_round.create_pairings()
            return True
        return False

    @TournamentAction.action
    def new_round(self) -> bool:
        """Starts a new round.

        Returns:
            True if a new round was successfully started, False otherwise.
        """
        if not self.last_round or self.last_round.done:
            return self.initialize_round()
        return False

    @TournamentAction.action
    def reset_pods(self) -> bool:
        """Resets the pods for the current round.

        Returns:
            True if pods were reset, False otherwise.
        """
        if not self.tour_round:
            return False
        if not self.tour_round.done:
            if not self.tour_round.reset_pods():
                return False
            return True
        return False

    @TournamentAction.action
    def manual_pod(self, players: list[Player]):
        """Creates a manual pod with the specified players.

        Args:
            players: A list of players to include in the manual pod.
        """
        if self.tour_round is None or self.tour_round.done:
            if not self.new_round():
                return
        assert isinstance(self.tour_round, Round)
        cap = min(self.config.max_pod_size, len(self.tour_round.unassigned))
        pod = Pod(self.tour_round, len(self.tour_round.pods), cap=cap)
        self.tour_round._pods.append(pod.uid)

        for player in players:
            pod.add_player(player)
        self.tour_round.pods.append(pod)

    @TournamentAction.action
    def report_win(self, players: list[Player] | Player):
        """Reports a win for the specified player(s) in the current round.

        Args:
            players: The player or list of players who won.
        """
        if self.tour_round:
            if not isinstance(players, list):
                players = [players]
            for p in players:
                self.tour_round.set_result(p, Player.EResult.WIN)

    @TournamentAction.action
    def report_draw(self, players: list[Player] | Player):
        """Reports a draw for the specified player(s) in the current round.

        Args:
            players: The player or list of players who drew.
        """
        if self.tour_round:
            if not isinstance(players, list):
                players = [players]
            for p in players:
                self.tour_round.set_result(p, Player.EResult.DRAW)

    @TournamentAction.action
    def random_results(self):
        """Generates random results for all incomplete pods in the current round."""
        if not self.tour_round:
            # Log.log(
            #    'A tour_round is not in progress.\nCreate pods first!',
            #    level=Log.Level.ERROR
            # )
            return
        if self.tour_round.pods:
            draw_rate = 1 - sum(self.config.global_wr_seats)
            # for each pod
            # generate a random result based on global_winrates_by_seat
            # each value corresponds to the pointrate of the player in that seat
            # the sum of percentages is less than 1, so there is a chance of a draw (1-sum(winrates))

            for pod in [x for x in self.tour_round.pods if not x.done]:
                # generate a random result
                result = random.random()
                rates = np.array(
                    self.config.global_wr_seats[0 : len(pod.players)] + [draw_rate]
                )
                rates = np.cumsum(rates / sum(rates))
                draw = result > rates[-2]
                if not draw:
                    win = np.argmax([result < x for x in rates])
                    # Log.log('won "{}"'.format(pod.players[win].name))
                    self.tour_round.set_result(pod.players[win], Player.EResult.WIN)
                    # player = random.sample(pod.players, 1)[0]
                    # Log.log('won "{}"'.format(player.name))
                    # self.tour_round.won([player])
                else:
                    players = pod.players
                    # Log.log('draw {}'.format(
                    #    ' '.join(['"{}"'.format(p.name) for p in players])))
                    for p in players:
                        self.tour_round.set_result(p, Player.EResult.DRAW)
                pass
        pass

    @TournamentAction.action
    def move_player_to_pod(
        self, pod: Pod, players: list[Player] | Player, manual=False
    ):
        """Moves a player or list of players to a specified pod.

        Args:
            pod: The target pod.
            players: The player or list of players to move.
            manual: If True, allows adding players even if the pod is full.
        """
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod(self.tour_round) == pod:
                continue
                # player.pod(self.tour_round).remove_player(player)
                # Log.log('Removed player {} from {}.'.format(
                #    player.name, old_pod), level=Log.Level.INFO)
            if ok := pod.add_player(player, manual=manual):
                pass
                # Log.log('Added player {} to {}'.format(
                #    player.name, pod.name), level=Log.Level.INFO)
                # else:
                #    Log.log('Failed to add palyer {} to Pod {}'.format(
                #        player.name, pod.table), level=Log.Level.ERROR)

    @TournamentAction.action
    def bench_players(self, players: Iterable[Player] | Player):
        """Removes player(s) from their current pod, effectively benching them.

        Args:
            players: The player or iterable of players to bench.
        """
        assert self.tour_round is not None
        if not isinstance(players, Iterable):
            players = [players]
        for player in players:
            self.remove_player_from_pod(player)

    @TournamentAction.action
    def toggle_game_loss(self, players: Iterable[Player] | Player):
        """Toggles the game loss status for player(s).

        If a player is assigned a game loss, they are removed from their pod and marked as having lost.
        If they already have a game loss, it is removed.

        Args:
            players: The player or iterable of players to toggle game loss for.
        """
        if not isinstance(players, Iterable):
            players = [players]

        for player in players:
            if player.uid in self.tour_round._game_loss:
                self.tour_round._game_loss.remove(player.uid)
            else:
                # if player.pod(self.tour_round) is not None:
                #    self.remove_player_from_pod(player)
                player.set_result(self.tour_round, Player.EResult.LOSS)
                # Log.log('{} assigned a game loss.'.format(
                #    player.name), level=Log.Level.INFO)

    @TournamentAction.action
    def toggle_bye(self, players: Iterable[Player] | Player):
        """Toggles the bye status for player(s).

        If a player is assigned a bye, they are removed from their pod and marked as having a bye.
        If they already have a bye, it is removed.

        Args:
            players: The player or iterable of players to toggle bye for.
        """
        if not isinstance(players, Iterable):
            players = [players]
        for player in players:
            if player.uid in self.tour_round._byes:
                self.tour_round._byes.remove(player.uid)
            else:
                if player.pod(self.tour_round) is not None:
                    self.remove_player_from_pod(player)
                self.tour_round.set_result(player, Player.EResult.BYE)

    @TournamentAction.action
    def delete_pod(self, pod: Pod):
        """Deletes a specified pod from the current round.

        Args:
            pod: The pod to delete.
        """
        if self.tour_round:
            self.tour_round.remove_pod(pod)

    def remove_player_from_pod(self, player: Player):
        """Removes a player from their current pod.

        Args:
            player: The player to remove.
        """
        assert self.tour_round is not None
        pod = player.pod(self.tour_round)
        if pod:
            pod.remove_player(player)
            # if player.uid not in self.tour_round._game_loss:
            #    self.tour_round.set_result(player, Player.EResult.BYE)
            # Log.log('Removed player {} from {}.'.format(
            #    player.name, pod.name), level=Log.Level.INFO)

    def rating(self, player: Player, tour_round: Round) -> float:
        """
        Calculate the rating of a player for a given round.
        The rating is the sum of the points for the player in the Swiss rounds up to and including the given round.
        If the round is not a Swiss round, the rating is the sum of the points for the player in the last Swiss round.

        Args:
            player: The player for whom to calculate the rating.
            tour_round: The round up to which to calculate the rating.

        Returns:
            The player's rating as a float.
        """
        points = 0
        for i, i_tour_round in enumerate(self.rounds):
            if i_tour_round.stage != Round.Stage.SWISS:
                break
            round_result = player.result(i_tour_round)
            if round_result == Player.EResult.WIN:
                points += self.config.win_points
            elif round_result == Player.EResult.DRAW:
                points += self.config.draw_points
            elif round_result == Player.EResult.BYE:
                points += self.config.bye_points
            if i_tour_round == tour_round:
                break
        return points

    # MISC ACTIONS

    def get_pods_str(self) -> str:
        """Generates a string representation of the current round's pods.

        Returns:
            A formatted string showing the pods and players, including byes if applicable.
        """
        if not self.tour_round:
            return ""
        standings = self.get_standings(self.tour_round)
        export_str = "\n\n".join(
            [
                pod.__repr__(TournamentContext(self, self.tour_round, standings))
                for pod in self.tour_round.pods
            ]
        )

        if self.config.allow_bye and self.tour_round.unassigned:
            export_str += "\n\nByes:\n" + "\n:".join(
                [
                    "\t{}\t| pts: {}".format(p.name, p.rating(self.tour_round) or "0")
                    for p in self.tour_round.unassigned
                ]
            )
        return export_str

    @override
    def get_standings(self, tour_round: Round | None = None) -> list[Player]:
        """Calculates and retrieves the standings for a specific round.

        Use this instead of accessing the players list directly, as this method ensures
        players are sorted according to the tournament's ranking configuration.

        Args:
            tour_round: The round for which to calculate standings.
                        If None, uses the current round.

        Returns:
            list[Player]: A list of players sorted by their current standing.
        """
        method = Player.SORT_METHOD
        order = Player.SORT_ORDER
        Player.SORT_METHOD = SortMethod.RANK
        Player.SORT_ORDER = SortOrder.ASCENDING
        playoffs = False
        if tour_round is None:
            tour_round = self.tour_round
        if tour_round.stage == Round.Stage.SWISS:
            standings = sorted(
                self.players,
                key=lambda x: self.config.ranking(x, tour_round),
                reverse=True,
            )
        else:
            final_swiss = self.final_swiss_round
            assert final_swiss is not None
            playoff_stage = tour_round.seq - final_swiss.seq - 1

            if playoff_stage > 0:
                # TODO: take the standings of previous playoff round and modify them to current results
                previous_round = self.previous_round(tour_round)
                assert previous_round is not None
                standings = self.get_standings(previous_round)
                advancing_players = tour_round.advancing_players(standings)
                non_advancing = [p for p in standings if p not in advancing_players]
                standings = advancing_players + non_advancing
                pass
            else:
                swiss_standings = self.get_standings(final_swiss)
                advancing_players = tour_round.advancing_players(swiss_standings)
                non_advancing = [
                    p for p in swiss_standings if p not in advancing_players
                ]

                # Sort non-advancing players: draws rank above losses, then by original standings
                standings_index = {
                    player: idx for idx, player in enumerate(swiss_standings)
                }
                non_advancing.sort(
                    key=lambda x: (
                        0
                        if tour_round and x.result(tour_round) == Player.EResult.DRAW
                        else 1,  # Draws first (0), losses second (1)
                        standings_index.get(
                            x, len(swiss_standings)
                        ),  # Then by original standings position
                    )
                )

                standings = advancing_players + non_advancing
                pass

            # TODO: Implement playoff standings
            pass

        Player.SORT_METHOD = method
        Player.SORT_ORDER = order
        return standings

    def get_standings_str(
        self,
        fields: list[StandingsExport.Field] = StandingsExport.DEFAULT_FIELDS,
        style: StandingsExport.Format = StandingsExport.Format.PLAIN,
        tour_round: Round | None = None,
        standings: list[Player] | None = None,
    ) -> str:
        """Generates a formatted string of the tournament standings.

        Args:
            fields: A list of StandingsExport.Field to include in the standings.
            style: The desired output format (e.g., PLAIN, CSV, DISCORD, JSON).
            tour_round: The round for which to generate standings. Defaults to the current round.
            standings: Pre-calculated standings. If None, standings will be calculated.

        Returns:
            A string containing the formatted standings.

        Raises:
            ValueError: If an invalid style is provided.
        """
        # raise DeprecationWarning("get_standings_str is deprecated. Use get_standings instead.")
        if tour_round is None:
            tour_round = self.tour_round
        if standings is None:
            standings = self.get_standings(tour_round)

        # Create context with all available data
        context = TournamentContext(
            tour=self,
            tour_round=tour_round,
            standings=standings,
        )

        lines = [[StandingsExport.info[f].name for f in fields]]
        lines += [
            [
                (StandingsExport.info[f].format).format(
                    StandingsExport.info[f].get(p, context)
                    if StandingsExport.info[f].denom is None
                    else StandingsExport.info[f].get(p, context)
                    * StandingsExport.info[f].denom
                )
                for f in fields
            ]
            for p in standings
        ]
        if style == StandingsExport.Format.PLAIN:
            col_len = [0] * len(fields)
            for col in range(len(fields)):
                for line in lines:
                    if len(line[col]) > col_len[col]:
                        col_len[col] = len(line[col])
            for line in lines:
                for col in range(len(fields)):
                    line[col] = line[col].ljust(col_len[col])
            # add new line at index 1
            lines.insert(1, ["-" * width for width in col_len])
            lines = "\n".join([" | ".join(line) for line in lines])
            return lines

            # Log.log('Log saved: {}.'.format(
            #    fdir), level=Log.Level.INFO)
        elif style == StandingsExport.Format.CSV:
            Log.log(
                "Log not saved - CSV not implemented.".format(fdir),
                level=Log.Level.WARNING,
            )
        elif style == StandingsExport.Format.DISCORD:
            Log.log(
                "Log not saved - DISCORD not implemented.".format(fdir),
                level=Log.Level.WARNING,
            )
        elif style == StandingsExport.Format.JSON:
            Log.log(
                "Log not saved - JSON not implemented.".format(fdir),
                level=Log.Level.WARNING,
            )

        raise ValueError("Invalid style: {}".format(style))

    @staticmethod
    def send_request(api, data, headers):
        """Sends a POST request to a specified API endpoint.

        Args:
            api: The API endpoint URL.
            data: The JSON data to send.
            headers: A dictionary of HTTP headers.
        """
        try:
            response = requests.post(api, json=data, headers=headers, timeout=10)
            if response.status_code == 200:
                Log.log("Data successfully sent to the server!")
            else:
                Log.log(f"Failed to send data. Status code: {response.status_code}")
        except Exception as e:
            Log.log(f"Error sending data: {e}", level=Log.Level.ERROR)

    def export_str(
        self,
        data: str,
        var_export_param: Any,
        target_type: StandingsExport.Target,
    ):
        """Exports a string of data to a specified target (file, web, discord, console).

        Args:
            data: The string data to export.
            var_export_param: Parameter specific to the target type (e.g., file path, log level).
            target_type: The target for the export (FILE, WEB, DISCORD, CONSOLE).
        """
        if StandingsExport.Target.FILE == target_type:
            if not os.path.exists(os.path.dirname(var_export_param)):
                os.makedirs(os.path.dirname(var_export_param))
            with open(var_export_param, "w", encoding="utf-8") as f:
                f.writelines(data)

        if StandingsExport.Target.WEB == target_type:
            api = os.getenv("EXPORT_ONLINE_API_URL")
            key = os.getenv("EXPORT_ONLINE_API_KEY")
            if not key or not api:
                Log.log(
                    "Error: EXPORT_ONLINE_API_URL or EXPORT_ONLINE_API_KEY not set in the environment variables."
                )
                return
            tournament_id = os.getenv("TOURNAMENT_ID")
            url = f"{api}?tournamentId={tournament_id}"

            # Send as POST request to the Express app with authentication
            headers = {"x-api-key": key}
            request_data = {
                "title": "Tournament Update",
                "timestamp": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "text": data,
                "tournament": self.serialize(),
            }

            thread = threading.Thread(
                target=self.send_request, args=(url, request_data, headers)
            )
            thread.start()

        if StandingsExport.Target.DISCORD == target_type:
            instance = DiscordPoster.instance()
            instance.post_message(data)

        if StandingsExport.Target.CONSOLE == target_type:
            if not isinstance(var_export_param, Log.Level):
                var_export_param = Log.Level.INFO
            Log.log(data, level=var_export_param)

    def serialize(self) -> dict[str, Any]:
        """Serializes the tournament state to a JSON-compatible dictionary.

        The serialization includes:
        - specific tournament configuration
        - list of players (including their state)
        - list of pods (including their state)
        - list of rounds (including pairings and results)

        All objects are cross-referenced by their unique IDs to maintain integrity upon restoration.

        Returns:
            dict: A dictionary representing the serialized tournament data.
        """

        data: dict[str, Any] = {}
        data["uid"] = str(self.uid)
        data["config"] = self.config.serialize()
        data["players"] = list(p.serialize() for p in self.players)
        # data['dropped'] = [p.serialize() for p in self.dropped_players]
        # data['disabled'] = [str(p.uid) for p in self.disabled_players]
        data["rounds"] = [r.serialize() for r in self.rounds]
        return data

    @classmethod
    def inflate(cls, data: dict[str, Any]) -> Tournament:
        """Creates a Tournament instance from serialized data.

        This method reconstructs the entire tournament state, including players, rounds, and pods,
        linking them back together using their UUIDs.

        Args:
            data: A dictionary containing the serialized tournament data (as produced by `serialize`).

        Returns:
            Tournament: The reconstructed Tournament instance.
        """
        config = TournamentConfiguration.inflate(data["config"])
        tour_uid = UUID(data["uid"])
        if tour_uid in Tournament.CACHE:
            tour = Tournament.CACHE[tour_uid]
        else:
            tour = cls(config, tour_uid)
        tour._players = {UUID(d_player["uid"]) for d_player in data["players"]}
        # tour._dropped = [UUID(d_player['uid']) for d_player in data['dropped']]
        # tour._disabled = [UUID(d_player['uid']) for d_player in data['disabled']]
        # tour._players.extend(tour._dropped)
        # tour._players.extend(tour._disabled)
        # Load disabled players (backward compatible: may not exist in old saves)
        # if 'disabled' in data:
        #    tour._disabled = {UUID(uid) for uid in data['disabled']}
        for d_player in data["players"]:
            Player.inflate(tour, d_player)
        # for d_player in data['dropped']:
        #    Player.inflate(tour, d_player)
        tour._rounds = [UUID(d_round["uid"]) for d_round in data["rounds"]]
        for _, d_round in tqdm(enumerate(data["rounds"]), desc="Inflating rounds"):
            r = Round.inflate(tour, d_round)
            tour._round = r.uid
        return tour


class Player(IPlayer):
    """
    Represents a player in the tournament.

    Attributes:
        SORT_METHOD (SortMethod): The method used for sorting players (ID, Name, Rank).
        SORT_ORDER (SortOrder): The order used for sorting (Ascending, Descending).
        CACHE (dict[UUID, Player]): Global cache of player instances.
    """

    SORT_METHOD: SortMethod = SortMethod.ID
    SORT_ORDER: SortOrder = SortOrder.ASCENDING
    FORMATTING = ["-p"]

    def __init__(
        self,
        tour: Tournament,
        name: str,
        uid: UUID | None = None,
        decklist: str | None = None,
    ):
        """Initializes a new Player instance.

        Args:
            tour: The tournament the player belongs to.
            name: The name of the player.
            uid: The unique identifier for the player. If None, a new UUID is generated.
            decklist: Optional URL or string representing the player's decklist.
        """
        self._tour = tour.uid
        super().__init__(uid=uid)
        self.name = name
        self.decklist = decklist
        self.CACHE[self.uid] = self
        self._pod_id: UUID | None = None  # Direct reference to current pod
        self.table_preference: list[int] = []

    # ACTIONS
    def set_result(self, tour_round: Round, result: Player.EResult) -> None:
        """Sets the result for the player in a specific round.

        Args:
            tour_round: The round where the result occurred.
            result: The result to set (WIN, LOSS, DRAW, BYE).
        """
        tour_round.set_result(self, result)

    def set_table_preference(self, table_preference: list[int]) -> None:
        self.table_preference = table_preference

    # QUERIES

    def pod(self, tour_round: Round) -> Pod | None:
        if tour_round is None:
            return None
        return tour_round.get_location(self)

    def result(self, tour_round: Round) -> Player.EResult:
        """Retrieves the player's result for a specific round.

        Args:
            tour_round: The round to query.

        Returns:
            Player.EResult: The result (WIN, LOSS, DRAW, BYE, PENDING).
        """
        if self.uid in tour_round._byes:
            return Player.EResult.BYE
        if self.uid in tour_round._game_loss:
            return Player.EResult.LOSS
        pod = self.pod(tour_round)
        if pod and len(pod._result) > 0:
            if self.uid in pod._result:
                if len(pod._result) == 1:
                    return Player.EResult.WIN
                return Player.EResult.DRAW
            else:
                return Player.EResult.LOSS

        return Player.EResult.PENDING

    def rating(self, tour_round: Round | None) -> float:
        """Calculates the player's rating (win percentage) up to a specific round.

        Args:
            tour_round: The round up to which to calculate rating. If None, uses current round.

        Returns:
            float: The rating as a decimal (0.0 to 1.0).
        """
        if tour_round is None:
            return 0
        return self.tour.rating(self, tour_round)

    def pods(self, tour_round: Round | None = None) -> list[Pod | Player.ELocation]:
        """Retrieves all pods or locations the player has been assigned to.

        Args:
            tour_round: The round up to which to include pods.

        Returns:
            list[Pod|Player.ELocation]: A list of pods or location markers (e.g. BYE).
        """
        if tour_round is None:
            tour_round = self.tour.tour_round
        pods: list[Pod | Player.ELocation] = [None for _ in range(tour_round.seq + 1)]  # type: ignore
        tour_rounds = self.tour.rounds

        for i, itr_round in enumerate(tour_rounds):
            pod = itr_round.get_location(self)
            if pod == None:
                pods[i] = itr_round.get_location_type(self)
            else:
                pods[i] = pod
            if itr_round == tour_round:
                break
        return pods

    def played(self, tour_round: Round | None = None) -> list[Player]:
        """Retrieves a list of unique opponents played against in completed pods.

        Args:
            tour_round: The round up to which to check.

        Returns:
            list[Player]: A list of unique opponents.
        """
        if tour_round is None:
            tour_round = self.tour.tour_round
        players = set()
        for p in self.pods(tour_round):
            if isinstance(p, Pod) and p.done:
                players.update(p.players)
        if players:
            players.discard(self)
        return list(players)

    def games(self, tour_round: Round | None = None):
        """Retrieves all completed games (pods) excluding byes and other non-game locations.

        Args:
            tour_round: The round up to which to check.

        Returns:
            list[Pod]: A list of actual completed game pods.
        """
        if tour_round is None:
            tour_round = self.tour.tour_round
        return [p for p in self.pods(tour_round) if isinstance(p, Pod) and p.done]

    def byes(self, tour_round: Round | None = None):
        """Counts the number of byes received.

        Args:
            tour_round: The round up to which to count.

        Returns:
            int: The number of byes.
        """
        if tour_round is None:
            tour_round = self.tour.tour_round
        return len([p for p in self.record(tour_round) if p is Player.EResult.BYE])

    def wins(self, tour_round: Round | None = None):
        """Counts the number of wins.

        Args:
            tour_round: The round up to which to count.

        Returns:
            int: The number of wins.
        """
        if tour_round is None:
            tour_round = self.tour.tour_round
        return len(
            [
                p
                for p in self.games(tour_round)
                if p.result_type == Pod.EResult.WIN and self.uid in p._result
            ]
        )

    def record(self, tour_round: Round | None = None) -> list[Player.EResult]:
        """Retrieves the full history of results.

        Args:
            tour_round: The round up to which to retrieve the record.

        Returns:
            list[Player.EResult]: A chronological list of results.
        """
        seq = list()
        if tour_round is None:
            tour_round = self.tour.tour_round
        pods: list[Pod | Player.ELocation] = []
        for i, p in enumerate(self.pods(tour_round)):
            # if i < tour_round.seq:
            pods.append(p)
        for pod in pods:
            if pod == Player.ELocation.BYE:
                seq.append(Player.EResult.BYE)
            elif pod == Player.ELocation.GAME_LOSS:
                seq.append(Player.EResult.LOSS)
            elif isinstance(pod, Pod):
                if pod.result_type != Pod.EResult.PENDING:
                    if pod.result_type == Pod.EResult.WIN and self.uid in pod._result:
                        seq.append(Player.EResult.WIN)
                    elif (
                        pod.result_type == Pod.EResult.DRAW and self.uid in pod._result
                    ):
                        seq.append(Player.EResult.DRAW)
                    else:
                        seq.append(Player.EResult.LOSS)
                else:
                    seq.append(Player.EResult.PENDING)
        return seq

    def seat_history(self, tour_round: Round | None = None) -> str:
        """Generates a string representation of the player's seat history.

        Format: "seat/pod_size" for each round.

        Args:
            tour_round: The round up to which to generate history.

        Returns:
            str: The seat history string.
        """
        if tour_round is None:
            tour_round = self.tour.tour_round
        pods = self.pods(tour_round)
        if sum([1 for p in pods if isinstance(p, Pod) and p.done]) == 0:
            return "N/A"
        ret_str = " ".join(
            [
                "{}/{}".format(
                    ([x.uid for x in p.players]).index(self.uid) + 1, len(p.players)
                )
                if isinstance(p, Pod)
                else "N/A"
                for p in pods
            ]
        )
        return ret_str

    def pointrate(self, tour_round: Round | None = None):
        """Calculates the point rate (actual points / maximum possible points).

        Args:
            tour_round: The round up to which to calculate.

        Returns:
            float: The point rate.
        """
        if len(self.games(tour_round)) == 0:
            return 0
        if tour_round is None:
            tour_round = self.tour.tour_round
        return self.rating(tour_round) / (
            self.tour.config.win_points * (tour_round.seq + 1)
        )

    def location(self, tour_round: Round | None = None) -> Player.ELocation:
        if tour_round is None:
            tour_round = self.tour.tour_round
        return tour_round.get_location_type(self)

    def players_beaten(self, tour_round: Round | None = None) -> list[Player]:
        if tour_round is None:
            tour_round = self.tour.tour_round
        players = set()
        games = self.games(tour_round)
        for pod in games:
            if pod.result_type == Pod.EResult.WIN and self.uid in pod._result:
                players.update(pod.players)

        players.discard(self)
        return list(players)

    def average_seat(self, rounds: list[Round]) -> np.float64:
        """
        Expressed in percentage.
        In a 4 pod game:
            seat 0: 100%
            seat 1: 66.66%
            seat 2: 33.33%
            seat 3: 0%
        In a 3 pod game:
            seat 0: 100%
            seat 1: 50%
            seat 2: 0%

        Higher percentage means better seats, statistically.
        In subsequent matching attempts, these will get lower priority on early seats.

        We are now using a weighted average of all the pods the player has been in.
        Weights are based on TC.global_wr_seats
        """
        pods = [
            self.pod(round)
            for round in rounds
            if self.pod(round) is not None and self.pod(round).done
        ]
        if not pods:
            return np.float64(0.5)
        n_pods = len([p for p in pods if isinstance(p, Pod)])
        if n_pods == 0:
            return np.float64(0.5)
        score = 0
        for pod in pods:
            if isinstance(pod, Pod):
                index = ([x.uid for x in pod.players]).index(self.uid)
                if index == 0:
                    score += 1
                elif index == len(pod) - 1:
                    continue
                else:
                    rates = self.tour.config.global_wr_seats[0 : len(pod)]
                    norm_scale = 1 - (np.cumsum(rates) - rates[0]) / (
                        np.sum(rates) - rates[0]
                    )
                    score += norm_scale[index]
        return np.float64(score / n_pods)

    def standing(
        self, tour_round: Round | None = None, standings: list[Player] | None = None
    ) -> int:
        if tour_round is None:
            tour_round = self.tour.tour_round
        if standings is None:
            standings = self.tour.get_standings(tour_round)
        if self not in standings:
            return -1
        return standings.index(self) + 1

    def not_played(self, tour_round: Round | None = None) -> list[Player]:
        if tour_round is None:
            tour_round = self.tour.tour_round
        return list(
            set(self.tour.tour_round.active_players) - set(self.played(tour_round))
        )

    def opponent_pointrate(self, tour_round: Round | None = None):
        if not self.played(tour_round):
            return 0
        oppwr = [opp.pointrate(tour_round) for opp in self.played(tour_round)]
        return sum(oppwr) / len(oppwr)

    # PROPERTIES

    @property
    def CACHE(self) -> dict[UUID, Player]:
        return self.tour.PLAYER_CACHE

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @tour.setter
    def tour(self, tour: Tournament):
        self._tour = tour.uid

    # STATICMETHOD

    @staticmethod
    def get(tour: Tournament, uid: UUID):
        return tour.PLAYER_CACHE[uid]

    def seated(self, tour_round: Round | None = None) -> bool:
        if tour_round is None:
            tour_round = self.tour.tour_round
        return tour_round.get_location(self) is not None

    @staticmethod
    def fmt_record(record: list[Player.EResult]) -> str:
        return "".join(
            [
                {
                    Player.EResult.WIN: "W",
                    Player.EResult.LOSS: "L",
                    Player.EResult.DRAW: "D",
                    Player.EResult.BYE: "B",
                    Player.EResult.PENDING: "_",
                }[r]
                for r in record
            ]
        )

    def __gt__(
        self,
        other: Player,
        tour_round: Round | None = None,
        context: TournamentContext | None = None,
    ) -> bool:
        b = False
        if self.SORT_METHOD == SortMethod.ID:
            b = self.uid > other.uid
        elif self.SORT_METHOD == SortMethod.NAME:
            b = self.name > other.name
        elif self.SORT_METHOD == SortMethod.RANK:
            if tour_round is None:
                tour_round = self.tour.tour_round
            if context is None:
                context = TournamentContext(
                    self.tour, tour_round, self.tour.get_standings(tour_round)
                )
            # Use index in standings to match the exact order provided in context.standings
            # standings[0] is best player, standings[-1] is worst player
            self_idx = context.standings.index(self)
            other_idx = context.standings.index(other)

            b = self_idx > other_idx

        return bool(b)

    def __lt__(
        self,
        other: Player,
        tour_round: Round | None = None,
        context: TournamentContext | None = None,
    ) -> bool:
        b = False
        if self.SORT_METHOD == SortMethod.ID:
            b = self.uid < other.uid
        elif self.SORT_METHOD == SortMethod.NAME:
            b = self.name < other.name
        elif self.SORT_METHOD == SortMethod.RANK:
            if tour_round is None:
                tour_round = self.tour.tour_round
            if context is None:
                context = TournamentContext(
                    self.tour, tour_round, self.tour.get_standings(tour_round)
                )
            # Use index in standings to match the exact order provided in context.standings
            # standings[0] is best player, standings[-1] is worst player
            self_idx = context.standings.index(self)
            other_idx = context.standings.index(other)

            b = self_idx < other_idx

        return bool(b)

    @override
    def __repr__(self, tokens=None, context: TournamentContext | None = None):
        if len(self.tour.tour_round.active_players) == 0:
            return ""
        if not tokens:
            tokens = self.FORMATTING
        parser_player = argparse.ArgumentParser()

        parser_player.add_argument(
            "-n", "--standi[n]g", dest="standing", action="store_true"
        )
        parser_player.add_argument("-i", "--id", dest="id", action="store_true")
        parser_player.add_argument("-w", "--win", dest="w", action="store_true")
        parser_player.add_argument(
            "-o", "--opponentwin", dest="ow", action="store_true"
        )
        parser_player.add_argument("-p", "--points", dest="p", action="store_true")
        parser_player.add_argument("-a", "--winr[a]te", dest="wr", action="store_true")
        parser_player.add_argument("-u", "--unique", dest="u", action="store_true")
        parser_player.add_argument(
            "-s", "--average_seat", dest="s", action="store_true"
        )
        parser_player.add_argument("-l", "--pod", dest="pod", action="store_true")
        parser_player.add_argument(
            "-r",
            "--round",
            dest="round",
            type=int,
            default=self.tour.tour_round.seq if self.tour.tour_round else 0,
        )
        """parser_player.add_argument(
            '-s', '--spaces',
            dest='spaces', type=int, default=0)"""
        # parser.add_argument('-n', '--notplayed',    dest='np', action='store_true')

        args, _ = parser_player.parse_known_args(tokens)

        fields = list()

        tsize = int(
            math.floor(math.log10(len(self.tour.tour_round.active_players))) + 1
        )
        pname_size = max([len(p.name) for p in self.tour.tour_round.active_players])

        tour_round = self.tour.rounds[args.round]
        if context is None:
            standings = self.tour.get_standings(tour_round)
        else:
            standings = context.standings
        if args.standing:
            fields.append("#{:>{}}".format(self.standing(tour_round, standings), tsize))
        if args.id:
            fields.append(
                "[{:>{}}] {}".format(self.uid, tsize, self.name.ljust(pname_size))
            )
        else:
            fields.append(self.name.ljust(pname_size))

        if args.pod and len(tour_round.pods) > 0:
            max_pod_id = max([len(str(p.table)) for p in tour_round.pods])
            pod = self.pod(tour_round)
            player_result = self.result(self.tour.tour_round)
            if self in self.tour.tour_round.dropped_players:
                fields.append("Drop".ljust(max_pod_id + 4))
            elif pod:
                # find number of digits in max pod id
                fields.append(
                    "{}".format(
                        f"P{str(pod.table).zfill(max_pod_id)}/S{pod.players.index(self)}"
                        if pod
                        else ""
                    )
                )
            elif player_result == Player.EResult.LOSS:
                fields.append("Loss".ljust(max_pod_id + 4))
            elif player_result == Player.EResult.BYE:
                fields.append("Bye".ljust(max_pod_id + 4))
            else:
                fields.append("".ljust(max_pod_id + 4))
        if args.p:
            fields.append("rating: {}".format(self.rating(self.tour.tour_round)))
        if args.w:
            fields.append("w: {}".format(self.wins(tour_round)))
        if args.ow:
            fields.append("o.wr.: {:.2f}".format(self.opponent_pointrate))
        if args.u:
            fields.append("uniq: {}".format(self.played))
        if args.s:
            fields.append(
                "seat: {:02.00f}%".format(
                    self.average_seat(
                        [r for r in self.tour.rounds if r.seq <= args.round]
                    )
                    * 100
                )
            )
        # if args.np:
        #    fields.append(''.format())
        # OUTPUT_BUFFER.append('\t{}'.format(' | '.join(fields)))

        return " | ".join(fields)

    def serialize(self) -> dict[str, Any]:
        return {
            #'tour': str(self._tour),
            "uid": str(self.uid),
            "name": self.name,
            "decklist": self.decklist,
        }

    @classmethod
    def inflate(cls, tour: Tournament, data: dict[str, Any]) -> Player:
        # assert tour.uid == UUID(data['tour'])
        return cls(tour, data["name"], UUID(data["uid"]), data.get("decklist"))


class Pod(IPod):
    """
    Represents a single pod (table) in a round.

    Attributes:
        table (int): The table number.
        cap (int): The capacity of the pod (maximum number of players).
        _players (list[UUID]): List of player UUIDs in the pod.
    """

    def __init__(self, tour_round: Round, table: int, cap=0, uid: UUID | None = None):
        """Initializes a new Pod instance.

        Args:
            tour_round: The round the pod belongs to.
            table: The table number.
            cap: The player capacity of the pod.
            uid: Optional UUID.
        """
        self._tour: UUID = tour_round.tour.uid
        self._round: UUID = tour_round.uid
        super().__init__(uid=uid)
        self.cap: int = cap
        self._players: list[UUID] = list()
        # self._players: list[UUID] = list() #TODO: make references to players
        # self.discord_message_id: None|int = None

    @property
    def table(self) -> int:
        """Returns the table number of the pod.
        The table number is determined by the pod's index in the round's pod list (+1).
        """
        try:
            return self.tour_round._pods.index(self.uid) + 1
        except ValueError:
            return -1

    @property
    def CACHE(self) -> dict[UUID, Pod]:
        return self.tour.POD_CACHE

    @staticmethod
    def get(tour: Tournament, uid: UUID) -> Pod:
        return tour.POD_CACHE[uid]

    def set_result(self, player: Player, result: IPlayer.EResult):
        if player.uid not in self._players:
            raise ValueError("Player {} not in pod {}".format(player.name, self.name))
        if result == IPlayer.EResult.WIN:
            self._result.clear()
        self._result.add(player.uid)

    def remove_result(self, player: Player):
        if player.uid in self._result:
            self._result.remove(player.uid)

    def reset_result(self):
        self._result.clear()

    @property
    def result(self) -> set[Player]:
        """Retrieves the players involved in the result of the pod (e.g., winners or drawers).

        Returns:
            set[Player]: A set of players.
        """
        return {Player.get(self.tour, x) for x in self._result}

    @property
    def result_type(self) -> Pod.EResult:
        if self._result:
            if len(self._result) == 1:
                return Pod.EResult.WIN
            return Pod.EResult.DRAW
        return Pod.EResult.PENDING

    @property
    def done(self) -> bool:
        return len(self._result) > 0

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @tour.setter
    def tour(self, tour: Tournament):
        self._tour = tour.uid

    @property
    def tour_round(self) -> Round:
        return Round.get(self.tour, self._round)

    @property
    def players(self) -> list[Player]:
        """Returns the list of players in the pod."""
        return [Player.get(self.tour, x) for x in self._players]

    @override
    def add_player(self, player: Player, manual=False, player_pod_map=None) -> bool:
        """Adds a player to the pod.

        Args:
            player: The player to add.
            manual: If True, allows exceeding the pod's capacity.
            player_pod_map: Optional map to update player locations (internal use).

        Returns:
            bool: True if the player was added, False otherwise (e.g., if full and not manual).
        """
        if len(self) >= self.cap and self.cap and not manual:
            return False
        if pod := player.pod(self.tour_round):
            pod.remove_player(player)
        self.tour_round._byes.discard(player.uid)
        self._players.append(player.uid)
        self.tour_round.player_locations_map[player.uid] = self
        # player.location = Player.ELocation.SEATED
        # player.pod = self  # Update player's pod reference
        return True

    def remove_player(self, player: Player, cleanup=True) -> Player | None:
        try:
            idx = self._players.index(player.uid)
        except ValueError:
            return None
        p = Player.get(self.tour, self._players.pop(idx))

        self.remove_result(player)
        del self.tour_round.player_locations_map[player.uid]
        # player.location = Player.ELocation.UNASSIGNED
        # player.pod = None  # Clear player's pod reference
        if len(self) == 0 and cleanup:
            self.tour_round.remove_pod(self)
        return player

    @property
    def average_seat(self) -> float:
        return np.average(
            [p.average_seat(self.tour.ended_rounds) for p in self.players]
        ).astype(float)

    @property
    def balance(self) -> np.ndarray:
        """
        Returns a list of count of players above 50% average seat and below 50% average seat
        """
        return np.array(
            [
                sum(
                    [1 for p in self.players if p.average_seat(self.tour.rounds) > 0.5]
                ),
                sum(
                    [1 for p in self.players if p.average_seat(self.tour.rounds) < 0.5]
                ),
            ]
        )

    @override
    def auto_auto_assign_seats(self):
        """Assigns seats to players in the pod.

        Seat assignment attempts to balance seating positions based on players' history,
        giving preference to players who have had poor seat variance in the past.
        """
        # Average seating positions
        average_positions = [
            p.average_seat(self.tour.ended_rounds) for p in self.players
        ]
        n = len(average_positions)

        if not any(average_positions):
            random.shuffle(self.players)
            return

        # partially sort players based on seating positions
        # those that have same average_seat should be randomly ordered
        seat_assignment = [0] * n
        for i in range(n):
            seat_assignment[i] = (
                sum([1 for x in average_positions if x < average_positions[i]]) + 1
            )
        # randomize players with same average_seat
        seat_assignment = [x + random.random() for x in seat_assignment]
        # sort players based on seat assignment
        self._players[:] = np.take(self._players, np.argsort(seat_assignment))

        pass

    def reorder_players(self, order: list[int]) -> None:
        """Reorders the players in the pod.

        Args:
            order: A list of integers representing the new order of the players.
        """
        if len(order) != len(self._players):
            raise ValueError("Order must have the same length as the number of players")
        if any([x not in range(len(self._players)) for x in order]):
            raise ValueError("Order must contain all integers from 0 to n-1")
        if len(set(order)) != len(order):
            raise ValueError("Order must not contain duplicate integers")

        self._players[:] = np.take(self._players, order)

    def clear(self):
        players = [p for p in self.players]
        for p in players:
            self.remove_player(p, cleanup=False)
        # self.players.clear()

    @property
    def name(self):
        return "Pod {}".format(self.table)

    @override
    def __repr__(self, context: TournamentContext | None = None):
        if not self.players:
            maxlen = 0
        else:
            maxlen = max([len(p.name) for p in self.players])
        ret = "Pod {} with {}/{} players:\n\t{}".format(
            self.table,
            len(self),
            self.cap,
            "\n\t".join(
                [
                    "[{}] {}\t".format(
                        " "
                        if not self._result
                        else "W"
                        if self.result_type == Pod.EResult.WIN and p.uid in self._result
                        else "D"
                        if self.result_type == Pod.EResult.DRAW
                        and p.uid in self._result
                        else "L",
                        p.__repr__(["-s", str(maxlen), "-p"], context=context),
                    )
                    for _, p in zip(range(1, len(self) + 1), self.players)
                ]
            ),
        )
        return ret

    def serialize(self) -> dict[str, Any]:
        return {
            "uid": str(self.uid),
            "tour_round": str(self._round),
            "table": self.table,
            "cap": self.cap,
            "result": sorted([str(p) for p in self._result]),
            "players": [str(p) for p in self._players],
        }

    @classmethod
    def inflate(cls, tour_round: Round, data: dict[str, Any]) -> Pod:
        assert tour_round.uid == UUID(data["tour_round"])
        pod = cls(tour_round, data["table"], data["cap"], UUID(data["uid"]))
        pod._players = [UUID(x) for x in data["players"]]
        pod._result = {UUID(x) for x in data["result"]}
        return pod


class Round(IRound):
    """
    Represents a single round in the tournament.

    Attributes:
        seq (int): The sequence number of the round (0-indexed).
        stage (Stage): The stage of the round (Swiss, Top X).
        logic (IPairingLogic): The pairing logic used for this round.
        CACHE (dict[UUID, Round]): Global cache of round instances.
    """

    class Stage(Enum):
        SWISS = 0
        TOP_4 = 4
        TOP_7 = 7
        TOP_10 = 10
        TOP_13 = 13
        TOP_16 = 16
        TOP_40 = 40

        @staticmethod
        def is_playoff(stage: Stage) -> bool:
            return stage in [
                Round.Stage.TOP_4,
                Round.Stage.TOP_7,
                Round.Stage.TOP_10,
                Round.Stage.TOP_13,
                Round.Stage.TOP_16,
                Round.Stage.TOP_40,
            ]

    def __init__(
        self,
        # Required
        tour: Tournament,
        seq: int,
        stage: Stage,
        pairing_logic: IPairingLogic,
        # Optional
        uid: UUID | None = None,
        dropped: set[UUID] | None = None,
        disabled: set[UUID] | None = None,
        byes: set[UUID] | None = None,
        game_loss: set[UUID] | None = None,
    ):
        """Initializes a new Round instance.

        Args:
            tour: The tournament the round belongs to.
            seq: The sequence number of the round.
            stage: The stage of the round (e.g., Swiss, Top 4).
            pairing_logic: The logic used for pairing players in this round.
            uid: Optional UUID.
            dropped: Optional set of dropped player UUIDs.
            disabled: Optional set of disabled player UUIDs.
            byes: Optional set of player UUIDs who have byes.
            game_loss: Optional set of player UUIDs who have game losses.
        """
        self._tour: UUID = tour.uid
        super().__init__(uid=uid)
        self.tour.ROUND_CACHE[self.uid] = self
        self.seq: int = seq
        self.stage: Round.Stage = stage

        self._logic = pairing_logic.name
        self._byes: set[UUID] = set() if byes is None else byes
        self._game_loss: set[UUID] = set() if game_loss is None else game_loss
        self._dropped: set[UUID] = set() if dropped is None else dropped
        self._disabled: set[UUID] = set() if disabled is None else disabled

        self.player_locations_map: dict[UUID, Pod] = {}

    def refresh_player_location_map(self):
        self.player_locations_map.clear()
        for pod in self.pods:
            for player in pod.players:
                self.player_locations_map[player.uid] = pod

    def get_location(self, player: Player) -> Pod | None:
        return self.player_locations_map.get(player.uid, None)

    def get_location_type(self, player: Player) -> Player.ELocation:
        if player.uid in self._game_loss:
            return Player.ELocation.GAME_LOSS
        if player.uid in self._byes:
            return Player.ELocation.BYE
        if player.uid in self.player_locations_map:
            return Player.ELocation.SEATED
        return Player.ELocation.UNASSIGNED

    @property
    def byes(self) -> set[Player]:
        return set(Player.get(self.tour, x) for x in self._byes)

    @property
    def game_loss(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._game_loss]

    @property
    def CACHE(self) -> dict[UUID, Round]:
        return self.tour.ROUND_CACHE

    @staticmethod
    def get(tour: Tournament, uid: UUID) -> Round:
        return tour.ROUND_CACHE[uid]

    @property
    def logic(self) -> IPairingLogic:
        return self.tour.get_pairing_logic(self._logic)

    @logic.setter
    def logic(self, logic: IPairingLogic):
        self._logic = logic.name

    @property
    def players(self) -> set[Player]:
        return self.tour.players

    @property
    def active_players(self) -> set[Player]:
        return {
            Player.get(self.tour, x)
            for x in self.tour._players - self._dropped - self._disabled
        }

    @property
    def dropped_players(self) -> set[Player]:
        return {Player.get(self.tour, x) for x in self._dropped}

    @property
    def disabled_players(self) -> set[Player]:
        return {Player.get(self.tour, x) for x in self._disabled}

    @property
    def pods(self) -> list[Pod]:
        return [Pod.get(self.tour, x) for x in self._pods]

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @property
    def done(self):
        """Checks if the round is completed.

        A round is considered done if all pods within it have reported results
        (i.e., no pending pods remain).

        Returns:
            bool: True if the round is done, False otherwise.
        """
        if len(self._pods) == 0:
            return False
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    @property
    def all_players_assigned(self):
        """Checks if all active players are assigned to pods.

        Returns:
            bool: True if all active players are assigned to pods, False otherwise.
        """
        seated = len(self.seated)
        n_players_to_play = seated + len(self.unassigned)
        if n_players_to_play == 0:
            return True
        if self.tour.get_pod_sizes(n_players_to_play) is None:
            return False
        if not (pod_sizes := self.tour.get_pod_sizes(n_players_to_play)):
            return True
        else:
            return sum(pod_sizes) == seated

    @property
    def seated(self) -> set[Player]:
        """Returns the set of players who are currently assigned to pods.

        Returns:
            set[Player]: A set of Player instances that are assigned to pods.
        """
        return {p for p in self.active_players if p.pod(self)}

    @property
    def unassigned(self) -> set[Player]:
        """Returns the set of players who are not currently assigned to pods.

        Returns:
            set[Player]: A set of Player instances that are not assigned to pods.
        """
        seated_player_uids = set()
        for pod in self.pods:
            seated_player_uids.update(pod._players)

        return set(
            Player.get(self.tour, x)
            for x in (
                self.tour._players
                - self._dropped
                - self._disabled
                - self._byes
                - self._game_loss
                - seated_player_uids
            )
        )

    def advancing_players(self, standings) -> list[Player]:
        """Determines which players advance to the next round.

        This is primarily used for transitions from Swiss to Top Cut, or between Top Cut rounds.

        Args:
           standings: A list of players sorted by their current standing.

        Returns:
            list[Player]: The list of players who advance.
                - For Swiss rounds, typically all active players return.
                - For Top Cut rounds, only winners (and potentially high-seeded byes) advance.
        """
        # Create index map for O(1) standings lookup instead of O(n) index() calls
        standings_index = {player: idx for idx, player in enumerate(standings)}

        if self.stage == Round.Stage.SWISS:
            return sorted(
                self.active_players,
                key=lambda x: standings_index.get(x, len(standings)),
            )
        else:
            active_players_set = self.active_players  # Cache set lookup

            # Collect players in three groups to maintain proper ordering:
            # 1. Byes (sorted by standings)
            # 2. Wins (sorted by standings)
            # 3. Draws (one per draw pod, sorted by standings)
            bye_players: list[Player] = []
            win_players: list[Player] = []
            draw_players: list[Player] = []
            processed_draw_pods: set[Pod] = (
                set()
            )  # Track processed draw pods to avoid duplicates

            for p in standings:
                if p not in active_players_set:
                    continue

                # Handle byes
                if p in self.byes:
                    bye_players.append(p)
                    continue

                # Get pod once per player
                pod = p.pod(self)
                if pod is None:
                    continue

                # Handle WIN results
                if pod.done and pod.result_type == Pod.EResult.WIN:
                    if p in pod.result:
                        win_players.append(p)

                # Handle DRAW results (only process once per pod)
                elif (
                    pod.result_type == Pod.EResult.DRAW
                    and pod not in processed_draw_pods
                ):
                    processed_draw_pods.add(pod)
                    if pod.result:
                        # Filter to only active players in the draw result
                        active_in_draw = [
                            p for p in pod.result if p in active_players_set
                        ]
                        if active_in_draw:
                            advancing_player = min(
                                active_in_draw,
                                key=lambda x: standings_index.get(x, len(standings)),
                            )
                            draw_players.append(advancing_player)

            # Sort each group by standings and concatenate in order
            bye_players.sort(key=lambda x: standings_index.get(x, len(standings)))
            win_players.sort(key=lambda x: standings_index.get(x, len(standings)))
            draw_players.sort(key=lambda x: standings_index.get(x, len(standings)))

            return bye_players + win_players + draw_players

    def repeat_pairings(self):
        prev_round = self.tour.rounds[-2]
        data = {}
        for pod in self.pods:
            # data[pod] = {
            #    sum(
            #        sum([
            #            0.5
            #            for p2 in pod.players
            #            if p2 in p1.played(prev_round)
            #        ])
            #    for p1 in pod.players)
            # }
            data[pod] = 0
            for i in range(len(pod.players) - 1):
                for j in range(i + 1, len(pod.players)):
                    p1 = pod.players[i]
                    p2 = pod.players[j]

                    if p2 in p1.played(prev_round):
                        # data[pod][p1].add(p2)
                        data[pod] += 1
        pass

        return data

    def reset_pods(self) -> bool:
        """Resets all pods in the round, clearing their assignments.

        This method removes all players from all pods and clears the bye list.
        It is useful for resetting the round before creating new pairings.

        Returns:
            bool: Always returns True, as the reset is always successful.
        """
        pods = [Pod.get(self.tour, x) for x in self._pods]
        # if any([not pod.done for pod in pods]):
        #    return False
        self._byes.clear()
        for pod in pods:
            self.remove_pod(pod)
        return True

    def remove_pod(self, pod: Pod) -> bool:
        """Removes a pod from the round, clearing its assignments.

        Args:
            pod: The pod to remove.

        Returns:
            bool: True if the pod was successfully removed, False otherwise.
        """
        # if not pod.done:
        pod.clear()
        self._pods.remove(pod.uid)
        return True
        # return False

    def create_pods(self) -> None:
        """Creates empty pod slots for the round.

        This method calculates the number and size of pods required based on the number of
        active players and the tournament configuration, then initializes these pods.
        """
        seats_required = len(self.unassigned) - sum(
            [pod.cap - len(pod) for pod in self.pods if not pod.done]
        )
        if seats_required == 0:
            return
        pod_sizes = self.tour.get_pod_sizes(seats_required)
        if pod_sizes is None:
            Log.log("Can not make pods.", level=Log.Level.WARNING)
            return None
        start_table = len(self._pods) + 1
        for i, size in enumerate(pod_sizes):
            pod = Pod(self, start_table + i, cap=size)
            self._pods.append(pod.uid)

    def disable_topcut(self, standings: list[Player]):
        """Disable players who don't advance to top cut.
        They remain in the tournament but won't participate in top cut rounds."""
        standings = self.tour.get_standings(self.tour.previous_round(self))

        # Disable players from bottom of standings until we reach top_cut size
        for p in standings[self.stage.value : :]:
            self.disable_player(p, set_disabled=True)

    def create_pairings(self) -> None:
        """Executes the pairing logic to assign players to pods.

        This method uses the round's `pairing_logic` to determine match-ups and assigns
        players to the pods created by `create_pods`.
        """
        if self.stage != Round.Stage.SWISS:
            standings = self.tour.get_standings(self.tour.previous_round(self))
            self.disable_topcut(standings)
            if self.stage in [
                Round.Stage.TOP_7,
                Round.Stage.TOP_10,
                Round.Stage.TOP_13,
                Round.Stage.TOP_16,
                Round.Stage.TOP_40,
            ]:
                self.logic.advance_topcut(self, cast(list[IPlayer], standings))

        self.create_pods()
        pods = [p for p in self.pods if all([not p.done, len(p) < p.cap])]

        self.logic.make_pairings(self, cast(set[IPlayer], self.unassigned), pods)

        if self.seq < self.tour.config.n_rounds:
            for pod in self.pods:
                pod.auto_auto_assign_seats()

        self.sort_pods()
        self.apply_table_preference()

    def sort_pods(self) -> None:
        """Sort pods by number of players and average rating."""

        pods_sorted = sorted(
            self.pods,
            key=lambda x: (
                len(x.players),
                np.average([p.rating(self) for p in x.players]),
            ),
            reverse=True,
        )
        self._pods[:] = [pod.uid for pod in pods_sorted]

    def apply_table_preference(self) -> bool:
        """Try to apply table preferences for players. Pod index is table number.
        Preserves the relative power-sorted order of non-locked pods.
        Prioritizes maximizing the number of satisfied preferences.

        Returns:
            bool: True if table preferences were applied, False otherwise.
        """
        pods = self.pods  # Already sorted by power
        n = len(pods)
        if n == 0:
            return False

        result_pods = [None] * n
        assigned_pod_uids = set()
        any_swapped = False

        # Pass 1: Handle Locked Pods (Best effort satisfaction)
        # We want to satisfy as many preferences as possible.
        # Primary priority: Number of players satisfied in that pod.
        # Secondary priority: Original power order (tie-breaker).

        possibilities = []
        for rank, pod in enumerate(pods):
            # Aggregated preferences for players in this pod (1-indexed)
            counts = {}
            for p in pod.players:
                for pref in p.table_preference:
                    target_idx = pref - 1  # Convert to 0-indexed
                    if 0 <= target_idx < n:
                        counts[target_idx] = counts.get(target_idx, 0) + 1
                    else:
                        Log.log(
                            f"Player {p.name} has invalid table preference: {pref}. Max table index is {n}",
                            level=Log.Level.WARNING,
                        )

            for target_idx, count in counts.items():
                # Sort key: (-count, rank, target_idx)
                # (Higher count first, then higher power first, then lower index first)
                possibilities.append((-count, rank, pod, target_idx))

        possibilities.sort()

        for _, _, pod, target_idx in possibilities:
            if pod.uid not in assigned_pod_uids and result_pods[target_idx] is None:
                result_pods[target_idx] = pod
                assigned_pod_uids.add(pod.uid)
                any_swapped = True

        # Pass 2: Fill gaps with remaining pods (Preserving relative power order)
        pod_iter = iter(pods)
        for i in range(n):
            if result_pods[i] is None:
                # Find the next pod that hasn't been assigned yet
                try:
                    while True:
                        next_pod = next(pod_iter)
                        if next_pod.uid not in assigned_pod_uids:
                            result_pods[i] = next_pod
                            break
                except StopIteration:
                    break  # Should not happen

        # Update the underlying UUID list
        self._pods[:] = [p.uid for p in result_pods if p is not None]

        return any_swapped

    def set_result(self, player: Player, result: IPlayer.EResult) -> None:
        if result == IPlayer.EResult.BYE:
            self._byes.add(player.uid)
        else:
            self._byes.discard(player.uid)

        if result == IPlayer.EResult.LOSS:
            self._game_loss.add(player.uid)
        else:
            self._game_loss.discard(player.uid)

        if result == IPlayer.EResult.WIN:
            if pod := player.pod(self):
                pod.set_result(player, result)
            else:
                raise ValueError("Player {} not in any pod".format(player.name))
        elif result == IPlayer.EResult.DRAW:
            if pod := player.pod(self):
                pod.set_result(player, result)
            else:
                raise ValueError("Player {} not in any pod".format(player.name))

    def remove_result(self, player: Player):
        if pod := player.pod(self):
            pod.remove_result(player)

    def drop_player(self, player: Player):
        if self.done and self != self.tour.last_round:
            raise ValueError("Can't drop player in a completed round.")
        if (pod := self.get_location(player)) is not None:
            if pod.done and self != self.tour.last_round:
                raise ValueError(
                    "Can't drop player in a completed pod in a previous round."
                )
            # pod.remove_player(player)
        self._dropped.add(player.uid)

    def disable_player(self, player: Player, set_disabled: bool = True):
        if self.done and self != self.tour.last_round:
            raise ValueError("Can't disable player in a completed round.")
        if (pod := self.get_location(player)) is not None:
            if pod.done and self != self.tour.last_round:
                raise ValueError(
                    "Can't disable player in a completed pod in a previous round."
                )
            # pod.remove_player(player)
        if set_disabled:
            self._disabled.add(player.uid)
        else:
            self._disabled.discard(player.uid)

    def serialize(self) -> dict[str, Any]:
        return {
            "tour": str(self._tour),
            "seq": self.seq,
            "stage": self.stage.value,
            "logic": self._logic,
            "uid": str(self.uid),
            "dropped": [str(p) for p in self._dropped],
            "disabled": [str(p) for p in self._disabled],
            "byes": [str(p) for p in self._byes],
            "game_loss": [str(p) for p in self._game_loss],
            "pods": [pod.serialize() for pod in self.pods],
        }

    @classmethod
    def inflate(cls, tour: Tournament, data: dict[str, Any]) -> Round:
        assert tour.uid == UUID(data["tour"])
        tour.discover_pairing_logic()
        stage = Round.Stage(data["stage"])
        logic = tour.get_pairing_logic(data["logic"])

        new_round: Round = cls(
            tour,
            data["seq"],
            stage=stage,
            pairing_logic=logic,
            uid=UUID(data["uid"]),
            dropped={UUID(x) for x in data["dropped"]},
            disabled={UUID(x) for x in data["disabled"]},
            byes={UUID(x) for x in data["byes"]},
            game_loss={UUID(x) for x in data["game_loss"]},
        )
        pods = [Pod.inflate(new_round, pod) for pod in data["pods"]]
        new_round._pods = [pod.uid for pod in pods]
        new_round.refresh_player_location_map()
        return new_round

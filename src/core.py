from __future__ import annotations
import inspect
from typing import List, Sequence, Union, Callable, Any
from typing_extensions import override

import argparse
import math
import os
import pickle
import random
from copy import deepcopy
from datetime import datetime
from enum import Enum

from .discord_engine import DiscordPoster
from .interface import IPlayer, ITournament, IPod, IRound, IPairingLogic, ITournamentConfiguration
from .misc import Json2Obj, generate_player_names, timeit
import numpy as np
from tqdm import tqdm # pyright: ignore
from uuid import UUID, uuid4
import json

from dotenv import load_dotenv
import requests
import threading



# Load configuration from .env file
load_dotenv()

#import sys
#sys.setrecursionlimit(5000)  # Increase recursion limit

class DataExport:
    class Format(Enum):
        PLAIN = 0
        DISCORD = 1
        CSV = 2
        JSON = 3

    class Target(Enum):
        CONSOLE = 0
        FILE = 1
        WEB = 2
        DISCORD = 3


class PodsExport(DataExport):
    @classmethod
    def auto_export(cls, func):
        def auto_pods_export_wrapper(self: Tournament, *original_args, **original_kwargs):
            tour_round = self.tour_round
            ret = func(self, *original_args, **original_kwargs)
            tour_round = tour_round or self.tour_round
            if self.config.auto_export:
                logf = TournamentAction.LOGF
                if logf and tour_round:
                    # Export pods to a file named {tournament_name}_round_{round_number}.txt
                    # And also export it into {log_directory}/pods.txt
                    context = TournamentContext(self, tour_round, self.get_standings(tour_round))
                    export_str: str = '\n\n'.join([
                        pod.__repr__(context=context)
                        for pod in tour_round.pods
                    ])
                    game_lost: list[Player] = [x for x in tour_round.active_players if x.result == Player.EResult.LOSS]
                    byes = [x for x in tour_round.unseated if x.location == Player.ELocation.UNASSIGNED and x.result == Player.EResult.BYE]
                    if len(game_lost) + len(byes) > 0:
                        max_len = max([len(p.name) for p in game_lost + byes])
                        if self.config.allow_bye and byes:
                            export_str += '\n\nByes:\n' + '\n'.join([
                                "\t{} | pts: {}".format(p.name.ljust(max_len), p.rating(tour_round) or '0')
                                for p in tour_round.unseated
                                if p.result == Player.EResult.BYE
                            ])
                        if game_lost:
                            export_str += '\n\nGame losses:\n' + '\n'.join([
                                "\t{} | pts: {}".format(
                                    p.name.ljust(max_len),
                                    p.rating(tour_round) or '0'
                                )
                                for p in game_lost
                            ])

                    path = os.path.join(
                        os.path.dirname(logf),
                        os.path.basename(logf).replace('.json', ''),
                        os.path.basename(logf).replace('.json', '_R{}.txt'.format(tour_round.seq)),
                    )
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))

                    self.export_str(export_str, path, DataExport.Target.FILE)
                    self.export_str(export_str, None, DataExport.Target.WEB)


                    path = os.path.join(os.path.dirname(logf), 'pods.txt')
                    self.export_str(export_str, path, DataExport.Target.FILE)

            return ret
        return auto_pods_export_wrapper


class TournamentContext:
    def __init__(self, tour: Tournament, tour_round: Round, standings: list[Player]):
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
        def __init__(self,
                     label: str,
                     format: str,
                     denom: int|None,
                     description: str,
                     getter: Callable[..., Any]
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
    def _get_rating(player: Player, context: TournamentContext) -> float|None:
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
            label='#',
            format='{:d}',
            denom=None,
            description='Player\'s standing in the tournament.',
            getter=_get_standing
        ),
        Field.ID: Formatting(
            label='ID',
            format='{:s}',
            denom=None,
            description='Player ID',
            getter=_get_id
        ),
        Field.NAME: Formatting(
            label='name',
            format='{:s}',
            denom=None,
            description='Player name',
            getter=_get_name
        ),
        Field.OPP_POINTRATE: Formatting(
            label='opp. win %',
            format='{:.2f}%',
            denom=100,
            description='Opponents\' point rate',
            getter=_get_opp_winrate
        ),
        Field.RATING: Formatting(
            label='pts',
            format='{:d}',
            denom=None,
            description='Player rating',
            getter=_get_rating
        ),
        Field.WINS: Formatting(
            label='# wins',
            format='{:d}',
            denom=None,
            description='Number of games won',
            getter=_get_wins
        ),
        Field.POINTRATE: Formatting(
            label='win %',
            format='{:.2f}%',
            denom=100,
            description='Player\'s point rate',
            getter=_get_winrate
        ),
        Field.UNIQUE: Formatting(
            label='uniq. opp.',
            format='{:d}',
            denom=None,
            description='Number of unique opponents',
            getter=_get_unique_opponents
        ),
        Field.GAMES: Formatting(
            label='# games',
            format='{:d}',
            denom=None,
            description='Number of games played',
            getter=_get_games
        ),
        Field.OPP_BEATEN: Formatting(
            label='# opp. beat',
            format='{:d}',
            denom=None,
            description='Number of opponents beaten',
            getter=_get_opponents_beaten
        ),
        Field.SEAT_HISTORY: Formatting(
            label='seat record',
            format='{:s}',
            denom=None,
            description='Seat record',
            getter=_get_seat_history
        ),
        Field.AVG_SEAT: Formatting(
            label='avg. seat',
            format='{:03.2f}%',
            denom=100,
            description='Average seat',
            getter=_get_avg_seat
        ),
        Field.RECORD: Formatting(
            label='record',
            format='{:s}',
            denom=None,
            description='Player\'s record',
            getter=_get_record
        ),
    }

    ext = {
        DataExport.Format.DISCORD: '.txt',
        DataExport.Format.PLAIN: '.txt',
        DataExport.Format.CSV: '.csv'
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

    def __init__(self, fields=None, format: DataExport.Format = DataExport.Format.PLAIN, dir: Union[str, None] = None):
        if fields is None:
            self.fields = self.DEFAULT_FIELDS
        else:
            self.fields = fields
        self.format = format
        if dir is None:
            self.dir = './logs/standings' + self.ext[self.format]
        else:
            self.dir = dir

    @classmethod
    def auto_export(cls, func):
        def auto_standings_export_wrapper(self: Tournament, *original_args, **original_kwargs):
            ret = func(self, *original_args, **original_kwargs)
            if self.config.auto_export:
                self.export_str(
                    self.get_standings_str(),
                    self.config.standings_export.dir,
                    DataExport.Target.FILE
                )
            return ret
        return auto_standings_export_wrapper

    def serialize(self):
        return {
            'fields': [f.value for f in self.fields],
            'format': self.format.value,
            'dir': self.dir
        }

    @classmethod
    def inflate(cls, data:dict):
        return cls(
            [StandingsExport.Field(f) for f in data['fields']],
            StandingsExport.Format(data['format']),
            data['dir']
        )


class SortMethod(Enum):
    ID = 0
    NAME = 1
    RANK = 2


class SortOrder(Enum):
    ASCENDING = 0
    DESCENDING = 1


class Log:
    class Level(Enum):
        NONE = 0
        INFO = 1
        WARNING = 2
        ERROR = 3

    class LogEntry:
        def __init__(self, msg, level):
            self.msg = msg
            self.level = level

        def short(self):
            if self.level == Log.Level.NONE:
                return ''
            if self.level == Log.Level.INFO:
                return 'I'
            if self.level == Log.Level.WARNING:
                return 'W'
            if self.level == Log.Level.ERROR:
                return 'E'

        @override
        def __repr__(self):
            return '{}> {}'.format(self.short(), self.msg)

    output = []

    PRINT = False
    DISABLE = False

    @classmethod
    def log(cls, str_log, level=Level.NONE):
        if cls.DISABLE:
            return
        entry = Log.LogEntry(str_log, level)
        cls.output.append(entry)
        if cls.PRINT:
            print(entry)

    @classmethod
    def print(cls):
        for entry in cls.output:
            print(entry)

    @classmethod
    def export(cls, fpath):
        try:
            with open(fpath, 'w') as f:
                f.writelines([str(s)+'\n' for s in cls.output])
        except Exception as e:
            cls.log(str(e), level=cls.Level.ERROR)


class TournamentAction:
    '''Serializable action that will be stored in tournament log and can be restored
    '''
    ACTIONS: List = []
    LOGF: bool|str|None = None
    DEFAULT_LOGF = 'logs/default.json'

    def __init__(self, before: dict, ret, after: dict, func_name, *nargs, **kwargs):
        self.before = before
        self.ret = ret
        self.after = after
        self.func_name = func_name
        self.nargs = nargs
        self.kwargs = kwargs
        self.ret = ret
        self.time = datetime.now()

    @classmethod
    def reset(cls) -> None:
        TournamentAction.ACTIONS = []
        TournamentAction.store()

    @classmethod
    def action(cls, func) -> Callable:
        @StandingsExport.auto_export
        @PodsExport.auto_export
        def wrapper(self: Tournament, *original_args, **original_kwargs):
            before = self.serialize()
            ret = func(self, *original_args, **original_kwargs)
            after = self.serialize()
            cls.ACTIONS.append(TournamentAction(
                before, ret, after, func.__name__, *original_args, **original_kwargs,
            ))
            cls.store()
            return ret
        return wrapper

    @classmethod
    def store(cls):
        if cls.LOGF is None:
            cls.LOGF = cls.DEFAULT_LOGF
        if cls.LOGF:
            assert isinstance(cls.LOGF, str)
            if not os.path.exists(os.path.dirname(cls.LOGF)):
                os.makedirs(os.path.dirname(cls.LOGF))
            #with open(cls.LOGF, 'w') as f:
                #pickle.dump(cls.ACTIONS, f)
            if cls.ACTIONS:
                with open(cls.LOGF, 'w') as f:
                    json.dump(cls.ACTIONS[-1].after, f, indent=4)

    @classmethod
    def load(cls, logdir='logs/default.json'):
        if os.path.exists(logdir):
            cls.LOGF = logdir
            try:
                with open(cls.LOGF, 'rb') as f:
                    cls.ACTIONS = pickle.load(f)

                    Tournament.CACHE.clear()
                    for action in cls.ACTIONS:
                        Tournament.CACHE[action.before.uid] = action.before
                        Tournament.CACHE[action.after.uid] = action.after
                if not cls.ACTIONS:
                    return False
                return True
            except Exception as e:
                Log.log(str(e), level=Log.Level.ERROR)
                return False
        return False

    @override
    def __repr__(self, *nargs, **kwargs):
        ret_str = (
            '{}'
            '{}'
            '{}'
        ).format(
            self.func_name,
            '' if not nargs else ', '.join([str(arg) for arg in nargs]),
            '' if not kwargs else ', '.join([
                '{}={}' for _, _ in kwargs.items()
            ])
        )
        return ret_str


class TournamentConfiguration(ITournamentConfiguration):
    def __init__(self, **kwargs):
        self.pod_sizes = kwargs.get('pod_sizes', [4, 3])
        self.allow_bye = kwargs.get('allow_bye', True)
        self.win_points = kwargs.get('win_points', 5)
        self.bye_points = kwargs.get('bye_points', 2)
        self.draw_points = kwargs.get('draw_points', 1)
        self.snake_pods = kwargs.get('snake_pods', True)
        self.n_rounds = kwargs.get('n_rounds', 5)
        self.max_byes = kwargs.get('max_byes', 2)
        self.auto_export = kwargs.get('auto_export', True)
        self.standings_export = kwargs.get('standings_export', StandingsExport())
        self.global_wr_seats = kwargs.get('global_wr_seats', [
            0.2553,
            0.2232,
            0.1847,
            0.1428,
        ])

    @property
    def min_pod_size(self):
        return min(self.pod_sizes)

    @property
    def max_pod_size(self):
        return max(self.pod_sizes)

    @staticmethod
    @override
    def ranking(x:Player, tour_round: Round) -> tuple:
        return (
            x.rating(tour_round),
            len(x.games(tour_round)),
            np.round(x.opponent_pointrate(tour_round), 10),
            len(x.players_beaten(tour_round)),
            -x.average_seat(x.tour.rounds),
            -x.uid if isinstance(x.uid, int) else -int(x.uid.int)
        )

    @override
    def __repr__(self):
        return "Tour. cfg:" + '|'.join([
            '{}:{}'.format(key, val)
            for key, val in self.__dict__.items()
        ])

    def serialize(self):
        return {
            'pod_sizes': self.pod_sizes,
            'allow_bye': self.allow_bye,
            'win_points': self.win_points,
            'bye_points': self.bye_points,
            'draw_points': self.draw_points,
            'snake_pods': self.snake_pods,
            'n_rounds': self.n_rounds,
            'max_byes': self.max_byes,
            'auto_export': self.auto_export,
            'standings_export': self.standings_export.serialize(),
            'global_wr_seats': self.global_wr_seats
        }

    @classmethod
    def inflate(cls, data:dict):
        return cls(
            pod_sizes=data['pod_sizes'],
            allow_bye=data['allow_bye'],
            win_points=data['win_points'],
            bye_points=data['bye_points'],
            draw_points=data['draw_points'],
            snake_pods=data['snake_pods'],
            n_rounds=data['n_rounds'],
            max_byes=data['max_byes'],
            auto_export=data['auto_export'],
            standings_export=StandingsExport.inflate(data['standings_export']),
            global_wr_seats=data['global_wr_seats'],
        )


class Tournament(ITournament):
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

        import importlib
        import pkgutil
        import os
        from pathlib import Path

        # Get the base directory of the project
        base_dir = Path(__file__).parent.parent
        pairing_logic_dir = base_dir / 'src' / 'pairing_logic'

        # Walk through all Python files in the pairing_logic directory
        for module_info in pkgutil.iter_modules([str(pairing_logic_dir)]):
            try:
                # Import the module
                module = importlib.import_module(f'src.pairing_logic.{module_info.name}')

                # Find all classes that implement IPairingLogic
                for name, obj in module.__dict__.items():
                    if (isinstance(obj, type) and
                        issubclass(obj, IPairingLogic) and
                        obj != IPairingLogic and
                        obj.IS_COMPLETE
                    ):
                        if obj.__name__ in cls._pairing_logic_cache:
                            raise ValueError(f"Pairing logic {obj.__name__} already exists")
                        cls._pairing_logic_cache[obj.__name__] = obj(name=f'{obj.__name__}')
            except Exception as e:
                Log.log(f"Failed to import pairing logic module {module_info.name}: {e}",
                       level=Log.Level.WARNING)

    @classmethod
    def get_pairing_logic(cls, logic_name: str) -> IPairingLogic:
        """Get a pairing logic instance by name."""
        cls.discover_pairing_logic()

        if logic_name not in cls._pairing_logic_cache:
            raise ValueError(f"Unknown pairing logic: {logic_name}")

        return cls._pairing_logic_cache[logic_name]

    def __init__(self, config: Union[TournamentConfiguration, None]=None, uid: UUID|None=None) :  # type: ignore
        TournamentAction.reset()
        if config is None:
            config = TournamentConfiguration()
        self._config = config
        self.uid: UUID = uuid4() if uid is None else uid
        self.CACHE[self.uid] = self

        self.PLAYER_CACHE: dict[UUID, Player] = {}
        self.POD_CACHE: dict[UUID, Pod] = {}
        self.ROUND_CACHE: dict[UUID, Round] = {}

        self._rounds: list[UUID] = list()
        self._players: list[UUID] = list()
        self._dropped: list[UUID] = list()
        self._round: UUID|None = None

        # Direct setting - don't want to overwrite old log file

        self.initialize_round()

    # TOURNAMENT ACTIONS
    # IMPORTANT: No nested tournament actions

    @override
    @classmethod
    def get(cls, uid: UUID) -> Tournament:
        return cls.CACHE[uid]

    @property
    def active_players(self) -> list[Player]:
        return [Player.get(self, x) for x in self._players if x not in self._dropped]

    @property
    def players(self) -> list[Player]:
        return [Player.get(self, x) for x in self._players]

    @property
    def dropped(self) -> list[Player]:
        return [Player.get(self, x) for x in self._dropped]

    @property
    def tour_round(self) -> Round:
        return Round.get(self, self._round) # type: ignore

    @tour_round.setter
    def tour_round(self, tour_round: Round):
        self._round = tour_round.uid

    @property
    def pods(self) -> list[Pod]|None:
        if not self.tour_round:
            return None
        return self.tour_round.pods

    @property
    def rounds(self) -> list[Round]:
        return [Round.get(self, x) for x in self._rounds]

    @rounds.setter
    def rounds(self, rounds: list[Round]):
        self._rounds = [r.uid for r in rounds]

    @property
    def ended_rounds(self) -> list[Round]:
        return [r for r in self.rounds if r.done]

    @property
    def draw_rate(self):
        n_draws = 0
        n_matches = 0
        for tour_round in self.rounds:
            for pod in tour_round.pods:
                if pod.done:
                    n_matches += 1
                    if pod.result_type == Pod.EResult.DRAW:
                        n_draws += len(pod._result)
        return n_draws/n_matches

    @property
    def config(self) -> TournamentConfiguration:
        return self._config

    @config.setter
    @TournamentAction.action
    def config(self, config: TournamentConfiguration):
        self._config = config

    @TournamentAction.action
    def add_player(self, names: str|list[str]|None=None):
        new_players = []
        if isinstance(names, str):
            names = [names]
        existing_names = set([p.name for p in self.active_players])
        for name in names:
            if name in existing_names:
                Log.log('\tPlayer {} already enlisted.'.format(
                    name), level=Log.Level.WARNING)
                continue
            if name:
                p = Player(self, name)
                self._players.append(p.uid)
                if p.uid not in self.tour_round._players:
                    self.tour_round._players.append(p.uid)
                new_players.append(p)
                existing_names.add(name)
                Log.log('\tAdded player {}'.format(
                    p.name), level=Log.Level.INFO)
        return new_players

    @TournamentAction.action
    def drop_player(self, players: list[Player]|Player) -> bool:
        if not isinstance(players, list):
            players = [players]
        for p in players:
            if self.tour_round and p.seated(self.tour_round):
                if self.tour_round.done:
                    #Log.log('Can\'t drop {} during an active tour_round.\nComplete the tour_round or remove player from pod first.'.format(
                    #    p.name), level=Log.Level.WARNING)
                    return False
            # If player has not played yet, it can safely be deleted without being saved
            if p.played(self.tour_round):
                self._dropped.append(p.uid)
            else:
                self._players.remove(p.uid)
            self.tour_round.drop_player(p)
        return True
            #Log.log('\tRemoved player {}'.format(p.name), level=Log.Level.INFO)

    @TournamentAction.action
    def rename_player(self, player, new_name):
        if player.name == new_name:
            return
        if new_name in [p.name for p in self.active_players]:
            Log.log('\tPlayer {} already enlisted.'.format(
                new_name), level=Log.Level.WARNING)
            return
        if new_name:
            player.name = new_name
            for tour_round in self.rounds:
                for pod in tour_round.pods:
                    for p in pod.players:
                        if p.name == player.name:
                            p.name = new_name
            Log.log('\tRenamed player {} to {}'.format(
                player.name, new_name), level=Log.Level.INFO)

    def get_pod_sizes(self, n) -> list[int]|None:
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
        if self._round is not None and not self.tour_round.done:
            return False
        seq = len(self.rounds)
        if seq == 0:
            logic = self.get_pairing_logic("PairingRandom")
        elif seq == 1 and self.config.snake_pods:
            logic = self.get_pairing_logic("PairingSnake")
        else:
            logic = self.get_pairing_logic("PairingDefault")
        new_round = Round(
            self,
            len(self.rounds),
            logic,
        )
        self._rounds.append(new_round.uid)
        self.tour_round = new_round
        return True

    @TournamentAction.action
    def create_pairings(self) -> bool:
        if self.tour_round.done:
            ok = self.initialize_round()
            if not ok:
                return False
        self.tour_round._byes.clear()
        if not self.tour_round.all_players_assigned:
            self.tour_round.create_pairings()
            return True
        return False

    @TournamentAction.action
    def new_round(self) -> bool:
        if not self.tour_round or self.tour_round.done:
            return self.initialize_round()
        return False

    @TournamentAction.action
    def reset_pods(self) -> bool:
        if not self.tour_round:
            return False
        if not self.tour_round.done:
            if not self.tour_round.reset_pods():
                return False
            return True
        return False

    @TournamentAction.action
    def manual_pod(self, players: list[Player]):
        if self.tour_round is None or self.tour_round.done:
           if not self.new_round():
                return
        assert isinstance(self.tour_round, Round)
        cap = min(self.config.max_pod_size, len(self.tour_round.unseated))
        pod = Pod(self.tour_round, len(self.tour_round.pods), cap=cap)
        self.tour_round._pods.append(pod.uid)

        for player in players:
            pod.add_player(player)
        self.tour_round.pods.append(pod)

    @TournamentAction.action
    def report_win(self, players: list[Player]|Player):
        if self.tour_round:
            if not isinstance(players, list):
                players = [players]
            for p in players:
                self.tour_round.set_result(p, Player.EResult.WIN)

    @TournamentAction.action
    def report_draw(self, players: list[Player]|Player):
        if self.tour_round:
            if not isinstance(players, list):
                players = [players]
            for p in players:
                self.tour_round.set_result(p, Player.EResult.DRAW)

    @TournamentAction.action
    def random_results(self):
        if not self.tour_round:
            #Log.log(
            #    'A tour_round is not in progress.\nCreate pods first!',
            #    level=Log.Level.ERROR
            #)
            return
        if self.tour_round.pods:
            draw_rate = 1-sum(self.config.global_wr_seats)
            #for each pod
            #generate a random result based on global_winrates_by_seat
            #each value corresponds to the pointrate of the player in that seat
            #the sum of percentages is less than 1, so there is a chance of a draw (1-sum(winrates))

            for pod in [x for x in self.tour_round.pods if not x.done]:
                #generate a random result
                result = random.random()
                rates = np.array(self.config.global_wr_seats[0:len(pod.players)] + [draw_rate])
                rates = rates/sum(rates)
                draw = result > np.cumsum(rates)[-2]
                if not draw:
                    win = np.argmax([result < x for x in rates])
                    #Log.log('won "{}"'.format(pod.players[win].name))
                    self.tour_round.set_result(pod.players[win], Player.EResult.WIN)
                    #player = random.sample(pod.players, 1)[0]
                    #Log.log('won "{}"'.format(player.name))
                    #self.tour_round.won([player])
                else:
                    players = pod.players
                    #Log.log('draw {}'.format(
                    #    ' '.join(['"{}"'.format(p.name) for p in players])))
                    for p in players:
                        self.tour_round.set_result(p, Player.EResult.DRAW)
                pass
        pass

    @TournamentAction.action
    def move_player_to_pod(self, pod: Pod, players: list[Player]|Player, manual=False):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod(self.tour_round) == pod:
                continue
                #player.pod(self.tour_round).remove_player(player)
                #Log.log('Removed player {} from {}.'.format(
                #    player.name, old_pod), level=Log.Level.INFO)
            if ok:=pod.add_player(player, manual=manual):
                pass
                    #Log.log('Added player {} to {}'.format(
                    #    player.name, pod.name), level=Log.Level.INFO)
                #else:
                #    Log.log('Failed to add palyer {} to Pod {}'.format(
                #        player.name, pod.table), level=Log.Level.ERROR)

    @TournamentAction.action
    def bench_players(self, players: list[Player]|Player):
        assert self.tour_round is not None
        if not isinstance(players, list):
            players = [players]
        for player in players:
            self.remove_player_from_pod(player)

    @TournamentAction.action
    def toggle_game_loss(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]

        for player in players:
            if player.uid in self.tour_round._game_loss:
                self.tour_round._game_loss.remove(player.uid)
            else:
                #if player.pod(self.tour_round) is not None:
                #    self.remove_player_from_pod(player)
                player.set_result(self.tour_round, Player.EResult.LOSS)
                #Log.log('{} assigned a game loss.'.format(
                #    player.name), level=Log.Level.INFO)

    @TournamentAction.action
    def delete_pod(self, pod: Pod):
        if self.tour_round:
            self.tour_round.remove_pod(pod)

    def remove_player_from_pod(self, player: Player):
        assert self.tour_round is not None
        pod = player.pod(self.tour_round)
        if pod:
            pod.remove_player(player)
            if player.uid not in self.tour_round._game_loss:
                self.tour_round.set_result(player, Player.EResult.BYE)
            #Log.log('Removed player {} from {}.'.format(
            #    player.name, pod.name), level=Log.Level.INFO)

    def rating(self, player: Player, tour_round: Round) -> float:
        points = 0
        for i, i_tour_round in enumerate(self.rounds):
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
        if not self.tour_round:
            return ''
        export_str = '\n\n'.join([
            pod.__repr__()
            for pod in self.tour_round.pods
        ])

        if self.config.allow_bye and self.tour_round.unseated:
            export_str += '\n\nByes:\n' + '\n:'.join([
                "\t{}\t| pts: {}".format(p.name, p.rating(self.tour_round) or '0')
                for p in self.tour_round.unseated
            ])
        return export_str

    @timeit
    def get_standings(self, tour_round:Round|None=None) -> list[Player]:
        method = Player.SORT_METHOD
        order = Player.SORT_ORDER
        Player.SORT_METHOD = SortMethod.RANK
        Player.SORT_ORDER = SortOrder.ASCENDING
        if tour_round is None:
            tour_round = self.tour_round
        standings = sorted(self.players, key=lambda x: self.config.ranking(x, tour_round), reverse=True)
        Player.SORT_METHOD = method
        Player.SORT_ORDER = order
        return standings

    def get_standings_str(
            self,
            fields: list[StandingsExport.Field] = StandingsExport.DEFAULT_FIELDS,
            style: StandingsExport.Format = StandingsExport.Format.PLAIN,
            tour_round: Round|None = None,
            standings: list[Player]|None = None
    ) -> str:
        #raise DeprecationWarning("get_standings_str is deprecated. Use get_standings instead.")
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
                    else StandingsExport.info[f].get(p, context) * StandingsExport.info[f].denom
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
            lines.insert(1, ['-' * width for width in col_len])
            lines = '\n'.join([' | '.join(line) for line in lines])
            return lines


            # Log.log('Log saved: {}.'.format(
            #    fdir), level=Log.Level.INFO)
        elif style == StandingsExport.Format.CSV:
            Log.log('Log not saved - CSV not implemented.'.format(
                fdir), level=Log.Level.WARNING)
        elif style == StandingsExport.Format.DISCORD:
            Log.log('Log not saved - DISCORD not implemented.'.format(
                fdir), level=Log.Level.WARNING)
        elif style == StandingsExport.Format.JSON:
            Log.log('Log not saved - JSON not implemented.'.format(
                fdir), level=Log.Level.WARNING)

        raise ValueError('Invalid style: {}'.format(style))

    @staticmethod
    def send_request(api, data, headers):
        try:
            response = requests.post(
                api,
                json=data,
                headers=headers,
                timeout=10
            )
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
        if StandingsExport.Target.FILE == target_type:
            if not os.path.exists(os.path.dirname(var_export_param)):
                os.makedirs(os.path.dirname(var_export_param))
            with open(var_export_param, 'w', encoding='utf-8') as f:
                f.writelines(data)

        if StandingsExport.Target.WEB == target_type:
            api = os.getenv("EXPORT_ONLINE_API_URL")
            key = os.getenv("EXPORT_ONLINE_API_KEY")
            tournament_id = os.getenv("TOURNAMENT_ID")
            url = f"{api}?tournamentId={tournament_id}"
            if not key or not api:
                Log.log("Error: EXPORT_ONLINE_API_URL or EXPORT_ONLINE_API_KEY not set in the environment variables.")
                return

            # Send as POST request to the Express app with authentication
            headers = {
                "x-api-key": key
            }
            request_data = {
                "title": "Tournament Update",
                "timestamp": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "text": data,
            }

            thread = threading.Thread(target=self.send_request, args=(url, request_data, headers))
            thread.start()

        if StandingsExport.Target.DISCORD == target_type:
            instance = DiscordPoster.instance()
            instance.post_message(data)

        if StandingsExport.Target.CONSOLE == target_type:
            if not isinstance(var_export_param, Log.Level):
                var_export_param = Log.Level.INFO
            Log.log(data, level=var_export_param)

    def serialize(self) -> dict[str, Any]:
        """
        Returns a JSON string of the tournament, that can be serialized.
        It contains:
            - tournament configuration json (serialized from class)
            - a list of player jsons (serialized from class)
            - a list of pod jsons (serialized from class)
            - a list of tour_round jsons (serialized from class)

        Objects are referenced to each other by ids.
        """

        data: dict[str, Any] = {}
        data['uid'] = str(self.uid)
        data['config'] = self.config.serialize()
        data['players'] = [p.serialize() for p in self.active_players]
        data['dropped'] = [p.serialize() for p in self.dropped]
        data['rounds'] = [r.serialize() for r in self.rounds]
        return data

    @classmethod
    def inflate(cls, data: dict[str, Any]) -> Tournament:
        config = TournamentConfiguration.inflate(data['config'])
        tour_uid = UUID(data['uid'])
        if tour_uid in Tournament.CACHE:
            tour = Tournament.CACHE[tour_uid]
        else:
            tour = cls(config, tour_uid)
        tour._players = [UUID(d_player['uid']) for d_player in data['players']]
        tour._dropped = [UUID(d_player['uid']) for d_player in data['dropped']]
        for d_player in data['players']:
            Player.inflate(tour, d_player)
        tour._rounds = [UUID(d_round['uid']) for d_round in data['rounds']]
        for d_round in data['rounds']:
            Round.inflate(tour, d_round)
        return tour


class Player(IPlayer):
    SORT_METHOD: SortMethod = SortMethod.ID
    SORT_ORDER: SortOrder = SortOrder.ASCENDING
    FORMATTING = ['-p']

    def __init__(self, tour: Tournament, name:str, uid: UUID|None = None):
        super().__init__()
        self._tour = tour.uid
        if uid is not None:
            self.uid = uid
        self.name = name
        self.CACHE[self.uid] = self
        self._pod_id: UUID|None = None  # Direct reference to current pod

    #ACTIONS
    def set_result(self, tour_round: Round, result: Player.EResult) -> None:
        tour_round.set_result(self, result)


    #QUERIES

    def pod(self, tour_round:Round) -> Pod|None:
        if tour_round is None:
            return None
        return tour_round.get_location(self)

    def result(self, tour_round: Round) -> Player.EResult:
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

    def rating(self, tour_round: Round|None) -> float|None:
        if tour_round is None:
            return None
        return self.tour.rating(self, tour_round)

    def pods(self, tour_round: Round|None=None) -> list[Pod|Player.ELocation]:
        if tour_round is None:
            tour_round = self.tour.tour_round
        pods:list[Pod|Player.ELocation] = [None for _ in self.tour._rounds] # type: ignore
        tour_rounds = self.tour.rounds
        for i, tour_round in enumerate(tour_rounds):
            pod = tour_round.get_location(self)
            if pod == None:
                pods[i] = tour_round.get_location_type(self)
            else:
                pods[i] = pod
        return pods

    def played(self, tour_round: Round|None=None) -> list[Player]:
        if tour_round is None:
            tour_round = self.tour.tour_round
        players = set()
        for p in self.pods(tour_round):
            if isinstance(p, Pod):
                players.update(p.players)
        return list(players)

    def games(self, tour_round: Round|None=None):
        if tour_round is None:
            tour_round = self.tour.tour_round
        return [p for p in self.pods(tour_round) if isinstance(p, Pod)]

    def byes(self, tour_round: Round|None=None):
        if tour_round is None:
            tour_round = self.tour.tour_round
        return len([p for p in self.pods(tour_round) if p is Player.EResult.BYE])

    def wins(self, tour_round: Round|None=None):
        if tour_round is None:
            tour_round = self.tour.tour_round
        return len([p for p in self.games(tour_round) if p._result is self.uid])

    def record(self, tour_round: Round|None=None) -> list[Player.EResult]:
        seq = list()
        if tour_round is None:
            tour_round = self.tour.tour_round
        pods: list[Pod|Player.ELocation] = []
        for i, p in enumerate(self.pods(tour_round)):
            #if i < tour_round.seq:
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
                    elif pod.result_type == Pod.EResult.DRAW and self.uid in pod._result:
                        seq.append(Player.EResult.DRAW)
                    else:
                        seq.append(Player.EResult.LOSS)
                else:
                    seq.append(Player.EResult.PENDING)
        return seq

    def seat_history(self, tour_round: Round|None=None) -> str:
        if tour_round is None:
            tour_round = self.tour.tour_round
        pods = self.pods(tour_round)
        if sum([1 for p in pods if isinstance(p, Pod) and p.done]) == 0:
            return 'N/A'
        ret_str = ' '.join([
            '{}/{}'.format(
                ([x.uid for x in p.players]).index(self.uid)+1,
                len(p.players)
            )
            if isinstance(p, Pod)
            else 'N/A'
            for p in pods
        ])
        return ret_str

    def pointrate(self, tour_round: Round|None=None):
        if len(self.games(tour_round)) == 0:
            return 0
        if tour_round is None:
            tour_round = self.tour.tour_round
        return self.rating(tour_round) / (self.tour.config.win_points * (tour_round.seq+1))

    def location(self, tour_round: Round|None=None) -> Player.ELocation:
        if tour_round is None:
            tour_round = self.tour.tour_round
        return tour_round.get_location_type(self)

    def players_beaten(self, tour_round: Round|None=None) -> list[Player]:
        if tour_round is None:
            tour_round = self.tour.tour_round
        players = set()
        for pod in self.games(tour_round):
            if pod.result_type == Pod.EResult.WIN and self in pod._result:
                players.update(pod.players)

        players.discard(self)
        return list(players)

    def average_seat(self, rounds:list[Round]) -> np.float64:
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

        Lower percentage means higher benefits.

        We are now using a weighted average of all the pods the player has been in.
        Weights are based on TC.global_wr_seats
        """
        pods = [
            self.pod(round)
            for
            round
            in rounds
            if self.pod(round) is not None
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
                    rates = self.tour.config.global_wr_seats[0:len(pod)]
                    norm_scale = 1-(np.cumsum(rates)-rates[0])/(np.sum(rates)-rates[0])
                    score += norm_scale[index]
        return np.float64(score/n_pods)

    def standing(self, tour_round: Round|None=None, standings:list[Player]|None=None) -> int:
        if tour_round is None:
            tour_round = self.tour.tour_round
        if standings is None:
            standings = self.tour.get_standings(tour_round)
        if self not in standings:
            return -1
        return standings.index(self) + 1

    def not_played(self, tour_round: Round|None=None    ) -> list[Player]:
        if tour_round is None:
            tour_round = self.tour.tour_round
        return list(set(self.tour.active_players) - set(self.played(tour_round)))

    def opponent_pointrate(self, tour_round: Round|None=None):
        if not self.played(tour_round):
            return 0
        oppwr = [opp.pointrate(tour_round) for opp in self.played(tour_round)]
        return sum(oppwr)/len(oppwr)


    #PROPERTIES

    @property
    def CACHE(self) -> dict[UUID, Player]:
        return self.tour.PLAYER_CACHE

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @tour.setter
    def tour(self, tour: Tournament):
        self._tour = tour.uid

    #STATICMETHOD

    @staticmethod
    def get(tour: Tournament, uid: UUID):
        return tour.PLAYER_CACHE[uid]


    def seated(self, tour_round: Round|None=None) -> bool:
        if tour_round is None:
            tour_round = self.tour.tour_round
        return tour_round.get_location(self) is not None

    @staticmethod
    def fmt_record(record:list[Player.EResult]) -> str:
        return ''.join([{
            Player.EResult.WIN: 'W',
            Player.EResult.LOSS: 'L',
            Player.EResult.DRAW: 'D',
            Player.EResult.BYE: 'B',
            Player.EResult.PENDING: '_',
        }[r] for r in record])

    def __gt__(self, other: Player, tour_round:Round|None=None):
        b = False
        if self.SORT_METHOD == SortMethod.ID:
            b = self.uid > other.uid
        elif self.SORT_METHOD == SortMethod.NAME:
            b = self.name > other.name
        elif self.SORT_METHOD == SortMethod.RANK:
            if tour_round is None:
                tour_round = self.tour.tour_round
            my_score = self.tour.config.ranking(self, tour_round)
            other_score = self.tour.config.ranking(other, tour_round)
            b = None
            for i in range(len(my_score)):
                if my_score[i] != other_score[i]:
                    b = my_score[i] > other_score[i]
                    break
        return b

    def __lt__(self, other: Player, tour_round:Round|None=None):
        b = False
        if self.SORT_METHOD == SortMethod.ID:
            b = self.uid < other.uid
        elif self.SORT_METHOD == SortMethod.NAME:
            b = self.name < other.name
        elif self.SORT_METHOD == SortMethod.RANK:
            if tour_round is None:
                tour_round = self.tour.tour_round
            my_score = self.tour.config.ranking(self, tour_round)
            other_score = self.tour.config.ranking(other, tour_round)
            b = None
            for i in range(len(my_score)):
                if my_score[i] != other_score[i]:
                    b = my_score[i] > other_score[i]
                    break
        return b

    @override
    def __repr__(self, tokens=None, context: TournamentContext|None=None):
        if len(self.tour.active_players) == 0:
            return ''
        if not tokens:
            tokens = self.FORMATTING
        parser_player = argparse.ArgumentParser()

        parser_player.add_argument(
            '-n', '--standi[n]g',
            dest='standing', action='store_true')
        parser_player.add_argument(
            '-i', '--id',
            dest='id', action='store_true')
        parser_player.add_argument(
            '-w', '--win',
            dest='w', action='store_true')
        parser_player.add_argument(
            '-o', '--opponentwin',
            dest='ow', action='store_true')
        parser_player.add_argument(
            '-p', '--points',
            dest='p', action='store_true')
        parser_player.add_argument(
            '-a', '--winr[a]te',
            dest='wr', action='store_true')
        parser_player.add_argument(
            '-u', '--unique',
            dest='u', action='store_true')
        parser_player.add_argument(
            '-s', '--average_seat',
            dest='s', action='store_true')
        parser_player.add_argument(
            '-l', '--pod',
            dest='pod', action='store_true')
        parser_player.add_argument(
            '-r', '--round',
            dest='round', type=int, default=self.tour.tour_round.seq if self.tour.tour_round else 0)
        '''parser_player.add_argument(
            '-s', '--spaces',
            dest='spaces', type=int, default=0)'''
        # parser.add_argument('-n', '--notplayed',    dest='np', action='store_true')

        args, _ = parser_player.parse_known_args(tokens)

        fields = list()

        tsize = int(math.floor(math.log10(len(self.tour.active_players))) + 1)
        pname_size = max([len(p.name) for p in self.tour.active_players])

        tour_round = self.tour.rounds[args.round]
        if context is None:
            standings = self.tour.get_standings(tour_round)
        else:
            standings = context.standings
        if args.standing:
            fields.append('#{:>{}}'.format(self.standing(tour_round, standings), tsize))
        if args.id:
            fields.append('[{:>{}}] {}'.format(
                self.uid, tsize, self.name.ljust(pname_size)))
        else:
            fields.append(self.name.ljust(pname_size))

        if args.pod and len(tour_round.pods) > 0:
            max_pod_id = max([len(str(p.table)) for p in tour_round.pods])
            pod = self.pod(tour_round)
            if self in self.tour.dropped:
                fields.append('Drop'.ljust(max_pod_id+4))
            elif pod:
                #find number of digits in max pod id
                fields.append('{}'.format(
                    f'P{str(pod.table).zfill(max_pod_id)}/S{pod.players.index(self)}' if pod else ''))
            elif self.result(self.tour.tour_round) == Player.EResult.LOSS:
                fields.append('Loss'.ljust(max_pod_id+4))
            else:
                fields.append('Bye'.ljust(max_pod_id+4))
        if args.p:
            fields.append('pts: {}'.format(self.rating(self.tour.tour_round) or '0'))
        if args.w:
            fields.append('w: {}'.format(self.wins(tour_round)))
        if args.ow:
            fields.append('o.wr.: {:.2f}'.format(self.opponent_pointrate))
        if args.u:
            fields.append('uniq: {}'.format(self.played))
        if args.s:
            fields.append('seat: {:02.00f}%'.format(self.average_seat([r for r in self.tour.rounds if r.seq <= args.round])*100))
        # if args.np:
        #    fields.append(''.format())
        # OUTPUT_BUFFER.append('\t{}'.format(' | '.join(fields)))

        return ' | '.join(fields)

    def serialize(self) -> dict[str, Any]:
        return {
            #'tour': str(self._tour),
            'uid': str(self.uid),
            'name': self.name,
        }

    @classmethod
    def inflate(cls, tour: Tournament, data: dict[str, Any]) -> Player:
        #assert tour.uid == UUID(data['tour'])
        return cls(tour, data['name'], UUID(data['uid']))


class Pod(IPod):
    def __init__(self, tour_round: Round, table:int, cap=0, uid: UUID|None = None):
        super().__init__()
        self._tour: UUID = tour_round.tour.uid
        self._round: UUID = tour_round.uid
        if uid is not None:
            self.uid = uid
        self.CACHE[self.uid] = self
        self.table:int = table
        self.cap:int = cap
        self._players: list[UUID] = list()
        #self._players: list[UUID] = list() #TODO: make references to players
        #self.discord_message_id: None|int = None

    @property
    def CACHE(self) -> dict[UUID, Pod]:
        return self.tour.POD_CACHE

    @staticmethod
    def get(tour: Tournament, uid: UUID) -> Pod:
        return tour.POD_CACHE[uid]

    def set_result(self, player: Player, result: IPlayer.EResult):
        if player.uid not in self._players:
            raise ValueError('Player {} not in pod {}'.format(player.name, self.name))
        if result == IPlayer.EResult.WIN:
            self._result.clear()
        self._result.add(player.uid)

    def remove_result(self, player: Player):
        if player.uid in self._result:
            self._result.remove(player.uid)

    @property
    def result(self) -> set[Player]:
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
        return [Player.get(self.tour, x) for x in self._players]

    @override
    def add_player(self, player: Player, manual=False, player_pod_map=None) -> bool:
        if len(self) >= self.cap and self.cap and not manual:
            return False
        if pod:=player.pod(self.tour_round):
            pod.remove_player(player)
        self.tour_round._byes.discard(player.uid)
        self._players.append(player.uid)
        self.tour_round.player_locations_map[player.uid] = self
        #player.location = Player.ELocation.SEATED
        #player.pod = self  # Update player's pod reference
        return True

    def remove_player(self, player: Player, cleanup=True) -> Player|None:
        try:
            idx = self._players.index(player.uid)
        except ValueError:
            return None
        p = Player.get(self.tour, self._players.pop(idx))

        self.remove_result(player)
        del self.tour_round.player_locations_map[player.uid]
        #player.location = Player.ELocation.UNASSIGNED
        #player.pod = None  # Clear player's pod reference
        if len(self) == 0 and cleanup:
            self.tour_round.remove_pod(self)
        return player

    @property
    def average_seat(self) -> float:
        return np.average([p.average_seat(self.tour.ended_rounds) for p in self.players]).astype(float)

    @property
    def balance(self) -> np.ndarray:
        '''
        Returns a list of count of players above 50% average seat and below 50% average seat
        '''
        return np.array([
            sum([1 for p in self.players if p.average_seat(self.tour.rounds) > 0.5]),
            sum([1 for p in self.players if p.average_seat(self.tour.rounds) < 0.5])
        ])

    @override
    def assign_seats(self):
        # Average seating positions
        average_positions = [p.average_seat(self.tour.ended_rounds) for p in self.players]
        n = len(average_positions)

        if not any(average_positions):
            random.shuffle(self.players)
            return

        '''distribution = [1] * n
        for i in range(n):
            distribution[i] += (2*(1-average_positions[i]))**3
        #normalize
        distribution = [x/sum(distribution) for x in distribution]

        # Generate random seat assignment based on probabilities
        seat_assignment = np.random.choice(
            range(1, n+1, 1),
            size=n,
            replace=False,
            p=distribution
        )'''
        #partially sort players based on seating positions
        #those that have same average_seat should be randomly ordered
        seat_assignment = [0] * n
        for i in range(n):
            seat_assignment[i] = sum([1 for x in average_positions if x < average_positions[i]]) + 1
        #randomize players with same average_seat
        seat_assignment = [x + random.random() for x in seat_assignment]
        #sort players based on seat assignment
        self._players[:] = np.take(self._players, np.argsort(seat_assignment))

        pass

    def clear(self):
        players = [p for p in self.players]
        for p in players:
            self.remove_player(p, cleanup=False)
        #self.players.clear()

    @property
    def name(self):
        return 'Pod {}'.format(self.table)

    @override
    def __repr__(self, context: TournamentContext|None=None):
        if not self.players:
            maxlen = 0
        else:
            maxlen = max([len(p.name) for p in self.players])
        ret = 'Pod {} with {}/{} players:\n\t{}'.format(
            self.table,
            len(self),
            self.cap,
            '\n\t'.join(
                [
                    '[{}] {}\t'.format(
                        ' ' if not self._result else
                        'W' if self.result_type == Pod.EResult.WIN and p.uid in self._result else
                        'D' if self.result_type == Pod.EResult.DRAW and p.uid in self._result else
                        'L',
                        p.__repr__(['-s', str(maxlen), '-p'], context=context))
                    for _, p in
                    zip(range(1, len(self)+1), self.players)
                ]
            ))
        return ret

    def serialize(self) -> dict[str, Any]:
        return {
            'uid': str(self.uid),
            'tour_round': str(self._round),
            'table': self.table,
            'cap': self.cap,
            'result': sorted([str(p) for p in self._result]),
            'players': [str(p) for p in self._players],
        }

    @classmethod
    def inflate(cls, tour_round: Round, data: dict[str, Any]) -> Pod:
        assert tour_round.uid == UUID(data['tour_round'])
        pod = cls(tour_round, data['table'], data['cap'], UUID(data['uid']))
        pod._players = [x for x in data['players']]
        pod._result = {x for x in data['result']}
        return pod


class Round(IRound):
    def __init__(self, tour: Tournament, seq: int, pairing_logic:IPairingLogic, uid: UUID|None = None):
        super().__init__()
        if uid is not None:
            self.uid = uid
        self._tour: UUID = tour.uid
        self.tour.ROUND_CACHE[self.uid] = self
        self._players: list[UUID] = [p.uid for p in self.tour.active_players]
        self._dropped: set[UUID] = set([p.uid for p in self.tour.dropped])
        self.seq:int = seq
        self._logic = pairing_logic.name
        self.player_locations_map: dict[UUID, Pod] = {}
        self._game_loss: set[UUID] = set()
        self._byes: set[UUID] = set()


    def get_location(self, player: Player) -> Pod|None:
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
    def byes(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._byes]

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
    def active_players(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._players if x not in self._dropped]

    @property
    def players(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._players]

    #@active_players.setter
    #def active_players(self, players: list[Player]):
    #    self._players = [p.uid for p in players]

    @property
    def pods(self) -> list[Pod]:
        return [Pod.get(self.tour, x) for x in self._pods]

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @property
    def done(self):
        if len(self._pods) == 0:
            return False
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    @property
    def all_players_assigned(self):
        seated = len(self.seated)
        n_players_to_play = seated + len(self.unseated)
        if n_players_to_play == 0:
            return True
        if self.tour.get_pod_sizes(n_players_to_play) is None:
            return False
        if not (pod_sizes:=self.tour.get_pod_sizes(n_players_to_play)):
            return True
        else:
            return sum(pod_sizes) == seated

    @property
    def seated(self) -> list[Player]:
        return [p for p in self.active_players if p.pod(self)]

    @property
    def unseated(self) -> list[Player]:
        return [
            p
            for p in self.active_players
            if p.location(self) == Player.ELocation.UNASSIGNED
        ]

    def reset_pods(self) -> bool:
        pods = [Pod.get(self.tour, x) for x in self._pods]
        #if any([not pod.done for pod in pods]):
        #    return False
        self._byes.clear()
        for pod in pods:
            self.remove_pod(pod)
        return True

    def remove_pod(self, pod: Pod) -> bool:
        #if not pod.done:
        pod.clear()
        self._pods.remove(pod.uid)
        return True
        #return False

    def create_pods(self):
        seats_required = len(self.unseated) - sum([pod.cap-len(pod) for pod in self.pods if not pod.done])
        if seats_required == 0:
            return
        pod_sizes = self.tour.get_pod_sizes(seats_required)
        if pod_sizes is None:
            Log.log('Can not make pods.', level=Log.Level.WARNING)
            return None
        start_table = len(self._pods) + 1
        for i, size in enumerate(pod_sizes):
            pod = Pod(self, start_table + i, cap=size)
            self._pods.append(pod.uid)

    def create_pairings(self):
        self.create_pods()
        pods = [p for p in self.pods
                if all([
                    not p.done,
                    len(p) < p.cap
        ])]

        self.logic.make_pairings(self, self.unseated, pods)

        for pod in self.pods:
            pod.assign_seats()
        self.sort_pods()
        #for pod in self.pods:
        #    Log.log(pod, Log.Level.NONE)

    def sort_pods(self):
        pods_sorted = sorted(self.pods, key=lambda x: (len(x.players), np.average([p.rating(self) for p in x.players])), reverse=True)
        self._pods[:] = [pod.uid for pod in pods_sorted]

    def set_result(self, player: Player, result: IPlayer.EResult) -> None:
        if result == IPlayer.EResult.BYE:
            self._byes.add(player.uid)
        elif result == IPlayer.EResult.LOSS:
            self._game_loss.add(player.uid)
        elif result == IPlayer.EResult.WIN:
            if pod:=player.pod(self):
                pod.set_result(player, result)
            else:
                raise ValueError('Player {} not in any pod'.format(player.name))
        elif result == IPlayer.EResult.DRAW:
            if pod:=player.pod(self):
                pod.set_result(player, result)
            else:
                raise ValueError('Player {} not in any pod'.format(player.name))

    def remove_result(self, player: Player):
        if pod:=player.pod(self):
            pod.remove_result(player)

    def drop_player(self, player: Player):
        if self.done:
            raise ValueError('Can\'t drop player in a completed round.')
        if (pod:=self.get_location(player)) is not None:
            if pod.done:
                raise ValueError('Can\'t drop player in a completed pod.')
            pod.remove_player(player)
        self._dropped.add(player.uid)

    def serialize(self) -> dict[str, Any]:
        return {
            'tour': str(self._tour),
            'uid': str(self.uid),
            'seq': self.seq,
            'pods': [pod.serialize() for pod in self.pods],
            'players': [str(p) for p in self._players],
            'logic': self._logic,
            'game_loss': [str(p) for p in self._game_loss],
            'byes': [str(p) for p in self._byes],
            'dropped': [str(p) for p in self._dropped],
        }

    @classmethod
    def inflate(cls, tour: Tournament, data: dict[str, Any]) -> Round:
        assert tour.uid == UUID(data['tour'])
        tour.discover_pairing_logic()
        logic = tour.get_pairing_logic(data['logic'])
        new_round = cls(tour, data['seq'], logic, UUID(data['uid']))
        pods = [Pod.inflate(new_round, pod) for pod in data['pods']]
        new_round._pods = [pod.uid for pod in pods]
        new_round._game_loss = {UUID(x) for x in data['game_loss']}
        new_round._byes = {UUID(x) for x in data['byes']}
        new_round._dropped = {UUID(x) for x in data['dropped']}
        return new_round

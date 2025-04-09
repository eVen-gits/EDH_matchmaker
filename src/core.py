from __future__ import annotations
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
from .misc import Json2Obj, generate_player_names
import numpy as np
from tqdm import tqdm # pyright: ignore
from uuid import UUID, uuid4

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
            tour_round = self.round
            ret = func(self, *original_args, **original_kwargs)
            tour_round = tour_round or self.round
            if self.config.auto_export:
                logf = TournamentAction.LOGF
                if logf and tour_round:
                    # Export pods to a file named {tournament_name}_round_{round_number}.txt
                    # And also export it into {log_directory}/pods.txt

                    export_str: str = '\n\n'.join([
                        pod.__repr__()
                        for pod in tour_round.pods
                    ])
                    game_lost: list[Player] = [x for x in tour_round.players if x.result == Player.EResult.LOSS]
                    byes = [x for x in tour_round.unseated if x.location == Player.ELocation.UNSEATED and x.result == Player.EResult.BYE]
                    if len(game_lost) + len(byes) > 0:
                        max_len = max([len(p.name) for p in game_lost + byes])
                        if self.config.allow_bye and byes:
                            export_str += '\n\nByes:\n' + '\n'.join([
                                "\t{} | pts: {}".format(p.name.ljust(max_len), p.points)
                                for p in tour_round.unseated
                                if p.result == Player.EResult.BYE
                            ])
                        if game_lost:
                            export_str += '\n\nGame losses:\n' + '\n'.join([
                                "\t{} | pts: {}".format(
                                    p.name.ljust(max_len),
                                    p.points
                                )
                                for p in game_lost
                            ])

                    path = os.path.join(
                        os.path.dirname(logf),
                        os.path.basename(logf).replace('.log', ''),
                        os.path.basename(logf).replace('.log', '_R{}.txt'.format(tour_round.seq)),
                    )
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))

                    self.export_str(export_str, path, DataExport.Target.FILE)
                    self.export_str(export_str, None, DataExport.Target.WEB)


                    path = os.path.join(os.path.dirname(logf), 'pods.txt')
                    self.export_str(export_str, path, DataExport.Target.FILE)

            return ret
        return auto_pods_export_wrapper


class StandingsExport(DataExport):
    class Field(Enum):
        STANDING = 0  # Standing
        ID = 1  # Player ID
        NAME = 2  # Player name
        RECORD = 3  # Record
        POINTS = 4  # Number of points
        WINS = 5  # Number of wins
        OPP_BEATEN = 6  # Number of opponents beaten
        OPP_WINRATE = 7  # Opponents' win percentage
        UNIQUE = 8  # Number of unique opponents
        WINRATE = 9  # Winrate
        GAMES = 10  # Number of games played
        SEAT_HISTORY = 11  # Seat record
        AVG_SEAT = 12  # Average seat

    info = {
        Field.STANDING: Json2Obj({
            'name': '#',
            'format': '{:d}',
            'denom': None,
            'description': 'Player\'s standing in the tournament.',
            'getter': lambda p: p.standing,
        }),
        Field.ID: Json2Obj({
            'name': 'ID',
            'format': '{:s}',
            'denom': None,
            'description': 'Player ID',
            'getter': lambda p: p.uid.hex
        }),
        Field.NAME: Json2Obj({
            'name': 'name',
            'format': '{:s}',
            'denom': None,
            'description': 'Player name',
            'getter': lambda p: p.name
        }),
        Field.OPP_WINRATE: Json2Obj({
            'name': 'opp. win %',
            'format': '{:.2f}%',
            'denom': 100,
            'description': 'Opponents\' win percentage',
            'getter': lambda p: p.opponent_winrate
        }),
        Field.POINTS: Json2Obj({
            'name': 'pts',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of points',
            'getter': lambda p: p.points
        }),
        Field.WINS: Json2Obj({
            'name': '# wins',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of games won',
            'getter': lambda p: p.games_won
        }),
        Field.WINRATE: Json2Obj({
            'name': 'win %',
            'format': '{:.2f}%',
            'denom': 100,
            'description': 'Winrate',
            'getter': lambda p: p.winrate
        }),
        Field.UNIQUE: Json2Obj({
            'name': 'uniq. opp.',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of unique opponents',
            'getter': lambda p: len(p.games)
        }),
        Field.GAMES: Json2Obj({
            'name': '# games',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of games played',
            'getter': lambda p: len(p.games)
        }),
        Field.OPP_BEATEN: Json2Obj({
            'name': '# opp. beat',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of opponents beaten',
            'getter': lambda p: len(p.players_beaten)
        }),
        Field.SEAT_HISTORY: Json2Obj({
            'name': 'seat record',
            'format': '{:s}',
            'denom': None,
            'description': 'Seat record',
            'getter': lambda p: p.seat_history
        }),
        Field.AVG_SEAT: Json2Obj({
            'name': 'avg. seat',
            'format': '{:03.2f}%',
            'denom': None,
            'description': 'Average seat',
            'getter': lambda p: p.average_seat*100
        }),
        Field.RECORD: Json2Obj({
            'name': 'record',
            'format': '{:s}',
            'denom': None,
            'description': 'Player\'s record',
            'getter': lambda p: Player.fmt_record(p.record)
        }),
    }

    ext = {
        DataExport.Format.DISCORD: '.txt',
        DataExport.Format.PLAIN: '.txt',
        DataExport.Format.CSV: '.csv'
    }

    DEFAULT_FIELDS = [
        Field.STANDING,
        Field.NAME,
        Field.POINTS,
        Field.RECORD,
        Field.OPP_WINRATE,
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
    DEFAULT_LOGF = 'logs/default.log'

    def __init__(self, before: Tournament, ret, after: Tournament, func_name, *nargs, **kwargs):
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
        def wrapper(self, *original_args, **original_kwargs):
            before = deepcopy(self)
            ret = func(self, *original_args, **original_kwargs)
            after = deepcopy(self)
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
            with open(cls.LOGF, 'wb') as f:
                pickle.dump(cls.ACTIONS, f)

    @classmethod
    def load(cls, logdir='logs/default.log'):
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
    def ranking(x):
        return (
            x.points,
            len(x.games),
            np.round(x.opponent_winrate, 10),
            len(x.players_beaten),
            -x.average_seat,
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
    # then opponent winrate, - CHANGE - moved this upwards and added dummy opponents with 33% winrate
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
                        cls._pairing_logic_cache[obj.__name__] = obj(name=f'src.pairing_logic.{module_info.name}')
            except Exception as e:
                Log.log(f"Failed to import pairing logic module {module_info.name}: {e}",
                       level=Log.Level.WARNING)

    @classmethod
    def get_pairing_logic(cls, logic_name: str) -> IPairingLogic:
        """Get a pairing logic instance by name."""
        cls.discover_pairing_logic()

        if logic_name not in cls._pairing_logic_cache:
            Log.log(f"Unknown pairing logic: {logic_name}, falling back to default",
                   level=Log.Level.WARNING)
            logic_name = "PairingDefault"

        return cls._pairing_logic_cache[logic_name]

    def __init__(self, config: Union[TournamentConfiguration, None]=None, uid: UUID|None=None) :  # type: ignore
        TournamentAction.reset()
        if config is None:
            config = TournamentConfiguration()
        self._config = config
        self.uid: UUID = uuid4() if uid is None else uid

        self.CACHE[self.uid] = self
        self._rounds: list[UUID] = list()
        self._players: list[UUID] = list()
        self._dropped: list[UUID] = list()
        self._round: UUID|None = None

        self.PLAYER_CACHE: dict[UUID, Player] = {}
        self.POD_CACHE: dict[UUID, Pod] = {}
        self.ROUND_CACHE: dict[UUID, Round] = {}
        # Direct setting - don't want to overwrite old log file


    # TOURNAMENT ACTIONS
    # IMPORTANT: No nested tournament actions

    @override
    @classmethod
    def get(cls, uid: UUID) -> Tournament:
        return cls.CACHE[uid]

    @property
    def players(self) -> list[Player]:
        return [Player.get(self, x) for x in self._players]

    @property
    def dropped(self) -> list[Player]:
        return [Player.get(self, x) for x in self._dropped]

    @property
    def round(self) -> Round|None:
        return Round.get(self, self._round) if self._round else None

    @round.setter
    def round(self, round: Round):
        self._round = round.uid

    @property
    def pods(self) -> list[Pod]|None:
        if not self.round:
            return None
        return self.round.pods

    @property
    def rounds(self) -> list[Round]:
        return [Round.get(self, x) for x in self._rounds]

    @rounds.setter
    def rounds(self, rounds: list[Round]):
        self._rounds = [r.uid for r in rounds]

    @property
    def draw_rate(self):
        n_draws = 0
        n_matches = 0
        for round in self.rounds:
            for pod in round.pods:
                if pod.done:
                    n_matches += 1
                    if pod.result_type == Pod.EResult.DRAW:
                        n_draws += len(pod.result)
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
        existing_names = set([p.name for p in self.players])
        for name in names:
            if name in existing_names:
                Log.log('\tPlayer {} already enlisted.'.format(
                    name), level=Log.Level.WARNING)
                continue
            if name:
                p = Player(self, name)
                self._players.append(p.uid)
                new_players.append(p)
                existing_names.add(name)
                Log.log('\tAdded player {}'.format(
                    p.name), level=Log.Level.INFO)
        return new_players

    @TournamentAction.action
    def remove_player(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for p in players:
            if self.round and p.seated:
                if not self.round.concluded:
                    Log.log('Can\'t drop {} during an active round.\nComplete the round or remove player from pod first.'.format(
                        p.name), level=Log.Level.WARNING)
                    continue
            if p.played:
                self.dropped.append(p)
            self._players.remove(p.uid)

            Log.log('\tRemoved player {}'.format(p.name), level=Log.Level.INFO)

    @TournamentAction.action
    def rename_player(self, player, new_name):
        if player.name == new_name:
            return
        if new_name in [p.name for p in self.players]:
            Log.log('\tPlayer {} already enlisted.'.format(
                new_name), level=Log.Level.WARNING)
            return
        if new_name:
            player.name = new_name
            for round in self.rounds:
                for pod in round.pods:
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

    @TournamentAction.action
    def create_pairings(self):
        if self.round is None or self.round.concluded:
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
            self.round = new_round
        assert self.round is not None
        if not self.round.all_players_seated:
            self.round.create_pairings()

            for p in self.round.unseated:
                if p.result!= Player.EResult.LOSS:
                    p.result = Player.EResult.BYE
        else:
            Log.log(30*'*', level=Log.Level.WARNING)
            Log.log('Please report results of following pods: {}'.format(
                ', '.join([
                    str(pod.table)
                    for pod in self.round.pods
                    if not pod.done
                ])
            ), level=Log.Level.WARNING)
            Log.log(30*'*', level=Log.Level.WARNING)

    @TournamentAction.action
    def new_round(self):
        if not self.round or self.round.concluded:
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
            self._round = new_round.uid
            self._rounds.append(new_round.uid)
            return True
        else:
            if self.round.pods:
                Log.log(
                    '{}\n{}\n{}'.format(
                        30*'*',
                        'Please report results of following pods:',
                        30*'*',
                    )
                )
                for pod in self.round.pods:
                    if not pod.done:
                        Log.log(str(pod))
            else:
                Log.log(
                    'Round has no pods - add some or cancel round.'
                )
            return False

    @TournamentAction.action
    def reset_pods(self):
        if self.round:
            self.round._pods = []
            for p in self.round.players:
                if p.result != Player.EResult.LOSS:
                    p.result = Player.EResult.PENDING
                p.pod = None
            self._round = None
            for player in self.players:
                player.location = Player.ELocation.UNSEATED

    @TournamentAction.action
    def manual_pod(self, players: list[Player]):
        if self.round is None or self.round.concluded:
           if not self.new_round():
                return
        assert isinstance(self.round, Round)
        cap = min(self.config.max_pod_size, len(self.round.unseated))
        pod = Pod(self.round, len(self.round.pods), cap=cap)
        self.round._pods.append(pod.uid)

        for player in players:
            pod.add_player(player)
        self.round.pods.append(pod)

    @TournamentAction.action
    def report_win(self, players: list[Player]|Player):
        if self.round:
            if not isinstance(players, list):
                players = [players]
            for p in players:
                Log.log('Reporting player {} won this round.'.format(p.name))
            self.round.assign_win(players)

    @TournamentAction.action
    def report_draw(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        if self.round:
            self.round.draw(players)

    @TournamentAction.action
    def random_results(self):
        if not self.round:
            #Log.log(
            #    'A round is not in progress.\nCreate pods first!',
            #    level=Log.Level.ERROR
            #)
            return
        if self.round.pods:
            draw_rate = 1-sum(self.config.global_wr_seats)
            #for each pod
            #generate a random result based on global_winrates_by_seat
            #each value corresponds to the winrate of the player in that seat
            #the sum of percentages is less than 1, so there is a chance of a draw (1-sum(winrates))

            for pod in [x for x in self.round.pods if not x.done]:
                #generate a random result
                result = random.random()
                rates = np.array(self.config.global_wr_seats[0:len(pod.players)] + [draw_rate])
                rates = rates/sum(rates)
                draw = result > np.cumsum(rates)[-2]
                if not draw:
                    win = np.argmax([result < x for x in rates])
                    #Log.log('won "{}"'.format(pod.players[win].name))
                    self.round.assign_win([pod.players[win]])
                    #player = random.sample(pod.players, 1)[0]
                    #Log.log('won "{}"'.format(player.name))
                    #self.round.won([player])
                else:
                    players = pod.players
                    #Log.log('draw {}'.format(
                    #    ' '.join(['"{}"'.format(p.name) for p in players])))
                    self.round.assign_draw([p for p in players])
        pass

    @TournamentAction.action
    def move_player_to_pod(self, pod: Pod, players: list[Player]|Player, manual=False):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod and player.pod != pod:
                old_pod = player.pod.name
                player.pod.remove_player(player)
                Log.log('Removed player {} from {}.'.format(
                    player.name, old_pod), level=Log.Level.INFO)
            if player.pod != pod:
                if pod.add_player(player, manual=manual):
                    player.result = Player.EResult.PENDING
                    player.location = Player.ELocation.SEATED
                    Log.log('Added player {} to {}'.format(
                        player.name, pod.name), level=Log.Level.INFO)
                else:
                    Log.log('Failed to add palyer {} to Pod {}'.format(
                        player.name, pod.table), level=Log.Level.ERROR)

    @TournamentAction.action
    def bench_players(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            self.remove_player_from_pod(player)

    @TournamentAction.action
    def toggle_game_loss(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.result == Player.EResult.LOSS:
                player.result = Player.EResult.PENDING
                Log.log('{} game loss removed.'.format(
                    player.name), level=Log.Level.INFO)
            else:
                if player.pod is not None:
                    self.remove_player_from_pod(player)
                player.result = Player.EResult.LOSS
                Log.log('{} assigned a game loss.'.format(
                    player.name), level=Log.Level.INFO)

    @TournamentAction.action
    def delete_pod(self, pod: Pod):
        if self.round:
            self.round.remove_pod(pod)

    def remove_player_from_pod(self, player):
        pod = player.pod
        if pod:
            pod.remove_player(player)
            Log.log('Removed player {} from {}.'.format(
                player.name, pod.name), level=Log.Level.INFO)
        return None

    # MISC ACTIONS

    def get_pods_str(self) -> str:
        if not self.round:
            return ''
        export_str = '\n\n'.join([
            pod.__repr__()
            for pod in self.round.pods
        ])

        if self.config.allow_bye and self.round.unseated:
            export_str += '\n\nByes:\n' + '\n:'.join([
                "\t{}\t| pts: {}".format(p.name, p.points)
                for p in self.round.unseated
            ])
        return export_str

    def get_standings(self) -> list[Player]:
        method = Player.SORT_METHOD
        order = Player.SORT_ORDER
        Player.SORT_METHOD = SortMethod.RANK
        Player.SORT_ORDER = SortOrder.ASCENDING
        standings = sorted(self.players, key=self.config.ranking, reverse=True)
        Player.SORT_METHOD = method
        Player.SORT_ORDER = order
        return standings

    def get_standings_str(
            self,
            fields: list[StandingsExport.Field] = StandingsExport.DEFAULT_FIELDS,
            style: StandingsExport.Format = StandingsExport.Format.PLAIN
    ) -> str:
        #fdir = os.path.join(self.TC.log_dir, 'standings.txt')
        standings = self.get_standings()
        lines = [[StandingsExport.info[f].name for f in fields]] # pyright: ignore
        lines += [
            [
                (StandingsExport.info[f].format).format(
                    StandingsExport.info[f].getter(p)
                    if StandingsExport.info[f].denom is None
                    else StandingsExport.info[f].getter(p) * StandingsExport.info[f].denom
                )
                for f
                in fields
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
            - a list of round jsons (serialized from class)

        Objects are referenced to each other by ids.
        """

        data: dict[str, Any] = {}
        data['uid'] = str(self.uid)
        data['config'] = self.config.serialize()
        data['players'] = [p.serialize() for p in self.players]
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
        self.points = 0
        self.CACHE[self.uid] = self
        self._pod_id: UUID|None = None  # Direct reference to current pod

    @property
    def CACHE(self) -> dict[UUID, Player]:
        return self.tour.PLAYER_CACHE

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @tour.setter
    def tour(self, tour: Tournament):
        self._tour = tour.uid

    @staticmethod
    def get(tour: Tournament, uid: UUID):
        return tour.PLAYER_CACHE[uid]

    @property
    def players_beaten(self) -> list[Player]:
        players = set()
        for pod in self.games:
            if pod.result_type == Pod.EResult.WIN and self in pod.result:
                players.update(pod.players)

        players.discard(self)
        return list(players)

    @property
    def average_seat(self) -> float:
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
        if not self.pods:
            return 0.5
        n_pods = len([p for p in self.pods if isinstance(p, Pod)])
        if n_pods == 0:
            return 0.5
        score = 0
        for pod in self.pods:
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
        return score/n_pods

    @property
    def standing(self) -> int:
        standings = self.tour.get_standings()
        if self not in standings:
            return -1
        return standings.index(self) + 1

    @property
    def seated(self) -> bool:
        if self.tour.round is None:
            return False
        for pod in self.tour.round.pods:
            if self in pod.players:
                return True
        return False

    @property
    def pod(self) -> Pod|None:
        if self._pod_id is None:
            return None
        return Pod.get(self.tour, self._pod_id)

    @pod.setter
    def pod(self, pod: Pod|None):
        self._pod_id = pod.uid if pod is not None else None

    @property
    @override
    def played(self) -> list[Player]:
        players = set()
        for p in self.pods:
            if isinstance(p, Pod):
                players.update(p.players)
        return list(players)

    @property
    def not_played(self) -> list[Player]:
        return list(set(self.tour.players) - set(self.played))

    @property
    def winrate(self):
        if len(self.games) == 0:
            return 0
        return self.wins/len(self.games)

    @property
    def opponent_winrate(self):
        if not self.played:
            return 0
        oppwr = [opp.winrate for opp in self.played]
        return sum(oppwr)/len(oppwr)

    @property
    def games(self):
        return [p for p in self.pods if isinstance(p, Pod)]

    @property
    @override
    def byes(self):
        return len([p for p in self.pods if p is Player.EResult.BYE])

    @property
    def wins(self):
        return len([p for p in self.games if p.result is self.uid])

    @property
    def record(self) -> list[Player.EResult]:
        #total_rounds = len(self.tour.rounds) + (1 if self.tour.round else 0)
        seq = list()
        for _, pod in enumerate(self.pods + ([self.pod] if self.tour.round else [])):
            if pod == Player.EResult.BYE:
                seq.append(Player.EResult.BYE)
            elif pod is None:
                if self.result == Player.EResult.LOSS:
                    seq.append(Player.EResult.LOSS)
                else:
                    seq.append(Player.EResult.BYE)
            elif isinstance(pod, Pod):
                if pod.result_type:
                    if pod.result_type == Pod.EResult.WIN and self.uid in pod.result:
                        seq.append(Player.EResult.WIN)
                    elif pod.result_type == Pod.EResult.DRAW and self.uid in pod.result:
                        seq.append(Player.EResult.DRAW)
                    seq.append(Player.EResult.LOSS)
                else:
                    seq.append(Player.EResult.PENDING)

        '''record_sequence = ''.join(seq)
        return ('{} ({}/{}/{})'.format(
            record_sequence.ljust(total_rounds),
            record_sequence.count('W') + record_sequence.count('B'),
            record_sequence.count('L'),
            record_sequence.count('D'),
        ))'''
        return seq

    @staticmethod
    def fmt_record(record:list[Player.EResult]) -> str:
        return ''.join([{
            Player.EResult.WIN: 'W',
            Player.EResult.LOSS: 'L',
            Player.EResult.DRAW: 'D',
            Player.EResult.BYE: 'B',
            Player.EResult.PENDING: '_',
        }[r] for r in record])

    @property
    def seat_history(self) -> str:
        if sum([1 for p in self.pods if isinstance(p, Pod)]) == 0:
            return 'N/A'
        ret_str = ' '.join([
            '{}/{}'.format(
                ([x.uid for x in p.players]).index(self.uid)+1,
                len(p.players)
            )
            if isinstance(p, Pod)
            else 'N/A'
            for p in self.pods

        ])
        return ret_str

    def __gt__(self, other: Player):
        b = False
        if self.SORT_METHOD == SortMethod.ID:
            b = self.uid > other.uid
        elif self.SORT_METHOD == SortMethod.NAME:
            b = self.name > other.name
        elif self.SORT_METHOD == SortMethod.RANK:
            my_score = self.tour.config.ranking(self)
            other_score = self.tour.config.ranking(other)
            b = None
            for i in range(len(my_score)):
                if my_score[i] != other_score[i]:
                    b = my_score[i] > other_score[i]
                    break
        return b

    def __lt__(self, other: Player):
        b = False
        if self.SORT_METHOD == SortMethod.ID:
            b = self.uid < other.uid
        elif self.SORT_METHOD == SortMethod.NAME:
            b = self.name < other.name
        elif self.SORT_METHOD == SortMethod.RANK:
            my_score = self.tour.config.ranking(self)
            other_score = self.tour.config.ranking(other)
            b = None
            for i in range(len(my_score)):
                if my_score[i] != other_score[i]:
                    b = my_score[i] > other_score[i]
                    break
        return b

    @override
    def __repr__(self, tokens=None):
        if len(self.tour.players) == 0:
            return ''
        if not tokens:
            tokens = self.FORMATTING
        parser_player = argparse.ArgumentParser()

        parser_player.add_argument(
            '-n', '--stanti[n]g',
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
            '-r', '--winrate',
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
        '''parser_player.add_argument(
            '-s', '--spaces',
            dest='spaces', type=int, default=0)'''
        # parser.add_argument('-n', '--notplayed',    dest='np', action='store_true')

        args, _ = parser_player.parse_known_args(tokens)

        fields = list()

        tsize = int(math.floor(math.log10(len(self.tour.players))) + 1)
        pname_size = max([len(p.name) for p in self.tour.players])

        if args.standing:
            fields.append('#{:>{}}'.format(self.standing, tsize))
        if args.id:
            fields.append('[{:>{}}] {}'.format(
                self.uid, tsize, self.name.ljust(pname_size)))
        else:
            fields.append(self.name.ljust(pname_size))

        if args.pod and self.tour.round and len(self.tour.round.pods) > 0:
            max_pod_id = max([len(str(p.table)) for p in self.tour.round.pods])
            if self.pod:
                #find number of digits in max pod id
                fields.append('{}'.format(
                    f'P{str(self.pod.table).zfill(max_pod_id)}/S{self.pod.players.index(self)}' if self.pod else ''))
            elif self.result == Player.EResult.LOSS:
                fields.append('Loss'.ljust(max_pod_id+4))
            else:
                fields.append('Bye'.ljust(max_pod_id+4))
        if args.p:
            fields.append('pts: {}'.format(self.points))
        if args.w:
            fields.append('w: {}'.format(self.wins))
        if args.ow:
            fields.append('o.wr.: {:.2f}'.format(self.opponent_winrate))
        if args.u:
            fields.append('uniq: {}'.format(self.played))
        if args.s:
            fields.append('seat: {:02.00f}%'.format(self.average_seat*100))
        # if args.np:
        #    fields.append(''.format())
        # OUTPUT_BUFFER.append('\t{}'.format(' | '.join(fields)))

        return ' | '.join(fields)

    def serialize(self) -> dict[str, Any]:
        return {
            'tour': str(self._tour),
            'uid': str(self.uid),
            'name': self.name,
        }

    @classmethod
    def inflate(cls, tour: Tournament, data: dict[str, Any]) -> Player:
        assert tour.uid == UUID(data['tour'])
        return cls(tour, data['name'], UUID(data['uid']))


class Pod(IPod):
    def __init__(self, round: Round, table:int, cap=0, uid: UUID|None = None):
        super().__init__()
        self._tour: UUID = round.tour.uid
        self._round: UUID = round.uid
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

    @property
    def result_type(self) -> None|Pod.EResult:
        if self.result:
            if len(self.result) == 1:
                return Pod.EResult.WIN
            return Pod.EResult.DRAW
        return None

    @property
    def done(self) -> bool:
        return len(self.result) > 0

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @tour.setter
    def tour(self, tour: Tournament):
        self._tour = tour.uid

    @property
    def round(self) -> Round:
        return Round.get(self.tour, self._round)

    @property
    def players(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._players]

    @override
    def add_player(self, player: Player, manual=False, player_pod_map=None) -> bool:
        if len(self) >= self.cap and self.cap and not manual:
            return False
        if player.pod is not None:
            player.pod.remove_player(player)
        self._players.append(player.uid)
        player.location = Player.ELocation.SEATED
        player.pod = self  # Update player's pod reference
        return True

    def remove_player(self, player: Player, cleanup=True) -> Player|None:
        try:
            idx = self._players.index(player.uid)
        except ValueError:
            return None
        p = self._players.pop(idx)
        player.location = Player.ELocation.UNSEATED
        player.pod = None  # Clear player's pod reference
        if len(self) == 0 and cleanup:
            self.round.remove_pod(self)
        return player

    @property
    def average_seat(self) -> float:
        return np.average([p.average_seat for p in self.players]).astype(float)

    @property
    def balance(self) -> np.ndarray:
        '''
        Returns a list of count of players above 50% average seat and below 50% average seat
        '''
        return np.array([
            sum([1 for p in self.players if p.average_seat > 0.5]),
            sum([1 for p in self.players if p.average_seat < 0.5])
        ])

    @override
    def assign_seats(self):
        # Average seating positions
        average_positions = [p.average_seat for p in self.players]
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
        self.players[:] = np.take(self.players, np.argsort(seat_assignment))

        pass

    def clear(self):
        for p in self.players:
            p.location = Player.ELocation.UNSEATED
            p.pod = None  # Clear pod references
        self.players.clear()

    @property
    def name(self):
        return 'Pod {}'.format(self.table)

    @override
    def __repr__(self):
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
                        ' ' if not self.result else
                        'W' if self.result_type == Pod.EResult.WIN and p.uid in self.result else
                        'D' if self.result_type == Pod.EResult.DRAW and p.uid in self.result else
                        'L',
                        p.__repr__(['-s', str(maxlen), '-p']))
                    for _, p in
                    zip(range(1, len(self)+1), self.players)
                ]
            ))
        return ret

    def serialize(self) -> dict[str, Any]:
        return {
            'uid': str(self.uid),
            'round': str(self._round),
            'table': self.table,
            'cap': self.cap,
            'result': sorted([str(p) for p in self.result]),
            'players': [str(p) for p in self._players],
        }

    @classmethod
    def inflate(cls, round: Round, data: dict[str, Any]) -> Pod:
        assert round.uid == UUID(data['round'])
        pod = cls(round, data['table'], data['cap'], UUID(data['uid']))
        pod._players = [x for x in data['players']]
        pod.result = {x for x in data['result']}
        return pod


class Round(IRound):

    def __init__(self, tour: Tournament, seq: int, pairing_logic:IPairingLogic, uid: UUID|None = None):
        super().__init__()
        if uid is not None:
            self.uid = uid
        self._tour: UUID = tour.uid
        self.tour.ROUND_CACHE[self.uid] = self
        self._players: list[UUID] = [p.uid for p in self.tour.players]
        self.seq:int = seq
        self._logic = pairing_logic.name

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
    def players(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._players]

    @players.setter
    def players(self, players: list[Player]):
        self._players = [p.uid for p in players]

    @property
    def pods(self) -> list[Pod]:
        return [Pod.get(self.tour, x) for x in self._pods]

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @property
    def done(self):
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    @property
    def all_players_seated(self):
        seated = len(self.seated)
        n_players_to_play = seated + len(self.unseated)
        if self.tour.get_pod_sizes(n_players_to_play) is None:
            return False
        if not (pod_sizes:=self.tour.get_pod_sizes(n_players_to_play)):
            return True
        else:
            return sum(pod_sizes) == seated

    @property
    def seated(self) -> list[Player]:
        return [p for p in self.players if p.location == Player.ELocation.SEATED]

    @property
    def unseated(self) -> list[Player]:
        return [
            p
            for p in self.players
            if p.location == Player.ELocation.UNSEATED
            and not p.result == Player.EResult.LOSS
        ]

    def remove_pod(self, pod: Pod):
        if not pod.done:
            pod.clear()
            self.pods.remove(pod)
            return True
        return False

    def create_pods(self):
        seats_required = len(self.unseated) - sum([pod.cap-len(pod) for pod in self.pods if not pod.done])
        if seats_required == 0:
            return
        pod_sizes = self.tour.get_pod_sizes(seats_required)
        if pod_sizes is None:
            Log.log('Can not make pods.', level=Log.Level.WARNING)
            return None
        start_table = len(self.pods) + 1
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
        self.logic.make_pairings(self.unseated, pods)
        for pod in self.pods:
            pod.assign_seats()
        self.sort_pods()
        for pod in self.pods:
            Log.log(pod, Log.Level.NONE)

    def sort_pods(self):
        pods_sorted = sorted(self.pods, key=lambda x: (len(x.players), np.average([p.points for p in x.players])), reverse=True)
        self._pods[:] = [pod.uid for pod in pods_sorted]

    def assign_win(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            pod = player.pod

            if not player or not pod:
                Log.log('Player {} not found in any pod'.format(
                    player.name), Log.Level.WARNING)
                continue

            if not pod.done:
                player.points = player.points + self.tour.config.win_points

                pod.result.add(player.uid)

            #if self.done:
            #    self.conclude()

    def assign_draw(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod is not None:
                player.points = player.points + self.tour.config.draw_points

                player.pod.result.add(player.uid)

        #if self.done and not self.concluded:
        #    self.conclude()

    @property
    def concluded(self) -> bool:
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    def conclude(self):
        raise DeprecationWarning('Round.conclude is deprecated. Rounds are no longer concluded manually.')
        for pod in self.pods:
           for p in pod.players:
               p.pods.append(pod)

        for p in self.unseated:
            if p.result == Player.EResult.LOSS:
                p.pods.append(Player.EResult.LOSS)
            elif self.tour.config.allow_bye:
                p.points += self.tour.config.bye_points
                p.result = Player.EResult.BYE
                p.pods.append(Player.EResult.BYE)

        self.tour.rounds.append(self)
        self.concluded = datetime.now()
        Log.log('{}{}{}'.format(
            30*'*', '\nRound completed!\n', 30*'*',), Log.Level.INFO)
        self.tour._round = None
        for p in self.tour.players:
            p.location = Player.ELocation.UNSEATED
            p.pod = None
            #p.result = Player.EResult.PENDING
        pass

    def serialize(self) -> dict[str, Any]:
        return {
            'tour': str(self._tour),
            'uid': str(self.uid),
            'seq': self.seq,
            'pods': [pod.serialize() for pod in self.pods],
            'players': [str(p) for p in self._players],
            'logic': self._logic,
        }

    @classmethod
    def inflate(cls, tour: Tournament, data: dict[str, Any]) -> Round:
        assert tour.uid == UUID(data['tour'])
        tour.discover_pairing_logic()
        logic = tour.get_pairing_logic(data['logic'])
        new_round = cls(tour, data['seq'], logic, UUID(data['uid']))
        pods = [Pod.inflate(new_round, pod) for pod in data['pods']]
        new_round._pods = [pod.uid for pod in pods]
        return new_round

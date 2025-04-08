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
from .pairing_logic.examples import PairingRandom, PairingSnake, PairingDefault
from uuid import UUID, uuid4

from dotenv import load_dotenv
import requests
import threading

from eventsourcing.application import Application, Aggregate


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
            if self.TC.auto_export:
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
                        if self.TC.allow_bye and byes:
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
            'getter': lambda p: p.ID.hex
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
            if self.TC.auto_export:
                self.export_str(
                    self.get_standings_str(),
                    self.TC.standings_export.dir,
                    DataExport.Target.FILE
                )
            return ret
        return auto_standings_export_wrapper


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


class ID:
    def __init__(self):
        self._last_ID = 0

    def next(self) -> UUID:
        #self._last_ID += 1
        #return self._last_ID
        return uuid4()


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
                        Tournament.CACHE[action.before.ID] = action.before
                        Tournament.CACHE[action.after.ID] = action.after
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


class Core(Application):
    def get_tournament(self, id: UUID) -> Tournament:
        return Tournament.from_aggregate(self, self.repository.get(id))

    def get_player(self, id: UUID) -> Player:
        return Player.from_aggregate(self, self.repository.get(id))

class AggregateWrapper:
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.aggregate, item)

class TournamentConfiguration(AggregateWrapper):
    def __init__(
        self,
        core: Core,
        aggregate: ITournamentConfiguration|None=None,
        **kwargs
    ):
        self._core = core
        if aggregate is None:
            self.aggregate = ITournamentConfiguration(kwargs)
            self._core.save(self.aggregate)
        else:
            self.aggregate = aggregate
        pass
        #self.pod_sizes = kwargs.get('pod_sizes', [4, 3])
        #self.allow_bye = kwargs.get('allow_bye', False)
        #self.win_points = kwargs.get('win_points', 5)
        #self.bye_points = kwargs.get('bye_points', 2)
        #self.draw_points = kwargs.get('draw_points', 1)
        #self.snake_pods = kwargs.get('snake_pods', False)
        #self.n_rounds = kwargs.get('n_rounds', 5)
        #self.max_byes = kwargs.get('max_byes', 2)
        #self.auto_export = kwargs.get('auto_export', False)
        #self.standings_export = kwargs.get('standings_export', StandingsExport())
        #self.player_id = kwargs.get('player_id', ID())
        #self.global_wr_seats = kwargs.get('global_wr_seats', [
        #    0.2553,
        #    0.2232,
        #    0.1847,
        #    0.1428,
        #])

    @classmethod
    def from_aggregate(cls, core: Core, aggregate: ITournamentConfiguration) -> 'TournamentConfiguration':
        return cls(core, aggregate)

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
            -x.ID if isinstance(x.ID, int) else -int(x.ID.int)
        )

    @override
    def __repr__(self):
        return "Tour. cfg:" + '|'.join([
            '{}:{}'.format(key, val)
            for key, val in self.__dict__.items()
        ])


class Tournament(AggregateWrapper):
    is_snapshotting_enabled = True
    _instance_cache: dict[UUID, 'Tournament'] = {}

    def __init__(
        self,
        core: Core,
        config: TournamentConfiguration,
        aggregate: ITournament|None=None
    ):
        """Wrap ITournament with a repository for dynamic behavior."""
        self._core = core  # Store repository
        if aggregate is None:
            self.aggregate = ITournament(config.id)
            self._core.save(self.aggregate)
        else:
            self.aggregate = aggregate
        # Cache the instance
        Tournament._instance_cache[self.aggregate.id] = self

    def save(self, aggregate: Aggregate):
        self._core.save(aggregate)

    @property
    def id(self) -> UUID:
        return self.aggregate.id

    @property
    def config(self) -> TournamentConfiguration:
        return TournamentConfiguration.from_aggregate(self._core, self._core.repository.get(self.aggregate.config))

    @classmethod
    def from_aggregate(cls, core: Core, aggregate: ITournament) -> "Tournament":
        # Check if we already have this instance cached
        if aggregate.id in cls._instance_cache:
            return cls._instance_cache[aggregate.id]

        tournament = core.repository.get(aggregate.id)
        config = TournamentConfiguration.from_aggregate(core, core.repository.get(tournament.config))

        return cls(core, config, aggregate)

    @property
    def players(self) -> list[Player]:
        """Automatically resolve player objects from repository."""
        return [Player.from_aggregate(self._core, self._core.repository.get(player_id)) for player_id in self.aggregate.players]

    def add_player(self, names: str|list[str]|None=None):
        new_players = []
        if isinstance(names, str):
            names = [names]
        existing_names = set([p.name for p in self.players])
        for name in names:
            if name in existing_names:
                #Log.log('\tPlayer {} already enlisted.'.format(
                #    name), level=Log.Level.WARNING)
                continue
            if name:
                p = Player(self, name)
                self.aggregate.add_player(p.aggregate)
                new_players.append(p)
                existing_names.add(name)
                #Log.log('\tAdded player {}'.format(
                #    p.name), level=Log.Level.INFO)
        return new_players

    def create_pairings(self):
        if self.round is None or self.round.concluded:
            seq = len(self.rounds)
            if seq == 0:
                logic = PairingRandom()
            elif seq == 1 and self.TC.snake_pods:
                logic = PairingSnake()
            else:
                logic = PairingDefault()
            self._round = Round(
                len(self.rounds),
                logic,
                self
            ).ID
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


class Player(AggregateWrapper):
    def __init__(
        self,
        tour: Tournament,
        name: str,
        aggregate: IPlayer|None = None
    ):
        self.tour = tour
        if aggregate is None:
            self.aggregate = IPlayer(name, tour.id)
            self.tour.save(self.aggregate)
        else:
            self.aggregate = aggregate

    @classmethod
    def from_aggregate(cls, core: Core, aggregate: IPlayer) -> 'Player':
        tour = core.get_tournament(aggregate.tour)
        return cls(tour, aggregate.name, aggregate)


class Pod(AggregateWrapper):
    def __init__(self, tour: Tournament, round: Round, table:int, cap=0):
        super().__init__(tour.ID, round.ID, table, cap)


    @property
    def done(self) -> bool:
        return len(self.result) > 0

    @property
    def tour(self) -> Tournament:
        return Tournament.get(self._tour)

    @tour.setter
    def tour(self, tour: Tournament):
        self._tour = tour.ID

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
        super().add_player(player)
        player.location = Player.ELocation.SEATED
        player.pod = self  # Update player's pod reference
        return True

    def remove_player(self, player: Player, cleanup=True) -> Player|None:
        try:
            idx = self._players.index(player.ID)
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
                        'W' if self.result_type == Pod.EResult.WIN and p.ID in self.result else
                        'D' if self.result_type == Pod.EResult.DRAW and p.ID in self.result else
                        'L',
                        p.__repr__(['-s', str(maxlen), '-p']))
                    for _, p in
                    zip(range(1, len(self)+1), self.players)
                ]
            ))
        return ret


class Round(AggregateWrapper):
    def __init__(
        self,
        seq: int,
        pairing_logic: IPairingLogic,
        tour: Tournament,
        aggregate: IRound|None = None
    ):
        self.tour = tour
        if aggregate is None:
            self.aggregate = IRound(tour.id, seq, pairing_logic)
            self.tour.save(self.aggregate)
        else:
            self.aggregate = aggregate

    @classmethod
    def from_aggregate(cls, core: Core, aggregate: IRound) -> 'Round':
        tour = core.get_tournament(aggregate._tour)
        return cls(
            aggregate.seq,
            aggregate.logic,
            tour,
            aggregate
        )

    @property
    def CACHE(self) -> dict[UUID, Round]:
        return self.tour.ROUND_CACHE

    @staticmethod
    def get(tour: Tournament, ID: UUID) -> Round:
        return tour.ROUND_CACHE[ID]

    @property
    def players(self) -> list[Player]:
        return [Player.get(self.tour, x) for x in self._players]

    @players.setter
    def players(self, players: list[Player]):
        self._players = [p.ID for p in players]

    @property
    def pods(self) -> list[Pod]:
        return [Pod.get(self.tour, x) for x in self._pods]

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
            self._pods.append(pod.ID)

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
        self._pods[:] = [pod.ID for pod in pods_sorted]

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
                player.points = player.points + self.tour.TC.win_points

                pod.result.add(player.ID)

            if self.done:
                self.conclude()

    def assign_draw(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod is not None:
                player.points = player.points + self.tour.TC.draw_points

                player.pod.result.add(player.ID)

        if self.done and not self.concluded:
            self.conclude()

    def conclude(self):
        for pod in self.pods:
           for p in pod.players:
               p.pods.append(pod)

        for p in self.unseated:
            if p.result == Player.EResult.LOSS:
                p.pods.append(Player.EResult.LOSS)
            elif self.tour.TC.allow_bye:
                p.points += self.tour.TC.bye_points
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

    def __del__(self):
        # Remove from cache when deleted
        if self.aggregate.id in Tournament._instance_cache:
            del Tournament._instance_cache[self.aggregate.id]
from __future__ import annotations
from typing import List, Union, Callable, Any
from typing_extensions import override

import sys
import argparse
import math
import os
import pickle
import random
from copy import deepcopy
from datetime import datetime
from enum import Enum

from .interface import IPlayer, ITournament, IPod, IRound, IPairingLogic
from .misc import Json2Obj
import numpy as np
from tqdm import tqdm # pyright: ignore
import json # pyright: ignore
from .pairing_logic.examples import PairingRandom, PairingSnake, PairingDefault

class PodsExport:
    @classmethod
    def auto_export(cls, func):
        def auto_pods_export_wrapper(self: Tournament, *original_args, **original_kwargs):
            tour_round = self.round
            ret = func(self, *original_args, **original_kwargs)
            tour_round = tour_round or self.round
            if self.TC.auto_export:
                logf = TournamentAction.logf
                if logf and tour_round:
                    # Export pods to a file named {tournament_name}_round_{round_number}.txt
                    # And also export it into {log_directory}/pods.txt

                    export_str: str = '\n\n'.join([
                        pod.__repr__()
                        for pod in tour_round.pods
                    ])

                    game_lost: list[Player] = [x for x in self.players if x.game_loss]
                    byes = [x for x in tour_round.unseated if not x.game_loss]
                    if len(game_lost) + len(byes) > 0:
                        max_len = max([len(p.name) for p in game_lost + byes])
                        if self.TC.allow_bye and byes:
                            export_str += '\n\nByes:\n' + '\n'.join([
                                "\t{} | pts: {}".format(p.name.ljust(max_len), p.points)
                                for p in tour_round.unseated
                                if not p.game_loss
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
                    self.export_str(path, export_str)

                    path = os.path.join(os.path.dirname(logf), 'pods.txt')
                    self.export_str(path, export_str)

            return ret
        return auto_pods_export_wrapper


class StandingsExport:
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

    class Format(Enum):
        TXT = 0
        CSV = 1
        DISCORD = 2
        JSON = 3

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
            'format': '{:d}',
            'denom': None,
            'description': 'Player ID',
            'getter': lambda p: p.ID
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
            'getter': lambda p: p.unique_opponents
        }),
        Field.GAMES: Json2Obj({
            'name': '# games',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of games played',
            'getter': lambda p: p.games_played
        }),
        Field.OPP_BEATEN: Json2Obj({
            'name': '# opp. beat',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of opponents beaten',
            'getter': lambda p: p.n_opponents_beaten
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
        Format.DISCORD: '.txt',
        Format.TXT: '.txt',
        Format.CSV: '.csv'
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

    def __init__(self, fields=None, format: Format = Format.TXT, dir: Union[str, None] = None):
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
                self.export_standings(
                    fdir=self.TC.standings_export.dir,
                    fields=self.TC.standings_export.fields,
                    style=self.TC.standings_export.format,
                )
            return ret
        return auto_standings_export_wrapper


class SORT_METHOD(Enum):
    ID = 0
    NAME = 1
    RANK = 2


class SORT_ORDER(Enum):
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

    @classmethod
    def log(cls, str_log, level=Level.NONE):
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

    def next(self) -> int:
        self._last_ID += 1
        return self._last_ID


class TournamentAction:
    '''Serializable action that will be stored in tournament log and can be restored
    '''
    LOGF: Union[str, bool, None] = None
    ACTIONS: List = []
    logf: Union[str, None] = None
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
            after = deepcopy(self) #TODO: Crash
            cls.ACTIONS.append(TournamentAction(
                before, ret, after, func.__name__, *original_args, **original_kwargs,
            ))
            cls.store()
            return ret
        return wrapper

    @classmethod
    def store(cls):
        if cls.logf is None:
            cls.logf = cls.DEFAULT_LOGF
        if cls.logf:
            if not os.path.exists(os.path.dirname(cls.logf)):
                os.makedirs(os.path.dirname(cls.logf))
            with open(cls.logf, 'wb') as f:
                pickle.dump(cls.ACTIONS, f)

    @classmethod
    def load(cls, logdir='logs/default.log'):
        if os.path.exists(logdir):
            cls.logf = logdir
            try:
                with open(cls.logf, 'rb') as f:
                    cls.ACTIONS = pickle.load(f)
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


class TournamentConfiguration:
    def __init__(self, **kwargs):
        self.pod_sizes = kwargs.get('pod_sizes', [4, 3])
        self.allow_bye = kwargs.get('allow_bye', False)
        self.win_points = kwargs.get('win_points', 5)
        self.bye_points = kwargs.get('bye_points', 2)
        self.draw_points = kwargs.get('draw_points', 1)
        self.snake_pods = kwargs.get('snake_pods', False)
        self.n_rounds = kwargs.get('n_rounds', 5)
        self.max_byes = kwargs.get('max_byes', 2)
        self.auto_export = kwargs.get('auto_export', False)
        self.standings_export = kwargs.get(
            'standings_export', StandingsExport())
        self.player_id = kwargs.get('player_id', ID())
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
    def ranking(x):
        return (
            x.points,
            x.games_played,
            np.round(x.opponent_winrate, 2),
            x.n_opponents_beaten,
            -x.average_seat,
            -x.ID
        )

    @staticmethod
    def matching(x):
        return (
            -x.games_played,
            x.points,
            -x.unique_opponents,
            x.opponent_winrate
        )

    @override
    def __repr__(self):
        return "Tour. cfg:" + '|'.join([
            '{}:{}'.format(key, val)
            for key, val in self.__dict__.items()
        ])


class Tournament(ITournament):
    # CONFIGURATION
    # Logic: Points is primary sorting key,
    # then opponent winrate, - CHANGE - moved this upwards and added dummy opponents with 33% winrate
    # then number of opponents beaten,
    # then ID - this last one is to ensure deterministic sorting in case of equal values (start of tournament for example)

    def __init__(self, config: Union[TournamentConfiguration, None] = None) :  # type: ignore
        if config is None:
            config = TournamentConfiguration()
        self.rounds: list[Round] = list()
        self.players: list[Player] = list()
        self.dropped: list[Player] = list()
        self.round: Round|None = None

        # Direct setting - don't want to overwrite old log file
        self._tc = config

    # TOURNAMENT ACTIONS
    # IMPORTANT: No nested tournament actions

    @property
    def draw_rate(self):
        n_draws = 0
        n_matches = 0
        for round in self.rounds:
            for pod in round.pods:
                if pod.done:
                    n_matches += 1
                    if pod.draw:
                        n_draws += len(pod.draw)
        return n_draws/n_matches

    @property
    def TC(self):
        return self._tc

    @TC.setter
    @TournamentAction.action
    def TC(self, config):
        self._tc = config

    @TournamentAction.action
    def add_player(self, names: str|list[str]|None=None):
        new_players = []
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name in [p.name for p in self.players]:
                Log.log('\tPlayer {} already enlisted.'.format(
                    name), level=Log.Level.WARNING)
                continue
            if name:
                p = Player(name, tour=self)
                self.players.append(p)
                new_players.append(p)
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
            self.players.remove(p)

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
        # tails = {}
        for pod_size in self.TC.pod_sizes:
            rem = n-pod_size
            if rem < 0:
                continue
            if rem == 0:
                return [pod_size]
            if rem < self.TC.min_pod_size:
                if self.TC.allow_bye and rem <= self.TC.max_byes:
                    return [pod_size]
                elif pod_size == self.TC.pod_sizes[-1]:
                    return None
            if rem >= self.TC.min_pod_size:
                # This following code prefers smaller pods over byes
                # tails[(rem, pod_size)] = self.get_pod_sizes(rem)
                # if tails[(rem, pod_size)] is not None:
                #    if sum(tails[(rem, pod_size)]) == rem:
                #        return sorted([pod_size] + tails[(rem, pod_size)], reverse=True)
                tail = self.get_pod_sizes(rem)
                if tail is not None:
                    return [pod_size] + tail

        return None

    @TournamentAction.action
    def create_pairings(self):
        if self.round is None or self.round.concluded:
            seq = len(self.rounds)
            if seq == 0:
                logic = PairingRandom()
            elif seq == 1 and self.TC.snake_pods:
                logic = PairingSnake()
            else:
                logic = PairingDefault()
            self.round = Round(
                len(self.rounds),
                logic,
                self
            )
        if not self.round.all_players_seated:
            self.round.create_pairings()
        else:
            Log.log(30*'*', level=Log.Level.WARNING)
            Log.log('Please report results of following pods: {}'.format(
                ', '.join([
                    str(pod.id)
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
                logic = PairingRandom()
            elif seq == 1 and self.TC.snake_pods:
                logic = PairingSnake()
            else:
                logic = PairingDefault()
            self.round = Round(seq, logic, self)
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
            self.round.pods = []
            self.round = None

    @TournamentAction.action
    def manual_pod(self, players: list[Player]):
        if self.round is None or self.round.concluded:
            seq = len(self.rounds)
            if seq == 0:
                logic = PairingRandom()
            elif seq == 1 and self.TC.snake_pods:
                logic = PairingSnake()
            else:
                logic = PairingDefault()
            self.round = Round(seq, logic, self)
        if not self.round.pods:
            self.round.pods = []

        pod = Pod(self.round, len(self.round.pods))

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
            self.round.won(players)

    @TournamentAction.action
    def report_draw(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        if self.round:
            self.round.draw(players)

    @TournamentAction.action
    def random_results(self):
        if not self.round:
            Log.log(
                'A round is not in progress.\nCreate pods first!',
                level=Log.Level.ERROR
            )
            return
        if self.round.pods:
            draw_rate = 1-sum(self.TC.global_wr_seats)
            #for each pod
            #generate a random result based on global_winrates_by_seat
            #each value corresponds to the winrate of the player in that seat
            #the sum of percentages is less than 1, so there is a chance of a draw (1-sum(winrates))

            for pod in [x for x in self.round.pods if not x.done]:
                #generate a random result
                result = random.random()
                rates = np.array(self.TC.global_wr_seats[0:len(pod.players)] + [draw_rate])
                rates = rates/sum(rates)
                draw = result > np.cumsum(rates)[-2]
                if not draw:
                    win = np.argmax([result < x for x in rates])
                    Log.log('won "{}"'.format(pod.players[win].name))
                    self.round.won([pod.players[win]])
                    #player = random.sample(pod.players, 1)[0]
                    #Log.log('won "{}"'.format(player.name))
                    #self.round.won([player])
                else:
                    #players = random.sample(
                    #    pod.players, random.randint(1, pod.p_count))
                    players = pod.players
                    Log.log('draw {}'.format(
                        ' '.join(['"{}"'.format(p.name) for p in players])))
                    self.round.draw([p for p in players])
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
                    player.game_loss = False
                    Log.log('Added player {} to {}'.format(
                        player.name, pod.name), level=Log.Level.INFO)
                else:
                    Log.log('Failed to add palyer {} to Pod {}'.format(
                        player.name, pod.id), level=Log.Level.ERROR)

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
            player.game_loss = not player.game_loss
            if player.game_loss:
                if player.pod is not None:
                    self.remove_player_from_pod(player)
                Log.log('{} assigned a game loss.'.format(
                    player.name), level=Log.Level.INFO)
            else:
                Log.log('{} game loss removed.'.format(
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

    def show_pods(self):
        if self.round and self.round.pods:
            self.round.print_pods()
        else:
            Log.log('No pods currently created.')

    def export_str(self, fdir, str):
        if not os.path.exists(os.path.dirname(fdir)):
            os.makedirs(os.path.dirname(fdir))
        with open(fdir, 'w', encoding='utf-8') as f:
            f.writelines(str)

    def get_standings(self) -> list[Player]:
        method = Player.SORT_METHOD
        order = Player.SORT_ORDER
        Player.SORT_METHOD = SORT_METHOD.RANK
        Player.SORT_ORDER = SORT_ORDER.ASCENDING
        standings = sorted(self.players, key=self.TC.ranking, reverse=True)
        Player.SORT_METHOD = method
        Player.SORT_ORDER = order
        return standings

    def export_standings(
        self,
        fdir: str,
        fields: list[StandingsExport.Field]|None = None,
        style: StandingsExport.Format|None = None,
    ):
        if fields is None:
            fields = StandingsExport.DEFAULT_FIELDS
        if style is None:
            style = StandingsExport.Format.TXT
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
        if style == StandingsExport.Format.TXT:
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

            self.export_str(fdir, lines)
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

    def parsable_log(self) -> dict[int, dict]:
        data = {}
        for r in self.rounds:
            data[r.seq] = {}
            for p in r.pods:
                data[r.seq][p.id] = {
                    'players': {
                        i: {
                            'id': x.ID,
                            'name': x.name,
                            'points': x.points,
                        }
                        for i, x in enumerate(p.players)
                    },
                    'result': (
                        [p.won.ID] if p.won else [pl.ID for pl in p.draw]
                    )
                }
            data[r.seq]['bye'] = [x.ID for x in r.unseated if not x.game_loss]
            data[r.seq]['game_loss'] = [x.ID for x in r.unseated if x.game_loss]
            data[r.seq]['drop'] = [x.ID for x in self.dropped]

        return data


class TournamentLog: #TODO: Implement
    class Format(Enum):
        TXT = 0
        DISCORD = 1
        JSON = 2

    def __init__(self, tournament: Tournament):
        self.tournament = tournament
        self.log = []

    '''def construct(self):
        players = set()
        for round in self.tournament.rounds:
            pass'''


class Player(IPlayer):
    SORT_METHOD: SORT_METHOD = SORT_METHOD.ID
    SORT_ORDER: SORT_ORDER = SORT_ORDER.ASCENDING
    FORMATTING = ['-p']

    def __init__(self, name:str, tour: Tournament):
        super().__init__()
        self.tour = tour
        self.name = name
        self.points = 0
        self.played = list()
        self.ID = tour.TC.player_id.next()
        self.games_played = 0
        self.games_won = 0
        self.opponents_beaten = set()
        self.game_loss = False
        self.byes = 0

    @property
    def average_seat(self):
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
                index = pod.players.index(self)
                if index == 0:
                    score += 1
                elif index == pod.p_count - 1:
                    continue
                else:
                    rates = self.tour.TC.global_wr_seats[0:pod.p_count]
                    norm_scale = 1-(np.cumsum(rates)-rates[0])/(np.sum(rates)-rates[0])
                    score += norm_scale[index]
        return score/n_pods

    @property
    def standing(self):
        standings = self.tour.get_standings()
        if self not in standings:
            return -1
        return standings.index(self) + 1

    @property
    def n_opponents_beaten(self):
        return len(self.opponents_beaten)

    @property
    def seated(self):
        if self.tour.round is None:
            return False
        for pod in self.tour.round.pods:
            if self in pod.players:
                return not pod.done
        return False

    @property
    def pod(self) -> Pod|None:
        if self.tour.round is None:
            return None
        for pod in self.tour.round.pods:
            if self in pod.players:
                return pod
        return None

    @property
    def not_played(self):
        return list(set(self.tour.players) - set(self.played))

    @property
    def unique_opponents(self):
        return len(set(self.played))

    @property
    def winrate(self):
        if self.games_played == 0:
            return 0
        return self.games_won/self.games_played

    @property
    def opponent_winrate(self):
        if not self.played:
            return 0
        oppwr = [opp.winrate for opp in self.played]
        return sum(oppwr)/len(oppwr)

    @property
    def record(self) -> list[Player.EResult]:
        #total_rounds = len(self.tour.rounds) + (1 if self.tour.round else 0)
        seq = list()
        for _, pod in enumerate(self.pods + ([self.pod] if self.tour.round else [])):
            if pod == Player.EResult.BYE:
                seq.append(Player.EResult.BYE)
            elif pod is None:
                if self.game_loss:
                    seq.append(Player.EResult.LOSS)
                else:
                    seq.append(Player.EResult.BYE)
            elif isinstance(pod, Pod):
                if pod.done:
                    if pod.won is not None:
                        if pod.won is self:
                            seq.append(Player.EResult.WIN)
                        else:
                            seq.append(Player.EResult.LOSS)
                    else:
                        if self in pod.draw:
                            seq.append(Player.EResult.DRAW)
                        else:
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
    def fmt_record(record:list[IPlayer.EResult]) -> str:
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
                p.players.index(self)+1, len(p.players)
            )
            if isinstance(p, Pod)
            else 'N/A'
            for p in self.pods

        ])
        return ret_str

    def evaluate_pod(self, pod):
        score = 0
        if pod.p_count == pod.cap:
            return -sys.maxsize
        for player in pod.players:
            score -= self.played.count(player) ** 2
        if pod.cap < self.tour.TC.max_pod_size:
            for pod in self.pods:
                if isinstance(pod, Pod):
                    score -= sum([10 for _ in pod.players if pod.cap < self.tour.TC.max_pod_size])
        return score

    def __gt__(self, other):
        b = False
        if self.SORT_METHOD == SORT_METHOD.ID:
            b = self.ID > other.ID
        elif self.SORT_METHOD == SORT_METHOD.NAME:
            b = self.name > other.name
        elif self.SORT_METHOD == SORT_METHOD.RANK:
            my_score = self.tour.TC.ranking(self)
            other_score = self.tour.TC.ranking(other)
            b = None
            for i in range(len(my_score)):
                if my_score[i] != other_score[i]:
                    b = my_score[i] > other_score[i]
                    break
        return b

    def __lt__(self, other):
        b = False
        if self.SORT_METHOD == SORT_METHOD.ID:
            b = self.ID < other.ID
        elif self.SORT_METHOD == SORT_METHOD.NAME:
            b = self.name < other.name
        elif self.SORT_METHOD == SORT_METHOD.RANK:
            my_score = self.tour.TC.ranking(self)
            other_score = self.tour.TC.ranking(other)
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
                self.ID, tsize, self.name.ljust(pname_size)))
        else:
            fields.append(self.name.ljust(pname_size))

        if args.pod and self.tour.round:
            max_pod_id = max([len(str(p.id)) for p in self.tour.round.pods])
            if self.pod:
                #find number of digits in max pod id
                fields.append('{}'.format(
                    f'P{str(self.pod.id).zfill(max_pod_id)}/S{self.pod.players.index(self)}' if self.pod else ''))
            elif self.game_loss:
                fields.append('Loss'.ljust(max_pod_id+4))
            else:
                fields.append('Bye'.ljust(max_pod_id+4))
        if args.p:
            fields.append('pts: {}'.format(self.points))
        if args.w:
            fields.append('w: {}'.format(self.games_won))
        if args.ow:
            fields.append('o.wr.: {:.2f}'.format(self.opponent_winrate))
        if args.u:
            fields.append('uniq: {}'.format(self.unique_opponents))
        if args.s:
            fields.append('seat: {:02.00f}%'.format(self.average_seat*100))
        # if args.np:
        #    fields.append(''.format())
        # OUTPUT_BUFFER.append('\t{}'.format(' | '.join(fields)))

        return ' | '.join(fields)


class Pod(IPod):
    def __init__(self, round: Round, id, cap=0):
        super().__init__()
        self.id = id
        self.cap = cap
        self.players: list[Player] = list()
        self.round: Round = round
        self.won: None|Player = None
        self.draw: list[Player] = list()
        self.pods: None|list[Pod] = None

    @property
    def p_count(self):
        return len(self.players)

    @override
    def add_player(self, player: Player, manual=False) -> bool:
        if player.pod:
            player.pod.remove_player(player)
        if self.p_count >= self.cap and self.cap and not manual:
            return False
        self.players.append(player)
        return True

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
    def sort(self):
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
        self.players.clear()

    def remove_player(self, player: Player, cleanup=True) -> Player|None:
        try:
            idx = self.players.index(player)
        except ValueError:
            return None
        p = self.players.pop(idx)
        if self.p_count == 0 and cleanup:
            self.round.remove_pod(self)
        return p

    def update_player_history(self, player):
        round_opponents = [p for p in self.players if not p is player]
        player.played += round_opponents
        player.games_played += 1
        while len(player.pods) < self.round.seq+1:
            player.pods.append(None)
        player.pods[self.round.seq] = self
        if player == self.won:
            player.games_won += 1
            player.opponents_beaten.update(set(round_opponents))

    @property
    def name(self):
        return 'Pod {}'.format(self.id)

    @override
    def __repr__(self):
        if not self.players:
            maxlen = 0
        else:
            maxlen = max([len(p.name) for p in self.players])
        ret = 'Pod {} with {}/{} players:\n\t{}'.format(
            self.id,
            self.p_count,
            self.cap,
            '\n\t'.join(
                [
                    '[{}] {}\t'.format(
                        '  ' if not self.done else
                        'W' if p == self.won else
                        'D' if p in self.draw else
                        'L',
                        p.__repr__(['-s', str(maxlen), '-p']))
                    for _, p in
                    zip(range(1, self.p_count+1), self.players)
                ]
            ))
        return ret


class Round(IRound):
    def __init__(self, seq: int, pairing_logic:IPairingLogic, tour: Tournament):
        super().__init__()
        self.pods: list[Pod] = list()
        self.seq = seq
        self.logic = pairing_logic
        self.tour = tour


    def next_pod_id(self):
        i = 0
        while True:
            for pod in self.pods:
                if pod.id == i:
                    i += 1
                    break
            else:
                return i

    @property
    def done(self):
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    @property
    def all_players_seated(self):
        pod_sizes = self.tour.get_pod_sizes(len([p for p in self.tour.players if not p.game_loss]))
        if pod_sizes is not None:
            return sum([pod.p_count for pod in self.pods]) == sum(pod_sizes)

    @property
    def seated(self):
        s = []
        if self.pods:
            for pod in self.pods:
                s += pod.players
        return s

    @property
    def unseated(self):
        return list(set([p for p in self.tour.players if not p.game_loss]) - set(self.seated))

    def remove_pod(self, pod: Pod):
        if not pod.done:
            pod.clear()
            self.pods.remove(pod)
            return True
        return False

    def create_pods(self):
        self.pods = []
        pod_sizes = self.tour.get_pod_sizes(len(self.unseated))
        if pod_sizes is None:
            Log.log('Can not make pods.', level=Log.Level.WARNING)
            return None
        pods:list[Pod] = []
        for size in pod_sizes:
            pod = Pod(self, self.next_pod_id(), cap=size)
            pods.append(pod)
            self.pods.append(pod)

    def create_pairings(self):
        self.create_pods()
        self.logic.make_pairings(self.unseated, self.pods)
        self.sort_pods()
        self.print_pods()

    def sort_pods(self):
        self.pods[:] = sorted(self.pods, key=lambda x: (len(x.players), np.average([p.points for p in x.players])), reverse=True)

    def is_better_swap(self, p1:Player, p2:Player) -> bool:
        # criteria 1:
        # if the swap results in a lower average seat for both players
        current_p1_pod = p1.pod.average_seat
        current_p2_pod = p2.pod.average_seat

        new_p1_pod = np.average([p.average_seat for p in p1.pod.players if p != p1] + [p2.average_seat])
        new_p2_pod = np.average([p.average_seat for p in p2.pod.players if p != p2] + [p1.average_seat])

        current_value = np.average(np.abs(0.5 - current_p1_pod) + np.abs(0.5 - current_p2_pod))
        new_value = np.average(np.abs(0.5 - new_p1_pod) + np.abs(0.5 - new_p2_pod))

        if new_value > current_value:
            return False
        # criteria 2:
        # if the swap doesn't introduce more rematches
        current_p1_rematches = len([p for p in p1.played if p in p2.pod.players])
        current_p2_rematches = len([p for p in p2.played if p in p1.pod.players])

        new_p1_rematches = len([p for p in p1.played if p in p2.pod.players if p != p2])
        new_p2_rematches = len([p for p in p2.played if p in p1.pod.players if p != p1])

        rematch_difference = (new_p1_rematches + new_p2_rematches) - (current_p1_rematches + current_p2_rematches)
        if rematch_difference > 0:
            return False

        return True

    def optimize_seatings(self):
        remaining = self.seated
        bucket_ranking = lambda x: (x.points, -x.unique_opponents)
        remaining = sorted(remaining, key=bucket_ranking, reverse=True)
        bucket_order = sorted(
            list(set(
                [bucket_ranking(p) for p in remaining]
            )), reverse=True)
        buckets = {
            k: [
                p for p in remaining
                if bucket_ranking(p) == k
            ]
            for k in bucket_order
        }
        for b in buckets.values():
            for i, player1 in enumerate(b):
                for player2 in b[i+1:]:
                    #pod1 = next(pod for pod in pods if player1 in pod.players)
                    #pod2 = next(pod for pod in pods if player2 in pod.players)
                    pod1 = player1.pod
                    pod2 = player2.pod
                    if pod1 != pod2 and self.is_better_swap(player1, player2):
                        # Perform swap
                        pod1.players.remove(player1)
                        pod2.players.remove(player2)
                        if pod1.add_player(player2) and pod2.add_player(player1):
                            continue
                        else:
                            # Rollback if the swap was not successful
                            pod1.players.append(player1)
                            pod2.players.append(player2)

    def print_pods(self):
        for p in self.pods:
            Log.log(p)
            Log.log('-'*80)

    def conclude(self):
        for pod in self.pods:
            for p in pod.players:
                pod.update_player_history(p)

        if self.unseated and self.tour.TC.allow_bye:
            for p in self.unseated:
                Log.log('bye "{}"'.format(p.name))
                p.points += self.tour.TC.bye_points
                p.byes += 1
                p.pods.append(Player.EResult.BYE)
        for p in [p for p in self.tour.players if p.game_loss]:
            p.games_played += 1
            p.game_loss = False
            p.pods.append(Player.EResult.LOSS)
        self.tour.rounds.append(self)
        self.concluded = True
        Log.log('{}{}{}'.format(
            30*'*', '\nRound completed!\n', 30*'*',), Log.Level.INFO)
        self.tour.round = None

    def won(self, players: list[Player]|Player):
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

                pod.won = player
                pod.done = True

            if self.done:
                self.conclude()

    def draw(self, players: list[Player]|Player):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod is not None:
                player.points = player.points + self.tour.TC.draw_points

                player.pod.draw.append(player)
                player.pod.done = True

        if self.done and not self.concluded:
            self.conclude()

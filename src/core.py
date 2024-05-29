from __future__ import annotations
import glob
from typing import *

import argparse
import math
import os
import pickle
import random
import sys
from copy import deepcopy
from datetime import datetime
from enum import Enum
from src.misc import Json2Obj
import numpy as np
import names

class PodsExport:
    @classmethod
    def auto_export(cls, func):
        def auto_pods_export_wrapper(self: Tournament, *original_args, **original_kwargs):
            tour_round = self.round
            ret = func(self, *original_args, **original_kwargs)
            tour_round = tour_round or self.round
            if self.TC.auto_export:
                file = TournamentAction.LOGF
                if file and tour_round:
                    # Export pods to a file named {tournament_name}_round_{round_number}.txt
                    # And also export it into {log_directory}/pods.txt

                    export_str = '\n\n'.join([
                        pod.__repr__()
                        for pod in tour_round.pods
                    ])

                    game_lost = [x for x in self.players if x.game_loss]
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
                        os.path.dirname(file),
                        os.path.basename(file).replace('.log', ''),
                        os.path.basename(file).replace('.log', '_R{}.txt'.format(tour_round.seq)),
                    )
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))
                    self.export_str(path, export_str)

                    path = os.path.join(os.path.dirname(TournamentAction.LOGF), 'pods.txt') # type: ignore
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
            'getter': lambda p: p.record
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
                self.export(
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
    lastID = 0

    def __init__(self):
        pass

    @staticmethod
    def next():
        ID.lastID += 1
        return ID.lastID


class TournamentAction:
    '''Serializable action that will be stored in tournament log and can be restored
    '''
    ACTIONS = []
    LOGF = None
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
    def reset(cls):
        TournamentAction.ACTIONS = []
        TournamentAction.store()

    @classmethod
    def action(cls, func):

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
            if not os.path.exists(os.path.dirname(cls.LOGF)):
                os.makedirs(os.path.dirname(cls.LOGF))
            with open(cls.LOGF, 'wb') as f:
                pickle.dump(cls.ACTIONS, f)

    @classmethod
    def load(cls, logdir='logs/default.log'):
        if os.path.exists(logdir):
            cls.LOGF = logdir
            with open(cls.LOGF, 'rb') as f:
                cls.ACTIONS = pickle.load(f)
            return True
        return False

    def __repr__(self, *nargs, **kwargs):
        ret_str = (
            '{}'
            '{}'
            '{}'
        ).format(
            self.func_name,
            '' if not nargs else ', '.join([str(arg) for arg in nargs]),
            '' if not kwargs else ', '.join([
                '{}={}' for key, val in kwargs.items()
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

    @property
    def min_pod_size(self):
        return min(self.pod_sizes)

    def ranking(_, x):
        return (
            x.points,
            x.games_played,
            np.round(x.opponent_winrate, 2),
            x.n_opponents_beaten,
            -x.average_seat,
            -x.ID
        )

    def matching(_, x):
        return (
            -x.games_played,
            x.points,
            -x.unique_opponents,
            x.opponent_winrate
        )

    def __repr__(self):
        return "Tour. cfg:" + '|'.join([
            '{}:{}'.format(key, val)
            for key, val in self.__dict__.items()
        ])


class Tournament:
    # CONFIGURATION
    # Logic: Points is primary sorting key,
    # then opponent winrate, - CHANGE - moved this upwards and added dummy opponents with 33% winrate
    # then number of opponents beaten,
    # then ID - this last one is to ensure deterministic sorting in case of equal values (start of tournament for example)

    def __init__(self, config: TournamentConfiguration = None):  # type: ignore
        if config is None:
            config = TournamentConfiguration()
        self.rounds = list()
        self.players = list()
        self.dropped = list()
        self.round = None

        # Direct setting - don't want to overwrite old log file
        self._TC = config

    # TOURNAMENT ACTIONS
    # IMPORTANT: No nested tournament actions

    @property
    def TC(self):
        return self._TC

    @TC.setter
    @TournamentAction.action
    def TC(self, config):
        self._TC = config

    @TournamentAction.action
    def add_player(self, names=None):
        new_players = []
        if not isinstance(names, list):
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
    def remove_player(self, players: list[Player] = []):
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

    @TournamentAction.action
    def make_pods(self):
        if self.round is None or self.round.concluded:
            self.round = Round(len(self.rounds), self)
        if not self.round.all_players_seated:
            self.round.make_pods()
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
            self.round = Round(len(self.rounds), self)
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
    def manual_pod(self, players: list[Player] = []):
        if self.round is None or self.round.concluded:
            self.round = Round(len(self.rounds), self)
        if not self.round.pods:
            self.round.pods = []

        pod = Pod(self.round, len(self.round.pods))

        for player in players:
            pod.add_player(player)
        self.round.pods.append(pod)

    @TournamentAction.action
    def report_win(self, players: list[Player] = []):
        if self.round:
            if not isinstance(players, list):
                players = [players]
            for p in players:
                Log.log('Reporting player {} won this round.'.format(p.name))
            self.round.won(players)

    @TournamentAction.action
    def report_draw(self, players: list[Player] = []):
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
        if self.round.pods is not None:
            global_winrates_by_seat = [
                0.2553,
                0.2232,
                0.1847,
                0.1428,
            ]
            draw_rate = 1-sum(global_winrates_by_seat)
            #for each pod
            #generate a random result based on global_winrates_by_seat
            #each value corresponds to the winrate of the player in that seat
            #the sum of percentages is less than 1, so there is a chance of a draw (1-sum(winrates))

            for pod in [x for x in self.round.pods if not x.done]:
                #generate a random result
                result = random.random()
                rates = np.array(global_winrates_by_seat[0:len(pod.players)] + [draw_rate])
                rates = rates/sum(rates)
                draw = result > np.cumsum(rates)[-1]
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
    def move_player_to_pod(self, pod: Pod, players: list[Player] = [], manual=False):
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
    def bench_players(self, players: list[Player] = []):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            self.remove_player_from_pod(player)

    @TournamentAction.action
    def toggle_game_loss(self, players: list[Player] = []):
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

    def get_pod_sizes(self, n):
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

    def export(
        self,
        fdir: str,
        fields: list[StandingsExport.Field] = StandingsExport.DEFAULT_FIELDS,
        style: StandingsExport.Format = StandingsExport.Format.TXT,
    ):
        standings = self.get_standings()
        lines = [[StandingsExport.info[f].name for f in fields]]
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


class Player:
    SORT_METHOD = SORT_METHOD.ID
    SORT_ORDER = SORT_ORDER.ASCENDING
    FORMATTING = ['-p']

    def __init__(self, name, tour: Tournament):
        self.tour = tour
        self.name = name
        self.points = 0
        self.played = list()
        self.ID = ID.next()
        self.games_played = 0
        self.games_won = 0
        self.opponents_beaten = set()
        self.game_loss = False
        self.pods = []
        self.byes = 0

    @property
    def average_seat(self):
        """
        Expressed in percentage
        In a 4 pod game:
            seat 0: 100%
            seat 1: 66.66%
            seat 2: 33.33%
            seat 3: 0%
        In a 3 pod game:
            seat 0: 100%
            seat 1: 50%
            seat 2: 0%

        Lower percentage means higher benefits
        """
        if not self.pods:
            return 0.5
        n_pods = len([p for p in self.pods if isinstance(p, Pod)])
        if n_pods == 0:
            return 0.5
        return sum([
            1 - (p.players.index(self) / (len(p.players) - 1))
            if len(p.players) > 1
            else 1  #If for some reason, there is a signle player in a pod (probably user error)
            for p in self.pods
            if isinstance(p, Pod)
        ])/n_pods

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
    def pod(self) -> Pod:
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
    def record(self) -> str:
        total_rounds = len(self.tour.rounds) + (1 if self.tour.round else 0)
        seq = [' '] * total_rounds
        for round, pod in enumerate(self.pods + ([self.pod] if self.tour.round else [])):
            if pod is Round.Result.BYE:
                seq[round] = 'B'
            elif pod is Round.Result.LOSS:
                seq[round] = 'L'
            elif pod is None:
                if self.game_loss:
                    seq[round] = 'L'
                else:
                    seq[round] = 'B'
            elif isinstance(pod, Pod):
                if pod.done:
                    if pod.won is not None:
                        if pod.won is self:
                            seq[round] = 'W'
                        else:
                            seq[round] = 'L'
                    else:
                        if self in pod.draw:
                            seq[round] = 'D'
                        else:
                            seq[round] = 'L'
                else:
                    seq[round] = '_'
        record_sequence = ''.join(seq)
        return ('{} ({}/{}/{})'.format(
            record_sequence.ljust(total_rounds),
            record_sequence.count('W') + record_sequence.count('B'),
            record_sequence.count('L'),
            record_sequence.count('D'),
        ))

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
            score = score - self.played.count(player) ** 2
        return score

    def __gt__(self, other):
        if self.SORT_METHOD == SORT_METHOD.ID:
            b = self.id > other.id
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
        '''parser_player.add_argument(
            '-s', '--spaces',
            dest='spaces', type=int, default=0)'''
        # parser.add_argument('-n', '--notplayed',    dest='np', action='store_true')

        try:
            args, unknown = parser_player.parse_known_args(tokens)
        except:
            # args = None
            # OUTPUT_BUFFER.append('Invalid argumets')
            pass

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


class Pod:
    def __init__(self, round: Round, id, cap=0):
        self.round = round
        self.players = list()
        self.cap = cap
        self.id = id
        self.done = False
        self.won = None
        self.draw = list()
        self.pods = None

    @property
    def p_count(self):
        return len(self.players)

    def add_player(self, player: Player, manual=False) -> bool:
        if player.seated:
            player.pod.remove_player(player)
        if self.p_count >= self.cap and self.cap and not manual:
            return False
        self.players.append(player)
        return True

    @property
    def average_seat(self) -> np.floating[Any]:
        return np.average([p.average_seat for p in self.players])

    @property
    def balance(self) -> np.ndarray:
        '''
        Returns a list of count of players above 50% average seat and below 50% average seat
        '''
        return np.array([
            sum([1 for p in self.players if p.average_seat > 0.5]),
            sum([1 for p in self.players if p.average_seat < 0.5])
        ])

    def sort_players_by_avg_seat(self):
        # Average seating positions
        average_positions = [p.average_seat for p in self.players]
        n = len(average_positions)

        if not any(average_positions):
            random.shuffle(self.players)
            return True

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


        self.players = [p for _, p in sorted(zip(seat_assignment, self.players))]
        pass

    def clear(self):
        self.players = list()

    def remove_player(self, player: Player, cleanup=True):
        p = None
        '''for i in range(len(self.players)):
            pi = self.players[i]
            if pi.name == player.name:
                p = self.players.pop(i)
                break'''
        self.players.remove(player)
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

    def __repr__(self):
        if not self.players:
            maxlen = 0
        else:
            maxlen = max([len(p.name) for p in self.players])
        ret = 'Pod {} with {} players and seats:\n\t{}'.format(
            self.id,
            self.p_count,
            # self.score,
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


class Round:
    class Result(Enum):
        DRAW = 0
        WIN = 1
        BYE = -1
        LOSS = -2

    def __init__(self, seq, tour: Tournament):
        self.tour = tour
        self.seq = seq
        self.pods = []
        # self.players = None
        self.concluded = False

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
        return sum([pod.p_count for pod in self.pods]) == len([1 for x in self.tour.players if not x.game_loss])

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

    def make_pods(self):
        remaining = self.unseated
        n_plyr = len(remaining)

        pod_sizes = self.tour.get_pod_sizes(n_plyr)
        if pod_sizes is None:
            Log.log('Can not make pods.', level=Log.Level.WARNING)
            return None

        bye_count = n_plyr - sum(pod_sizes)
        pods:list[Pod] = []
        for size in pod_sizes:
            pod = Pod(self, self.next_pod_id(), cap=size)
            pods.append(pod)
            self.pods.append(pod)

        #First round pods are completely random
        if self.seq == 0:
            random.shuffle(remaining)
            for pod in pods:
                for _ in range(pod.cap):
                    pod.add_player(remaining.pop(0))

        #Snake pods logic for 2nd round
        #First bucket is players with most points and least unique opponents
        #Players are then distributed in buckets based on points and unique opponents
        #Players are then distributed in pods based on bucket order
        elif self.tour.TC.snake_pods and self.seq == 1:
            snake_ranking = lambda x: (x.points, -x.unique_opponents)
            remaining = sorted(remaining, key=snake_ranking, reverse=True)
            bucket_order = sorted(
                list(set(
                    [snake_ranking(p) for p in remaining]
                )), reverse=True)
            buckets = {
                k: [
                    p for p in remaining
                    if snake_ranking(p) == k
                ]
                for k in bucket_order
            }
            for b in buckets.values():
                random.shuffle(b)

            for order_idx, b in enumerate(bucket_order):
                if (
                    order_idx == 0  # if not first bucket
                    # and not same points as previous bucket
                    or b[0] != bucket_order[order_idx-1][0]
                ):
                    i = 0

                for p in buckets[b]:
                    ok = False
                    if b == bucket_order[-1] and p in buckets[b][-1:-bye_count-1:-1]:
                        ok = True
                    if sum(pod_sizes) == sum(len(pod_x.players) for pod_x in pods):
                        ok = True
                    while not ok:
                        ok = pods[i % len(pods)].add_player(p)
                        i += 1

            #at this point, pods are created and filled with players
            #but seating order is not yet determined
            #swaps between pods need to be made first - your code here
            # Attempt to swap equivalent players between pods

            # Swapping equivalent players between pods to optimize seats

            self.optimize_seatings()
            for pod in pods:
                pod.sort_players_by_avg_seat()

            pass
        else:
            for p in sorted(random.sample(remaining, len(remaining)), key=self.tour.TC.matching, reverse=True):
                pod_scores = [p.evaluate_pod(pod) for pod in pods]
                index = pod_scores.index(max(pod_scores))
                pods[index].add_player(p)
            self.optimize_seatings()
            for pod in pods:
                pod.sort_players_by_avg_seat()

        self.sort_pods()
        self.print_pods()

    def sort_pods(self):
        self.pods = sorted(self.pods, key=lambda x: sum([p.points for p in x.players]), reverse=True)

    def is_better_swap(self, p1:Player, p2:Player) -> bool:
        old_p1_pod = p1.pod.average_seat
        old_p2_pod = p2.pod.average_seat

        new_p1_pod = np.average([p.average_seat for p in p1.pod.players if p != p1] + [p2.average_seat])
        new_p2_pod = np.average([p.average_seat for p in p2.pod.players if p != p2] + [p1.average_seat])

        current_value = np.average(np.abs(0.5 - old_p1_pod) + np.abs(0.5 - old_p2_pod))
        new_value = np.average(np.abs(0.5 - new_p1_pod) + np.abs(0.5 - new_p2_pod))

        return new_value < current_value

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
                p.pods.append(Round.Result.BYE)
        for p in [p for p in self.tour.players if p.game_loss]:
            p.games_played += 1
            p.game_loss = False
            p.pods.append(Round.Result.LOSS)
        self.tour.rounds.append(self)
        self.concluded = True
        Log.log('{}{}{}'.format(
            30*'*', '\nRound completed!\n', 30*'*',), Log.Level.INFO)
        self.tour.round = None

    def won(self, players: list[Player] = []):
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

    def draw(self, players: list[Player] = []):
        for player in players:
            pod = player.pod

            player.points = player.points + self.tour.TC.draw_points

            pod.draw.append(player)
            pod.done = True

        if self.done and not self.concluded:
            self.conclude()


if __name__ == "__main__":
    Log.PRINT = True
    tour = Tournament()
    loaded = TournamentAction.load()
    if loaded:
        tour = TournamentAction.ACTIONS[-1].after
        for action in TournamentAction.ACTIONS:
            print(action.func_name)
            for p in action.after.players:
                print('[{}]{} - {} '.format(
                    p.ID,
                    p.name,
                    p.points
                ))
            print()
        print()

    else:
        tour.add_player([
            names.get_full_name()
            for i in range(11)
        ])
        for i in range(2):
            tour.make_pods()
            tour.random_results()
            # tour.remove_player(tour.players[0])

        print()
    # tour.show_pods()

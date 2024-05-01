from __future__ import annotations

import argparse
import inspect
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


class Export:
    __instance = None

    @staticmethod
    def instance():
        """ Static access method. """
        if Export.__instance == None:
            Export()
        return Export.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Export.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Export.__instance = self

    class Field(Enum):
        STANDING = 0  # Standing
        ID = 1  # Player ID
        NAME = 2  # Player name
        RECORD = 3 # Record
        POINTS = 4  # Number of points
        WINS = 5  # Number of wins
        OPPONENTSBEATEN = 6  # Number of opponents beaten
        OPPONENTWIN = 7  # Opponents' win percentage
        UNIQUE = 8  # Number of unique opponents
        WINRATE = 9  # Winrate
        GAMES = 10  # Number of games played
        AVG_SEAT = 11  # Average seat

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
        Field.OPPONENTWIN: Json2Obj({
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
        Field.OPPONENTSBEATEN: Json2Obj({
            'name': '# opp. beat',
            'format': '{:d}',
            'denom': None,
            'description': 'Number of opponents beaten',
            'getter': lambda p: p.n_opponents_beaten
        }),
        Field.AVG_SEAT: Json2Obj({
            'name': 'avg. seat',
            'format': '{:.2f}',
            'denom': None,
            'description': 'Average seat',
            'getter': lambda p: p.average_seat
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
        Field.OPPONENTWIN,
        Field.OPPONENTSBEATEN,
        Field.AVG_SEAT,
    ]

    fields = [f for f in DEFAULT_FIELDS]
    format = Format.TXT
    dir = './logs/standings' + ext[format]


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
    STATE = []
    LOGF = None
    DEFAULT_LOGF = 'logs/default.log'

    def __init__(self, before, ret, after, func_name, *nargs, **kwargs):
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
        if not cls.LOGF:
            cls.LOGF = cls.DEFAULT_LOGF
        if cls.LOGF:
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

    @property
    def min_pod_size(self):
        return min(self.pod_sizes)

    def ranking(_, x):
        return (
                x.points,
                x.games_played,
                np.round(x.opponent_winrate, 2),
                x.n_opponents_beaten,
                x.average_seat,
                -x.ID
        )

    def scoring(_, x):
        return (
            -x.games_played,
            x.points,
            -x.unique_opponents,
            x.opponent_winrate
        )


class Tournament:
    # CONFIGURATION
    # Logic: Points is primary sorting key,
    # then opponent winrate, - CHANGE - moved this upwards and added dummy opponents with 33% winrate
    # then number of opponents beaten,
    # then ID - this last one is to ensure deterministic sorting in case of equal values (start of tournament for example)

    def __init__(self, config:TournamentConfiguration=TournamentConfiguration()):
        self.rounds = list()
        self.players = list()
        self.dropped = list()

        self.TC = config

        self.round = None

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
            for pod in [x for x in self.round.pods if not x.done]:
                draw_or_win = random.random() < 0.8
                if draw_or_win:
                    player = random.sample(pod.players, 1)[0]
                    Log.log('won "{}"'.format(player.name))
                    self.round.won([player])
                else:
                    players = random.sample(
                        pod.players, random.randint(1, pod.p_count))
                    Log.log('draw {}'.format(
                        ' '.join(['"{}"'.format(p.name) for p in players])))
                    self.round.draw([p for p in players])

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
        fields: list[Export.Field] = Export.DEFAULT_FIELDS,
        style: Export.Format = Export.Format.TXT,
    ):
        standings = self.get_standings()
        lines = [[Export.info[f].name for f in fields]]
        lines += [
            [
                (Export.info[f].format).format(
                    Export.info[f].getter(p)
                    if Export.info[f].denom is None
                    else Export.info[f].getter(p) * Export.info[f].denom
                )
                for f
                in fields
            ]
            for p in standings
        ]
        if style == Export.Format.TXT:
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
            Log.log('Log saved: {}.'.format(
                fdir), level=Log.Level.INFO)
        elif style == Export.Format.CSV:
            Log.log('Log not saved - CSV not implemented.'.format(
                fdir), level=Log.Level.WARNING)
        elif style == Export.Format.DISCORD:
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
        if not self.pods:
            return 0
        return sum([
            p.players.index(self)+1
            for p in self.pods
            if p is not None
        ])/len(self.pods)

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
        if len(self.tour.rounds) < 1:
            return ''
        total_rounds = len(self.tour.rounds)
        seq = [' '] * total_rounds
        for round, pod in enumerate(self.pods):
            if pod is None:
                seq[round] = 'B'
            elif pod.won is not None:
                if pod.won is self:
                    seq[round] = 'W'
                else:
                    seq[round] = 'L'
            else:
                if self in pod.draw:
                    seq[round] = 'D'
                else:
                    seq[round] = 'L'
        record_sequence = ''.join(seq)
        return ('{} ({}/{}/{})'.format(
            record_sequence.ljust(total_rounds),
            record_sequence.count('W') + record_sequence.count('B'),
            record_sequence.count('L'),
            record_sequence.count('D'),
        ))


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
            my_score = Tournament.RANKING(None, self)
            other_score = Tournament.RANKING(None, other)
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
            my_score = Tournament.RANKING(None, self)
            other_score = Tournament.RANKING(None, other)
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
        '''parser_player.add_argument(
            '-s', '--spaces',
            dest='spaces', type=int, default=0)'''
        #parser.add_argument('-n', '--notplayed',    dest='np', action='store_true')

        try:
            args, unknown = parser_player.parse_known_args(tokens)
        except:
            #args = None
            #OUTPUT_BUFFER.append('Invalid argumets')
            pass

        fields = list()

        tsize = int(math.floor(math.log10(len(self.tour.players))) + 1)
        pname_size = max([len(p.name) for p in self.tour.players])

        if args.standing:
            fields.append('#{:>{}}'.format(self.standing, tsize))
        if args.id:
            fields.append('[{:>{}}] {}'.format(self.ID, tsize, self.name.ljust(pname_size)))
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
        # if args.np:
        #    fields.append(''.format())
        #OUTPUT_BUFFER.append('\t{}'.format(' | '.join(fields)))

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
        if self.p_count >= self.cap:
            # Average seating positions
            average_positions = [p.average_seat for p in self.players]
            if not all(average_positions):
                random.shuffle(self.players)
                return True

            n = len(average_positions)
            # Calculate inverse averages
            inverse_averages = [pos / max(average_positions) for pos in average_positions]

            # Normalize to get probabilities
            probabilities = [inv / sum(inverse_averages) for inv in inverse_averages]

            # Generate random seat assignment based on probabilities
            seat_assignment = np.random.choice(range(1, n+1, 1), size=n, replace=False, p=probabilities)
            self.players = [p for _, p in sorted(zip(seat_assignment, self.players))]
            pass
        return True

    def clear(self):
        self.players = list()

    def remove_player(self, player: Player):
        p = None
        '''for i in range(len(self.players)):
            pi = self.players[i]
            if pi.name == player.name:
                p = self.players.pop(i)
                break'''
        self.players.remove(player)
        if self.p_count == 0:
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
                    '{}: [{}] {}\t'.format(
                        i,
                        '  ' if not self.done else
                        'W' if p == self.won else
                        'D' if p in self.draw else
                        'L',
                        p.__repr__(['-s', str(maxlen), '-p']))
                    for i, p in
                    zip(range(1, self.p_count+1), self.players)
                ]
            ))
        return ret


class Round:
    def __init__(self, seq, tour: Tournament):
        self.tour = tour
        self.seq = seq
        self.pods = []
        #self.players = None
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
        return sum([pod.p_count for pod in self.pods]) == len(self.tour.players)

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
        pods = []
        for size in pod_sizes:
            pod = Pod(self, self.next_pod_id(), cap=size)
            pods.append(pod)
            self.pods.append(pod)

        if self.seq == 0:
            random.shuffle(remaining)
            for pod in pods:
                for _ in range(pod.cap):
                    pod.add_player(remaining.pop(0))

        elif self.tour.TC.snake_pods and self.seq == 1:
            ranking = lambda x: (x.points, x.unique_opponents)
            remaining = sorted(remaining, key=ranking, reverse=True)
            bucket_order = sorted(list(set([ranking(p) for p in remaining])), reverse=True)
            buckets = {k: [p for p in remaining if ranking(p) == k] for k in bucket_order}
            for b in buckets.values():
                random.shuffle(b)

            for b in bucket_order:
                i = 0
                for p in buckets[b]:
                    ok = False
                    if b == bucket_order[-1] and p in buckets[b][-1:-bye_count-1:-1]:
                        ok = True
                    while not ok:
                        ok = pods[i % len(pods)].add_player(p)
                        i += 1
                    #pods[i % len(pods)].add_player(p)

            pass
        else:
            for p in sorted(random.sample(remaining, len(remaining)), key=self.tour.MATCHING, reverse=True):
                pod_scores = [p.evaluate_pod(pod) for pod in pods]
                index = pod_scores.index(max(pod_scores))
                pods[index].add_player(p)


        self.print_pods()

    def print_pods(self):
        for p in self.pods:
            Log.log(p)
            Log.log('-'*80)

    def conclude(self):
        for pod in self.pods:
            for p in pod.players:
                pod.update_player_history(p)

        #self.players = deepcopy(self.players)
        if self.unseated and self.tour.TC.allow_bye:
            for p in self.unseated:
                Log.log('bye "{}"'.format(p.name))
                p.points += self.tour.TC.bye_points
                p.byes += 1
                p.pods.append(None)
        for p in [p for p in self.tour.players if p.game_loss]:
            p.games_played += 1
            p.game_loss = False
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

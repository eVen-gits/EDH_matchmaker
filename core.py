import sys, argparse
from copy import deepcopy
import random
import shlex
import jellyfish
from enum import Enum
import pickle
import os
import inspect
import names
from datetime import datetime

class Pod: pass
class Player: pass
class Round: pass

class SORT_METHOD(Enum):
    ID=0
    NAME=1
    RANK=2

class SORT_ORDER(Enum):
    ASCENDING=0
    DESCENDING=1

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
        TournamentAction.ACTIONS=[]
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

class Tournament:
    #CONFIG
    RANKING = lambda x: (-x.points, -x.opponent_winrate, -x.unique_opponents)
    MATCHING = lambda _, x: (-x.games_played, -x.unique_opponents, x.points, -x.opponent_winrate)

    POD_SIZES = [4, 3]

    ALLOW_BYE = False

    WIN_POINTS = 3
    BYE_POINTS = 3
    DRAW_POINTS = 1

    def __init__(self,
        pod_sizes=POD_SIZES,
        allow_bye=ALLOW_BYE,
        win_points=WIN_POINTS,
        draw_points=DRAW_POINTS,
        bye_points=BYE_POINTS,
    ):
        self.__class__.set_pod_sizes(pod_sizes)
        self.__class__.set_allow_bye(allow_bye)
        self.__class__.set_scoring([win_points, draw_points, bye_points])

        self.rounds = list()
        self.players = list()
        self.dropped = list()

        self.round = None

    #TOURNAMENT ACTIONS
    #IMPORTANT: No nested tournament actions

    @TournamentAction.action
    def add_player(self, names=[]):
        new_players = []
        if not isinstance(names, list):
            names =  [names]
        for name in names:
            if name in [p.name for p in self.players]:
                Log.log('\tPlayer {} already enlisted.'.format(name), level=Log.Level.WARNING)
                continue
            if name:
                p = Player(name, tour=self)
                self.players.append(p)
                new_players.append(p)
                Log.log('\tAdded player {}'.format(p.name), level=Log.Level.INFO)
        return new_players

    @TournamentAction.action
    def remove_player(self, players: list[Player]=[]):
        if not isinstance(players, list):
            players = [players]
        for p in players:
            if self.round and p.seated:
                if not self.round.concluded:
                    Log.log('Can\'t drop {} during an active round.\nComplete the round or remove player from pod first.'.format(p.name), level=Log.Level.WARNING)
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
            Log.log('\tPlayer {} already enlisted.'.format(new_name), level=Log.Level.WARNING)
            return
        if new_name:
            player.name = new_name
            for round in self.rounds:
                for pod in round.pods:
                    for p in pod.players:
                        if p.name == player.name:
                            p.name = new_name
            Log.log('\tRenamed player {} to {}'.format(player.name, new_name), level=Log.Level.INFO)

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
    def manual_pod(self, players: list[Player]=[]):
        if self.round is None or self.round.concluded:
            self.round = Round(len(self.rounds), self)
        if not self.round.pods:
            self.round.pods = []

        pod = Pod(self.round, len(self.round.pods))

        for player in players:
            pod.add_player(player)
        self.round.pods.append(pod)

    @TournamentAction.action
    def report_win(self, players: list[Player]=[]):
        if self.round:
            self.round.won(players)

    @TournamentAction.action
    def report_draw(self, players: list[Player]=[]):
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
                    self.round.won(player)
                else:
                    players = random.sample(pod.players, random.randint(1, pod.p_count))
                    Log.log('draw {}'.format(' '.join(['"{}"'.format(p.name) for p in players])))
                    self.round.draw([p for p in players])

    @TournamentAction.action
    def move_player_to_pod(self, pod:Pod, players: list[Player]=[], manual=False):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            if player.pod and player.pod != pod:
                old_pod = player.pod.name
                player.pod.remove_player(player)
                Log.log('\tRemoved player {} from {}.'.format(player.name, old_pod), level=Log.Level.INFO)
            if player.pod != pod:
                if pod.add_player(player, manual=manual):
                    Log.log('\tAdded player {} to {}'.format(player.name, pod.name), level=Log.Level.INFO)
                else:
                    Log.log('\tFailed to add palyer {} to Pod {}'.format(player.name, pod.id), level=Log.Level.ERROR)

    @TournamentAction.action
    def bench_players(self, players: list[Player]=[]):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            pod = player.pod
            if pod:
                pod.remove_player(player)
                Log.log('\tRemoved player {} from {}.'.format(player.name, pod.name), level=Log.Level.INFO)

    @TournamentAction.action
    def delete_pod(self, pod:Pod):
        if self.round:
            self.round.remove_pod(pod)

    def get_pod_sizes(self, n):
        for pod_size in self.POD_SIZES:
            if n-pod_size == 0:
            #or n-pod_size < self.MIN_POD_SIZE and self.ALLOW_BYE and pod_size == self.POD_SIZES[-1]:
                return [pod_size]
            if n-pod_size < self.MIN_POD_SIZE and pod_size == self.POD_SIZES[-1]:
                if self.ALLOW_BYE and n-pod_size > 0:
                    return [pod_size]
                return None
            if n-pod_size >= self.MIN_POD_SIZE:
                tail = self.get_pod_sizes(n-pod_size)
                if tail is not None:
                    return [pod_size] + tail
        return None

    #MISC ACTIONS

    def show_pods(self, tokens=[]):
        if self.round and self.round.pods:
            self.round.print_pods()
        else:
            Log.log('No pods currently created.')

    def export_str(self, fdir, str):
        with open(fdir, 'w') as f:
            f.writelines(str)

    #PROPS AND CLASSMETHODS

    @classmethod
    def set_allow_bye(cls, allow_bye):
        cls.ALLOW_BYE = allow_bye

    @classmethod
    def set_pod_sizes(cls, sizes):
        cls.POD_SIZES = sizes

    @classmethod
    def set_scoring(cls, scoring: list):
        cls.WIN_POINTS, cls.DRAW_POINTS, cls.BYE_POINTS = scoring

    @classmethod
    @property
    def MIN_POD_SIZE(cls):
        return min(cls.POD_SIZES)

class Player:
    SORT_METHOD = SORT_METHOD.ID
    SORT_ORDER = SORT_ORDER.ASCENDING

    def __init__(self, name, tour: Tournament):
        self.tour = tour
        self.name = name
        self.points = 0
        self.played = list()
        self.ID = ID.next()
        self.games_played = 0
        self.games_won = 0

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

    def evaluate_pod(self, pod):
        score = 0
        if pod.p_count == pod.cap:
            return -sys.maxsize
        for player in pod.players:
            score = score - self.played.count(player) ** 2
        return score

    def __gt__ (self, other):
        if self.SORT_METHOD == SORT_METHOD.ID:
            b = self.id > other.id
        elif self.SORT_METHOD == SORT_METHOD.NAME:
            b =  self.name > other.name
        elif self.SORT_METHOD == SORT_METHOD.RANK:
            if self.points != other.points:
                b = self.points > other.points
            elif self.opponent_winrate != other.opponent_winrate:
                b = self.opponent_winrate > other.opponent_winrate
            elif self.unique_opponents != other.unique_opponents:
                b = self.unique_opponents > other.unique_opponents
            else:
                return False
        return b

    def __lt__ (self, other):
        if self.SORT_METHOD == SORT_METHOD.ID:
            b =  self.ID < other.ID
        elif self.SORT_METHOD == SORT_METHOD.NAME:
            b =  self.name < other.name
        elif self.SORT_METHOD == SORT_METHOD.RANK:
            if self.points != other.points:
                b = self.points < other.points
            elif self.opponent_winrate != other.opponent_winrate:
                b = self.opponent_winrate < other.opponent_winrate
            elif self.unique_opponents != other.unique_opponents:
                b = self.unique_opponents < other.unique_opponents
            else:
                return False
        return b

    def __repr__(self, tokens=['-p']):
        #ret = '{} | played: {} | pts: {}'.format(self.name, len(set(self.played)), self.points)
        parser_player = argparse.ArgumentParser()

        parser_player.add_argument('-i', '--id',           dest='id', action='store_true')
        parser_player.add_argument('-w', '--win',          dest='w', action='store_true')
        parser_player.add_argument('-o', '--opponentwin',   dest='ow', action='store_true')
        parser_player.add_argument('-p', '--points',       dest='p', action='store_true')
        parser_player.add_argument('-r', '--winrate',      dest='wr', action='store_true')
        parser_player.add_argument('-u', '--unique',       dest='u', action='store_true')
        parser_player.add_argument('-s', '--spaces',       dest='spaces', type=int, default=0)
        #parser.add_argument('-n', '--notplayed',    dest='np', action='store_true')

        try:
            args, unknown = parser_player.parse_known_args(tokens)
        except:
            #args = None
            #OUTPUT_BUFFER.append('Invalid argumets')
            pass

        fields = list()

        if args.id:
            fields.append('[{}] {}'.format(self.ID, self.name))
        else:
            fields.append(self.name.ljust(args.spaces))
        if args.p:
            fields.append('pts: {}'.format(self.points))
        if args.w:
            fields.append('w: {}'.format(self.games_won))
        if args.ow:
            fields.append('o.wr.: {:.2f}'.format(self.opponent_winrate))
        if args.u:
            fields.append('uniq: {}'.format(self.unique_opponents))
        #if args.np:
        #    fields.append(''.format())
        #OUTPUT_BUFFER.append('\t{}'.format(' | '.join(fields)))

        return ' | '.join(fields)

class Pod:
    def __init__(self, round:Round, id, cap=0):
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

    def add_player(self, player: Player, manual=False):
        if player.seated:
            player.pod.remove_player(player)
        if self.p_count >= self.cap and self.cap and not manual:
            return False
        self.players.append(player)
        if self.p_count >= self.cap:
            random.shuffle(self.players)
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
        player.played = player.played + ([p for p in self.players if not p is player])
        player.games_played += 1
        if player == self.won:
            player.games_won += 1

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
            #self.score,
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
        return list(set(self.tour.players) - set(self.seated))

    def remove_pod(self, pod:Pod):
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

        pods = []
        for size in pod_sizes:
            pod = Pod(self, self.next_pod_id(), cap=size)
            pods.append(pod)
            self.pods.append(pod)

        random.shuffle(remaining)
        for p in sorted(remaining, key=self.tour.MATCHING, reverse=True):
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
        if self.unseated and self.tour.ALLOW_BYE:
            for p in self.unseated:
                p.points += self.tour.BYE_POINTS
        self.tour.rounds.append(self)
        self.concluded = True
        Log.log('{}{}{}'.format(30*'*', '\nRound completed!\n', 30*'*',), Log.Level.INFO)
        self.tour.round = None

    def find_player_pod(self, player):
        for i_pod in self.pods:
            for i_player in i_pod.players:
                if player is i_player:
                    return i_pod
        return None

    def won(self, players: list[Player]=[]):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            pod = self.find_player_pod(player)

            if not player or not pod:
                Log.log('Player {} not found in any pod'.format(player.name), Log.Level.WARNING)
                continue

            if not pod.done:
                player.points = player.points + self.tour.WIN_POINTS

                pod.won = player
                pod.done = True

            if self.done:
                self.conclude()

    def draw(self, players: list[Player]=[]):
        for player in players:
            pod = self.find_player_pod(player)

            player.points = player.points + self.tour.DRAW_POINTS

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
            #tour.remove_player(tour.players[0])

        print()
    #tour.show_pods()

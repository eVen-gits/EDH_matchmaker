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
    LOGF = 'logs/default.log'

    def __init__(self, before, ret, after, func_name, *nargs, **kwargs):
        self.before = before
        self.ret = ret
        self.after = after
        self.func_name = func_name
        self.nargs = nargs
        self.kwargs = kwargs
        self.ret = ret

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
    def set_log_dir(cls, logdir):
        cls.LOGF = logdir

    @classmethod
    def store(cls):
        with open(cls.LOGF, 'wb') as f:
            pickle.dump(cls.ACTIONS, f)

    @classmethod
    def load(cls, logdir):
        if os.path.exists(logdir):
            cls.LOGF = logdir
            with open(cls.LOGF, 'rb') as f:
                cls.ACTIONS = pickle.load(f)
            return True
        return False

    def __repr__(self):
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
    MATCHING = lambda _, x: (-x.unique_opponents, x.points, -x.opponent_winrate)

    POD_SIZES = [4, 3]

    def __init__(self, pod_sizes=POD_SIZES):
        self.__class__.set_pod_sizes(pod_sizes)

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
    def remove_player(self, players=[]):
        if self.round:
            if not self.round.concluded:
                Log.log('Can\'t drop player during an active round.\nComplete the round and remove player before creating new pods.', level=Log.Level.WARNING)
                return
        if not isinstance(players, list):
            players = [players]
        for p in players:
            if p.played:
                self.dropped.append(p)
            self.players.remove(p)

            Log.log('\tRemoved player {}'.format(p.name), level=Log.Level.INFO)

    @TournamentAction.action
    def make_pods(self):
        if self.round is None or self.round.concluded:
            self.round = Round(len(self.rounds), self)
        if not self.round.ready():
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

    @TournamentAction.action
    def manual_pod(self, players=[]):
        if self.round is None or self.round.concluded:
            self.round = Round(len(self.rounds), self)
        if not self.round.pods:
            self.round.pods = []

        pod = Pod(len(self.round.pods))

        for player in players:
            pod.add_player(player)
        self.round.pods.append(pod)

    @TournamentAction.action
    def report_win(self, players=[]):
        if self.round:
            self.round.won(players)

    @TournamentAction.action
    def report_draw(self, players=[]):
        if self.round:
            self.round.draw(players)

    @TournamentAction.action
    def random_results(self):
        if not self.round:
            Log.log(
                'A round is not in progress.\nStart a new round with "pods" command.',
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

    #MISC ACTIONS

    def show_pods(self, tokens=[]):
        if self.round and self.round.pods:
            self.round.print_pods()
        else:
            Log.log('No pods currently created.')

    #PROPS AND CLASSMETHODS

    @classmethod
    def set_pod_sizes(cls, sizes):
        cls.POD_SIZES = sizes

    @classmethod
    @property
    def MIN_POD_SIZE(cls):
        return min(cls.POD_SIZES)

class Player:
    def __init__(self, name, tour: Tournament):
        self.tour = tour
        self.name = name
        self.points = 0
        self.played = list()
        self.ID = ID.next()
        self.games_played = 0
        self.games_won = 0

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

    def __repr__(self, tokens=['-i', '-p']):
        #ret = '{} | played: {} | pts: {}'.format(self.name, len(set(self.played)), self.points)
        parser_player = argparse.ArgumentParser()

        parser_player.add_argument('-i', '--id',           dest='id', action='store_true')
        parser_player.add_argument('-w', '--win',          dest='w', action='store_true')
        parser_player.add_argument('-o', '--opponentwin',   dest='ow', action='store_true')
        parser_player.add_argument('-p', '--points',       dest='p', action='store_true')
        parser_player.add_argument('-r', '--winrate',      dest='wr', action='store_true')
        parser_player.add_argument('-u', '--unique',       dest='u', action='store_true')
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
            fields.append(self.name)
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
    def __init__(self, id, cap=0):
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

    def add_player(self, player: Player):
        if self.p_count >= self.cap and self.cap:
            return False
        self.players.append(player)
        if self.p_count == self.cap:
            random.shuffle(self.players)
        return True

    def remove_player(self, player: Player):
        for i in range(len(self.players)):
            p = self.players[i]
            if p.name == player.name:
                return self.players.pop(i)

    def swap_players(self, p1: Player, other, p2: Player):
        p1 = self.remove_player(p1)
        p2 = other.remove_player(p2)

        self.add_player(p2)
        other.add_player(p1)

    @property
    def score(self):
        score = self.p_count
        self.cap = self.cap + 1
        for player in self.players:
            score = score + player.evaluate_pod(self)
        self.cap = self.cap - 1
        return score

    def gain(self, player: Player):
        copy = deepcopy(self)
        copy.add_player(player)
        return copy.score - self.score

    def update_player_history(self, player):
        player.played = player.played + ([p for p in self.players if not p is player])
        player.games_played += 1
        if player == self.won:
            player.games_won += 1

    def __repr__(self):
        return 'Pod {} with {} players and seats:\n\t{}'.format(
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
                        #TODO: p.__repr__(['-w']))
                        str(p))
                    for i, p in
                    zip(range(1, self.p_count+1), self.players)
                ]
            ))

class Round:
    def __init__(self, seq, tour: Tournament):
        self.tour = tour
        self.seq = seq
        self.pods = []
        self.players = None
        self.concluded = False

    @property
    def done(self):
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    def seated(self):
        s = []
        if self.pods:
            for pod in self.pods:
                s += pod.players

        return s

    def make_pods(self):
        seated = set(self.seated())
        remaining = list(set(self.tour.players) - seated)

        n_plyr = len(remaining)
        pod_sizes = Round.get_pod_sizes(n_plyr)
        if pod_sizes is None:
            Log.log('Can not make pods.', level=Log.Level.WARNING)
            return None
        n_pods = len(pod_sizes)

        pods = [Pod(i, size) for size, i in zip(pod_sizes, range(n_pods))]
        random.shuffle(remaining)
        for p in sorted(remaining, key=self.tour.MATCHING, reverse=True):
            pod_scores = [p.evaluate_pod(pod) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)

        self.pods += pods

        self.print_pods()

    def print_pods(self):
        for p in self.pods:
            Log.log(p)
            Log.log('-'*80)

    #TODO: This doesn't work
    def optimize(self):
        pod = sorted(self.pods, key=lambda x: x.score)[0]
        player = sorted(pod.players, key=lambda x: x.evaluate_pod(pod))[0]

        swap_count = 0

        swap = True
        while swap:
            swap = False
            for i_pod in [x for x in self.pods if not x is pod]:
                for i_player in i_pod.players:
                    pod1_cp = deepcopy(pod)
                    pod2_cp = deepcopy(i_pod)

                    pod1_cp.swap_players(player, pod2_cp, i_player)

                    gain = pod1_cp.score + pod2_cp.score - pod.score - i_pod.score

                    if gain > 0:
                        pod.swap_players(player, i_pod, i_player)
                        swap_count = swap_count + 1
                        break
                if swap:
                    break
        Log.log('{} swaps in optimization stage.'.format(swap_count), Log.Level.INFO)

    @staticmethod
    def get_pod_sizes(n=None):
        if(isinstance(n, list)):
            n = n[0]
        for pod_size in Tournament.POD_SIZES:
            if n-pod_size == 0:
                return [pod_size]
            if n-pod_size < Tournament.MIN_POD_SIZE and pod_size == Tournament.POD_SIZES[-1]:
                return None
            if n-pod_size >= Tournament.MIN_POD_SIZE:
                tail = Round.get_pod_sizes(n-pod_size)
                if tail:
                    return [pod_size] + tail
        return None

    def conclude(self):
        for pod in self.pods:
            for p in pod.players:
                pod.update_player_history(p)

        self.players = deepcopy(self.players)
        self.tour.rounds.append(self)
        self.concluded = True
        Log.log('{}{}{}'.format(30*'*', '\nRound completed!\n', 30*'*',), Log.Level.INFO)

    def find_player_pod(self, player):
        for i_pod in self.pods:
            for i_player in i_pod.players:
                if player is i_player:
                    return i_pod
        return None

    def won(self, players=[]):
        if not isinstance(players, list):
            players = [players]
        for player in players:
            pod = self.find_player_pod(player)

            if not player or not pod:
                Log.log('Player {} not found in any pod'.format(player.name), Log.Level.WARNING)
                continue

            if not pod.done:
                player.points = player.points + 3

                pod.won = player
                pod.done = True

            if self.done:
                self.conclude()

    def draw(self, players=[]):
        for player in players:
            pod = self.find_player_pod(player)

            player.points = player.points + 1

            pod.draw.append(player)
            pod.done = True

        if self.done and not self.concluded:
            self.conclude()

    def ready(self):
        return sum([pod.p_count for pod in self.pods]) == len(self.tour.players)

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
        #    print(action.func_name)

    else:
        tour.add_player([
            names.get_full_name()
            for i in range(17)
        ])
        for i in range(5):
            tour.make_pods()
            tour.random_results()
            #tour.remove_player(tour.players[0])

        print()
    #tour.show_pods()

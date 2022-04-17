import sys, argparse
from copy import deepcopy
import random
import shlex
import jellyfish

POD_SIZES = [4, 3]
MIN_POD_SIZE = min(POD_SIZES)

players = list()
dropped = list()

ROUNDS = list()
ROUND = None

LAST = None
OUTPUT_BUFFER = list()

RANKING = lambda x: (-x.points, -x.opponent_winrate, -x.unique_opponents)
MATCHING = lambda x: (-x.unique_opponents, x.points, -x.opponent_winrate)

class ID:
    lastID = 0
    def __init__(self):
        pass

    @staticmethod
    def next():
        ID.lastID += 1
        return ID.lastID

class Player:
    def __init__(self, name):
        self.name = name
        self.points = 0
        self.played = list()
        self.ID = ID.next()
        self.games_played = 0
        self.games_won = 0

    @property
    def not_played(self):
        return list(set(players) - set(self.played))

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
        parser_player = subparsers.add_parser('player', help='player help')

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
            fields.append('({}){}'.format(self.ID, self.name))
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
                        ' ' if not self.done else
                        'W' if p == self.won else
                        'D' if p in self.draw else
                        'L',
                        p.__repr__(['-w']))
                    for i, p in
                    zip(range(1, self.p_count+1), self.players)
                ]
            ))

class Round:
    def __init__(self, seq):
        self.seq = seq
        self.pods = []
        self.players = None
        self.concluded = False
        self.tiebreaker = False

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
        remaining = list(set(players) - seated)

        n_plyr = len(remaining)
        pod_sizes = Round.get_pod_sizes(n_plyr)
        if pod_sizes is None:
            OUTPUT_BUFFER.append('Can not make pods.')
            return None
        n_pods = len(pod_sizes)

        pods = [Pod(i, size) for size, i in zip(pod_sizes, range(n_pods))]
        random.shuffle(remaining)
        for p in sorted(remaining, key=MATCHING, reverse=True):
            pod_scores = [p.evaluate_pod(pod) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)

        self.pods += pods

        self.print_pods()

    def print_pods(self):
        for p in self.pods:
            OUTPUT_BUFFER.append(p)
            OUTPUT_BUFFER.append('-'*80)

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
        OUTPUT_BUFFER.append(swap_count, 'swaps in optimization stage.')

    @staticmethod
    def get_pod_sizes(n=None):
        if(isinstance(n, list)):
            n = n[0]
        for pod_size in POD_SIZES:
            if n-pod_size == 0:
                return [pod_size]
            if n-pod_size < MIN_POD_SIZE and pod_size == POD_SIZES[-1]:
                return None
            if n-pod_size >= MIN_POD_SIZE:
                tail = Round.get_pod_sizes(n-pod_size)
                if tail:
                    return [pod_size] + tail
        return None

    def conclude(self):
        for pod in self.pods:
            for p in pod.players:
                pod.update_player_history(p)

        self.players = deepcopy(players)
        ROUNDS.append(self)
        self.concluded = True
        OUTPUT_BUFFER.append('{}{}{}'.format(30*'*', '\nRound completed!\n', 30*'*',))

        #global ROUND
        #ROUND = Round(len(ROUNDS))

    def find_player_pod(self, pname):
        pod = player = None

        for i_pod in self.pods:
                for i_p in i_pod.players:
                    if pname.lower() in i_p.name.lower():
                        pod = i_pod
                        player = i_p
                        break
                if pod:
                    break
        return player, pod

    def won(self, tokens):
        for pname in tokens:
            player, pod = self.find_player_pod(pname)


            if not player or not pod:
                OUTPUT_BUFFER.append('Player {} not found in any pod'.format(pname))
                continue

            if not pod.done:
                player.points = player.points + WIN_POINTS

                pod.won = player
                pod.done = True

            if self.done:
                self.conclude()

    def draw(self, tokens):
        for pname in tokens:
            player, pod = self.find_player_pod(pname)

            player.points = player.points + DRAW_POINTS

            pod.draw.append(player)
            pod.done = True

        if self.done and not self.concluded:
            self.conclude()

    def ready(self):
        return sum([pod.p_count for pod in self.pods]) == len(players)

def tokenize(stdin):
    tokens = shlex.split(stdin)
    if not tokens:
        return None, None

    if tokens[0].lower() not in options or tokens[0] == 'def':
        tokens = ['def'] + tokens

    for i in range(1, len(tokens)):
        try:
            tokens[i] = int(tokens[i])
        except:
            pass
    if len(tokens) > 1:
        return tokens[0].lower(), tokens[1::]
    return tokens[0].lower(), None

def unknown(tokens):
    OUTPUT_BUFFER.append('Uknown command: {} with arguments {}'.format(tokens[0], tokens[1::]))

def add_player(names=[]):
    added_players = []
    if not isinstance(names, list):
        names =  [names]
    for name in names:
        if name in [p.name for p in players]:
            OUTPUT_BUFFER.append('\tPlayer {} already enlisted.'.format(name))
            continue
        if name:
            p = Player(name)
            players.append(p)
            OUTPUT_BUFFER.append('\tAdded player {}'.format(p.name))
            added_players.append(p)
    return added_players

def get_player(search, helper=True):
    try:
        pid = int(search)
    except Exception as e:
        pid = None
    for p in players:
        if p.name == search:
            return p

        if p.ID == pid:
                return p

    if helper and pid is None:
        suggested = []
        for p in players:
            if jellyfish.damerau_levenshtein_distance(search, p.name) <= 4:
                suggested.append(p)
                continue
            for word in p.name.split(' '):
                if jellyfish.damerau_levenshtein_distance(search, word) <= 2:
                    suggested.append(p)
                    break
        if len(suggested) == 1:
            while True:
                choice = input('Did you mean "{}" with "{}"? (y/n)'.format(suggested[0].name, search))
                if choice == 'y' or choice == 'Y':
                    return suggested[0]
                elif choice == 'n' or choice == 'N':
                    return None
                else:
                    print('Unknown option. Please retry.')
        elif len(suggested) > 1:
            print('Optional alternatives for "{}":'.format(search))
            for i, p in zip(range(len(suggested)), suggested):
                print('\t{} : {}'.format(i, p.name))
            while True:
                try:
                    choice = int(input("Select number:"))
                    return suggested[choice]
                except Exception as e:
                    print(str(e))

    OUTPUT_BUFFER.append('\tPlayer {} does not exist and no suggestions found.'.format(search))
    return None

def remove_player(names=[], helper=True):
    if ROUND:
        if not ROUND.concluded:
            OUTPUT_BUFFER.append('ERROR: Can\'t drop player during an active round.\nComplete the round and remove player before creating new pods.')
            return
    if not isinstance(names, list):
        names = [names]
    for name in names:
        p = get_player(name, helper=helper)
        if p is None:
            return

        if p.played:
            dropped.append(p)
        players.remove(p)

        OUTPUT_BUFFER.append('\tRemoved player {}'.format(p.name))

def rename_player(name, new_name, helper=True):
    p = get_player(name, helper=helper)
    if p is None:
        OUTPUT_BUFFER.append('\tPlayer {} doesnt exist.'.format(p.name))
        return False

    if new_name in [p.name for p in players]:
        OUTPUT_BUFFER.append('\tPlayer {} already exists.'.format(new_name))
        return False

    p.name = new_name
    OUTPUT_BUFFER.append('\tRenamed player {} -> {}'.format(name, new_name))

    return True

def player_stats(tokens=['-i', '-p'], players=players):
    #parser.add_argument('-p', '--points', dest='p', action='store_true')
    #parser.add_argument('-u', '--unique', dest='u', action='store_true')
    #parser.add_argument('-l', '--log', dest='l', action='store_true')
    #parser.add_argument('-t', '--tiebreakers', dest='t', action='store_true')
    parser_stats = subparsers.add_parser('stats', help='stats help')
    parser_stats.add_argument('-s', '--sort', dest='s', default='s')

    sargs = None
    unknown = None
    try:
        sargs, unknown = parser_stats.parse_known_args(tokens)
    except:
        pass
        #args = None
        #OUTPUT_BUFFER.append('Invalid argumets')

    l = {
        'n': lambda x: x.name,
        'i': lambda x: x.ID,
        's': RANKING, #sort by standing
        'u': lambda x: (-x.unique_opponents, x.name),
    }
    if sargs:
        for player in sorted(players, key=l[sargs.s]):
            if unknown:
                OUTPUT_BUFFER.append('\t{}'.format(player.__repr__(unknown)))
            else:
                OUTPUT_BUFFER.append('\t{}'.format(player.__repr__()))
        #TODO: Dropped players somewhat broken anyway
        if dropped:
            OUTPUT_BUFFER.append('Dropped players:')
            for player in sorted(dropped, key=l[sargs.s]):
                if unknown:
                    OUTPUT_BUFFER.append('\t{}'.format(player.__repr__(unknown)))
                else:
                    OUTPUT_BUFFER.append('\t{}'.format(player.__repr__()))

def new_round(tokens=[]):
    global ROUND
    if not ROUND or ROUND.concluded:
        ROUND = Round(len(ROUNDS))
        return True
    else:
        if ROUND.pods:
            OUTPUT_BUFFER.append(
                '{}\n{}\n{}'.format(
                    30*'*',
                    'Please report results of following pods:',
                    30*'*',
                )
            )
            for pod in ROUND.pods:
                if not pod.done:
                    OUTPUT_BUFFER.append(str(pod))
        else:
            OUTPUT_BUFFER.append(
                'Round has no pods - add some or cancel round.'
            )
        return False

def reset_pods(tokens=[]):
    global ROUND
    if ROUND:
        ROUND.pods = []

def make_pods(tokens=[]):
    if ROUND is None or ROUND.concluded:
        new_round(len(ROUNDS))
    if not ROUND.ready():
        ROUND.make_pods()
    else:
        OUTPUT_BUFFER.append(
            '{}\n{}\n{}'.format(
                30*'*',
                'Please report results of following pods:',
                30*'*',
            )
        )
        for pod in ROUND.pods:
            if not pod.done:
                OUTPUT_BUFFER.append(str(pod))

def manual_pod(tokens=[]):
    if ROUND is None or ROUND.concluded:
        new_round(len(ROUNDS))
    if not ROUND.pods:
        ROUND.pods=[]
    pod = Pod(len(ROUND.pods))

    for pid in tokens:
        p = get_player(pid, helper=True)
        pod.add_player(p)
    ROUND.pods.append(pod)

def report_win(tokens=[]):
    if ROUND:
        ROUND.won(tokens)

def report_draw(tokens=[]):
    if ROUND:
        ROUND.draw(tokens)

def random_results(tokens=None):
    if not ROUND:
        OUTPUT_BUFFER.append('ERROR: A round is not in progress.\nStart a new round with "pods" command.')
        return
    if ROUND.pods is not None:
        for pod in [x for x in ROUND.pods if not x.done]:
            result = random.sample(pod.players, 1)
            #Selects one. If you want draws too, use
            #result = random.sample(pod.players, random.randint(1, pod.p_count))
            if len(result) == 1:
                OUTPUT_BUFFER.append('won "{}"'.format(result[0].name))
                report_win([result[0].name])
            else:
                OUTPUT_BUFFER.append('draw {}'.format(' '.join(['"{}"'.format(p.name) for p in result])))
                report_draw([p.name for p in result])

def log():
    x = 30
    OUTPUT_BUFFER.append('*'*x)
    OUTPUT_BUFFER.append('Tournament with {} attendants:'.format(len(players)))

    for p in players:
        OUTPUT_BUFFER.append('\t{}'.format(p.name))

    for i in range(len(ROUNDS)):
        r = ROUNDS[i]
        OUTPUT_BUFFER.append('*'*x)
        OUTPUT_BUFFER.append('ROUND {}'.format(i + 1))
        OUTPUT_BUFFER.append('*'*x)

        for pod in r.pods:
            OUTPUT_BUFFER.append(pod.__repr__())

        OUTPUT_BUFFER.append('\n Standings after round {}:'.format(i + 1))

        player_stats(players=r.players)


def print_output(tokens=[]):
    global LAST

    f = open('print.txt', 'w')
    f.write('\n'.join([str(x) for x in LAST]))
    f.close()

def rtfm(tokens=None):
    with open('README.md', 'r') as fin:
        OUTPUT_BUFFER.append(fin.read())

def show_pods(tokens=[]):
    if ROUND and ROUND.pods:
        ROUND.print_pods()
    else:
        OUTPUT_BUFFER.append('No pods currently created.')

options = {
    'add': add_player,
    'def': unknown,
    'draw': report_draw,
    'h': rtfm,
    'help': rtfm,
    'log': log,
    'pods': make_pods,
    'pod': manual_pod,
    'print': print_output,
    'q': quit,
    'random': random_results,
    'remove': remove_player,
    'resetpods': reset_pods,
    'showpods': show_pods,
    'spods': Round.get_pod_sizes,
    'stats': player_stats,
    'win': report_win,
}

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--players', dest='players', nargs='*')
parser.add_argument('-f', '--file', dest='file')
parser.add_argument('-i', '--input', dest='input', nargs='*')

subparsers = parser.add_subparsers()

if __name__ == "__main__":
    raise NotImplementedError('The CLI interface has to be updated to comply with UI+core changes. Use \'run_cli.py\'')
    args, unknown = parser.parse_known_args()

    #Reads file to parse players from
    if args.file:
        with open(args.file) as f:
            content = f.readlines()
        for p in [x.strip() for x in content]:
            add_player(p)

    if args.players:
        for p in args.players:
            add_player(p)

    while True:
        if OUTPUT_BUFFER:
            print('\n'.join([str(x) for x in OUTPUT_BUFFER]))
            LAST = OUTPUT_BUFFER.copy()
            OUTPUT_BUFFER.clear()
        ret = None
        if args.input:
            pre_in = args.input.pop(0)
            OUTPUT_BUFFER.append('> ' + pre_in)
            cmd, tokens = tokenize(pre_in)
        else:
            cmd, tokens = tokenize(input('> '))

        if cmd:
            if tokens:
                ret = options[cmd](tokens)
            else:
                ret = options[cmd]()

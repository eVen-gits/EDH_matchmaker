import sys, argparse
from copy import deepcopy
import random
import shlex

POD_SIZES = [4, 3]
MIN_POD_SIZE = min(POD_SIZES)

players = list()

ROUNDS = list()

ROUND = None

class Player:
    def __init__(self, name):
        self.name = name
        self.points = 0
        self.played = list()

    @property
    def not_played(self):
        return list(set(players) - set(self.played))

    @property
    def unique_opponents(self):
        return len(set(self.played))

    def evaluate_pod(self, pod):
        score = 0
        if pod.p_count == pod.cap:
            return -sys.maxsize
        for player in pod.players:
            score = score - self.played.count(player) ** 2
        return score

    def __repr__(self, detailed=False):
        ret = '{} | played: {} | pts: {}'.format(self.name, len(set(self.played)), self.points)
        if detailed:
            ret = ret + '\n\t' + '|'.join([p.name.split(' ')[0] for p in self.played])
        return ret

class Pod:
    def __init__(self, cap, id):
        self.players = list()
        self.cap = cap
        self.id = id
        self.done = False

    @property
    def p_count(self):
        return len(self.players)

    def add_player(self, player: Player):
        if self.p_count >= self.cap:
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

    def update_player_opponent_history(self, player):
        player.played = player.played + ([p for p in self.players if not p is player])

    def __repr__(self):
        return 'Pod {} of {} players ~ ({}):\n\t{}'.format(
            self.id,
            self.p_count,
            self.score,
            '\n\t'.join(
                [p.name for p in self.players]
            ))

class Round:
    def __init__(self, seq):
        self.seq = seq
        self.pods = None
        self.players = None
        self.concluded = False

    @property
    def done(self):
        for pod in self.pods:
            if not pod.done:
                return False
        return True

    def make_pods(self):
        n_plyr = len(players)
        pod_sizes = Round.get_pod_sizes(n_plyr)
        n_pods = len(pod_sizes)

        pods = [Pod(size, i) for size, i in zip(pod_sizes, range(n_pods))]
        for p in sorted(players, key=lambda x: (-len(set(x.played)), x.points), reverse=True):
            pod_scores = [p.evaluate_pod(pod) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            #print(pod_scores, index)
            #print('Adding {} to pod {}'.format(p.name, index))
            pods[index].add_player(p)

        self.pods = pods

        #self.optimize()

        for p in self.pods:
            print(p)

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
        print(swap_count, 'swaps in optimization stage.')

    @staticmethod
    def get_pod_sizes(n):
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
                pod.update_player_opponent_history(p)

        self.players = deepcopy(players)
        ROUNDS.append(self)
        ROUND = None
        self.concluded = True
        print('{}{}{}'.format(30*'*', '\nRound completed!\n', 30*'*',))

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
                print('Player {} not found in any pod'.format(pname))
                continue

            player.points = player.points + 3

            pod.done = True

            if self.done:
                self.conclude()

    def draw(self, tokens):
        for pname in tokens:
            player, pod = self.find_player_pod(pname)

            player.points = player.points + 1

            pod.done = True



        if self.done and not self.concluded:
            self.conclude()

def tokenize(stdin):
    tokens = shlex.split(stdin)
    if tokens[0].lower() not in options:
        tokens = ['def'] + tokens

    for i in range(1, len(tokens)):
        try:
            tokens[i] = int(tokens[i])
        except:
            pass
    if len(tokens) > 1:
        return tokens[0], tokens[1::]
    return tokens[0], None

def unknown(tokens):
    print('Uknown command: {} with arguments {}'.format(tokens[0], tokens[1::]))

def add_player(names):
    if not isinstance(names, list):
        names =  [names]
    for name in names:
        if name in [p.name for p in players]:
            print('\tPlayer {} already enlisted.'.format(token))
            #continue
        p = Player(name)
        players.append(p)
        print('\tAdded player {}'.format(p.name))

def player_stats(tokens=['-p', '-s', 'p'], players=players):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--points', dest='p', action='store_true')
    parser.add_argument('-u', '--unique', dest='u', action='store_true')
    parser.add_argument('-l', '--log', dest='l', action='store_true')
    parser.add_argument('-s', '--sort', dest='s', default='a')

    try:
        args, unknown = parser.parse_known_args(tokens)
    except:
        args = None
        print('Invalid argumets')

    l = {
        'a': lambda x: x.name,
        'p': lambda x: (-x.points, x.name),
        'u': lambda x: (-x.unique_opponents, x.name),
    }
    if args:
        for player in sorted(players, key=l[args.s]):
            fields = list()
            fields.append(player.name)
            if args.u:
                fields.append('unique: {}'.format(player.unique_opponents))
            if args.p:
                fields.append('pts: {}'.format(player.points))
            if args.l:
                fields.append('log: {}'.format('|'.join([p.name for p in player.played])))

            print('\t{}'.format(' | '.join(fields)))

def make_pods(tokens=len(ROUNDS)):
    global ROUND
    if not ROUND or ROUND.concluded:
        ROUND = Round(tokens)
        ROUND.make_pods()
    else:
        print(
            '{}\n{}\n{}'.format(
                30*'*',
                'Please report results of following pods:',
                30*'*',
            )
        )
        for pod in ROUND.pods:
            if not pod.done:
                print(str(pod))

def report_win(tokens):
    if ROUND:
        ROUND.won(tokens)

def report_draw(tokens):
    if ROUND:
        ROUND.draw(tokens)

def random_results(tokens=None):
    for pod in [x for x in ROUND.pods if not x.done]:
        players = pod.players
        result = random.sample(pod.players, random.randint(1, pod.p_count))
        if len(result) == 1:
            print('won "{}"'.format(result[0].name))
            report_win([result[0].name])
        else:
            print('draw', ' '.join(['"{}"'.format(p.name) for p in result]))
            report_draw([p.name for p in result])

def log():
    x = 30
    print('*'*x)
    print('Tournament with {} attendants:'.format(len(players)))

    for p in players:
        print('\t{}'.format(p.name))

    for i in range(len(ROUNDS)):
        r = ROUNDS[i]
        print('*'*x)
        print('ROUND {}'.format(i + 1))
        print('*'*x)

        for pod in r.pods:
            print(pod.__repr__())

        print('\n Standings after round {}:'.format(i + 1))

        player_stats(players=r.players)

        print()

def rtfm(tokens=None):
    with open('README.md', 'r') as fin:
        print(fin.read())

options = {
    'add': add_player,
    'list': player_stats,
    'pods': make_pods,
    'spods': Round.get_pod_sizes,
    'won': report_win,
    'draw': report_draw,
    'q': quit,
    'def': unknown,
    'log': log,
    'random': random_results,
    'h': rtfm
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', dest='players', nargs='*')
    parser.add_argument('-f', '--file', dest='file')
    parser.add_argument('-i', '--input', dest='input', nargs='*')

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
        ret = None
        if args.input:
            pre_in = args.input.pop(0)
            print('>', pre_in)
            cmd, tokens = tokenize(pre_in)
        else:
            cmd, tokens = tokenize(input('> '))


        if tokens:
            ret = options[cmd](tokens)
        else:
            ret = options[cmd]()


        if ret:
            print(ret)
        #print('>')

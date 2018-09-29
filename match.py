import sys, argparse
from copy import deepcopy
from random import shuffle

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
        return True

    def remove_player(self, player: Player):
        return self.players.pop(player)

    def swap_players(self, p1: Player, other, p2: Player):
        self.remove_player(p1)
        other.remove_player(p2)

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

    def __repr__(self):
        return 'Pod {} of {} players\n\t{}'.format(
            self.id,
            self.p_count,
            '\n\t'.join(
                [str(p) for p in self.players]
            ))

class Round:
    def __init__(self, seq):
        self.seq = seq
        self.pods = None

    def make_pods(self):
        n_plyr = len(players)
        pod_sizes = Round.get_pod_sizes(n_plyr)
        n_pods = len(pod_sizes)

        pods = [Pod(size, i) for size, i in zip(pod_sizes, range(n_pods))]
        #shuffle(players)
        for p in sorted(players, key=lambda x: (-len(set(x.played)), x.points), reverse=True):
            pod_scores = [p.evaluate_pod(pod) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            #print(pod_scores, index)
            #print('Adding {} to pod {}'.format(p.name, index))
            pods[index].add_player(p)

        for p in pods:
            print('Pod score:', p.score)
            print(p)

        self.pods = pods

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

    def won(self, tokens):
        for pname in tokens:
            pod = player = None

            for i_pod in self.pods:
                for i_p in i_pod.players:
                    if pname.lower() in i_p.name.lower():
                        pod = i_pod
                        player = i_p
                        break
                if pod:
                    break

            #[a.played.append([b for b in pod.players if not b is a]) for a in pod.players]

            for i_p in pod.players:
                i_p.played = i_p.played + ([p for p in pod.players if not p is i_p])
                print(i_p)
                if i_p == player:
                    player.points = player.points + 3

            pod.done = True

            done = True
            for pod in self.pods:
                if not pod.done:
                    done = False

            if done:
                ROUNDS.append(self)
                ROUND = None
                print('Round completed!\nStandings:')

    def draw(self, tokens):
        pod_id = tokens[0]
        raise NotImplementedError

def tokenize(stdin):
    tokens = stdin.split(' ')
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

def add_player(tokens):
    p = Player(tokens)
    players.append(p)
    print('Added player {}'.format(p.name))

def player_stats(tokens=['-p', '-s', 'p']):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--points', dest='p', action='store_true')
    parser.add_argument('-u', '--unique', dest='u', action='store_true')
    parser.add_argument('-s', '--sort', dest='s', default='a')

    args, unknown = parser.parse_known_args(tokens)

    l = {
        'a': lambda x: x.name,
        'p': lambda x: (-x.points, x.name),
        'u': lambda x: (x.unique_opponents, x.name),
    }

    for player in sorted(players, key=l[args.s]):
        fields = list()
        fields.append(player.name)
        if args.u:
            fields.append('unique: {}'.format(player.unique_opponents))
        if args.p:
            fields.append('pts: {}'.format(player.points))

        print('\t{}'.format(' | '.join(fields)))


def make_pods(tokens=len(ROUNDS)):
    global ROUND
    ROUND = Round(tokens)
    ROUND.make_pods()

def report_win(tokens):
    if ROUND:
        ROUND.won(tokens)

def report_draw(tokens):
    if ROUND:
        ROUND.draw(tokens)

options = {
    'add': add_player,
    'list': player_stats,
    'pods': make_pods,
    'spods': Round.get_pod_sizes,
    'won': report_win,
    'draw': report_draw,
    'q': exit,
    'def': unknown
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', dest='players', nargs='*')
    parser.add_argument('-i', '--input', dest='input', nargs='*')

    args, unknown = parser.parse_known_args()

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

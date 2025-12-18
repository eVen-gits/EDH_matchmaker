#This is requred so that relative imports work
import sys
sys.path.append(".")

from src.core import *

from tqdm import tqdm
import names
import numpy as np
import random


def create_tournament(players):
    """

    players: dict where keys are names and values are player 'skill'

    """
    t = Tournament()
    t.add_player(list(players.keys()))

    return t


def deliver_results(t: Tournament, winners):
    t.report_win(winners)


"""
from numpy.random import choice
draw = choice(list_of_candidates, number_of_items_to_pick,
              p=probability_distribution)
"""
def determine_winner(players, player_skill_obj):
    player_skills = [player_skill_obj[p.name] for p in players]
    player_probs = [p / sum(player_skills) for p in player_skills]
    #return np.random.choice(players, 1, p=player_probs)[0]
    return random.choices(players, weights=player_probs)


PAST_PLAYERS = [
    ("Pahor Samo (8)", 8),
    ("Birsa Luka (7)", 7),
    ("Denchy (7)", 7),
    ("Bibic Zan (6)", 6),
    ("DeadeyeOrb (6)", 6),
    ("Bradule Bernard (5)", 5),
    ("Karou (5)", 5),
    ("Krelj Ales (4)", 4),
    ("MrKing (4)", 4),
    ("Vencelj Egej (4)", 4),
    ("Gruden Gregor (4)", 4),
    ("zyzz (3)", 3),
    ("Kovacec Rene (2)", 2),
    ("Bartol Zan (2)", 2),
    ("Rakovic Damir (2)", 2),
    ("Evil (2)", 2),
    ("wudup (2)", 2),
    ("zoky (1)", 1),
    ("Bajzelj (1)", 1),
    ("Vodnik Primoz (1)", 1),
    ("Obup (1)", 1),
    ("Raso (1)", 1),
    ("Sentigon (1)", 1),
    ("Mai (1)", 1),
    ("Carnessa (0)", 0),
    ("Brighter (1)", 1),
    ("The5G (0)", 0),
    ("Vodnik Luka (3)", 3),
]


if __name__ == "__main__":
    #num_players = 30
    num_rounds = 4

    #player_names = [fkr.name() for _ in range(num_players)]
    #player_skill_obj = {n: 5 for n in player_names}

    """op_player = player_names[0]
    print(op_player)
    player_skill_obj[op_player] = 100000"""

    player_skill_obj = {n: s+1 for (n, s) in PAST_PLAYERS}

    t = create_tournament(
        player_skill_obj
    )

    for _ in tqdm(range(num_rounds)):
        t.make_pods()
        #[print(p) for p in t.tour_round.pods]
        for pod in t.tour_round.pods:
            if np.random.random() < 0:
                t.report_draw(pod.players)
            else:
                t.report_win(determine_winner(pod.players, player_skill_obj))

    for i, p in enumerate(t.get_standings(), 1):
        if i == 16:
            print(i, f"{str(p)}\n---------")
        else:
            print(i, str(p))

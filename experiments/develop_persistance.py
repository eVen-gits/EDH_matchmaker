import json
from src.core import Tournament, TournamentConfiguration, TournamentAction
from faker import Faker
from tqdm import tqdm

if __name__ == "__main__":
    t = Tournament(
        TournamentConfiguration(
            pod_sizes=[4, 3],
            allow_bye=True,
            snake_pods=True,
            max_byes=2,
            auto_export=False,
        )
    )
    fkr = Faker()
    players = [
        f"{i}:{fkr.name()}" for i in range(17)
    ]
    t.add_player(players)
    for i in tqdm(range(5)):
        t.create_pairings()
        t.random_results()
        tour_orig = t.serialize()
        json.dump(tour_orig, open('tournament.json', 'w'), indent=4)


    tour_orig = t.serialize()
    json.dump(tour_orig, open('tournament.json', 'w'), indent=4)
    del t

    t2 = Tournament.inflate(json.load(open('tournament.json')))
    tour_new = t2.serialize()
    # compare tour_orig and tour_new and print the differences
    json.dump(tour_new, open('tournament_new.json', 'w'), indent=4)

    #load both jsons and print lines that are different
    with open('tournament.json', 'r') as f:
        tour_orig = f.readlines()
    with open('tournament_new.json', 'r') as f:
        tour_new = f.readlines()
    print(tour_orig == tour_new)
    for i in range(len(tour_orig)):
        if tour_orig[i] != tour_new[i]:
            print(f"Line {i}: {tour_orig[i]} != {tour_new[i]}")
            pass


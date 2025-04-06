from tqdm import tqdm
import os

from src.misc import generate_player_names
from src.core import Tournament, TournamentConfiguration, Log, TournamentAction, Core

from uuid import UUID

if __name__ == "__main__":
    #for i in range(3000, 10000, 1000):

    os.environ['PERSISTENCE_MODULE'] = 'eventsourcing.sqlite'
    os.environ['SQLITE_DBNAME'] = 'database.db'

    n = 16
    print(f"Tour size: {n}")
    core = Core()

    #t = core.get_tournament(UUID('80261c28-aee8-4ce8-a2d3-d536b0107555'))
    pass

    config = TournamentConfiguration(
            core,
            #pod_sizes=[4, 3],
            #n_rounds=8,
            #auto_export=False,
            #allow_bye=True,
            #win_points=5,
            #draw_points=1,
            #bye_points=2,
            #snake_pods=True,
            #max_byes=2,
    )

    core.save(config.aggregate)
    cid = config.aggregate.id
    config = core.repository.get(cid)
    pass
    t1 = Tournament(
        core,
        config
    )
    config = t1.config
    tid = t1.aggregate.id
    t2 = core.get_tournament(tid)

    pass

    t1.add_player(generate_player_names(n))
    #TournamentAction.LOGF = os.path.normpath("/home/even/Dev/EDH_matchmaker/logs/testing.log")

    for n in tqdm(range(5), desc="Round", total=5):
        t1.create_pairings()
        t1.random_results()

    pass


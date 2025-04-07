from tqdm import tqdm
import os

from src.misc import generate_player_names
from src.core import Tournament, TournamentConfiguration, Log, TournamentAction, Core, Player

from uuid import UUID

if __name__ == "__main__":
    #for i in range(3000, 10000, 1000):

    os.environ['PERSISTENCE_MODULE'] = 'eventsourcing.sqlite'
    os.environ['SQLITE_DBNAME'] = 'database.db'

    restore = False
    core = Core()
    if not restore:
        n = 16

        config = TournamentConfiguration(
                core,
                pod_sizes=[4, 3],
                n_rounds=8,
                auto_export=False,
                allow_bye=True,
                win_points=5,
                draw_points=1,
                bye_points=2,
                snake_pods=True,
                max_byes=2,
        )
        core.save(config.aggregate)
        cid = config.aggregate.id
        config = core.repository.get(cid)
        pass
        t1 = Tournament(
            core,
            config
        )

        pass

        p = Player(t1, 'Player 1')

        t1.add_player(generate_player_names(n))

        print(t1.players)
        print(t1.id)
    else:
        tid = UUID('52631e2b-75ef-419d-8ef9-c142ae2fba24')
        t1 = core.get_tournament(tid)
        pass
    #TournamentAction.LOGF = os.path.normpath("/home/even/Dev/EDH_matchmaker/logs/testing.log")

    #for n in tqdm(range(5), desc="Round", total=5):
    #    t1.create_pairings()
    #    t1.random_results()

    pass


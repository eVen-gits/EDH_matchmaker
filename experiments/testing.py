from tqdm import tqdm
import os

from src.misc import generate_player_names
from src.core import Tournament, TournamentConfiguration, Log, TournamentAction, Core, Player
from src.interface import ITournament

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
        tour = Tournament(
            core,
            config
        )

        pass

        players = tour.add_player(generate_player_names(n))

        tour.create_pairings()

        pass
    else:
        # Show available tournaments
        print(f"\nAvailable tournaments:")
        tournaments = []
        start = 1
        while True:
            notifications = list(core.notification_log.select(start=start, limit=10))
            if not notifications:
                break
            for notification in notifications:
                event = core.mapper.to_domain_event(notification)
                try:
                    aggregate = core.repository.get(event.originator_id)
                    if isinstance(event, ITournament.Registered):
                        tournaments.append((event.originator_id, event.originator_version))
                        print(f"{len(tournaments)}. Tournament ID: {event.originator_id}")
                except:
                    continue
            start += 10

        if tournaments:
            choice = int(input("\nEnter tournament number to restore: ")) - 1
            if 0 <= choice < len(tournaments):
                tour_id, _ = tournaments[choice]

                # Print all events in chronological order
                print("\nAll events in chronological order:")
                start = 1
                while True:
                    notifications = list(core.notification_log.select(start=start, limit=10))
                    if not notifications:
                        break
                    for notification in notifications:
                        event = core.mapper.to_domain_event(notification)
                        print(f"\nEvent: {event.__class__.__name__}")
                        print(f"  Originator ID: {event.originator_id}")
                        print(f"  Version: {event.originator_version}")
                        print(f"  Topic: {notification.topic}")
                        print(f"  State: {event.__dict__}")
                    start += 10

                # Get the tournament aggregate directly
                try:
                    tour_aggregate = core.repository.get(tour_id)
                    print(f"\nTournament aggregate state:")
                    print(f"ID: {tour_aggregate.id}")
                    print(f"Config ID: {tour_aggregate.config}")
                    print(f"Player IDs: {tour_aggregate.players}")

                    # Try to get each player
                    print("\nAttempting to load players:")
                    for player_id in tour_aggregate.players:
                        try:
                            player_aggregate = core.repository.get(player_id)
                            print(f"Found player: {player_aggregate.name} (ID: {player_id})")
                        except:
                            print(f"Could not load player with ID: {player_id}")

                    # Now try to restore the tournament
                    tour = core.get_tournament(tour_id)
                    print(f"\nRestored tournament has {len(tour.players)} players")

                except Exception as e:
                    print(f"Error loading tournament: {e}")
            else:
                print("Invalid choice")
        else:
            print("No tournaments found")
    #TournamentAction.LOGF = os.path.normpath("/home/even/Dev/EDH_matchmaker/logs/testing.log")

    #for n in tqdm(range(5), desc="Round", total=5):
    #    t1.create_pairings()
    #    t1.random_results()

    pass


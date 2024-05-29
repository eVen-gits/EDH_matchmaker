import matplotlib.pyplot as plt
import numpy as np
from src.core import *
from tqdm import tqdm

TournamentAction.LOGF = False #type: ignore
np.set_printoptions(formatter={'float_kind': "{:.3f}".format})


N = 18
rounds = 3
sims = 7

# Initialize a list to store seating averages for each player across all simulations
all_player_seating_averages = np.zeros([sims, rounds, N])

max_rounds_per_player = [0] * N  # Initialize a list to store the maximum number of rounds for each player

for sim_n, sim in tqdm(enumerate(range(sims)), total=sims):
    t = Tournament(
        TournamentConfiguration(
            pod_sizes=[4, 3],
            allow_bye=True,
            snake_pods=True,
            max_byes=2,
        )
    )
    t.add_player([
        names.get_full_name()
        for _ in range(N)
    ])

    player_averages_per_sim = np.empty([rounds, N])  # Initialize a list to store seating averages for each player in this simulation

    for i in range(rounds):
        t.make_pods()
        t.random_results()
        player_averages_per_sim[i, :] = np.array([player.average_seat for player in t.players])
    #sort array by final average_seat values
    player_averages_per_sim = player_averages_per_sim.T[player_averages_per_sim[-1, :].argsort()].T

    pass

    # Extend the all_player_seating_averages with the player averages for this simulation
    all_player_seating_averages[sim_n, :, :] = player_averages_per_sim


# Plot histograms for each round
plt.figure(figsize=(12, 8))
for i in range(rounds):
    plt.subplot(3, 3, i + 1)  # Subplot index starts from 1
    plt.hist(all_player_seating_averages[:, i, :].ravel(), bins=np.arange(0, 1.1, 0.1), alpha=0.7)
    plt.title(f'Round {i + 1} Histogram')
    plt.xlabel('Average Seat')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

pass

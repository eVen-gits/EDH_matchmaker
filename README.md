# EDH Matchmaker

[![Documentation](https://img.shields.io/badge/docs-live-blue.svg)](https://even-gits.github.io/EDH_matchmaker/)
[![Build Status](https://github.com/eVen-gits/EDH_matchmaker/actions/workflows/run_tests.yml/badge.svg)](https://github.com/eVen-gits/EDH_matchmaker/actions/workflows/run_tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/eVen-gits/EDH_matchmaker/blob/master/LICENSE.md)

![image](https://user-images.githubusercontent.com/2612606/166250998-7b4c721c-8a35-4ac2-ad87-e8fe02c46d11.png)

A comprehensive tool for managing Swiss-pairing tournaments: Commander (EDH,
multiplayer pods), 1v1 Magic and Catan. A tournament's game is an explicit choice
(the ruleset), picked in the GUI when a new tournament is created.

## Features

- **Swiss Pairings**: Automated pairing logic per ruleset - EDH's pod-fill
  algorithm for 4-player pods (3-player fallback), and a maximum-weight
  matching for 1v1's 2-player pods. Pick the pairing logic per round in the
  tournament config.
- **1v1 Magic support**: Play/report best-of-N matches, single-elimination
  top cut seeded by Swiss standings, and MTR-standard tiebreakers (OMW, GW,
  OGW).
- **Catan support**: Catan tournaments (catanevents.com tournament rules):
  3-4 player tables with no byes, victory points in every report, ranking by
  wins, then total VP, VP%, 2nd and 3rd places, a 3-game preliminary schedule
  with different opponents and rotating play order, zero-win elimination
  before game 4 at 80+ players, serpentine top-16 semifinals and a final.
- **Pluggable Scoring**: Choose the scoring algorithm (Default, Hareruya,
  Modified Hareruya for Commander; a Magic Tournament Rules-based scoring for
  1v1). Each algorithm declares its own parameters in a sidecar file, and the
  config screen builds their fields automatically.
- **Tournament Management**: Track standings, drops, and round history.
- **Standings Export**: Export results for external use.
- **Cross-Platform**: Runs on Linux, Windows, and macOS (Python-based).

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/eVen-gits/EDH_matchmaker.git
   cd EDH_matchmaker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   To embed the engine (no GUI, docs or dev tools), install only the runtime
   list instead: `pip install -r requirements-core.txt`.

## Usage

Run the user interface:

```bash
python run_ui.py
```

Or with additional options:
```bash
python run_ui.py --help
```

Commander configuration accepts pod sizes of 3, 4, 5, or 6 players.
This restriction applies to new tournaments, configuration changes, CLI
options, and every attempt to load a saved tournament. Manual pod edits
can still leave two players in a pod. Use the 1v1 Magic ruleset for
configured two-player pairings.

### Running a Catan event

Pick the Catan ruleset when you create the tournament. Set the number of
games and the top cut yourself; the structure the rules recommend
(catanevents.com tournament rules 1.1.3 and 1.1.5) is:

| Players | Games | Top cut |
|---|---|---|
| 12-28 | 3 | 4: a final of the top 4 |
| 29-79 | 3 | 16: serpentine semifinals, then a final of the semifinal winners |
| 80+ | 4, with `eliminate_below_wins: 1` on game 4 | 16, as above |

At 80+ players, players without a win after game 3 are eliminated: they sit
out game 4 and the playoffs but keep their rank. Pairing game 1 with a Catan
pairing logic logs a warning, once, when the event has fewer than 12 players or
its structure differs from this table. Every Catan report needs every player's
victory points; a win reported without them is refused. With the engine:

```python
from src.core import Tournament, TournamentConfiguration
from src.interface import IGameResult

tour = Tournament(TournamentConfiguration(ruleset="CatanRuleset", n_rounds=3, top_cut=16))
tour.add_player([f"Player {i}" for i in range(40)])
tour.create_pairings()
pod = tour.tour_round.pods[0]
vp = dict(zip((p.uid for p in pod.players), (10, 8, 6, 5)))
tour.report_match(pod, [IGameResult(frozenset({pod.players[0].uid}), tuple(vp.items()))])
```

After the semifinals, places 5-16 follow the semifinal finishing places; after
the final, places 1-4 follow the final table (rule 1.4.4). The full rules
behind this are in [`docs/tournament-log-spec.md`](docs/tournament-log-spec.md#catanruleset).

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

The command is `pip install -r requirements.txt`

This will install the required libraries for the program to run.

## Running

Again, move to the directory and fire up:

`python run_ui.py`

On windows, you can also add a shortcut for this with different configurations.

## Runtime options

The program can also be run with some extra command parameters to set it up.

You can check it by running `python run_ui.py --help`. Again, you can add a shortcut WITH those parameters, if you don't want to add them every time manually.

Alternatively, you can also set everything up through GUI once the program is running.

## Testing

The project uses `pytest`. You can run tests locally using:

```bash
PYTHONPATH=. pytest
```

Automated test results are generated on every push via GitHub Actions. You can view the latest test status via the badge at the top of this README or in the **Actions** tab.

# Closing words

This software is still in development. The best thing you can do to help me is by testing it and submiting bugs. Best way to do it is here, on github - open an issue and describe what's happening and how to reproduce it, so I can fix it. You can also add your tournament log.

Also you can star this project for more exposure :)

Best regards,
/E
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

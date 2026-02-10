# EDH Matchmaker

[![Documentation](https://img.shields.io/badge/docs-live-blue.svg)](https://even-gits.github.io/EDH_matchmaker/)
[![Build Status](https://github.com/eVen-gits/EDH_matchmaker/actions/workflows/run_tests.yml/badge.svg)](https://github.com/eVen-gits/EDH_matchmaker/actions/workflows/run_tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

![image](https://user-images.githubusercontent.com/2612606/166250998-7b4c721c-8a35-4ac2-ad87-e8fe02c46d11.png)

A comprehensive tool for managing Commander (EDH) tournaments with Swiss pairings.

## Features

- **Swiss Pairings**: Automated pairing logic optimized for EDH (4-player pods).
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

## Usage

Run the user interface:

```bash
python run_ui.py
```

Or with additional options:
```bash
python run_ui.py --help
```

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

The project uses `unittest` for testing. You can run tests locally using:

```bash
PYTHONPATH=. python tests/run_tests.py
```

Automated test results are generated on every push via GitHub Actions. You can view the latest test status via the badge at the top of this README or in the **Actions** tab.

# Closing words

This software is still in development. The best thing you can do to help me is by testing it and submiting bugs. Best way to do it is here, on github - open an issue and describe what's happening and how to reproduce it, so I can fix it. You can also add your tournament log.

Also you can star this project for more exposure :)

Best regards,
/E
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

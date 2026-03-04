# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the application:**
```bash
python run_ui.py
python run_ui.py --help  # see all runtime options
```

**Run tests:**
```bash
PYTHONPATH=. python tests/run_tests.py
```

**Type checking:**
```bash
pyright  # uses pyrightconfig.json (basic mode)
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Build and deploy documentation:**
```bash
mkdocs gh-deploy --force
```

## Architecture

EDH Matchmaker manages Commander (EDH) Swiss-pairing tournaments with 4-player pods via a PyQt6 GUI.

### Key modules

- **`src/interface.py`** — Abstract interfaces (`IPlayer`, `ITournament`, `IPod`, `IRound`, `IPairingLogic`, `IStandingsExport`, `ITournamentConfiguration`). All major components implement these. `IHashable` provides UUID-based caching via a class-level `CACHE` dict.

- **`src/core.py`** (~3500 lines) — All concrete implementations:
  - `Tournament`: Orchestrates rounds, players, pairing, standings, JSON save/load, and export triggers.
  - `Player`: Tracks results (wins/draws/losses/byes), ratings, pod/opponent history. Seat normalization accounts for global win-rate advantages by seat position (~24.7%, 19.3%, 16.7%, 14.6%).
  - `Pod`: A game group (default 4 players, fallback 3).
  - `Round`: Holds pods; uses `IPairingLogic` to create pairings.
  - `TournamentConfiguration`: Scoring (win=5, bye=4, draw=1 by default), pod sizes, round count, top-cut thresholds, auto-export settings.
  - `StandingsExport` / `PodsExport`: Multi-format output (plain text, CSV, JSON).
  - `Log`: Event logging with severity levels for console and file output.
  - `TournamentAction`: Decorator that auto-persists tournament state to JSON after each action.

- **`src/pairing_logic/examples.py`** — Pairing algorithm implementations:
  - `CommonPairing`: Shared utilities (`evaluate_pod()`, `bye_matching()`, `assign_byes()`).
  - `PairingRandom`: Random pairing.
  - `PairingSnake`: Swiss-style snake seeding (default).

- **`src/discord_engine.py`** — Optional async Discord integration (`DiscordPoster` singleton) with background thread and message queue. Configured via environment variables (`DISCORD_TOKEN`, `GUILD_ID`, `CHANNEL_ID`).

- **`src/misc.py`** — `Json2Obj` (dict-to-object converter), `generate_player_names()` (Faker-based), `timeit` decorator.

- **`run_ui.py`** — PyQt6 GUI entry point. Loads `.ui` files from `ui/`. Key classes: `UILog` (decorator for status updates), `PlayerListItem` (color-coded list items).

### Data flow

```
run_ui.py (PyQt6 GUI)
  └─ Tournament (core.py)
       ├─ Round → Pod → Player
       ├─ IPairingLogic (pairing_logic/examples.py)
       ├─ TournamentConfiguration → StandingsExport / PodsExport
       └─ TournamentAction decorator → JSON persistence (logs/)
```

### Cross-cutting patterns

- **`@TournamentAction.action()`** — wraps mutating Tournament methods; auto-saves state to JSON after each call.
- **`@StandingsExport.auto_export()` / `@PodsExport.auto_export()`** — triggers file/Discord/API export after standings change.
- **`@UILog.with_status()`** — updates the GUI log display after operations.
- **UUID caching** — `IHashable.CACHE` allows `O(1)` object lookup by UUID across all cacheable types.

### Testing

Tests use `unittest`. `tests/blns.txt` (Big List of Naughty Strings) is used for edge-case input testing. Performance benchmarks in `test_performance.py` are filtered out of normal test runs.

### Type checking

Pyright runs in `basic` mode. Several noisy checks are suppressed (`reportUnusedCallResult`, `reportUnknownMemberType`, `reportUnknownVariableType`, `reportMissingTypeStubs`). Use `# pyright: ignore` or `cast()` rather than disabling rules globally.

### Docstrings

Use **Google-style docstrings** (required by MkDocs `mkdocstrings` auto-documentation).

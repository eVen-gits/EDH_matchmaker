# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Domain terms and their traps are in `CONTEXT.md`. Judgment calls that no
tool enforces (auto-discovery, param sidecars, test setup, etc.) are in
`CONTRIBUTING.md`'s "Judgment standards" section.

## Commands

**Run the application:**
```bash
python run_ui.py
python run_ui.py --help  # runtime options
```

**Run tests:**
```bash
PYTHONPATH=. pytest
```
`pytest.ini` skips slow and performance tests by default. Run them with
`pytest -m slow` or `pytest tests/test_performance.py -m performance`.

- **Type checking:** `pyright` (basic mode, config in `pyrightconfig.json`).
- **Dependencies:** `pip install -r requirements.txt`.
- **Docs site:** `mkdocs gh-deploy --force`.

## Architecture

EDH Matchmaker runs Commander (EDH) Swiss-pairing tournaments (4-player pods, 3-player fallback) through a PyQt6 GUI.

### Modules

- **`src/interface.py`** — Abstract interfaces (`IPlayer`, `ITournament`, `IPod`, `IRound`, `IPairingLogic`, `IScoringLogic`, `IStandingsExport`, `ITournamentConfiguration`). `IHashable` provides UUID-based `O(1)` object caching via a class-level `CACHE`.
- **`src/core.py`** — Concrete `Tournament`, `Player`, `Pod`, `Round`, `TournamentConfiguration`, `StandingsExport`/`PodsExport`, `Log`, and the `TournamentAction` decorator (auto-persists state to JSON in `logs/` after each mutating action).
- **`src/logic/<game>/`** — One directory per game/format, each holding `matching.py` (`IPairingLogic` implementations) and/or `scoring.py` (`IScoringLogic` implementations), auto-discovered by filename — no core or GUI change to register a new class or a new game. `src/logic/commander/matching.py` has `PairingRandom` / `PairingSnake` / `PairingDefault` for Swiss rounds and the `PairingTopN` family for top-cut (marked `SELECTABLE = False`); `src/logic/commander/scoring.py` has `ScoringDefault`, `ScoringHareruya`, `ScoringModifiedHareruya`. Swiss pairing logic and params come from `config.pairing_rounds`, else an adaptive default (which respects each logic's `SUPPORTED_POD_SIZES`); scoring logic is selected by `config.scoring_logic`.
- **`src/param_spec.py`** — Loads each algorithm's parameters from a sidecar `<ClassName>.params.yaml` file, next to the module that defines the class (the source of truth for names, defaults, types, ranges, GUI widget hints, and descriptions). The config GUI generates parameter widgets from it.
- **`src/misc.py`** — `Json2Obj`, `generate_player_names()` (Faker-based), `timeit`.
- **`run_ui.py`** — PyQt6 GUI entry point; loads `.ui` files from `ui/`.

Authoritative references (do not copy their values here — they drift):
`docs/tournament-log-spec.md` for the save format and scoring formulas, and the
`src/logic/<game>/*.params.yaml` sidecars for each algorithm's parameters.

### Data flow

```
run_ui.py (PyQt6 GUI)
  └─ Tournament (core.py)
       ├─ Round → Pod → Player
       ├─ IPairingLogic (pairing_logic/)   — how players are paired
       ├─ IScoringLogic (scoring_logic/)   — how points are computed
       └─ TournamentAction → JSON persistence (logs/)
```

### Conventions

- **`@TournamentAction.action()`** wraps mutating `Tournament` methods and auto-saves to JSON.
- **`@StandingsExport.auto_export()` / `@PodsExport.auto_export()`** run exports after standings change.
- **Adding a scoring or pairing algorithm:** add a class (set `IS_COMPLETE = True`) plus a `<ClassName>.params.yaml` sidecar if it has parameters. Auto-discovery and the config GUI pick it up — no core or GUI changes. A pairing algorithm may set `SUPPORTED_POD_SIZES` (a tuple; `None` = any) to limit which tournament pod sizes it is offered for.
- **Tests** use `unittest` (`unittest.TestCase`). A test module that builds a `Tournament` must set `TournamentAction.LOGF = False` at its top, or it writes a log file during the run.
- **Type checking:** pyright `basic` mode. Prefer `# pyright: ignore` or `cast()` over disabling rules globally.
- **Docstrings:** Google style (required by MkDocs `mkdocstrings`).

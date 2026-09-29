# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Params sidecars for the tournament configuration
  (`src/TournamentConfiguration.params.yaml`) and each game's config fields
  (`CommanderConfiguration.params.yaml`, `Mtg1v1Configuration.params.yaml`);
  their defaults are no longer hardcoded in Python. The sidecar format gains
  list/dict types (`item_type`), `default_from`, `choices_from`, and
  `multiselect`/`listedit`/`custom` widgets.
- `python -m src.param_catalog` (or `src.param_catalog.catalog()`) emits every
  parameter spec - tournament, per-game, ruleset, scoring, pairing - as JSON
  for front ends.
- 1v1 Magic tournament support (`Mtg1v1Ruleset`), alongside Commander/EDH.
  A tournament's game is now an explicit choice (`config.ruleset`, default
  `CommanderRuleset`), selected via a ruleset dropdown when creating a
  tournament in the GUI; it is fixed once the tournament has any pods, byes,
  or game losses. 1v1 adds:
  - `Pairing1v1`, a maximum-weight-matching Swiss pairing (via `networkx`,
    a new dependency) for 2-player pods, and `PairingBracket` for
    single-elimination top cut, seeded by final Swiss standings.
  - `Scoring1v1`, Magic Tournament Rules match points (3/1/0, bye as a 2-0
    win).
  - Best-of-N match reporting (`games_to_win`, default 2), with per-game
    winners/draws rather than Commander's single-game-per-pod report, plus
    a GUI score-entry dialog for it.
  - MTR standings tiebreakers - OMW, GW, OGW - shown as extra standings
    columns.
  - Game-specific config (e.g. 1v1's `match_wr_seats`/`match_draw_rate`)
    now lives on a `TournamentConfiguration` subclass per ruleset
    (`Mtg1v1Configuration`), declared via `GAME_FIELDS`.
  See `docs/tournament-log-spec.md` and `CONTEXT.md` for the save format,
  scoring formulas, and terminology (Pod, Game vs. match, Ruleset, Stage).
- Tunable pairing-logic parameters, using the same sidecar mechanism as scoring.
  A pairing algorithm declares its parameters in `<ClassName>.params.yaml`, reads
  them with `self._param(...)`, and the tournament config screen generates the
  widgets from the spec. `PairingDefault` ships two parameters:
  `rematch_penalty_exponent` (default `2`), which controls how hard pairing
  pushes against seating players who already met, and `small_pod_penalty`
  (default `10`), which controls how hard pairing avoids repeatedly seating a
  player in a pod smaller than the preferred size (for example a 3-player pod).
- Per-round pairing settings in the tournament config. Each round is a group box
  with a pairing-logic dropdown and, below it, the parameter widgets for the
  selected logic. Settings are per round, stored in `config.pairing_rounds` as
  one `{"logic", "params"}` object per round. When the adaptive default picks an
  incompatible logic (or a configured one is incompatible), the pairing engine
  falls back to a compatible logic or logs a warning. Old logs load unchanged.
- Pod-size-aware pairing selection. Each pairing algorithm declares the pod
  sizes it supports (`SUPPORTED_POD_SIZES`; `None` means any). A round offers
  only the logics that support the tournament's `config.pod_sizes`. For example,
  a tournament with 2-player pods offers only `PairingRandom`, because
  `PairingDefault` and `PairingSnake` support only `3`, `4`, and `5`. The config
  screen has one tournament-wide pod-size editor, and the per-round pairing
  dropdowns re-filter when the pod sizes change.

### Fixed
- Dropping a player who had a bye or a seat in a pod no longer removes them from
  the tournament; they are marked dropped and stay in history and standings (#27).
- Renaming a player: `Tournament.rename_player` no longer crashes, logs the old name, and refuses empty or duplicate names with a message; the player-list right-click menu has a "Rename player..." entry.
- Documented and locked in with regression tests that `pod_sizes` is an
  ordered preference list: `Tournament.get_pod_sizes()` already preferred
  earlier sizes over later ones and used a bye before backtracking to a
  later, evenly-dividing size, but this wasn't spelled out anywhere. A
  6-player Commander event configured as `[4, 3]` pairs one pod of 4 plus 2
  byes (4 preferred); `[3, 4]` pairs two pods of 3 instead. See `CONTEXT.md`
  ("Pod") and `TournamentConfiguration.params.yaml`'s `pod_sizes`
  description for the rule.
- File > Load tournament no longer crashes the app on a truncated or corrupt
  log file. `TournamentAction.load` now parses the file before updating
  `LOGF`, so a failed load leaves the current tournament's log path
  unchanged, and the GUI shows an error dialog instead of letting the
  exception abort the process (#31). `run_ui.py -o <file>` and the startup
  load of the last log now print a one-line error and exit non-zero on an
  unreadable log, without writing to it.
- File > Load players no longer crashes the app on a blank line in the input
  file; blank lines are skipped instead of being passed to `add_player` as an
  empty name (#26).
- Standings export in `CSV` and `JSON` formats now writes real CSV / JSON
  instead of the plain table or a `ValueError`. `get_standings_str()` defaults
  to the configured `standings_export.format`, so the Export > Standings dialog
  and auto-export honour the chosen format (#25).
- File > New tournament no longer crashes on an invalid config (e.g. no pod
  sizes, or Mtg1v1's required byes left at 0); it now shows the validation
  message and keeps the dialog open, like the edit path already did.
  Switching Game to Mtg1v1 also resets the hidden `max_byes` spin box to a
  valid value, since `BYES_REQUIRED` hides it without resetting it. Emptying
  the pod-size list no longer offers the other game's scoring/pairing
  algorithms (#28).
- Commander's scoring logics (`ScoringDefault`, `ScoringHareruya`,
  `ScoringModifiedHareruya`) left `SUPPORTED_POD_SIZES` unset (`None`, any
  size), so they were also offered - and loadable - for Mtg1v1's 2-player
  pods, where the MTR tiebreakers (OMW) would then read points from the
  wrong game's scoring. They now declare `(3, 4, 5, 6)`, so the config
  GUI's scoring dropdown filters them out for 1v1 the same way it already
  filtered pairing logic (a 6-player Commander pod pairs with
  `PairingRandom`; the default stays `[4, 3]`). Commander tournaments with
  2-player or 7+-player pods are therefore no longer supported: the config dialog refuses to apply a configuration no
  scoring logic supports, instead of saving one with no scoring logic.
- Mtg1v1's top-cut bracket no longer crashes on Create pods when the field
  is smaller than the cut (e.g. a Top 8 cut with only 5 players); missing
  seeds are now treated as already eliminated, giving the top remaining
  seeds a bye into the next round instead of an `IndexError` (#29).

### Changed
- `PairingDefault` now measures a "small" pod against the preferred (first) pod
  size in `config.pod_sizes`, not the largest. For the standard `[4, 3]` this is
  the same as before. It matters only when a larger size such as `5` is added:
  4-player pods are no longer treated as small.

### Fixed
- File > Save As no longer crashes the app; it writes the tournament to the
  chosen file (#22).
- Exporting standings to a bare filename (no directory) no longer crashes;
  the file is written to the working directory. A standings auto-export that
  fails with an OS error is now logged instead of aborting every later
  action. The standings and pods export dialogs now show an error message for
  an unwritable path instead of closing the app (#24).

## [3.1.0] - 2026-08-28

### Added
- Pluggable, spec-driven algorithm parameters. Each scoring/pairing algorithm
  declares its parameters in a sidecar file `<ClassName>.params.yaml` (name,
  default, type, range, GUI widget hint, and description) - the single source of
  truth, loaded at class-definition time so nothing is hard-coded, and inherited
  by subclasses. The tournament config screen generates each parameter's widget
  from that spec (widget kind, range, percentage display, conditional fields),
  so a new algorithm's settings appear with no GUI code. Adds a `PyYAML`
  dependency.
- Per-round pairing-logic selection in the tournament config: one dropdown per
  Swiss round (the list resizes with the round count), each Random, Snake, or
  Default. Stored in `config.pairing_logics`; an empty list keeps the adaptive
  default (round 1 Random, round 2 Snake, later rounds Default). Top-cut pairing
  stays automatic.
- Modified Hareruya scoring logic. This variant removes the round-order
  dependency of Hareruya. It averages the wagering economy over every
  permutation of the Swiss round order, so the record WDD scores the same as
  DDW. It is selectable in the scoring dropdown and reuses the Hareruya wager
  settings. Exact enumeration runs up to 7 rounds. Above that, it averages a
  fixed-seed random sample of orders.
- Standardized project documentation (README, CONTRIBUTING, CODE_OF_CONDUCT, LICENSE).
- Added GPLv3 License.
- `ScoringHareruya`: draw pot points left over after `draw_redistribution_fraction`
  can now be reclaimed instead of discarded, via a new "Reclaim discarded draw
  points" checkbox and pod/tournament split slider
  (`redistribute_discarded_draw_points`, `draw_discard_pod_fraction`). Off by
  default; existing tournament files load unaffected. See
  `docs/tournament-log-spec.md` for the exact formula.

### Changed
- Tournament log format `1.1`: `config.win_points`, `bye_points`,
  `draw_points`, `wager_percent`, `wagering_starting_points`,
  `draw_redistribution_fraction`, `draw_distribution_shape`,
  `redistribute_discarded_draw_points`, and `draw_discard_pod_fraction` moved
  into a new `config.scoring_params` object, nested by whichever scoring
  algorithm reads them, instead of sitting flat alongside universal
  tournament settings regardless of which algorithm is active. `1.1` still
  reads `1.0` files with these fields flat. See `docs/tournament-log-spec.md`
  for the full field list per algorithm.
- Optimized the rating computation on the standings, pairing, and pod-power
  sorts. Each now computes the full-field rating map once and passes it down,
  instead of recomputing it per player and per opponent. Modified Hareruya also
  extracts each pod result once and replays pure integer arithmetic per
  permutation. Wagering scorings are much faster. The results are identical.
- Documented that a pod smaller than the largest pod size uses only its real
  seated wagers under Hareruya and Modified Hareruya. There is no phantom
  player. A smaller pod has a smaller pot and a smaller reward.
- Removed deprecated Discord integration mentions from documentation.

### Fixed
- The `-x/--scoring` command-line flag now works again. It crashed on launch
  (it called a `TournamentConfiguration.scoring()` method that no longer
  exists); it now writes win/draw/bye into `config.scoring_params`.

[Unreleased]: https://github.com/eVen-gits/EDH_matchmaker/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/eVen-gits/EDH_matchmaker/compare/v3.0.0...v3.1.0

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Parameter specifications for scoring and pairing algorithms. Each algorithm
  declares its parameters (name, default, type, range, and a human description)
  in a sidecar file `<ClassName>.params.yaml` next to the class. The file is the
  single source of truth. The code loads it at class-definition time and derives
  the defaults from it, so nothing is hard-coded. A subclass with no sidecar
  inherits its parent's parameters. This lets a new-algorithm author add
  documented parameters in one modular place, and a future GUI render them.
  Adds a `PyYAML` dependency.
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
- Updated README to reflect current project state.
- Removed deprecated Discord integration mentions from documentation.

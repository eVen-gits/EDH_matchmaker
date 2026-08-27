# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modified Hareruya scoring logic. This variant removes the round-order
  dependency of Hareruya. It averages the wagering economy over every
  permutation of the Swiss round order, so the record WDD scores the same as
  DDW. It is selectable in the scoring dropdown and reuses the Hareruya wager
  settings. Exact enumeration runs up to 7 rounds. Above that, it averages a
  fixed-seed random sample of orders.
- Standardized project documentation (README, CONTRIBUTING, CODE_OF_CONDUCT, LICENSE).
- Added GPLv3 License.

### Changed
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

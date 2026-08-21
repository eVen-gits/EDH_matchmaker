# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Standardized project documentation (README, CONTRIBUTING, CODE_OF_CONDUCT, LICENSE).
- Added GPLv3 License.
- `ScoringHareruya`: draw pot points left over after `draw_redistribution_fraction`
  can now be reclaimed instead of discarded, via a new "Reclaim discarded draw
  points" checkbox and pod/tournament split slider
  (`redistribute_discarded_draw_points`, `draw_discard_pod_fraction`). Off by
  default; existing tournament files load unaffected. See
  `docs/tournament-log-spec.md` for the exact formula.

### Changed
- Updated README to reflect current project state.
- Removed deprecated Discord integration mentions from documentation.

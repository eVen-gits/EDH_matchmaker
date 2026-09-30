---
name: add-pairing-scoring-algorithm
description: Use this skill whenever adding a new pairing algorithm (IPairingLogic), a new scoring algorithm (IScoringLogic), or standing up a brand-new src/logic/<game>/ directory in the EDH Matchmaker repo. Trigger on requests like "add a new pairing algorithm", "add a scoring algorithm", "implement a new scoring logic class", "create a top-cut pairing method", or "set up src/logic/<game>/" — even if the user just names the algorithm they want without using this exact terminology.
---

# Adding a pairing or scoring algorithm

This repo discovers pairing, scoring, and ruleset methods from their own
`src/logic/*/<ClassName>.params.yaml` sidecars. There is no registry to edit.
`Tournament.discover_pairing_logic()`, `discover_scoring_logic()`, and
`discover_ruleset()` resolve these names in `matching.py`, `scoring.py`, and
`rules.py`. Only concrete classes with `IS_COMPLETE = True` can register.
Discovery warns about missing sidecars and sidecars without matching classes.
Do not change core or GUI code to register a new method.

**A brand-new game needs a `rules.py` with an `IRuleset`** (`IS_COMPLETE =
True`), not just a `matching.py`/`scoring.py` — the ruleset is what decides
whether a reported match result is valid, who won it, and the playoff plan.
Scoring algorithms own points, standings tiebreakers, and their export
columns. Do not put tiebreakers on the ruleset or `Player`.
See `Scoring1v1.standings_keys` (`src/logic/mtg/scoring.py`) for the pattern.

Use `src/logic/commander/matching.py` and `src/logic/commander/scoring.py`
as the reference implementations throughout — they show the established
shape for this repo.

## Steps

1. **Pick the game directory.** Use the existing one (e.g.
   `src/logic/commander/`) if the algorithm applies to an existing format,
   or create `src/logic/<game>/matching.py` and/or `scoring.py` for a new
   game. Add the class and its own sidecar; discovery is automatic.

2. **Subclass the shared base, not the raw interface.** Shared bases live in
   `src/logic/common/pairing.py` and `src/logic/common/scoring.py`.
   Import from `common`, never from another game's directory.
   `CommonPairing` and `CommonScoring` supply reusable helpers:
   - Fixed points: `FixedPointsScoring`, with game-specific standings and parameters.
   - Pairing: `field_ratings()`, `evaluate_pod()`,
     `bye_matching()`/`assign_byes()`.
   - Scoring: `_swiss_rounds_up_to()`, `rating()`.
   Reimplementing these from scratch is themselves a sign you should be
   subclassing instead.

3. **Set `IS_COMPLETE = True`.** Discovery requires this flag, a concrete
   class, and the class's own sidecar.

4. **Set `SUPPORTED_POD_SIZES` deliberately.** This is a tuple like `(4, 3)`
   or `None`. `None` means "works for any pod size" — only use it if that's
   actually true. The adaptive default pairing logic relies on this value
   to decide fallback behavior, so an inaccurate `None` is a correctness
   bug, not just a lie in a docstring.

5. **Set `SELECTABLE = False` only for top-cut algorithms** (the
   `PairingTopN` family's pattern) — ones chosen automatically by
   tournament stage rather than offered to the user in the config GUI.
   Regular Swiss pairing/scoring algorithms should leave this at its
   default.

6. **Implement the abstract methods.**
   - Pairing: `make_pairings(self, tour_round, players, pods) ->
     set[IPlayer]`. Only top-cut-style pairing logic meaningfully
     implements `advance_topcut(...)` — the base is a no-op, which is
     correct for ordinary Swiss pairing.
   - Scoring: `compute_ratings(self, tour, tour_round) -> Mapping[UUID,
     float]`, and typically `pointrate_denominator(tour_round)`.

7. **Add a `<ClassName>.params.yaml` sidecar in the same commit.** Every
   discoverable method requires its own file, including rulesets, subclasses,
   and top-cut methods with `SELECTABLE = False`.
   For a method without parameters, use `{}`.
   For a parameterized subclass, declare its full parameter contract.
   See the `params-yaml-sidecar` skill for the field contract.

8. **Follow the hot-path/cold-path parameter convention.** `self._param
   (name)` reads a single value with no dict allocation — use it inside
   per-player or per-pod loops. `self.params()` returns a merged dict and
   allocates — reserve it for cold, once-per-call code. Likewise, compute
   shared values like `field_ratings()` once outside a loop rather than
   recomputing them per player; `matching.py`/`scoring.py` do this
   consistently and any new algorithm should match.

9. **Write tests.**
   - Add a case to `tests/test_param_spec.py` named
     `test_<algo>_params_derived_from_sidecar` that asserts
     `Tournament.get_pairing_logic(...)` / `get_scoring_logic
     ("YourClassName").DEFAULT_PARAMS` matches the sidecar's literal
     values.
   - Add a logic-level test that exercises the algorithm through a real
     `Tournament` — `new_round()`, `add_player()`, `create_pairings()` /
     `random_results()` — rather than unit-testing `make_pairings`/
     `compute_ratings` in isolation. Follow the shape of
     `tests/test_pairing.py` or `tests/test_scoring_hareruya.py`.
   - **Any test module that constructs a `Tournament` must set
     `TournamentAction.LOGF = False  # type: ignore` at the top of the
     file, right after imports.** Without it, the test suite writes a real
     file under `logs/` on every run.

10. **Docstrings are Google style** (this repo uses mkdocstrings). If the
    algorithm implements a nonstandard scoring formula, cite
    `docs/tournament-log-spec.md` instead of re-deriving the formula in the
    docstring — that file is the authoritative source and duplicating
    values in two places invites them to drift apart.

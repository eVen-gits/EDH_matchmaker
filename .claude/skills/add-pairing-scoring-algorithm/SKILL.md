---
name: add-pairing-scoring-algorithm
description: Use this skill whenever adding a new pairing algorithm (IPairingLogic), a new scoring algorithm (IScoringLogic), or standing up a brand-new src/logic/<game>/ directory in the EDH Matchmaker repo. Trigger on requests like "add a new pairing algorithm", "add a scoring algorithm", "implement a new scoring logic class", "create a top-cut pairing method", or "set up src/logic/<game>/" — even if the user just names the algorithm they want without using this exact terminology.
---

# Adding a pairing or scoring algorithm

This repo auto-discovers pairing, scoring, and ruleset algorithms by
filename — there is no registry to edit. `Tournament.discover_pairing_logic()`
/ `discover_scoring_logic()` / `discover_ruleset()` (in `src/core.py`) scan
every `src/logic/*/matching.py`, `src/logic/*/scoring.py`, and
`src/logic/*/rules.py`, import them, and register any class with
`IS_COMPLETE = True`. **If your plan touches `src/core.py` or GUI code to
"register" the new algorithm, stop — that's not how this works, and you've
likely misunderstood the task.**

**A brand-new game needs a `rules.py` with an `IRuleset`** (`IS_COMPLETE =
True`), not just a `matching.py`/`scoring.py` — the ruleset is what decides
whether a reported match result is valid, who won it, and the Swiss
standings tiebreaker order beyond raw points. Do not put tiebreakers on a
scoring algorithm or on `Player`; that's ruleset work, even if it feels
like a scoring concern — see `Mtg1v1Ruleset.standings_keys`
(`src/logic/mtg/rules.py`) for the pattern.

Use `src/logic/commander/matching.py` and `src/logic/commander/scoring.py`
as the reference implementations throughout — they show the established
shape for this repo.

## Steps

1. **Pick the game directory.** Use the existing one (e.g.
   `src/logic/commander/`) if the algorithm applies to an existing format,
   or create `src/logic/<game>/matching.py` and/or `scoring.py` for a new
   game. Either way, the file only needs the class inside it — discovery is
   automatic.

2. **Subclass the shared base, not the raw interface.** If the game
   directory already has a `CommonPairing`/`CommonScoring` base class,
   subclass that instead of `IPairingLogic`/`IScoringLogic` directly — it
   carries reusable helpers you almost certainly need:
   - Pairing: `field_ratings()`, `evaluate_pod()`,
     `bye_matching()`/`assign_byes()`.
   - Scoring: `_swiss_rounds_up_to()`, `rating()`.
   Reimplementing these from scratch is themselves a sign you should be
   subclassing instead.

3. **Set `IS_COMPLETE = True`.** Without it, the class is invisible to the
   config GUI even if fully implemented and otherwise correct — an easy
   thing to forget and a silent failure mode.

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

7. **If the algorithm takes parameters, add a `<ClassName>.params.yaml`
   sidecar in the same commit.** This is the single source of truth for
   parameter defaults/ranges/GUI widgets — see the `params-yaml-sidecar`
   skill for the field contract. You can skip adding a new file only when
   subclassing an algorithm whose existing sidecar already covers your
   params unchanged (the loader walks the MRO and inherits it
   automatically).

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

# Context

This file is the project's ubiquitous language: terms that would otherwise
trip up a reader because the code's usage is narrower, stricter, or just
different from the plain-English word. It is vocabulary, not design — for
how the pieces fit together see `CLAUDE.md`'s Architecture section.

Entries without an `Invariant:` line have nothing non-obvious to add, so
they aren't listed — this is not a full data dictionary, only terms with a
genuine trap get an entry.

## Pod

A group of players (2 for 1v1, 3 or 4 for Commander) paired to play a match
together in a given round. A match is one or more games — see the "Game vs.
match" entry below — collapsing to exactly one game by default, which is why
Commander never needed to distinguish the two.

**Lives in:** `src/core.py` (`Pod`), `src/interface.py` (`IPod`)
**Invariant:** Pod size is 4 by default, 3 only as a fallback when player
count doesn't divide evenly; 2 for 1v1 tournaments. A pairing algorithm must
declare which sizes it supports via `SUPPORTED_POD_SIZES` — see that entry
below.

## Game vs. match

A **match** is a round's pod-level outcome — what standings, pairing, and
scoring see (`Pod.result_type`/`Pod.done`). A **game** is one hand played to
a single win/draw/loss. `Pod.games` (a tuple of `IGameResult`) is the
match's whole report, validated and set as a unit by
`Pod.record_result()`/`Tournament.report_match()` — see "Match report"
below. There is no per-game append API: a new report always **replaces**
the pod's previous one, it never appends to it.

Whether a match needs more than one game to decide it is a **ruleset**
question, not a pairing-logic one — see "Ruleset" below.
`Mtg1v1Ruleset.games_to_win` (`src/logic/mtg/Mtg1v1Ruleset.params.yaml`,
default `2`, best-of-3) is a per-round `ruleset_params` override
(`config.pairing_rounds[seq].ruleset_params` /
`config.playoff_rounds[stage].ruleset_params`), read through
`IRuleset._param(tour_round, "games_to_win")`. `CommanderRuleset` has no
such parameter at all: every Commander pod's report is exactly one game,
so `Pod.result_type` is just that one game's outcome.

Do not confuse this with `Round._game_loss` (also spelled "game_loss" in
the JSON format's `rounds[].game_loss`) — that's an unrelated, round-level
*match*-loss penalty (e.g. a no-show) with zero games played, despite the
name.

**Lives in:** `src/core.py` (`Pod.games`, `Pod.record_result`,
`Pod.result_type`), `src/interface.py` (`IGameResult`),
`src/logic/mtg/Mtg1v1Ruleset.params.yaml`

## Match report

The complete, whole-match content of `Pod.games`, set in one call to
`Pod.record_result()` (or the `Tournament.report_match()` /
`report_win()` / `report_draw()` actions that go through it). Reporting a
match again — correcting a mistake, or replacing an in-progress report —
**replaces** the previous report outright; there is no way to append one
game to an existing report. `IRuleset.validate_report()` decides whether a
given report is valid for a pod (seat count, per-game winners, a ruleset's
own rules like `games_to_win`); `IRuleset.match_winners()` derives the
match's winner(s) from an already-valid report.

**Lives in:** `src/core.py` (`Pod.record_result`, `Round.record_result`,
`Tournament.report_match`), `src/interface.py`
(`IRuleset.validate_report`, `IRuleset.match_winners`)

## Ruleset

The plugin kind that owns one game's rules: which match reports are
valid, how a match's winner is derived, Swiss standings tiebreakers
beyond raw points, the default Swiss pairing logic, whether seats get
auto-balanced, and the playoff plan for a given `top_cut`. Selected by
`config.ruleset` (default `"CommanderRuleset"`); the discovery mechanism
is identical to `IPairingLogic`/`IScoringLogic` — see that entry below.

**Lives in:** `src/interface.py` (`IRuleset`), `src/logic/<game>/rules.py`
**Invariant:** A tournament's ruleset is fixed once it has any pods, byes,
or game losses (`Tournament.__validate_config` rejects a config-setter
change past that point) — a result must never be re-read under a
different game's rules mid-tournament.
**Invariant:** Standings tiebreakers belong on the ruleset, never on the
scoring logic or on `Player` — see the `IPairingLogic` / `IScoringLogic`
entry below.

## Stage

`Round.stage` / `config.top_cut`'s non-zero values are the number of
players still in contention when that stage begins (`TOP_4`, `TOP_8`,
…) — not a size specific to any one game. The same stage value produces a
different pod layout per ruleset: Commander's `TOP_4` is one 4-player pod
(`PairingTop4`); MTG's `TOP_4` is two 2-player pods, a semifinal
(`PairingBracket`). Which stage values a ruleset accepts as `top_cut` is
that ruleset's `PLAYOFFS` mapping's keys.

**Lives in:** `src/core.py` (`Round.Stage`, `TournamentConfiguration.TopCut`)

## Round

One full cycle of pairing every active player into pods and recording
results, before standings are recomputed.

**Lives in:** `src/core.py` (`Round`), `src/interface.py` (`IRound`)

## Bye

A player who could not be seated in a pod this round (odd player count) and
receives bye points instead of playing.

**Lives in:** `src/core.py` (`TournamentConfiguration.allow_bye`,
`max_byes`)
**Invariant:** `max_byes` caps how many byes a single player can receive
across the tournament — exceeding it changes pairing behavior, it isn't just
a display limit.

## `IHashable` / `CACHE`

The base class every core domain object (`Player`, `Pod`, `Round`,
`Tournament`, …) inherits from, giving it a UUID (`uid`) and O(1) lookup by
that UUID.

**Lives in:** `src/interface.py` (`IHashable`)
**Invariant:** `CACHE` is a **class-level** dict (`dict[UUID, IHashable]`),
shared across every subclass unless a subclass redeclares it — instantiating
an object registers it in `CACHE` as a side effect of `__init__`, it isn't
opt-in.

## `TournamentAction`

The decorator (`@TournamentAction.action()`) that wraps every mutating
`Tournament` method to auto-persist the tournament's state to JSON after the
method runs.

**Lives in:** `src/core.py` (`TournamentAction`)
**Invariant:** `TournamentAction.LOGF` is a *class attribute*, not
per-instance — it's global mutable state for "where to write the log."
Tests that build a `Tournament` must set `TournamentAction.LOGF = False` at
module top, or a real log file gets written under `logs/` during the test
run (see `DEFAULT_LOGF = "logs/default.json"`).

## `IPairingLogic` / `IScoringLogic` / `IRuleset`

The three extension points for tournament behavior: how players get
grouped into pods (`IPairingLogic`), how results become points/standings-
rating (`IScoringLogic`), and how one game's rules decide what a valid
result is, who won, and the Swiss/playoff structure (`IRuleset` — see the
"Ruleset" entry above). All three are auto-discovered — for each game
directory under `src/logic/`, a `matching.py`/`scoring.py`/`rules.py`
module is picked up without touching core or GUI code.

**Trap:** standings tiebreakers go on the ruleset, never on the scoring
logic or on `Player` — `Mtg1v1Ruleset.standings_keys` owns the MTR's
OMW/GW/OGW chain, not `Scoring1v1`, so switching a tournament's scoring
logic never silently changes its tiebreaker order.

**Lives in:** `src/interface.py`, `src/logic/<game>/matching.py`,
`src/logic/<game>/scoring.py`, `src/logic/<game>/rules.py`
**Invariant:** A new implementation is *not* offered to the config GUI
unless `IS_COMPLETE = True` is set — an in-progress algorithm left at the
default `IS_COMPLETE = False` stays invisible rather than half-working.
**Invariant:** One game's `matching.py`/`scoring.py`/`rules.py` may
subclass another game's class directly (`mtg.scoring.Scoring1v1`
subclasses `commander.scoring.ScoringDefault`, for example) to reuse a
formula that isn't actually game-specific. Doing this safely requires
importing the *module*, not the class
(`from ..commander import scoring as _commander_scoring`, then
`class Scoring1v1(_commander_scoring.ScoringDefault)`) — importing the
class by name also binds it in the subclassing module's namespace, and
since the discovery scan (`Tournament._discover_logic`) keys registered
classes by `obj.__name__` (not the import alias), an already-`IS_COMPLETE`
parent class showing up in two modules' `__dict__` collides and silently
drops that whole module from discovery (caught, logged as a warning, easy
to miss).

## `SUPPORTED_POD_SIZES`

A tuple on a pairing-logic or scoring-logic class limiting which pod sizes
(3, 4, 5, …) it will be offered for; `None` means "any size." Both
`IPairingLogic` and `IScoringLogic` carry it independently — a scoring
formula tied to 1v1 (e.g. `Scoring1v1`) and a pairing algorithm tied to 1v1
(e.g. `Pairing1v1`) each declare their own tuple, so neither leaks into a
tournament of the wrong pod size on its own.

**Lives in:** `src/interface.py` (`IPairingLogic.SUPPORTED_POD_SIZES`,
`IScoringLogic.SUPPORTED_POD_SIZES`, `supports_pod_sizes()` on each)
**Invariant:** Must be set deliberately for any algorithm that can't handle
every pod size — the adaptive default pairing logic checks this to decide
what it's allowed to fall back to, and the config GUI checks both
`Tournament.selectable_pairing_logics()`/`selectable_scoring_logics()` to
filter their dropdowns, so leaving it `None` on an algorithm that actually
can't handle 3-player pods is a silent correctness bug, not just a GUI
omission.

## `SELECTABLE`

Whether a pairing-logic class is offered as a user choice in the config GUI.

**Lives in:** `src/interface.py` (`IPairingLogic.SELECTABLE`)
**Invariant:** Top-cut pairing logic (`PairingTopN` family) sets
`SELECTABLE = False` — it's chosen automatically by tournament stage, not by
the user, even though `IS_COMPLETE = True`. Don't add a top-cut algorithm to
a user-facing picker; that's the tournament stage's job.

## `<ClassName>.params.yaml` sidecar

The YAML file next to a scoring/pairing/ruleset class that is the source
of truth for that algorithm's (or ruleset's) parameter names, defaults,
types, ranges, and GUI widget hints — a ruleset's sidecar works exactly
like a pairing or scoring one, e.g. `Mtg1v1Ruleset.params.yaml`'s
`games_to_win`.

**Lives in:** `src/param_spec.py`, `src/*_logic/<ClassName>.params.yaml`
**Invariant:** A hardcoded default inside the Python class is *not*
authoritative once a sidecar exists — the sidecar is what the config GUI
reads to build widgets, so a param added only in code without updating the
sidecar won't be configurable and may silently diverge from what's shown.

_Avoid:_ don't restate sidecar values in `CLAUDE.md` or other docs — they
drift; point at the sidecar file instead.

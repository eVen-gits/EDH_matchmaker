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
a single win/draw/loss, recorded in `Pod._games`; a match is decided once
one player's game-win tally reaches `Round.games_to_win`.

`games_to_win` is a parameter of whichever pairing logic built that round
(`src/logic/commander/CommonPairing.params.yaml`, inherited by every
pairing class with no sidecar of its own; `PairingDefault.params.yaml`,
inherited by `mtg.Pairing1v1` since it subclasses commander's
`PairingDefault` directly — see the `IPairingLogic` / `IScoringLogic`
entry below) — not a bespoke tournament-config field. It follows the
existing per-round pairing-param override mechanism
(`config.pairing_rounds[seq].params`), same as any other pairing
parameter. It defaults to `1` everywhere, so Commander's
one-game-per-pod convention is unchanged unless a round's config
explicitly overrides it (a 1v1 tournament typically sets it to `2`,
best-of-3, on the rounds that need it).

Do not confuse this with `Round._game_loss` (also spelled "game_loss" in
the JSON format's `rounds[].game_loss`) — that's an unrelated, round-level
*match*-loss penalty (e.g. a no-show) with zero games played, despite the
name.

**Lives in:** `src/core.py` (`Pod._games`, `Pod.game_wins`,
`Round.games_to_win`, `Round._game_loss`),
`src/logic/commander/CommonPairing.params.yaml`,
`src/logic/commander/PairingDefault.params.yaml`

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

## `IPairingLogic` / `IScoringLogic`

The two extension points for tournament behavior: how players get grouped
into pods (`IPairingLogic`) and how results become points/standings
(`IScoringLogic`). Both are auto-discovered — for each game directory under
`src/logic/`, a `matching.py` and/or `scoring.py` module is picked up
without touching core or GUI code.

**Lives in:** `src/interface.py`, `src/logic/<game>/matching.py`,
`src/logic/<game>/scoring.py`
**Invariant:** A new implementation is *not* offered to the config GUI
unless `IS_COMPLETE = True` is set — an in-progress algorithm left at the
default `IS_COMPLETE = False` stays invisible rather than half-working.
**Invariant:** One game's `matching.py`/`scoring.py` may subclass another
game's class directly (`mtg.Pairing1v1` subclasses
`commander.matching.PairingDefault`, for example) to reuse a Swiss-pairing
or scoring formula that isn't actually game-specific. Doing this safely
requires importing the *module*, not the class
(`from ..commander import matching as _commander_matching`, then
`class Pairing1v1(_commander_matching.PairingDefault)`) — importing the
class by name also binds it in the subclassing module's namespace, and
since the discovery scan (`Tournament._discover_logic`) keys registered
classes by `obj.__name__` (not the import alias), an already-`IS_COMPLETE`
parent class showing up in two modules' `__dict__` collides and silently
drops that whole module from discovery (caught, logged as a warning, easy
to miss).

## `SUPPORTED_POD_SIZES`

A tuple on a pairing-logic class limiting which pod sizes (3, 4, 5, …) it
will be offered for; `None` means "any size."

**Lives in:** `src/interface.py` (`IPairingLogic.SUPPORTED_POD_SIZES`,
`supports_pod_sizes()`)
**Invariant:** Must be set deliberately for any algorithm that can't handle
every pod size — the adaptive default pairing logic checks this to decide
what it's allowed to fall back to, so leaving it `None` on an algorithm that
actually can't handle 3-player pods is a silent correctness bug, not just a
GUI omission.

## `SELECTABLE`

Whether a pairing-logic class is offered as a user choice in the config GUI.

**Lives in:** `src/interface.py` (`IPairingLogic.SELECTABLE`)
**Invariant:** Top-cut pairing logic (`PairingTopN` family) sets
`SELECTABLE = False` — it's chosen automatically by tournament stage, not by
the user, even though `IS_COMPLETE = True`. Don't add a top-cut algorithm to
a user-facing picker; that's the tournament stage's job.

## `TournamentConfiguration.game`

Which ruleset a tournament follows (`"commander"` or `"mtg"`) - selects the
`src/logic/<game>/` module whose `CUT_STAGES` table and `IScoringLogic.
ranking()` apply.

**Lives in:** `src/core.py` (`TournamentConfiguration.game`,
`Tournament.cut_stages`)
**Invariant:** Defaults to `"commander"` for backward file-compatibility -
every tournament log written before this field existed is implicitly
Commander. Adding a third game means adding its own `CUT_STAGES` table and
`ranking()` override in a new `src/logic/<game>/` module, not another
branch anywhere in `core.py`.

## `Round.Stage` / `stage_value` / `CUT_STAGES`

`Round.Stage` has exactly one member, `SWISS = 0` - every other stage value
is a plain `int`, not an enum member. `Round.stage` is `Round.Stage.SWISS`
for a Swiss round, or one of those plain ints for a playoff round.
`Round.stage_value` normalizes either into the raw int, so comparisons
should use `round.stage_value == Round.Stage.SWISS.value` (or `!=`), never
compare `round.stage` to a non-SWISS `Round.Stage` member - there isn't one.

Each game's `src/logic/<game>/matching.py` defines a module-level
`CUT_STAGES: dict[int, list[tuple[int, int, str]]]`, keyed by
`config.top_cut`, each entry a `(stage_value, n_players,
pairing_logic_name)` triple in play order (see
`src/logic/commander/matching.py` and `src/logic/mtg/matching.py` for the
two shipped tables). `Tournament.__compute_stage_and_logic` walks this
table to pick the next round's stage/logic; `Round.disable_topcut` walks it
to find the current round's `n_players`.

**Lives in:** `src/core.py` (`Round.Stage`, `Round.stage_value`,
`Round._cut_stage_entry`, `Tournament.cut_stages`,
`Tournament.__compute_stage_and_logic`), `src/logic/<game>/matching.py`
(`CUT_STAGES`)
**Invariant:** Stage values are only unique *within* one game's table, not
globally - Commander's `4` (a single 4-player pod) and `mtg`'s `104` (a
4-player bracket semifinal) are unrelated numbers that happen to both
exist. A reader (in code or on the wire, see
`docs/tournament-log-spec.md`'s `top_cut`/`stage` value tables) must know
`config.game` before a raw stage int means anything. `mtg`'s cut stages are
deliberately offset by `100` from Commander's `{0, 4, 7, 10, 13, 16, 40}` so
the two tables never collide by raw value even though a reader ignoring
`game` could otherwise misread one game's stage as the other's.

## 1v1 bracket seeding (`PairingBracketN`)

1v1's top cut is a standard power-of-2 single-elimination bracket (MTR
10.4), not Commander's cascading multiplayer-pod funnel. Seeding is fixed
once, from `tour.get_standings(final_swiss_round)`, when the cut starts;
results inside the cut never reseed it. Each round re-derives that round's
matchups by filtering the *same* fixed seed-order list (from
`_bracket_seed_order()`) down to survivors (whoever `Round.disable_topcut`
left in `players`/`active_players`), preserving the fixed list's relative
order, then pairing adjacent entries - not by re-sorting current standings
and pairing adjacent ranks, which would let seed 1 and 2 meet before the
final.

**Lives in:** `src/logic/mtg/matching.py` (`PairingBracketCommon`,
`_bracket_seed_order`)
**Invariant:** A pod's drawn result inside a bracket round has no defined
"who advances" answer (MTR disallows draws in single elimination) -
`Round.advancing_players` (`src/core.py`) raises for a drawn pod with
exactly 2 players rather than picking one, before any `PairingBracketN`
class runs. Don't add draw-tolerance to the bracket pairing classes; the
correct behavior is to raise, and it already happens one layer up.

## `<ClassName>.params.yaml` sidecar

The YAML file next to a scoring/pairing class that is the source of truth
for that algorithm's parameter names, defaults, types, ranges, and GUI
widget hints.

**Lives in:** `src/param_spec.py`, `src/*_logic/<ClassName>.params.yaml`
**Invariant:** A hardcoded default inside the Python class is *not*
authoritative once a sidecar exists — the sidecar is what the config GUI
reads to build widgets, so a param added only in code without updating the
sidecar won't be configurable and may silently diverge from what's shown.

_Avoid:_ don't restate sidecar values in `CLAUDE.md` or other docs — they
drift; point at the sidecar file instead.

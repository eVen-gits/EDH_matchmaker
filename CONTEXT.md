# Context

This file is the project's ubiquitous language: terms that would otherwise
trip up a reader because the code's usage is narrower, stricter, or just
different from the plain-English word. It is vocabulary, not design — for
how the pieces fit together see `CLAUDE.md`'s Architecture section.

Entries without an `Invariant:` line have nothing non-obvious to add, so
they aren't listed — this is not a full data dictionary, only terms with a
genuine trap get an entry.

## Pod

A group of players (3 or 4) paired to play one game together in a given
round.

**Lives in:** `src/core.py` (`Pod`), `src/interface.py` (`IPod`)
**Invariant:** Pod size is 4 by default, 3 only as a fallback when player
count doesn't divide evenly. A pairing algorithm must declare which sizes it
supports via `SUPPORTED_POD_SIZES` — see that entry below.

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

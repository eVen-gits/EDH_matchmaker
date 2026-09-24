# Tournament Log Format Specification

This page specifies the JSON file format that EDH_matchmaker uses to save
and load tournament state. Any program can use this page to read or write
compatible files, without reading the EDH_matchmaker source code.

## Status of this document

This page describes format version `1.2`, the format that EDH_matchmaker
writes today. [Known gaps](#known-gaps-and-open-decisions) lists the parts
of the format that a stricter cross-software standard must still decide.

### Requirement words

This page uses two words for requirements:

- **must** — a conforming file has to include the field or follow the
  rule.
- **optional** — a field can be absent. Each optional field states its
  default when absent.

## File semantics

A tournament log file is a full snapshot of one tournament. It is not an
event log or a change history.

Each save overwrites the complete file with the current state. The file
holds no record of earlier states.

### Write procedure

A writer must not write the target path directly. Direct writes leave a
truncated, unreadable file if the process stops mid-write.

A writer must use this procedure instead:

1. Write the full JSON document to a temporary file in the same
   directory as the target path.
2. Rename the temporary file onto the target path with an atomic
   rename operation (for example, POSIX `rename()`, or Python's
   `os.replace()`).

This format defines no file-locking or multi-writer protocol. If two
processes write the same path at the same time, the last rename wins and
the other writer's update is lost silently. Treat each file path as
having a single writer.

## Top-level object

| Field | Type | Required | Description |
|---|---|---|---|
| `format_version` | string | optional (default `"1.0"`) | The format version this file follows. See [Format versioning](#format-versioning). |
| `generator` | object | optional | Identifies the software that wrote the file. See below. |
| `created_at` | string (ISO 8601) | optional | When the tournament was first created. Absent in files written before format `1.0`. |
| `updated_at` | string (ISO 8601) | optional | When this file was last written. |
| `uid` | string (UUID) | must | The tournament's unique ID. |
| `config` | object | must | Tournament rules and settings. See [The config object](#the-config-object). |
| `players` | array of object | must | Every player in the tournament. See [Player objects](#player-objects). |
| `rounds` | array of object | must | Every round played, in order. See [Round objects](#round-objects). |

`generator`, when present, is an object with one field:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | must (within `generator`) | Name of the software that wrote the file, for example `"EDH_matchmaker"`. |

A future writer can add a `version` field to `generator`. This spec does
not require one, since EDH_matchmaker itself has no version-numbering
scheme yet.

### Format versioning

A reader must accept a file with no `format_version` field as `"1.0"`.

A reader must not reject a file whose `format_version` is a known older
version (`"1.0"`, `"1.1"`) - attempt to read it without warning, since
format changes are expected to stay additive (new optional fields) for
the foreseeable future. Only a `format_version` this reader does not
recognize at all warrants a warning. A breaking change bumps
`format_version` and documents the break on this page.

#### `1.0` → `1.1`

`config.win_points`, `bye_points`, `draw_points`, `wager_percent`,
`wagering_starting_points`, `draw_redistribution_fraction`,
`draw_distribution_shape`, `redistribute_discarded_draw_points`, and
`draw_discard_pod_fraction` moved from flat `config` fields into
`config.scoring_params`, nested by whichever scoring algorithm reads them.
See [The `config` object](#the-config-object) and [Scoring
logic](#scoring-logic). A `1.1` reader must also accept a `1.0` file with
these nine fields still flat on `config` (see [Scoring
logic](#scoring-logic) for exactly which fields belong to which
algorithm) and treat them as the equivalent `scoring_params` entries. A
`1.1` writer always writes the nested form.

#### `1.1` → `1.2`

Purely additive - a `1.1` file is already a valid `1.2` file:

- `config.ruleset`, `config.playoff_rounds`, and
  `pairing_rounds[].ruleset_params` (see [The `config`
  object](#the-config-object) and [Rulesets](#rulesets)).
- `pods[].games` entries changed shape, from a bare UID array to
  `{"winners": [...]}` (see [Pod objects](#pod-objects)) - never released
  before this version, so no compatibility burden.
- New `top_cut` / `stage` values `2` (`TOP_2`) and `8` (`TOP_8`) - see
  [`top_cut` and `stage` values](#top_cut-and-stage-values). A `1.1`
  reader would reject these two values; a `1.2` reader must accept them.

### Minimal example

```json
{
  "format_version": "1.1",
  "generator": {"name": "EDH_matchmaker"},
  "created_at": "2026-08-21T10:00:00+00:00",
  "updated_at": "2026-08-21T10:05:00+00:00",
  "uid": "11111111-1111-1111-1111-111111111111",
  "config": {
    "pod_sizes": [2],
    "allow_bye": true,
    "snake_pods": true,
    "n_rounds": 1,
    "max_byes": 1,
    "auto_export": false,
    "standings_export": {
      "fields": [0, 2, 4],
      "format": 0,
      "dir": "./logs/standings.txt"
    },
    "global_wr_seats": [0.2470, 0.1928, 0.1672, 0.1458],
    "top_cut": 0,
    "scoring_logic": "ScoringDefault",
    "scoring_params": {
      "win_points": 3,
      "bye_points": 3,
      "draw_points": 1
    }
  },
  "players": [
    {"uid": "aaaaaaaa-0000-0000-0000-000000000001", "name": "Alice", "decklist": null},
    {"uid": "aaaaaaaa-0000-0000-0000-000000000002", "name": "Bob", "decklist": null}
  ],
  "rounds": [
    {
      "tour": "11111111-1111-1111-1111-111111111111",
      "seq": 0,
      "stage": 0,
      "logic": "PairingRandom",
      "uid": "bbbbbbbb-0000-0000-0000-000000000001",
      "dropped": [],
      "disabled": [],
      "byes": [],
      "game_loss": [],
      "pods": [
        {
          "uid": "cccccccc-0000-0000-0000-000000000001",
          "tour_round": "bbbbbbbb-0000-0000-0000-000000000001",
          "table": 1,
          "cap": 2,
          "result": ["aaaaaaaa-0000-0000-0000-000000000001"],
          "players": [
            "aaaaaaaa-0000-0000-0000-000000000001",
            "aaaaaaaa-0000-0000-0000-000000000002"
          ]
        }
      ]
    }
  ]
}
```

This example holds two players, one round, and one pod. Alice won the
pod (see [Pod objects](#pod-objects) for how `result` encodes a win, a
draw, or a pending game).

## The `config` object

| Field | Type | Description |
|---|---|---|
| `pod_sizes` | array of int | The tournament's pod sizes, ordered (first = most preferred), for example `[4, 3]`. The pairing logic fills pods at the first size before falling back to the next. These sizes also decide which pairing logics a round may use - see [Pairing logic](#pairing-logic). |
| `allow_bye` | bool | If `true`, a leftover player can receive a bye instead of a pod. |
| `snake_pods` | bool | If `true`, round 2 seeding uses a Swiss snake order. |
| `n_rounds` | int | Number of Swiss rounds. |
| `max_byes` | int | Maximum number of byes one player can receive across the tournament. |
| `auto_export` | bool | If `true`, the writer also produces the plain-text exports described in [Adjacent outputs](#adjacent-outputs-not-part-of-this-format). Does not affect this JSON format. |
| `standings_export` | object | See below. |
| `global_wr_seats` | array of float | Seat-position win-rate adjustment, one value per seat, most-advantaged seat first. |
| `top_cut` | int | The playoff cut size. See [`top_cut` and `stage` values](#top_cut-and-stage-values). `0` means no playoff cut (Swiss only). |
| `scoring_logic` | string | optional, default `"ScoringDefault"`. Which formula computes player points. See [Scoring logic](#scoring-logic). |
| `scoring_params` | object | optional, default `{}`. Parameters for whichever algorithm `scoring_logic` names - field names, types, and defaults are owned by that algorithm, not by this format. See [Scoring logic](#scoring-logic) for the fields each shipped algorithm reads. |
| `pairing_rounds` | array of object | optional, default `[]`. The pairing configuration per Swiss round, one object per round: `{"logic": <name or null>, "params": {<field>: value}, "ruleset_params": {<field>: value}}`. `logic` is the pairing-logic name (or `null` / a missing/short list to use the ruleset's adaptive default - see [Rulesets](#rulesets)). `params` holds that pairing logic's overrides; a missing field uses the logic's default. `ruleset_params` holds the tournament's ruleset's overrides for this round (for example a per-round match format); a missing field uses the ruleset's default. A configured `logic` should support the tournament's `pod_sizes` - see [Pairing logic](#pairing-logic). Top-cut rounds ignore `logic`/`params`, but not `ruleset_params` - see `playoff_rounds`. |
| `ruleset` | string | optional, default `"CommanderRuleset"`. Which class owns this tournament's game rules: match-report validation, standings tiebreakers, and the playoff plan. See [Rulesets](#rulesets). |
| `playoff_rounds` | object | optional, default `{}`. Ruleset param overrides per playoff stage, keyed by stage value as a string (see [`top_cut` and `stage` values](#top_cut-and-stage-values)): `{"<stage>": {"ruleset_params": {<field>: value}}}`. A stage not in the ruleset's current playoff plan is ignored. |

`standings_export` fields:

| Field | Type | Description |
|---|---|---|
| `fields` | array of int | Which standings columns to include. See [`StandingsExport.Field` values](#standingsexportfield-values). |
| `format` | int | Output format for the plain-text export. See [`DataExport.Format` values](#dataexportformat-values). |
| `dir` | string | File path for the plain-text standings export. Not part of this JSON format — see [Adjacent outputs](#adjacent-outputs-not-part-of-this-format). |

These values are this implementation's defaults, given for reference.
A conforming file must set every `config` field explicitly, except
`scoring_params` keys - each is optional and defaults per-algorithm
(see [Scoring logic](#scoring-logic)); a reader must not assume a
default for a missing `config` field otherwise.

## Player objects

Each entry in the top-level `players` array:

| Field | Type | Description |
|---|---|---|
| `uid` | string (UUID) | The player's unique ID. Unique within this file. |
| `name` | string | Display name. |
| `decklist` | string or `null` | Free text, commonly a URL to the player's decklist. This format places no constraint on its content. See [Known gaps](#known-gaps-and-open-decisions). |

A player object holds no per-round state. Seating, results, byes, drops,
and game losses all live on the round and pod objects instead. To find a
player's status in a given round, search that round's `dropped`,
`disabled`, `byes`, and `game_loss` arrays, and that round's `pods`
arrays, for the player's `uid`.

## Round objects

Each entry in the top-level `rounds` array:

| Field | Type | Description |
|---|---|---|
| `tour` | string (UUID) | The parent tournament's `uid`. |
| `seq` | int | Round number, starting at `0`. |
| `stage` | int | Which stage of the tournament this round belongs to. See [`top_cut` and `stage` values](#top_cut-and-stage-values). |
| `logic` | string | Name of the pairing algorithm that built this round's pods. See [`logic` values](#logic-values). |
| `uid` | string (UUID) | The round's unique ID. Unique within this file. |
| `dropped` | array of string (UUID) | Players who left the tournament as of this round. |
| `disabled` | array of string (UUID) | Players still in the tournament but not eligible to play this round, for example players eliminated from a playoff cut. |
| `byes` | array of string (UUID) | Players who received a bye this round instead of a pod. |
| `game_loss` | array of string (UUID) | Players who received a game loss this round. |
| `pods` | array of object | The pods played this round. See [Pod objects](#pod-objects). |

### Pod objects

Each entry in a round's `pods` array:

| Field | Type | Description |
|---|---|---|
| `uid` | string (UUID) | The pod's unique ID. Unique within this file. |
| `tour_round` | string (UUID) | The parent round's `uid`. |
| `table` | int | Table number, for display and seating. |
| `cap` | int | Maximum number of players this pod holds. |
| `result` | array of string (UUID) | The pod's *match* outcome, derived from `games`. See below. |
| `games` | array of object | optional, default `[]`. The pod's whole match report: one object per completed game, in play order, `{"winners": [<uid>, ...]}`. A new report always **replaces** the previous one - see [Match report](#match-report) below. |
| `players` | array of string (UUID) | Players seated at this pod, in seat order. |

Each game object's `winners` array encodes that game's outcome by how many
player UIDs it holds (`minItems: 1` - an empty array is not a valid game;
"no game played yet" is simply an empty `games` array):

- **One UID** — that player won the game.
- **Two or more UIDs** — a draw among those players.

`result` is the pod's *match* outcome — what standings, pairing, and
scoring read — derived from `games` by the tournament's ruleset (see
[Rulesets](#rulesets)): under `CommanderRuleset`, `games` always holds
exactly one game and `result` is simply that game's `winners`. An empty
`games` array means the match is pending, and `result` is empty. A reader
that only needs the final outcome can read `result` directly and ignore
`games` entirely, since `result` is always kept consistent with it.

#### Match report

A **match report** is the complete, whole-match content of `games` set in
one write. Reporting a match again (correcting a mistake, or replacing an
in-progress report) **replaces** `games` outright - this format does not
support appending one game to an existing report. Game order within one
report carries no meaning beyond what the ruleset assigns it.

A player UID listed in the round's `byes` array does not need to appear
in any pod that round.

## Rulesets

`config.ruleset` names the class (a `src/logic/<game>/rules.py` `IRuleset`
implementation) that owns this tournament's game-specific rules: which
match reports are valid for a pod (`games`, see [Pod objects](#pod-objects)
above), how a match's winners are derived from a report, the Swiss
tiebreaker order beyond raw points, and the playoff plan for a given
`top_cut`. A reader that only needs to display `pods[].result` and
`rounds[].byes`/`game_loss` does not need to know the ruleset; recomputing
Swiss standings order (beyond primary points) or replaying a playoff plan
does depend on it.

### `CommanderRuleset`

The default (`config.ruleset` absent or `"CommanderRuleset"`). Every pod's
`games` holds exactly one game; that game's `winners` is the match `result`
directly. Standings order beyond points follows this implementation's
existing tiebreaker chain (opponent point rate, games played, opponents
beaten, average seat, then UID) - see `src/logic/commander/rules.py` for
the exact tuple; this page does not restate it since it would only drift
(see [Where parameter definitions live](#where-parameter-definitions-live-implementation-note)
for the same rationale applied to scoring parameters).

### `Mtg1v1Ruleset`

Rules for 1v1 Magic tournaments (`config.ruleset: "Mtg1v1Ruleset"`), pod
size `2` only. `games_to_win` (default `2`, best of three) is a
`ruleset_params` field for this ruleset - see the `config` table and
`src/logic/mtg/Mtg1v1Ruleset.params.yaml` for its default, range, and
description; it can be overridden per Swiss round
(`pairing_rounds[i].ruleset_params`) or per playoff stage
(`playoff_rounds[stage].ruleset_params`).

A pod's `games` is that match's whole report: each game's `winners` holds
one UID (that player won the game) or both seated UIDs (a draw). A valid
report has at least one game, no player's single-game win count exceeding
`games_to_win`, and not both players reaching it - a report may end below
`games_to_win` when a round's time limit is called (a `1-0` at the time
limit is valid; a `2-2` tied outcome at `games_to_win: 2` is not, and
neither is an over-cap `3-0`). The match `result` is derived by tallying
single-winner games: the sole leader is the match winner; a tied leader
count (including an all-drawn report) is a match draw.

`src/logic/mtg/mtr-1v1-spec.md` has the full Magic Tournament Rules
citations this implements; this page states only what changes the wire
format's validity, not the rules text itself.

## Enum reference

The fields below carry integer values over the wire. A reader must
treat each value as fixed: renumbering any of them is a breaking change
that requires a new `format_version`.

### `top_cut` and `stage` values

`config.top_cut` and `rounds[].stage` share one value set. `top_cut`
names its zero value `NONE`; `stage` names the same value `SWISS`. A
non-zero value is the number of players still in contention when that
playoff stage begins - not a Commander-specific size. Each ruleset
accepts only some of these values as `top_cut` (its `PLAYOFFS` plan's
keys - see [Rulesets](#rulesets)); `rounds[].stage` can hold any of them,
whichever the tournament's ruleset produced.

| Value | `top_cut` name | `stage` name | Meaning | Accepted by |
|---|---|---|---|---|
| 0 | `NONE` | `SWISS` | No playoff cut / a Swiss round. | every ruleset |
| 2 | `TOP_2` | `TOP_2` | Top-2 (final) playoff. | `Mtg1v1Ruleset` |
| 4 | `TOP_4` | `TOP_4` | Top-4 playoff. | `CommanderRuleset`, `Mtg1v1Ruleset` |
| 7 | `TOP_7` | `TOP_7` | Top-7 playoff. | `CommanderRuleset` |
| 8 | `TOP_8` | `TOP_8` | Top-8 playoff. | `Mtg1v1Ruleset` |
| 10 | `TOP_10` | `TOP_10` | Top-10 playoff. | `CommanderRuleset` |
| 13 | `TOP_13` | `TOP_13` | Top-13 playoff. | `CommanderRuleset` |
| 16 | `TOP_16` | `TOP_16` | Top-16 playoff. | `CommanderRuleset`, `Mtg1v1Ruleset` |
| 40 | `TOP_40` | `TOP_40` | Top-40 playoff. | `CommanderRuleset` |

### `StandingsExport.Field` values

Column identifiers used in `config.standings_export.fields`:

| Value | Name |
|---|---|
| 0 | `STANDING` |
| 1 | `ID` |
| 2 | `NAME` |
| 3 | `RECORD` |
| 4 | `RATING` |
| 5 | `WINS` |
| 6 | `OPP_BEATEN` |
| 7 | `OPP_POINTRATE` |
| 8 | `UNIQUE` |
| 9 | `POINTRATE` |
| 10 | `GAMES` |
| 11 | `SEAT_HISTORY` |
| 12 | `AVG_SEAT` |

These values only affect the plain-text standings export (see
[Adjacent outputs](#adjacent-outputs-not-part-of-this-format)). They do
not change how this JSON format itself is read.

### `DataExport.Format` values

Used in `config.standings_export.format`:

| Value | Name |
|---|---|
| 0 | `PLAIN` |
| 2 | `CSV` |
| 3 | `JSON` |

Value `1` is not assigned. Reserve it; do not assign it a new meaning.

### Derived result values

These two value sets are not stored anywhere in the file as integers.
They describe the outcomes a reader computes from `pods[].result` and
the round's `byes`/`game_loss` arrays, given here so implementations
agree on names when they report results back to a user.

Per-pod result, derived from `pods[].result` as described in
[Pod objects](#pod-objects):

| Name | Meaning |
|---|---|
| `PENDING` | `result` is empty. |
| `WIN` | `result` holds exactly one UID. |
| `DRAW` | `result` holds two or more UIDs. |

Per-player result for one round, derived by checking, in order, whether
the player's UID appears in that round's `game_loss` array, any pod's
`result` array, the round's `byes` array, or none of those:

| Name | Meaning |
|---|---|
| `LOSS` | Player UID is in the round's `game_loss` array, or is seated in a pod but not in that pod's `result` array once the pod has a result. |
| `DRAW` | Player UID is in a pod's `result` array alongside one or more other UIDs. |
| `WIN` | Player UID is the sole entry in a pod's `result` array. |
| `BYE` | Player UID is in the round's `byes` array. |
| `PENDING` | The player's pod has an empty `result` array. |

### `logic` values

`rounds[].logic` names the pairing algorithm that built the round, for
example `"PairingRandom"` or `"PairingSnake"`. This format treats it as
a vendor-extension string: a reader can display or log it, but must not
depend on values beyond the two shown here matching any specific
behavior. See [Known gaps](#known-gaps-and-open-decisions).

## Identifiers

Every `uid` field is a UUID, given as a string in canonical
8-4-4-4-12 hyphenated form (for example
`"aaaaaaaa-0000-0000-0000-000000000001"`).

A conforming file must give every player, pod, and round a `uid` that
is unique within that file. The tournament's own top-level `uid` must
be unique across every file a reader expects to handle at once, since
files are commonly compared or merged by tournament ID.

This format defines no required relationship between a `uid` and any
previously seen file, database row, or in-memory object. One existing
implementation (EDH_matchmaker) additionally keeps a process-wide cache
that reuses an existing in-memory tournament object when it encounters
a `uid` it has already loaded, updating that object in place instead of
building a new one. That behavior is an implementation detail of one
codebase, not a requirement of this format. A conforming reader can
build a fresh object graph on every load.

## Scoring logic

A player's point total is never stored directly in the file (see
[Player objects](#player-objects)). A reader recomputes it from
`pods[].result`, `rounds[].byes`, and the `config` fields below, using
one of the formulas selected by `config.scoring_logic`. All formulas
only accumulate points for Swiss-stage rounds
(`stage` / `rounds[].stage` value `0`, see
[`top_cut` and `stage` values](#top_cut-and-stage-values)); a
non-Swiss round ends the accumulation for both.

### `ScoringDefault`

The default. `scoring_params` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `win_points` | int | `5` | Points awarded for a pod win. |
| `bye_points` | int | `4` | Points awarded for a bye. |
| `draw_points` | int | `1` | Points awarded to each player in a draw. |

A player's rating is the sum, over every Swiss round up to and
including the round in question, of:

- `scoring_params.win_points` for a round the player won.
- `scoring_params.draw_points` for a round the player drew.
- `scoring_params.bye_points` for a round the player received a bye.
- `0` for a round the player lost or has pending.

This is independent per player: no player's score affects any other
player's.

### `Scoring1v1`

Identical formula to `ScoringDefault` (`Scoring1v1` subclasses it directly,
`src/logic/mtg/scoring.py`) - the pod win/draw/bye a round's rating sums
over are unchanged. Only the `scoring_params` defaults differ, matching the
Magic Tournament Rules' match points (Appendix C) instead of Commander's:

| Field | Type | Default | Description |
|---|---|---|---|
| `win_points` | int | `3` | Points awarded for a pod (match) win. |
| `bye_points` | int | `3` | Points awarded for a bye (an automatic match win). |
| `draw_points` | int | `1` | Points awarded to each player in a draw. |

### `ScoringHareruya`

`scoring_params` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `wager_percent` | float | `0.07` | Fraction of a player's current stack wagered per pod. |
| `wagering_starting_points` | float | `1000` | Starting stack, before round 1. |
| `draw_redistribution_fraction` | float, 0-1 | `1.0` | `R` below - fraction of a draw's pot actually paid out to the drawers. |
| `draw_distribution_shape` | float, 0-1 | `1.0` | `S` below - even split (`1`) vs. wager-proportional split (`0`) of a draw payout. |
| `redistribute_discarded_draw_points` | bool | `false` | If `true`, the pot left over after `draw_redistribution_fraction` is applied is reclaimed instead of discarded. |
| `draw_discard_pod_fraction` | float, 0-1 | `1.0` | `P` below - only used when `redistribute_discarded_draw_points` is `true`. |

Each player carries a running point stack, seeded at
`scoring_params.wagering_starting_points` before round 1. For every pod a
player is seated in, once that pod has a result:

1. Every seated player wagers `scoring_params.wager_percent × their current
   stack`, deducted immediately. The **pot** is the sum of every
   seated player's wager. The pot uses the real seated players only. A
   pod smaller than the largest `config.pod_sizes` value has fewer
   wagers, so a smaller pot and a smaller reward. No phantom or
   placeholder player is added to a short pod.
2. **Win** (`pods[].result` holds one UID): that player receives the
   full pot. Every other seated player has already lost their wager
   and receives nothing further.
3. **Draw** (`pods[].result` holds two or more UIDs, the *drawers*):
   any seated player not listed is a loser under the same rule as
   step 2. Their wager stays in the pot, and they receive nothing.
   The pot (drawers' wagers plus any losers' forfeited wagers) is
   then split among the *N* drawers. For drawer *i*, whose own wager
   was `w_i`:
   ```
   payout_i = R × ( S × (pot / N) + (1 − S) × w_i )
   ```
   where `R` = `scoring_params.draw_redistribution_fraction` and `S` =
   `scoring_params.draw_distribution_shape`, each `0`-`1`. At `R = 1, S = 1`
   the pot splits evenly regardless of individual wager size. At
   `R = 1, S = 0` each drawer gets exactly their own wager back, with
   no net change. At `R = 0` every drawer's wager is lost regardless
   of `S`.

   By default, `(1 − R) × pot` — whatever `payout_i` above never
   pays out — leaves the economy entirely: it is not returned to the
   original wagerers. If `scoring_params.redistribute_discarded_draw_points`
   is `true`, that amount is reclaimed instead of discarded, split by
   `scoring_params.draw_discard_pod_fraction` (`P`, `0`-`1`):
   ```
   discarded    = (1 − R) × pot
   pod_share    = P × discarded
   global_share = discarded − pod_share
   ```
   `pod_share` is paid to the same *N* drawers only (not the pod's
   other seated losers), as a bonus on top of `payout_i`, split by the
   same `S` shape but normalized to each drawer's share of the total
   drawer wager `w_i / Σw_j` rather than the raw wager `w_i`:
   ```
   bonus_i = pod_share × ( S × (1 / N) + (1 − S) × (w_i / Σw_j) )
   ```
   `global_share` is paid out as an equal per-capita dividend to every
   player in the tournament — including this pod's own drawers, and
   including players not seated in this round at all (a bye recipient
   this round included, per step 4). At `P = 1` all of `discarded`
   goes to `bonus_i`; at `P = 0` all of it goes to the dividend.
   Either way, `bonus_i` and the dividend sum back to exactly
   `discarded`, so the total ever paid out on a draw
   (`Σ payout_i + discarded`) never exceeds `pot`.
4. **Bye** (player listed in `rounds[].byes`): receiving the bye
   itself is a no-op — no wager, no `bye_points`. It does not exempt
   the player from step 3's per-capita dividend, if any other pod in
   the round has a redistributed draw this round.
5. **Pending** (`pods[].result` is empty): no wager is assessed for
   that pod this round.

A win's payout depends on every other seated player's current stack.
Because of this, one player's rating cannot be computed from that
player's own history alone. Computing it requires replaying every
player's stack up to the round in question. A reader implementing
this must build the whole tournament's running stacks together, not
player by player.

`pointrate` (used for standings tie-breaking, see
[Reference implementation](#reference-implementation)) has no exact
`0`-`1` bound under this formula, since a single win is not capped at
a fixed maximum the way `ScoringDefault`'s is. EDH_matchmaker's
reference implementation approximates it as
`rating / (wagering_starting_points × (round number + 1))`. This is
shaped like the default formula, but it is not a strict bound. Treat
it as a rough ordering signal, not a percentage.

### `ScoringModifiedHareruya`

`ScoringHareruya` depends on the round order. Each wager is a
percentage of the current stack, so the stack compounds. The record
`WDD` gives a different total than `DDW`. The round order is arbitrary,
so this dependency is unwanted.

`ScoringModifiedHareruya` removes the dependency on round order. It runs
the `ScoringHareruya` economy once for each permutation of the Swiss
round order. Then it averages each player's final stack across all
permutations. For 3 rounds, it averages the 6 orders.

A pod result does not change with the round order. Only the stack
values change. So a reader extracts each pod result one time, then
replays the wager math for each order.

The permutation count is the factorial of the round count. For more
than 7 rounds, the reference implementation does not use every order. It
averages a fixed sample of random orders instead. This is an
implementation choice, not a requirement of this format.

This variant uses the same `scoring_params` fields as `ScoringHareruya`. It
shares the same `pointrate` approximation.

### Where parameter definitions live (implementation note)

This part is about the reference implementation, not the JSON format. Each
algorithm's parameter names, defaults, types, ranges, and human descriptions
live in a sidecar file next to the algorithm class, named
`<ClassName>.params.yaml` (for example
`src/logic/commander/ScoringHareruya.params.yaml`). The file is the single source
of truth. The code loads it at class-definition time and derives the defaults
from it - no parameter values are hard-coded in the class.

To add a new scoring or pairing algorithm with tunable parameters:

1. Write the algorithm class in `src/logic/<game>/scoring.py` or
   `src/logic/<game>/matching.py`. Set `IS_COMPLETE = True`. Read a parameter
   with `self._param(tour, "name")`.
2. Add `<ClassName>.params.yaml` next to it. Each entry needs a `default` and a
   `description`. Optional keys: `type`, `min`, `max`, `step`, `label`,
   `widget`, `choices`, `scale`, `suffix`, `visible_when`.
   The config GUI generates a widget per entry from these fields, so the field
   appears with no GUI code. `widget` is inferred from `type` when omitted
   (`bool`->checkbox, `int`->spinbox, `float`->doublespinbox, `str`->lineedit,
   `choices`->combobox). `scale`/`suffix` set a display transform (a fraction
   shown as a percentage). `visible_when: {other_param: value}` shows the field
   only while another param holds that value.
3. A subclass with no sidecar of its own inherits its parent's parameters (as
   `ScoringModifiedHareruya` inherits `ScoringHareruya`'s).

An algorithm with no parameters needs no sidecar file.

## Pairing logic

Pairing logic decides how players are grouped into pods each round.
`config.pairing_rounds` holds one object per round (see the `config` table),
`{"logic": <name>, "params": {...}, "ruleset_params": {...}}`. Entry `[i]`
sets the logic and its overrides for round `i`. The per-round list differs
from the flat `scoring_params`, because pairing logic and its settings can
differ per round. `ruleset_params` is a ruleset concern, not a pairing-logic
one - see [Rulesets](#rulesets).

A reader that does not re-pair rounds can ignore this section. Pairing affects
how a file was produced, not how a stored result is read back.

### Pod sizes

`config.pod_sizes` sets the tournament's pod sizes (ordered, preference first).
Each pairing algorithm declares which sizes it supports; an algorithm is offered
for a round only if it supports every size in `config.pod_sizes`. `PairingDefault`
and `PairingSnake` support `3`, `4`, and `5`; `Pairing1v1` supports `2`;
`PairingRandom` supports any size. So a tournament with pod sizes `[2]` can use
`Pairing1v1` or `PairingRandom`. This is a constraint the writer applies; the
format itself does not enforce it.

A "small" pod, for the `small_pod_penalty` below, is one smaller than the
preferred (first) size in `config.pod_sizes`. So with `[4, 3]` the 3-player pod
is small; with `[5, 4, 3]` the 4-player and 3-player pods are both smaller than
the preferred `5`.

### `PairingDefault`

Additional fields in the `params` object for a round that uses `PairingDefault`
(or `Pairing1v1`, which shares the same parameters):

| Field | Type | Default | Description |
|---|---|---|---|
| `rematch_penalty_exponent` | int | `2` | Exponent on the repeat-opponent count when scoring how well a player fits a pod. A higher value pushes harder against seating players who already met. |
| `small_pod_penalty` | int | `10` | Penalty for seating a player in a pod smaller than the preferred (first) pod size, for example a 3-player pod, when the player already sat in a small pod. A higher value spreads the small pods across more players. |

`PairingRandom`, `PairingSnake`, and the top-cut pairings read no
parameters. A per-round match format (such as games needed to win a match)
is a ruleset concern, not a pairing-logic parameter - see `ruleset_params`
in the `config` table and [Rulesets](#rulesets).

## Adjacent outputs (not part of this format)

EDH_matchmaker writes several other files alongside the JSON log. None
of them are part of this specification. A conforming reader/writer only
needs the sections above.

- A plain-text standings table, written to `config.standings_export.dir`.
- A plain-text pairings dump per round.
- An optional webhook POST of pairings data, controlled by environment
  variables outside this file.
- An in-memory event log (severity levels: none, info, warning, error),
  used for status messages. Nothing writes this to disk by default.

## Machine-checkable schema

[`docs/tournament-log.schema.json`](tournament-log.schema.json) is a
JSON Schema (draft 2020-12) that formalizes the tables above. Use it to
validate a file programmatically instead of checking each field by
hand.

`docs/tournament-log-examples/` holds example files for testing an
implementation against the schema:
[`valid-minimal.json`](tournament-log-examples/valid-minimal.json),
[`valid-topcut.json`](tournament-log-examples/valid-topcut.json), and
[`valid-hareruya.json`](tournament-log-examples/valid-hareruya.json)
(exercises `config.scoring_logic: "ScoringHareruya"`) must pass
validation;
[`invalid-missing-required.json`](tournament-log-examples/invalid-missing-required.json)
and [`invalid-bad-enum.json`](tournament-log-examples/invalid-bad-enum.json)
must fail it, each for the specific reason its name states.

## Governance

This repository is the source of truth for this format. A format
change is a pull request against this repository that updates
`docs/tournament-log.schema.json`, this page, and `core.py`'s
`serialize`/`inflate` methods together, in the same pull request. A
change that is not backward compatible must bump `format_version` and
add an entry to `CHANGELOG.md`. A change that only adds optional
fields does not need a version bump, per
[Format versioning](#format-versioning).

## Security considerations

This format defines no size or depth limit for any field. A reader
that parses a file from an untrusted source must apply its own limits
(maximum file size, maximum array length, a sane ceiling on `n_rounds`
and pod counts) before trusting values taken from the file.
EDH_matchmaker itself does not enforce such limits today, since it
only reads files a user selects from local disk.

## Reference implementation

EDH_matchmaker's Python implementation of this format, for a reader who
wants working code to compare against:

| Class | `serialize` | `inflate` | File |
|---|---|---|---|
| `Tournament` | writes the top-level object | reads the top-level object | `src/core.py` |
| `Player` | writes a `players[]` entry | reads a `players[]` entry | `src/core.py` |
| `Round` | writes a `rounds[]` entry | reads a `rounds[]` entry | `src/core.py` |
| `Pod` | writes a `pods[]` entry | reads a `pods[]` entry | `src/core.py` |
| `TournamentConfiguration` | writes `config` | reads `config` | `src/core.py` |
| `StandingsExport` | writes `standings_export` | reads `standings_export` | `src/core.py` |
| `TournamentAction` | `store()` writes the file | `load()` reads the file | `src/core.py` |

`ScoringDefault`, `ScoringHareruya`, and `ScoringModifiedHareruya` (all
in `src/logic/commander/scoring.py`), and `Scoring1v1`
(`src/logic/mtg/scoring.py`, subclassing `ScoringDefault`) implement the
formulas from [Scoring logic](#scoring-logic); they are not part of the
JSON schema themselves, only the `config.scoring_logic` string that names
one of them.

`CommanderRuleset` (`src/logic/commander/rules.py`) and `Mtg1v1Ruleset`
(`src/logic/mtg/rules.py`) implement [Rulesets](#rulesets); like the
scoring logics above, neither is part of the JSON schema itself, only the
`config.ruleset` string that names one of them.

`tests/test_serialization.py` holds round-trip tests that double as
executable proof of this contract, including a test that loads a real
captured tournament file end to end. `tests/test_scoring_wagering.py`
holds the same proof for the two scoring logic formulas, including the
draw payout formula at its `R`/`S` boundary values.

## Worked example: load, change, save

```python
from src.core import TournamentAction

tournament = TournamentAction.load("logs/default.json")
assert tournament is not None

# Read a field.
print(tournament.config.n_rounds)

# Change a field and save. TournamentAction.store() always writes the
# full current state, using the atomic write procedure described above.
tournament.config.n_rounds = 6
TournamentAction.store(tournament)
```

## Known gaps and open decisions

Format `1.0` already includes `format_version`, `generator`,
`created_at`, `updated_at`, and the atomic write procedure. The items
below remain open for a future version aimed at multiple independent
implementations.

- **Enum value stability is a convention, not an enforced contract.**
  Nothing in the file format stops a future writer from reusing an
  integer for a different meaning. Treat the tables in
  [Enum reference](#enum-reference) as frozen, and bump
  `format_version` before ever changing one.
- **`decklist` has no defined format.** It is free text today — often
  a URL, sometimes plain text, sometimes absent. A future version must
  pick one convention (for example, "must be a URL or `null`") for
  cross-software decklist exchange to work.
- **`rounds[].logic` names one codebase's pairing-algorithm classes.**
  A future version needing pairing-algorithm portability across
  software must define a fixed, software-independent vocabulary
  instead of accepting any string.
- **`config.scoring_logic` has the same limitation.** `"ScoringDefault"`
  and `"ScoringHareruya"` are this codebase's class names, not a fixed
  cross-software vocabulary. A reader that does not recognize a
  `scoring_logic` value cannot know how to reproduce that tournament's
  point totals from `pods[].result` alone.
- **Under `ScoringHareruya`, computing one player's rating requires
  replaying every player's stack**, since payouts are interdependent
  (see [Scoring logic](#scoring-logic)). EDH_matchmaker's reference
  implementation accepts this cost rather than adding a cache, since
  it is only paid per standings computation, not per game reported;
  see the `# ponytail:` comment on `Tournament.rating()` in
  `src/core.py` for the exact tradeoff.
- **No concurrent-write protection.** Two writers on the same path can
  silently overwrite each other. A future version needing multi-writer
  safety must add file locking or a compare-and-swap mechanism; this
  version does not attempt it. This is a deliberate choice, not an
  oversight: EDH_matchmaker has no current multi-writer use case, and
  building locking infrastructure for a need nobody has yet is not
  worth the added complexity.
- **No file-naming convention.** This format is a description of file
  *content*; it says nothing about file names or where files live on
  disk. Software that wants to auto-discover tournament files from a
  directory must agree on a naming convention separately from this
  page.
- **One reference implementation, in one language.** Every claim on
  this page is checked against EDH_matchmaker's Python code and the
  schema in [Machine-checkable schema](#machine-checkable-schema), but
  no second, independent implementation exists to confirm the page is
  sufficient on its own. This is a deliberate choice, not an
  oversight: building a second implementation for a format with one
  current consumer is not worth the effort yet. The conformance
  example files exist so a future second implementation has something
  concrete to test against.

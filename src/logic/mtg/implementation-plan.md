# 1v1 Magic and game rulesets: core implementation plan

For an agent to execute. **No GUI work.** The Magic rules come from
[`mtr-1v1-spec.md`](mtr-1v1-spec.md) (the "spec"). "Spec §3" means section
3 there.

Before you start, read:

- `CLAUDE.md`, `CONTEXT.md` ("Game vs. match", "Bye", `SELECTABLE`,
  "sidecar", the import-the-module discovery trap), `CONTRIBUTING.md`,
  `AGENTS.md`.
- The skills `.claude/skills/add-pairing-scoring-algorithm/SKILL.md`,
  `params-yaml-sidecar/SKILL.md`, `tournament-log-format/SKILL.md`. Several
  phases change the save format, so the last one applies.
- `src/logic/commander/matching.py` and `scoring.py` as reference
  implementations.

## 0. Rules for the implementing agent

- **Scope: core only.** `src/core.py`, `src/interface.py`,
  `src/param_spec.py`, `src/logic/commander/`, `src/logic/mtg/`, the log
  format (docs, schema, examples), tests, project docs. Do **not** touch
  `run_ui.py` or `ui/`, and keep every core API they call working
  (section 10).
- Repo rules: never write the em dash character; never edit `CHANGELOG.md`;
  commit only when the user asks, without an agent co-author line;
  Google-style docstrings; `unittest` tests; `TournamentAction.LOGF = False
  # type: ignore` at the top of every test module that builds a
  `Tournament`; the `*.params.yaml` sidecar is the only source of parameter
  defaults; no new pyright errors in touched code.
- Do the phases in order. Each phase ends with a green `PYTHONPATH=. pytest`.
  A phase that changes the saved JSON also updates
  `docs/tournament-log.schema.json` and `docs/tournament-log-spec.md` in the
  same phase (governance rule of that doc).
- For each existing bug you fix, write the failing test first.
- Commander behaviour must not change, except where a step says so
  explicitly.
- Find code by name. Line numbers are hints from commit `0f55acf`.
- Baseline at `0f55acf`: default run 6 failed / 189 passed / 2 skipped;
  `-m slow` also fails (`test_snake_no_repeat_matching`). All are stale tests
  from commit `f2ceb67`. Phase 1 fixes them.
- `.claude/skills/add-pairing-scoring-algorithm/SKILL.md` has an uncommitted
  user edit (`(3, 4)` -> `(4, 3)`). Keep it.
- Do not merge or cherry-pick the older MTG branches (section 11).

---

## 1. Context: what is wrong today

| Area | Today | Needed |
|---|---|---|
| Match format | `games_to_win` is a **pairing** param (`CommonPairing.params.yaml`, `PairingDefault.params.yaml`, default 1). Its default depends on the round's pairing logic; playoff rounds cannot set it; Commander shows a useless knob. | A rule of the game, configurable per round (spec §1): MTG default 2 in Swiss, each Swiss round and each playoff round configurable (Bo3/Bo5/Bo7). Commander: always one game, not configurable. |
| Result entry | `Pod.set_result` appends one game per call; the match closes only at `games_to_win` or after `2n-1` games. Re-reporting a Commander pod raises. | Report the whole match at once (`2-1-0`, time-called `1-0`, ID `0-0-3`), replacing any earlier report (spec §1). |
| Standings | `TournamentConfiguration.ranking` (`core.py:737`): the Commander chain, in core. | Spec §3: MP, OMW, GW, OGW, owned by the game's rules. |
| Swiss pairing | `mtg.Pairing1v1` subclasses commander `PairingDefault`: greedy pod fill, rematches only penalised. | Spec §4.1: score groups, pair-down, no rematches, bye to the lowest-ranked player without a bye. |
| Top cut | `TopCut` / `Stage` are Commander pod cuts (4/7/10/13/16/40), stage chain hard-coded in `Tournament.__compute_stage_and_logic`. | Spec §4: single-elimination bracket 1v8 / 4v5 / 2v7 / 3v6, no draws. |
| `random_results` | One seat-weighted game per pod (about 56% draws for 2-player pods). | Valid whole matches. |
| Game-specific code in core | Standings chain, stage chain, Swiss defaults (`_adaptive_pairing_logic`), random results. | In the game's plugin. |

Next game family after Magic is Catan-like (points decide the match). The
design below must let it be added as a plugin (section 9).

---

## 2. Architecture

### 2.1 Vocabulary

| Term | Meaning |
|---|---|
| Game | One play-through with one outcome per player: won, drew, lost. |
| Match | Everything one pod plays in one round: one or more games. Standings, pairing and scoring see only the match outcome. |
| Match report | The complete list of a match's games, entered at once. A new report **replaces** the previous one. |
| Ruleset | The rules of one game system. New plugin kind in `src/logic/<game>/rules.py`, selected by `config.ruleset`. |
| Stage | `SWISS` (0), or a playoff stage whose value is the number of players in contention when that round starts. Already true for Commander (`disable_topcut` uses `stage.value`). |

### 2.2 Who owns what

| Question | Owner | Commander | MTG 1v1 | Catan (later) |
|---|---|---|---|---|
| Pod sizes | Ruleset `DEFAULT_POD_SIZES` / `ALLOWED_POD_SIZES` | (4, 3) / any | (2,) / (2,) | (4, 3) / (3, 4) |
| Match format per round | Ruleset params (sidecar) + per-round overrides | one game, nothing to configure | `games_to_win` per Swiss round and playoff stage | one game |
| Is a report valid? | `IRuleset.validate_report` | exactly one game | 2 players, win caps, no drawn playoff match | points for every player |
| Who won the match? | `IRuleset.match_winners` | the game's winners | more game wins; equal = draw | the game winner |
| Points | Scoring logic (unchanged) | `ScoringDefault`, Hareruya | `Scoring1v1` (3/1/0) | e.g. placement points |
| Order on equal points | `IRuleset.standings_keys` (+ `standings_columns`) | today's chain | OMW, GW, OGW | VP totals |
| Seat balancing | `IRuleset.SEAT_BALANCING` | yes | no (spec §4: seat means nothing) | yes |
| Swiss pairing | Pairing logic; default per round from `IRuleset.swiss_pairing_logic` | Random, Snake, Default | `Pairing1v1` (weighted matching) | Random, Default |
| Playoffs | `IRuleset.PLAYOFFS` + top-cut pairing logics | Top 4/7/10/13/16/40 | single elimination 2/4/8/16 | final table |
| Random results (dev/tests) | `IRuleset.random_report` | seat-weighted | simulated games | random VP |
| Rounds, byes, drops, persistence | Core | | | |

Rule of thumb: if the answer depends on the game ("who won?", "who ranks
higher on equal points?", "what comes after Swiss?"), it belongs to the
ruleset. "How many points is that worth?" belongs to scoring. "Who plays
whom?" belongs to pairing.

### 2.3 Data flow

```
config.ruleset -------------> IRuleset  (src/logic/<game>/rules.py)
                                 |  validate_report / match_winners / report_from_winners
report_match(pod, games) ---> Pod._games: list[GameResult] --> Pod._result (match winners)
report_win / report_draw --^                                      |
  (ruleset shorthand)                                             v
                               Player.result(round): WIN / DRAW / LOSS / BYE / PENDING
                                                                  |
        IScoringLogic.rating --> Tournament.get_standings --> IRuleset.standings_keys
        IPairingLogic.make_pairings   (Swiss default: IRuleset.swiss_pairing_logic)
        playoffs: IRuleset.PLAYOFFS[top_cut] --> (stage, top-cut pairing logic)
```

### 2.4 Decisions (and rejected alternatives)

1. **Tiebreakers belong to the ruleset, not to scoring.** If they lived on
   `Scoring1v1` (as branch `fm/1v1-topcut-recover-92f` did), an MTG event
   that picks another scoring logic would silently get Commander
   tiebreakers. Scoring answers "how many points"; the standings order is a
   rule of the game.
2. **`games_to_win` is a per-round ruleset parameter, not a pairing
   parameter.** Its existence and default must not depend on the pairing
   logic; playoff rounds need it; Commander has none.
3. **Whole-match reports that replace.** Matches MTG match slips; a
   time-called match is simply reported as its score (no time tracking);
   re-reporting a Commander pod overwrites again, as before `f2ceb67`.
   Game-by-game entry is not supported.
4. **Stage value = players in contention, one enum for every game.** Add
   `TOP_2 = 2`, `TOP_8 = 8`. No per-game offset (branch 92f used 102,
   104, ...): the ruleset maps each stage to its pairing logic, and pods are
   stored in the log anyway.
5. **One `PairingBracket` class**, not one per size: the stage value tells
   it which bracket level it pairs.
6. **Swiss 1v1 pairing is a maximum-weight perfect matching (networkx).**
   No rematch when avoidable, fewest and smallest pair-downs, bye chosen
   jointly. Measured: 128 players 0.4 s, 256 players 2.4 s.
7. **`GameResult` is a frozen dataclass; `pods[].games[]` entries are JSON
   objects.** `games` was never released (`f2ceb67` is not on `master`), so
   its shape can change now; Catan's points become an additive field later.
8. **A tournament's game is explicit (`config.ruleset`)**, not inferred from
   `pod_sizes == [2]`.

---

## 3. Core contracts (`src/interface.py`)

### 3.1 `GameResult` (new)

```python
@dataclass(frozen=True)
class GameResult:
    """One game of a match: the players who did not lose it.

    One UID means that player won the game. Two or more mean the game was
    drawn among them. Every other player seated in the pod lost the game.
    """
    winners: frozenset[UUID]
```

`__post_init__` coerces any iterable to `frozenset` (`object.__setattr__`)
and raises `ValueError` when empty.

### 3.2 `IRuleset` (new ABC)

Same plugin shape as `IScoringLogic` (`IS_COMPLETE`, `name`,
`PARAM_SPEC` / `DEFAULT_PARAMS` loaded from the sidecar in
`__init_subclass__`):

| Member | Contract |
|---|---|
| `__init__(self, name: str)` | Stores `name` (discovery calls `cls(name=cls.__name__)`). |
| `DEFAULT_POD_SIZES: tuple[int, ...]` | Pod sizes of a new config, preferred first. Required. |
| `ALLOWED_POD_SIZES: tuple[int, ...] \| None = None` | `config.pod_sizes` must be a subset. `None` = any. |
| `DEFAULT_SCORING_LOGIC: str` | Scoring logic of a new config. Required. |
| `DEFAULT_STANDINGS_FIELDS: tuple[str, ...] = ("STANDING", "NAME", "RATING", "RECORD")` | `StandingsExport.Field` names for a new config's export. |
| `SEAT_BALANCING: bool = True` | Whether core calls `Pod.auto_assign_seats` after Swiss pairing. |
| `PLAYOFFS: Mapping[int, tuple[tuple[int, str], ...]] = {}` | `top_cut` -> playoff rounds in play order, each `(stage value, top-cut pairing logic name)`. Its keys are the only non-zero `top_cut` values the ruleset accepts. |
| `params(self, tour_round) -> dict` | `{**DEFAULT_PARAMS, **config.ruleset_overrides(tour_round)}`. Cold path. |
| `_param(self, tour_round, key) -> Any` | One param without allocating (same pattern as `CommonPairing._param`). Hot path. |
| `validate_report(self, pod, games) -> None` | Raises `ValueError` (message fit for a tournament organiser) if `games` is not a valid complete report for `pod` in its round. Never mutates. |
| `match_winners(self, pod) -> frozenset[UUID]` | Players who did not lose the match: one = winner, several = drew. Called only when `pod.games` is non-empty. Must be total: it runs on stored data (old files, rosters edited after reporting) and must never raise. |
| `report_from_winners(self, pod, winners) -> list[GameResult]` | Turns "A won" / "A and B drew" into a canonical report. Backs `report_win` / `report_draw`. |
| `random_report(self, pod) -> list[GameResult]` | A plausible valid report for `Tournament.random_results`. Use the `random` module so tests can seed it. |
| `swiss_pairing_logic(self, tour, seq) -> str` | Default pairing logic for Swiss round `seq` when `config.pairing_rounds` sets none. |
| `standings_keys(self, tour, tour_round, ratings) -> Mapping[UUID, tuple]` | One sort key per player, compared descending. Swiss rounds only. `ratings` is the scoring logic's field map; the first element should be the rating. |
| `standings_columns(self, tour, tour_round) -> list[tuple[str, Mapping[UUID, str]]]` | Extra standings-export columns `(header, formatted cell per player)`. Default `[]`. |

### 3.3 Other interface changes

- `IPod`: `_games: list[GameResult]`; new abstract property
  `games -> tuple[GameResult, ...]`.
- `IPairingLogic.advance_topcut`: no longer abstract; default no-op,
  documented as "called once per playoff round before `make_pairings`;
  top-cut logics override it to give seeded byes". Delete the raising
  `CommonPairing.advance_topcut`.
- `IRound`: add abstract `remove_pod(pod)` (exists on `Round`;
  `PairingBracket` needs it).
- `ITournament`: add abstract properties `players` and `final_swiss_round`
  (plugins already use both by duck typing).
- `ITournamentConfiguration`: delete `ranking`; add
  `ruleset: str = "CommanderRuleset"`,
  `playoff_rounds: dict[int, dict[str, Any]] = {}`, and abstract
  `ruleset_overrides(tour_round) -> Mapping[str, Any]`.

---

## 4. Configuration and persistence

### 4.1 Config fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ruleset` | str | `"CommanderRuleset"` | Class name of the ruleset. Also provides the defaults of `pod_sizes`, `scoring_logic` and `standings_export.fields` when the caller does not pass them. |
| `pairing_rounds[i].ruleset_params` | object | `{}` | Ruleset param overrides for Swiss round `i`, next to its `logic` and `params`. |
| `playoff_rounds` | object `{"<stage>": {"ruleset_params": {...}}}` | `{}` | Overrides per playoff stage (key = stage value, e.g. `"2"` = MTG final). In memory `dict[int, dict]`. Stages not in the current playoff plan are ignored. |

`TournamentConfiguration._build_pairing_rounds` must keep `ruleset_params`
(today it rebuilds entries with `logic` and `params` only). `pairing_logics`
/ `pairing_params` stay as read-only views.

### 4.2 Resolving a round's ruleset params

`TournamentConfiguration.ruleset_overrides(tour_round)`:
- Swiss round: `pairing_rounds[seq].get("ruleset_params", {})` when
  `seq < len(pairing_rounds)`, else `{}`.
- Playoff round: `playoff_rounds.get(stage.value, {}).get("ruleset_params", {})`.

Override wins over the sidecar default. No tournament-wide layer: the
sidecar default is the tournament default.

### 4.3 Validation, in one place

`Tournament.__validate_config(config)` is called from the `config` setter
(as today) and, new, from `Tournament.__init__` once the round list exists
(`inflate` goes through `__init__`; every existing valid Commander log
passes). It raises `ValueError` when:

1. more Swiss rounds were played than `n_rounds` (existing);
2. `config.ruleset` is unknown;
3. `pod_sizes` is empty or not a subset of `ALLOWED_POD_SIZES`;
4. `top_cut` is neither 0 nor a key of `ruleset.PLAYOFFS`;
5. any `ruleset_params` (Swiss or playoff) fails
   `validate_values(ruleset.PARAM_SPEC, ...)`: unknown name, wrong type,
   outside `min`/`max`, not in `choices`. So Commander rejects every
   override, including `games_to_win`;
6. the tournament already has a round with pods, byes or game losses and
   `config.ruleset` differs from the current one (a result must never be
   re-read under another game's rules).

New helper `src/param_spec.py: validate_values(specs, values, where)`. Types:
`int` = `type(v) is int`; `float` accepts `int` or `float` but not `bool`;
`bool` and `str` exact.

### 4.4 Log format 1.2

- Additive (optional on read): `config.ruleset`, `config.playoff_rounds`,
  `pairing_rounds[].ruleset_params`.
- Never released, so the shape may change: `pods[].games` is an array of
  objects:
  ```json
  "games": [{"winners": ["<uid A>"]}, {"winners": ["<uid B>"]}, {"winners": ["<uid A>"]}]
  ```
- `pods[].result` stays the derived match outcome: a reader that ignores
  `games` still reads correct results.
- New `top_cut` / `stage` values `2` (`TOP_2`) and `8` (`TOP_8`). A 1.1
  reader would reject them, so set `Tournament.LOG_FORMAT_VERSION = "1.2"`
  (Phase 5).

### 4.5 Reading old files

- No `ruleset`: `CommanderRuleset`. No `playoff_rounds` / `ruleset_params`: `{}`.
- No `games` (files before `f2ceb67`): one game from `result` (existing).
- `games` entries that are UID arrays (unreleased branch files): read as
  `{"winners": [...]}`.
- `pairing_rounds[].params.games_to_win` (unreleased branch files): ignored;
  such files load as Commander. Deliberate; document it.
- Warn only for an unknown `format_version` (today every non-current version
  warns, including valid older ones).

---

## 5. Plugins

### 5.1 `CommanderRuleset` (new `src/logic/commander/rules.py`, no sidecar)

A pure move: Commander must behave byte for byte as today.

- `DEFAULT_POD_SIZES = (4, 3)`, `ALLOWED_POD_SIZES = None`,
  `DEFAULT_SCORING_LOGIC = "ScoringDefault"`, `SEAT_BALANCING = True`.
- `DEFAULT_STANDINGS_FIELDS`: the names of today's
  `StandingsExport.DEFAULT_FIELDS`, same order.
- `PLAYOFFS`: the table `__compute_stage_and_logic` encodes today:
  `{4: ((4, "PairingTop4"),), 7: ((7, "PairingTop7"), (4, "PairingTop4")), 10: ((10, "PairingTop10"), (4, "PairingTop4")), 13: ((13, "PairingTop13"), (4, "PairingTop4")), 16: ((16, "PairingTop16"), (4, "PairingTop4")), 40: ((40, "PairingTop40"), (16, "PairingTop16"), (4, "PairingTop4"))}`.
- `validate_report`: exactly one game; its winners are seated in the pod.
- `match_winners`: the last game's winners.
- `report_from_winners`: `[GameResult(winners)]`.
- `random_report`: the body of today's `Tournament.random_results` for one
  pod, unchanged (same `random.random()` call, same numpy arithmetic on
  `config.global_wr_seats`).
- `swiss_pairing_logic`: today's choice: `PairingRandom` at seq 0,
  `PairingSnake` at seq 1 when `config.snake_pods`, `PairingDefault` after.
  The pod-size fallback stays in core.
- `standings_keys`: today's `TournamentConfiguration.ranking` tuple,
  verbatim, for every player in `tour.players`.

### 5.2 `Mtg1v1Ruleset` (new `src/logic/mtg/rules.py` + `Mtg1v1Ruleset.params.yaml`)

Sidecar:
```yaml
games_to_win:
  default: 2
  type: int
  min: 1
  max: 4
  label: Games to win
  description: >
    Game wins a player needs to win the match in this round (Magic Tournament
    Rules 2.1): 2 is best of three, 3 best of five, 4 best of seven. Drawn
    games do not count toward it. A report may end below it when time is called.
```

- `DEFAULT_POD_SIZES = ALLOWED_POD_SIZES = (2,)`,
  `DEFAULT_SCORING_LOGIC = "Scoring1v1"`, `SEAT_BALANCING = False` (spec §4:
  a random method or the higher seed chooses play/draw; seats mean nothing),
  `DEFAULT_STANDINGS_FIELDS = ("STANDING", "NAME", "RATING", "RECORD")`,
  `PLAYOFFS = {2: ((2, B),), 4: ((4, B), (2, B)), 8: ((8, B), (4, B), (2, B)), 16: ((16, B), (8, B), (4, B), (2, B))}`
  with `B = "PairingBracket"`.
- `games_to_win(tour_round) -> int` = `self._param(tour_round, "games_to_win")`.
  Always read through the round.
- `validate_report(pod, games)`, with `g = games_to_win(pod.tour_round)`:
  1. the pod seats exactly 2 players;
  2. `games` is non-empty;
  3. each game has one winner or both players (drawn game); every winner is
     seated in the pod;
  4. each player has at most `g` game wins, and not both have `g`;
  5. in a playoff round (`stage != SWISS`), the game-win tally must not be
     tied (spec §1: no drawn single-elimination match). Added in Phase 5.

  Game order carries no meaning and is not validated.
- `match_winners`: count single-winner games per seated player; the sole
  leader wins; tied leaders drew. `1-0` = win, `1-1`, `1-1-1` and `0-0-3` =
  draw.
- `report_from_winners`: one winner -> `g` games won by them (`2-0` in Bo3,
  `3-0` in Bo5); both players -> one drawn game (`0-0-1`). Document that this
  shorthand exists for the old API; exact scores go through `report_match`.
- `random_report`: simulate games (about 5% drawn, otherwise a coin flip)
  until someone reaches `g`. In Swiss, stop early about 5% of the time after
  a game ("time called"); never in playoffs. Always passes `validate_report`.
- `swiss_pairing_logic`: always `"Pairing1v1"` (round 1 comes out random:
  everyone is in one score group, spec §4.1 step 1).
- `standings_keys`: `(rating, omw, gw, ogw, -uid.int)` from `mtr_stats`
  (section 7). The last element is the stable final tie-break (spec §3,
  not MTR; uids are random per tournament).
- `standings_columns`: `OMW`, `GW`, `OGW`, formatted `f"{float(x):.4f}"`.
- Module function
  `games_from_score(pod, wins: Mapping[IPlayer, int], draws: int = 0) -> list[GameResult]`:
  every key seated in `pod`, counts >= 0; returns single-winner games in
  seat order, then `draws` drawn games. It does not check `games_to_win`
  (`report_match` does). Usage:
  ```python
  t.report_match(pod, games_from_score(pod, {alice: 2, bob: 1}))  # 2-1
  t.report_match(pod, games_from_score(pod, {alice: 1}))          # time called at 1-0
  t.report_match(pod, games_from_score(pod, {}, draws=3))         # ID, 0-0-3
  t.report_win(alice)                                             # shorthand, 2-0
  ```

### 5.3 `Pairing1v1`, rewritten (`src/logic/mtg/matching.py`)

Keep the **class name** (saved logs refer to it). Subclass
`_commander_matching.CommonPairing` instead of `PairingDefault` (import the
module, not the class). `IS_COMPLETE = True`, `SUPPORTED_POD_SIZES = (2,)`,
no sidecar (the inherited `rematch_penalty_exponent` / `small_pod_penalty`
go away). Import networkx **inside** `make_pairings`, so a missing install
only breaks MTG Swiss pairing, not discovery of the whole `mtg` package.
Add `networkx` to `requirements.txt` (check the version with `pypi-axi`).

`make_pairings(tour_round, players, pods)`:

1. `ratings = self.field_ratings(tour_round)`;
   `standings = tour.get_standings(tour_round)`. Use only the generic API:
   a user may pair with `Pairing1v1` and score with any logic.
2. Anchored players: the single player of each pod that seats exactly 1.
   Free players: `players`. Byes needed:
   `B = len(free) - (2 * empty_pods + partial_pods)`; `B < 0` raises.
3. Graph nodes: free players, anchored players, `B` dummy `BYE` nodes.
   Edges: free-free, free-anchored, free-BYE. None between two anchored
   players, anchored-BYE or BYE-BYE.
4. Terms: `g(p)` = index of `p`'s rating among the distinct ratings of all
   player nodes, highest first (score group); `Gn` = number of groups;
   `g_low` = largest `g` among free players; `met(a, b)` = earlier matches
   between them (count opponents over `p.games(tour_round)`); `byes(p)` =
   `p.byes(tour_round)`; `rank_from_bottom(p)` = 0 for the lowest-standing
   player node, counting up; `P` = player nodes; `E = (P + B) // 2`.
5. Integer weights make the priorities strictly lexicographic (Python ints;
   networkx is exact with ints):
   ```
   D  = 1000                       # jitter range
   U  = D * (E + 1)                # 1 cost unit outweighs all jitter combined
   K  = 1                          # rank unit
   Gc = B * P + 1                  # 1 squared-gap unit outweighs all rank terms
   Rc = Gc * (E + 1) * (Gn**2 + 1) # 1 rematch or repeat bye outweighs all gap cost
   cost(a, b)   = Rc * met(a, b) + Gc * (g(a) - g(b))**2
   cost(p, BYE) = Rc * byes(p)   + Gc * (g_low - g(p))**2 + K * rank_from_bottom(p)
   weight       = -cost * U + random.randrange(D)
   ```
   Priority: (1) no rematch and no second bye, (2) smallest total squared
   score gap (pair-down as little as possible), (3) bye to the lowest-ranked
   player without a bye (spec §4.1 step 5), (4) random among equals.
6. `networkx.max_weight_matching(G, maxcardinality=True)`. Add nodes in
   standings order so a seeded `random` reproduces pairings. Any unmatched
   player or `BYE` node raises `ValueError("No valid 1v1 pairing: ...")`.
7. Apply: the partner of `BYE` gets
   `p.set_result(tour_round, IPlayer.EResult.BYE)`; the partner of an
   anchored player joins that pod; any other pair takes the next empty pod,
   higher standing first. Return `players`.
8. Comment the ceiling:
   `# ponytail: complete graph, O(P^3); 256 players ~2.4 s. If large events need it, keep only edges within +-2 score groups and fall back to the full graph when the matching is not perfect.`

### 5.4 `PairingBracket` (new, `src/logic/mtg/matching.py`)

`IS_COMPLETE = True`, `SELECTABLE = False`, `SUPPORTED_POD_SIZES = (2,)`,
no sidecar. Module function:
```python
def bracket_seed_order(n: int) -> list[int]:
    """Seeds in bracket order, e.g. 8 -> [1, 8, 4, 5, 2, 7, 3, 6] (MTR 10.4)."""
    order = [1]
    while len(order) < n:
        m = 2 * len(order)
        order = [x for s in order for x in (s, m + 1 - s)]
    return order
```

`make_pairings(tour_round, players, pods)`:
1. `n = int(tour.config.top_cut)`, `s = tour_round.stage.value`,
   `seeds` = the cut in seed order: the top `n` of
   `tour.get_standings(tour.final_swiss_round)` who were eligible at the cut
   (see 6, Phase 5 step 5). `slots = [seeds[i - 1] for i in bracket_seed_order(n)]`.
   The bracket is fixed at the cut and never reseeded.
2. `block = 2 * n // s` (slots that feed one match at this stage);
   `half = block // 2`. For each block, `left` / `right` = this round's
   survivors (`players`) in each half. Each side has at most one player (a
   sub-bracket has one winner); otherwise raise `ValueError`.
3. Both present: a match, higher seed first (seat 1 chooses play/draw,
   spec §4; informational). One present (the other dropped after the cut):
   that player gets a BYE and advances (see open decision 1). Neither:
   nothing.
4. Fill `pods` with the matches; remove pods left empty with
   `tour_round.remove_pod`. More matches than pods raises `ValueError`.

With a cut of 8: stage 8 blocks `[1,8] [4,5] [2,7] [3,6]`, stage 4 blocks
`[1,8,4,5] [2,7,3,6]`, stage 2 one block. The existing machinery does the
rest: `Round.disable_topcut`, `Round.advancing_players`, and the playoff
branch of `Tournament.get_standings` (winner, finalist, then losers by
elimination round, each by Swiss seed; 3rd/4th by Swiss seed, not MTR).

### 5.5 `Scoring1v1`

Behaviour unchanged (`ScoringDefault` with MTR point defaults, spec §2).
Update the docstring: tiebreakers live in `Mtg1v1Ruleset`, cite spec §2-3.

---

## 6. Phases

Files:
- **New:** `src/logic/commander/rules.py`, `src/logic/mtg/rules.py`,
  `src/logic/mtg/Mtg1v1Ruleset.params.yaml`, `tests/test_ruleset.py`,
  `tests/test_mtg.py`, `docs/tournament-log-examples/valid-mtg.json`.
- **Changed:** `src/interface.py`, `src/core.py`, `src/param_spec.py`,
  `src/logic/commander/matching.py`,
  `src/logic/commander/PairingDefault.params.yaml`,
  `src/logic/mtg/matching.py`, `src/logic/mtg/scoring.py` (docstring),
  `requirements.txt`, `tests/test_pairing.py`, `tests/test_topcut.py`,
  `tests/test_performance.py`, `docs/tournament-log-spec.md`,
  `docs/tournament-log.schema.json`, `CONTEXT.md`, `CLAUDE.md`,
  `.claude/skills/add-pairing-scoring-algorithm/SKILL.md`.
- **Deleted:** `src/logic/commander/CommonPairing.params.yaml`.
- **Untouched:** `run_ui.py`, `ui/`, `CHANGELOG.md`, `mtr-1v1-spec.md`,
  `docs/mtr-annotated.md`.

### Phase 1: ruleset extension point, match reports, Commander move

Goal: game-agnostic core, Commander unchanged, stale tests fixed.

**Snapshot first (before any edit).** Seeding `random` does not reproduce a
run (uids are `uuid4`, set order depends on object ids), so snapshot through
saved logs: on the unchanged code, generate a 16-player, 4-round Commander
log per Commander scoring logic in a scratch directory
(`TournamentAction.store`), plus the `docs/tournament-log-examples/valid-*.json`
logs. For each log, `TournamentAction.load` it (then set
`TournamentAction.LOGF = False` again, or the next action overwrites the
file) and save `get_standings_str(tour_round=r)` for every round. After the
phase, reload and diff: identical. Do not commit the scratch files.

1. `src/interface.py`: everything in section 3.
2. `src/param_spec.py`: `validate_values`.
3. `src/logic/commander/rules.py`: `CommanderRuleset` (5.1).
4. `src/core.py`:
   - Discovery: `_ruleset_cache`, `discover_ruleset()`
     (`_discover_logic("rules.py", IRuleset, ...)`), `get_ruleset(name)`, a
     `ruleset` property on `Tournament`.
   - `TournamentConfiguration`: `ruleset` field and its defaults (4.1),
     `pairing_rounds[].ruleset_params`, `playoff_rounds` (int keys in
     memory, str keys on the wire), `ruleset_overrides`, `serialize` /
     `inflate`; delete `ranking`.
   - `Tournament.__validate_config`: section 4.3; also call it from
     `__init__`.
   - `_adaptive_pairing_logic`: name from
     `self.ruleset.swiss_pairing_logic(self, seq)`; keep the pod-size
     fallback.
   - `__compute_stage_and_logic`, playoff branch:
     `plan = self.ruleset.PLAYOFFS.get(int(top_cut))`; `None` raises the
     existing "Unknown top cut" error; previous stage `SWISS` -> `plan[0]`;
     previous stage in the plan -> the next entry; otherwise, or past the
     end, log "Tournament completed." and return `None` (today's semantics).
   - `Round.create_pairings`: in playoff rounds call
     `self.logic.advance_topcut(...)` unconditionally (drop the hard-coded
     stage list); run `pod.auto_assign_seats()` only when
     `self.tour.ruleset.SEAT_BALANCING` (keep the `seq < n_rounds` check).
   - `Pod`: `_games: list[GameResult]`; `games` property;
     `record_result(games)` validates via
     `self.tour.ruleset.validate_report` then replaces `_games`;
     `remove_result(player)` removes the uid from every game and drops games
     left empty; `reset_result()` clears. `_result` = `frozenset()` without
     games, else `self.tour.ruleset.match_winners(self)`. `result_type` =
     PENDING if `_result` empty, WIN if one member, else DRAW. Delete
     `set_result`, `game_wins`, `games_to_win`, `max_games`. Serialize and
     inflate per 4.4 / 4.5.
   - `Round`: delete `games_to_win`. New `record_result(pod, games)`:
     `pod.record_result(games)`, then remove every uid in `pod._result` from
     `_byes` and `_game_loss` (the side effect today's `set_result` has on
     reported players). `set_result(players, WIN | DRAW)`: group players by
     pod (unseated raises, as today); WIN needs exactly one player per pod;
     build every pod's report with `ruleset.report_from_winners`, validate
     all, then `record_result` each (no half-applied report). BYE / LOSS
     branch unchanged.
   - `Tournament`: new actions `report_match(pod, games)` (->
     `pod.tour_round.record_result`) and `reset_result(pod)`.
     `random_results`: `tour_round.record_result(pod, self.ruleset.random_report(pod))`
     for each pending pod, in today's pod order. `get_standings`, Swiss
     branch: `keys = self.ruleset.standings_keys(self, tour_round, ratings)`,
     sort by `keys[p.uid]` descending.
5. `src/logic/commander/`: delete `CommonPairing.params.yaml` (only holds
   `games_to_win`); remove `games_to_win` from `PairingDefault.params.yaml`;
   delete the raising `CommonPairing.advance_topcut`. In
   `src/logic/mtg/matching.py`, drop `games_to_win` from the `Pairing1v1`
   docstring.
6. Stale tests (the only allowed edits to existing tests in this phase):
   - `tests/test_pairing.py`: the two `selectable_pairing_logics`
     expectations gain `"Pairing1v1"` (for `[2]` and for no filter;
     `[4, 3, 2]` stays `["PairingRandom"]`).
   - `test_snake_no_repeat_matching` (slow): replace the per-player
     `pod.set_result` loop with one
     `t.tour_round.set_result(winners, WIN if one winner else DRAW)` per pod.
   - The other four stale tests pass unchanged once `games_to_win` leaves
     the pairing sidecars and reports replace.
7. New `tests/test_ruleset.py`: discovery and unknown-name error; Commander
   config defaults (pod sizes, scoring, export fields equal
   `StandingsExport.DEFAULT_FIELDS`); `report_win` twice on one pod
   replaces; `report_draw` after a win replaces; a two-game Commander report
   raises and leaves the pod unchanged; `report_win` with two players of one
   pod raises; `reset_result`; Commander rejects any `ruleset_params`;
   ruleset change allowed before pairings, rejected after; serialize /
   inflate round trip of `ruleset`, `pairing_rounds[].ruleset_params`,
   `playoff_rounds`, game objects; legacy `games` arrays and result-only
   pods load; for every discovered ruleset, all `PLAYOFFS` names and
   `swiss_pairing_logic(t, 0..3)` resolve and all `DEFAULT_STANDINGS_FIELDS`
   are `StandingsExport.Field` names.
8. Schema and spec doc for this phase's wire changes (`config.ruleset`,
   `pairing_rounds` items incl. `ruleset_params`, `playoff_rounds`,
   `pod.games` items as objects with `winners`: uidArray, `minItems` 1).
   Remove the `games_to_win` subsection of "Pairing logic".

Done when: full suite green (also `-m slow`), snapshot diff empty.

### Phase 2: `Mtg1v1Ruleset`

1. `src/logic/mtg/rules.py`: `Mtg1v1Ruleset` without the playoff rule and
   without standings (5.2), plus `games_from_score`; sidecar
   `Mtg1v1Ruleset.params.yaml`. Until Phase 3, `standings_keys` may return
   the rating plus the uid tie-break.
2. New `tests/test_mtg.py` (rules): table test of `validate_report` (valid:
   `2-0`, `2-1`, `1-0`, `1-1`, `0-0-3`, `2-0-1`; invalid: 3 players, empty,
   `3-0` in Bo3, `2-2`, a game won by an unseated player, a one-player
   "draw"); table test of `match_winners`; shorthand (`report_win` records
   `g`-0 with the round's `g`, `report_draw` records `0-0-1`); per-round
   override (Swiss round 2 Bo1 via `pairing_rounds[1].ruleset_params`, read
   back through `games_to_win(round)`); invalid overrides (`0`, `5`, `"2"`)
   raise; 1000 seeded `random_report` samples all validate; config defaults
   (`pod_sizes == [2]`, `Scoring1v1`, export fields); pod sizes `[4]`
   rejected; an 8-player, 3-round MTG tournament runs through
   `create_pairings` / `random_results`; pods keep the order
   `make_pairings` produced (no seat balancing).
3. Spec doc: the MTG match-outcome rule in "Pod objects" and ruleset params
   in the config section.

### Phase 3: MTR standings (spec §2-3)

1. In `src/logic/mtg/rules.py`: `mtr_stats` / `MtrStats` (section 7), exact
   arithmetic with `fractions.Fraction` (floor = `Fraction(33, 100)`);
   `standings_keys`; `standings_columns`.
2. `src/core.py` `get_standings_str`: `fields=None` defaults to
   `self.config.standings_export.fields` (today it ignores configured fields
   and always uses `DEFAULT_FIELDS`, auto-export included); append
   `self.ruleset.standings_columns(...)` after the core fields in PLAIN
   output.
3. Tests: spec §8 cases word for word (MW 0.667 / floor / 0.60, GW 0.70 /
   floor, OMW 0.62 / 0.63 with the bye) against the pure helpers; section 7
   vector A through a real tournament with `report_match` (exact values and
   order); a drop (MW denominator = rounds actually played); a bye (in own
   MW/GW, absent from OMW); MTG standings string has `OMW` / `GW` / `OGW`
   headers, a Commander one does not; configured export fields are honoured.

### Phase 4: Swiss pairing (spec §4.1)

1. `requirements.txt` += `networkx`. Rewrite `Pairing1v1` (5.3).
2. Tests (seed `random`, loop over several seeds): round 1 seats everyone
   at random; 8 players / 3 rounds and 16 players / 5 rounds have no
   rematches; 4 players / 3 rounds is a perfect round robin; 4 players / 4
   rounds still pairs everyone with the fewest rematches; after round 1 with
   8 players (4 at 3 MP, 4 at 0), every round-2 match stays in its group;
   6 players with an odd group give exactly one pair-down; 5 players: the
   bye goes to a player with 0 byes and the lowest standing, and over 5
   rounds nobody gets a second bye while a bye-less player is available; a
   1-player manual pod (anchored) gets filled; adaptive default with no
   `pairing_rounds` uses `Pairing1v1` every Swiss round. In
   `tests/test_performance.py` (performance marker): 256 players pair in
   under 5 s.

### Phase 5: top cut (spec §4, §5)

1. `TournamentConfiguration.TopCut` and `Round.Stage`: add `TOP_2 = 2`,
   `TOP_8 = 8`; `Stage.is_playoff` = `stage != SWISS`.
2. `Mtg1v1Ruleset.PLAYOFFS`, and validation rule 5 (no tied playoff match).
3. `PairingBracket`, `bracket_seed_order` (5.4).
4. `LOG_FORMAT_VERSION = "1.2"`; warn only for unknown versions (known:
   `"1.0"`, `"1.1"`, `"1.2"`). Schema `stageOrTopCut` enum
   `[0, 2, 4, 7, 8, 10, 13, 16, 40]`. Spec doc: a "1.1 -> 1.2" section and
   the enum table (stage = players in contention; allowed values per
   ruleset).
5. **Cut uses eligible players only** (`Round.disable_topcut`). Today it
   disables `standings[stage.value:]`, so a player who dropped or was
   disabled before the cut still takes a top-N slot. Spec §5: no
   replacement *after* a cut, which implies the next player advances when
   someone drops *before* it. Filter to active players only for the first
   playoff round (previous round is Swiss):
   ```python
   if self.tour.previous_round(self).stage == Round.Stage.SWISS:
       standings = [p for p in standings if p in self.active_players]
   ```
   Do not filter in later playoff rounds (a QF winner who drops must not
   pull the best QF loser into the SF). This also changes Commander: it is
   a deliberate fix. Add a test in `tests/test_topcut.py`: drop a top-4
   player after the last Swiss round; the 5th player advances.
6. Tests: 16 players, `n_rounds=4`, `top_cut=8`: stage sequence
   TOP_8 -> TOP_4 -> TOP_2, then `create_pairings()` returns False; QF pods
   are exactly {s1,s8}, {s4,s5}, {s2,s7}, {s3,s6}, higher seed first; forced
   results (s8 beats s1, others to the higher seed) give SF {s8,s4} and
   {s2,s3} (no reseeding); `report_draw` and a tied `report_match` raise in
   playoffs; Bo5 final via `playoff_rounds[2]` (`report_win` records `3-0`,
   `2-1` accepted); a player drops after the QF is paired but before the SF:
   their SF opponent gets a BYE, no 9th seed is pulled in; final standings
   `[0]` champion, `[1]` finalist; save and load between playoff rounds; top
   cuts 2 and 16 work; Commander rejects `top_cut=8`, MTG rejects
   `top_cut=7`. Existing Commander top-cut tests pass unchanged.

### Phase 6: documentation

- `docs/tournament-log-spec.md`: new "Rulesets" section (what a ruleset
  decides; `CommanderRuleset`; `Mtg1v1Ruleset`: `games_to_win` per round,
  report rules, match-outcome rule, playoff plans, and a short pointer to
  `src/logic/mtg/mtr-1v1-spec.md` §2-3 for the tiebreaker formulas, which
  readers must recompute because standings are not stored; do not copy the
  formulas); pairing section (`Pairing1v1` weighted matching,
  `PairingBracket`); new example `valid-mtg.json` (Bo3 results including a
  drawn game, a time-called `1-0`, an ID `0-0-3`, a bye, a top 4 with a Bo5
  final; generate from a real run, then trim); the doc fixes in section 12.
- `CONTEXT.md`: rewrite "Game vs. match" (whole-match reports that replace;
  `games_to_win` is a per-round `Mtg1v1Ruleset` param); new entries
  "Ruleset" (fixed once pairings exist; owns match rules, standings,
  playoffs, Swiss defaults, seat balancing), "Stage" (value = players in
  contention; same value, different pod layouts per ruleset), "Match
  report"; update "IPairingLogic / IScoringLogic" to three extension points
  with the trap "tiebreakers go on the ruleset, never on scoring or on
  `Player`"; the import-the-module example must stay valid (`Pairing1v1` no
  longer subclasses `PairingDefault`); sidecars also apply to rulesets.
- `CLAUDE.md`: `rules.py` in Modules; the ruleset in the data-flow diagram;
  convention "a new game = `src/logic/<game>/rules.py` with an `IRuleset`
  (`IS_COMPLETE = True`), plus matching/scoring as needed".
- `.claude/skills/add-pairing-scoring-algorithm/SKILL.md`: a new game
  directory needs `rules.py`; match results and tiebreakers are ruleset
  work.
- Tell the user (do not edit it) that the format bump needs a
  `CHANGELOG.md` entry under the log spec's governance rule.

### Phase 7: verification

1. `PYTHONPATH=. pytest` and `PYTHONPATH=. pytest -m slow`: green.
2. `PYTHONPATH=. pytest tests/test_performance.py -m performance` before
   Phase 1 and after Phase 6; investigate a Commander slowdown above 20%
   (`Pod._result` now goes through `tour.ruleset`).
3. `pyright`: no new errors in touched files (compare with the count before
   Phase 1). `pre-commit run --all-files`.
4. End-to-end `tests/test_mtg.py::TestEndToEnd`: 13 players,
   `Mtg1v1Ruleset`, 4 Swiss rounds, `top_cut=8`, Bo5 final via
   `playoff_rounds`, seeded `random_results`, `TournamentAction.store` /
   `load` to a temp file after round 2 and after the final. Check: every
   report validates, no rematches, nobody has two byes, each bye goes to a
   lowest-group player, champion first and finalist second, the saved file
   validates against the schema and reloads to the same standings.
5. Commander regression: snapshot diff from Phase 1 still empty; `git diff
   tests/` shows only the Phase 1 step 6 edits and the Phase 5 step 5 test;
   `test_load_real_tournament_file` passes when its fixture exists.
6. `git grep -n "games_to_win" src` matches only `src/logic/mtg/`, and
   `grep -n "def ranking\|PairingTop\|PairingSnake\|PairingDefault" src/core.py`
   finds nothing (only the generic `PairingRandom` fallback may remain).

---

## 7. MTR tiebreakers (spec §2-3)

Over Swiss rounds only, from round 0 up to and including `tour_round` (stop
at the first non-Swiss round), with `Player.result(r)`:

| Result | Match points | Rounds | Games / game points | Opponents |
|---|---|---|---|---|
| BYE | `bye_points` | +1 | +2 games, +6 (a 2-0 win, spec §2) | none |
| WIN / DRAW / LOSS | win / draw / 0 points (scoring sidecar) | +1 | if seated in a decided pod: per game +1 game; +3 if won alone, +1 if drawn, 0 if lost | the pod's other player, once per round (a list: a forced rematch counts twice) |
| LOSS from `game_loss` with no decided pod | 0 | +1 | none | none |
| PENDING (unassigned, dropped, pending pod) | skipped | | | |

```
FLOOR = 0.33                                   # as written in the MTR, not 1/3
pct(points, possible) = max(FLOOR, points / possible) if possible else FLOOR
MW  = pct(MP, 3 * rounds)          GW  = pct(GP, 3 * games)
OMW = mean(MW of each opponent)    OGW = mean(GW of each opponent)   # FLOOR if no opponents
```
Each opponent's MW uses that opponent's own rounds played, so a dropped
opponent is not diluted. Sort key, descending: `(rating, OMW, GW, OGW,
-uid.int)`. Use `Fraction(33, 100)` and `Fraction` arithmetic for exact
comparisons; round only for display. Keep `pct` and `mean_pct` module-level
so the spec §8 cases test them without a tournament.

**Vector A** (4 players, 2 rounds, `Scoring1v1`). R1: A beats B `2-0`, C
beats D `2-1`. R2: A beats C `2-1`, B and D draw `1-1`.

| Player | MP | MW | GW | OMW | OGW |
|---|---|---|---|---|---|
| A | 6 | 1 | 0.8 | 0.415 | 0.415 |
| C | 3 | 0.5 | 0.5 | 0.665 | 0.6 |
| B | 1 | 0.33 (raw 1/6) | 0.33 (raw 0.25) | 0.665 | 0.6 |
| D | 1 | 0.33 (raw 1/6) | 0.4 | 0.415 | 0.415 |

Expected order: A, C, B, D (B ahead of D on OMW).

---

## 8. Open decisions (defaults used above; confirm before the phase)

1. **Drop after the cut** (Phase 5): the bye goes to the dropped player's
   bracket opponent (MTR annotation). The rule text says "the highest
   ranked remaining player"; following it would mean re-bracketing.
   User input: Yes, the bye should go to the opponent.
2. **OMW / OGW with no opponents** (only byes or penalties) = 0.33 (the
   floor), not 0 (Phase 3).
   User input: Yes, do the 0.33, but this is for 1v1 only. For commander, use as is.
3. **A match loss penalty** (`game_loss`) with no decided pod counts as a
   round played with no games (Phase 3). If the player was seated, the TO
   reports the match (normally `0-2`); the penalty does not auto-report it.
   User input: I think in such cases, the player should not be put in a pod... but I am not sure how the math would work. I guess we can do it in calculations as a loss?
4. **`report_draw` shorthand for MTG** records `0-0-1`, `report_win` records
   `g-0`. Exact scores go through `report_match` (Phase 2).
   User input: Sounds good.

---

## 9. Catan fit check (design only, do not implement)

- Core, one additive change: `GameResult.points: Mapping[UUID, float] |
  None = None`, saved as an optional `"points"` object per game (additive in
  1.2).
- `src/logic/catan/rules.py` `CatanRuleset`: pod sizes (4, 3), allowed
  (3, 4); per-round params such as `victory_points_to_win`;
  `validate_report` requires points for every seated player and winners
  among the top scorers; `standings_keys` = (rating, total VP, ...);
  `PLAYOFFS` reuses Commander's multiplayer cut logics (e.g.
  `{4: ((4, "PairingTop4"),)}`); `random_report` draws random VP;
  `SEAT_BALANCING = True`.
- `src/logic/catan/scoring.py`: e.g. placement points from
  `pod.games[-1].points`.
- Pairing: `PairingRandom` / `PairingDefault` already support pods of 3-5.

---

## 10. GUI compatibility (out of scope, but do not break it)

`run_ui.py`, `tests/test_config_dialog.py` and `tests/test_cli.py` call
these; keep them working unchanged: `TournamentConfiguration(**kwargs)` (no
`ruleset` = Commander); the `TopCut` members; `get_scoring_logic`,
`get_pairing_logic`, `selectable_pairing_logics`; `logic.PARAM_SPEC`,
`DEFAULT_PARAMS`, `params(core)`; `config.pairing_logics`,
`pairing_params`, `snake_pods`; `report_win`, `report_draw`,
`random_results`, `toggle_bye`, `toggle_game_loss`, `move_player_to_pod`,
`bench_players`, `delete_pod`, `manual_pod`, `create_pairings`,
`reset_pods`, `new_round`; `Pod.reset_result`, `result`, `result_type`,
`done`; `Pod.EResult`, `Player.EResult`, the `StandingsExport` fields.

The GUI config editor builds Commander configs, so validation rule 6
rejects it for an MTG tournament that has started. That is a safe failure
until the GUI knows rulesets.

Later GUI work (not now): ruleset selector; `games_to_win` widgets per
Swiss round and playoff stage from `Mtg1v1Ruleset.PARAM_SPEC`; top-cut
choices from `ruleset.PLAYOFFS`; scoring list from discovery (today it is
hard-coded and omits `Scoring1v1`); MTG score entry (`games_from_score` +
`report_match`); `Tournament.reset_result` instead of the direct
`pod.reset_result()` call, which skips autosave; delete the duplicated
`_adaptive_pairing_default`.

---

## 11. Older MTG branches: do not merge

| Branch | Verdict |
|---|---|
| `feature/mtg-tiebreakers` (`cf9f010`) | Discard: "MOTC / OP1-3" are not MTR tiebreakers, and it changes Commander standings. |
| `origin/feature/mtg-gamepoints` (`db3131e`) | Discard: MTG rules in core; its GW ignores drawn games. |
| `origin/feature/mtg-topcut` (`2da42db`) | Superseded. |
| `origin/fm/1v1-best-of-3-default-games-to-win-d4` | Discard: `games_to_win` moves to the ruleset. |
| `origin/fm/1v1-topcut-recover-92f` (`381009e`..) | Salvage only the bracket seed order and the "filter a fixed seed order to survivors" idea (5.4 generalises it for byes). Discard the rest: decisions 1, 4, 5; its GW ignores drawn games, its OMW divides by the tournament's round count, and it changes `ScoringDefault` to 5/4. |

---

## 12. Existing issues found

Fix during this work (you touch the code or doc anyway):
1. `get_standings_str` ignores `config.standings_export.fields` (Phase 3).
2. `Tournament.inflate` warns on every older `format_version` (Phase 5).
3. `Round.disable_topcut` counts players who dropped before the cut
   (Phase 5 step 5).
4. `Player.__repr__` `-o` flag formats the method object instead of calling
   it (`core.py:2842`); `-u` has the same bug with `self.played`. Fix both
   (Phase 1).
5. `docs/tournament-log-spec.md` gives `ScoringDefault` defaults 5/4/1; the
   sidecar (source of truth, pinned by
   `tests/test_tournament.py::test_default_scoring_values`) says win 7,
   draw 1, bye 7. It also cites `tests/test_scoring_wagering.py` (the file
   is `tests/test_scoring_hareruya.py`) and a `# ponytail:` comment on
   `Tournament.rating()` that does not exist (Phase 6).

Report to the user, do not fix (needs a product decision):
6. `CONTEXT.md` says `max_byes` caps byes per player across the tournament;
   `get_pod_sizes` uses it as a per-round cap.

---

## Out of scope (say so if asked)

- Booster-draft playoff by seat (spec §4, Limited); early advance at Pro
  Tour / Worlds.
- Top cuts that are not a power of two (byes for top seeds in 1v1).
- Appendix E round recommendations, round timers, extra turns, play/draw
  tracking, game-by-game live entry.
- The GUI and CLI flags.

# MTG 1v1 Tournament Rules - Implementation Spec

Condensed from the Magic Tournament Rules (MTR), effective Feb 27, 2026,
last updated Jul 30, 2026. Only the rules that affect pairing, scoring,
standings, and result entry are here. The full annotated text is in
[`docs/mtr-annotated.md`](../../../docs/mtr-annotated.md). The source is
<https://blogs.magicjudges.org/rules/mtr/>.

Section numbers are MTR sections. Items marked **(not MTR)** are our own
choices, where the MTR says nothing.

## 1. Match structure (2.1, 2.4, 2.5)

- A match is played until one player wins a set number of games, usually
  **2**. Drawn games do **not** count toward that number. It is not "best
  of 3": draws can cause a 4th or 5th game.
- The TO may change the number of games to win for any part of the
  tournament. They must announce it before the tournament starts (for
  example, finals played to 3 wins).
- If the round ends before a player reaches the required number of wins,
  the player with more game wins wins the match. Equal game wins
  (0-0, 1-1, 1-1-1, …) = **match draw**.
- A game that is not complete after the end-of-round extra turns is a
  **drawn game**. No new game starts after time is called.
- **Single-elimination matches cannot end in a draw.** If game wins are
  tied after the extra turns, the player with the higher life total wins
  the current game. The result entry must reject a draw in a playoff match.
- Concessions and intentional draws are allowed until the result is
  recorded:
  - If the conceding player won a game, report the match as **2-1**.
  - An intentional draw with no games played is reported as **0-0-3**.
    This is three drawn games, so each player gets 3 game points out of
    3 games.
  - A player who refuses to play has conceded the match.
- Only match results go to Wizards. We still need game results, because
  two tiebreakers use them.

Result format: `W-L-D` in games, from the reporting player's point of view.

## 2. Points (Appendix C)

| | Match points | Game points |
|---|---|---|
| Win | 3 | 3 per game won |
| Draw | 1 | 1 per game drawn |
| Loss | 0 | 0 per game lost |
| Bye | 3 (counts as a 2-0 win) | 6 (2 games won) |

- An unfinished game is a draw (1 game point each).
- An unplayed game is worth 0 game points and is not a game played.
- Examples: 2-0-0 → 6 / 0 · 2-1-0 → 6 / 3 · 2-0-1 → 7 / 1.

## 3. Standings and tiebreakers (3.1, Appendix C)

Rank by, in order:

1. Match points
2. Opponents' match-win % (OMW)
3. Game-win % (GW)
4. Opponents' game-win % (OGW)

Some tiebreakers may not apply to formats with single-game matches (3.1).

Definitions:

```
MW(p)  = max(0.33, match_points(p) / (3 × rounds_played(p)))
GW(p)  = max(0.33, game_points(p)  / (3 × games_played(p)))
OMW(p) = mean(MW(o) for each opponent o of p)
OGW(p) = mean(GW(o) for each opponent o of p)
```

- The floor is **0.33** as written in the MTR. It is not 1/3.
- `rounds_played(o)` is the rounds **that opponent** played, including
  their byes. It is not the tournament's round count. If a player drops
  after 4 of 8 rounds, the denominator is 4 × 3.
- A bye counts toward the player's own MW and GW (3 match points,
  6 game points, 2 games, 1 round).
- A player's byes are **ignored** in their own OMW and OGW. A bye is not an
  opponent and is not in the mean (the divisor is the number of real
  opponents).
- The floor is applied to each opponent's MW or GW **before** averaging.

**(not MTR)** If everything above is equal, the order is undefined. Use a
stable random tiebreak that is seeded per tournament.

## 4. Pairing (10.4, 2.10, 7.6)

- The default is **Swiss**. The MTR does not specify the Swiss algorithm.
  See 4.1.
- An optional single-elimination playoff (top 2, 4, 8, or other) follows
  the Swiss rounds. It is seeded by the final Swiss standings.
- **Top 8:** QF 1v8, 4v5, 2v7, 3v6. SF (1/8 winner) v (4/5 winner),
  (2/7 winner) v (3/6 winner). Then the final.
- **Top 4:** SF 1v4, 2v3. Then the final.
- **Top 8 with booster-draft playoff (Limited):** players get random seats
  at the draft table. QF by seat: 1v5, 2v6, 3v7, 4v8. SF (1/5) v (3/7),
  (2/6) v (4/8).
- In a playoff, the **higher-seeded** player chooses play or draw for
  game 1. In Swiss, a random method decides who chooses. After each game,
  the loser of that game chooses. After a drawn game, the same player
  chooses again (2.2).
- Booster Draft: players may play only against players in their own draft
  pod. At Regular REL, the TO may lift this rule (7.6).
- Pro Tour, Limited Championship, and Worlds only: a player who reaches
  the announced match points can advance to the playoff before the Swiss
  rounds end. **(not modelled)**

### 4.1 Swiss pairing algorithm (not MTR)

Source: [Swiss-system tournament](https://en.wikipedia.org/wiki/Swiss-system_tournament)
on Wikipedia. The article does not mention Magic.

Common rules for all Swiss variants:

- Fix the number of rounds in advance. Pair each round only after all
  results of the previous round are in.
- Round 1 is random, or follows a seeding pattern.
- Later rounds: sort players by score. Pair players with the same or a
  similar score.
- Two players never meet twice.
- If the player count is odd, one player gets a **bye**, scored as a win.
  A player never gets a second bye.
- Rank by score after the last round. Break ties with tiebreakers.
- `ceil(log2(N))` rounds leave one undefeated player (no draws). Fewer
  rounds leave two or more players with a perfect score.

Variants. They differ in how players inside a score group are matched:

| System | Within a score group (sorted by rating or tiebreakers) |
|---|---|
| Dutch (FIDE default) | Top half v bottom half: with 8 players, 1v5, 2v6, 3v7, 4v8 |
| Burstein | Fold: 1v6, 2v5, 3v4 |
| Monrad | Sort all players (not groups), then 1v2, 3v4, … and skip rematches |
| Danish | Monrad, but rematches are allowed |
| Accelerated | Rounds 1–2 only: top half gets +1 virtual point for pairing |

Wikipedia does not say what Magic uses. Our choice for Magic, not
verified against Wizards' software:

1. Round 1: random pairing.
2. Group players by match points. Shuffle each group (Magic has no
   ratings, so the order within a group is random, as in Monrad with
   random start numbers).
3. Pair from the highest group down. If a group has an odd player left,
   that player **pairs down** into the next lower group.
4. Never pair a rematch. If you cannot avoid one, pair down further.
5. With an odd field, the bye goes to the **lowest-ranked player who has
   not had a bye**.

## 5. Drops (2.10)

- A player can drop at any time. The drop affects the **next** pairing
  only if it is reported before that pairing is generated.
- If a player drops before round 1, they did not participate and are not
  in the final standings.
- A player who does not show up for their match is dropped, unless they
  report to the scorekeeper.
- If a player drops after a cut, **no one replaces them**. In single
  elimination, the highest-ranked remaining player gets a bye instead.
- After a cut, a dropped player cannot come back in. Before a cut, the
  Head Judge decides.

## 6. Tournament size and rounds (10.1, 10.2, Appendix E)

- A rated tournament needs at least **4 players** and at least
  **3 rounds**.
- Announce the number of rounds at or before the start of round 1. After
  that, you cannot change it. A variable number with clear stop rules is
  allowed.
- Appendix E. This is required for Premier events and optional for other
  events:

| Players | Swiss rounds | Playoff |
|---|---|---|
| 5–8 | none - 3 single-elimination rounds | - |
| 9–16 | 5 (4 if Limited with a booster-draft playoff) | Top 4 (Top 8 if booster-draft playoff) |
| 17–32 | 5 | Top 8 |
| 33–64 | 6 | Top 8 |
| 65–128 | 7 | Top 8 |
| 129–226 | 8 | Top 8 |
| 227–409 | 9 | Top 8 |
| 410+ | 10 | Top 8 |

  - 4 players/teams applies to team and 2HG only: 2 single-elimination
    rounds.
  - Awarded byes change the count: a player with a 1-round bye counts as
    2 players, a 2-round bye as 4, and a 3-round bye as 8.

## 7. Time limits (Appendix B, 2.4, 2.6)

- The minimum time for any match is 40 minutes. The recommended time for
  a Swiss round is **50 minutes**. For a QF or SF, it is 90 minutes. A
  final has no time limit.
- When time is called, the current turn ends. Then players take **5 extra
  turns** in total (3 for team and 2HG). Slow-play warnings add more
  turns.
- A judge gives a time extension for a pause of more than 1 minute. For
  a deck check, the extension is the check time + 3 minutes.

## 8. Verification cases

```python
# Match points (Appendix C)
assert 4*3 + 2*0 + 2*1 == 14          # 4-2-2
assert 6*3 == 18                      # 6-2-0

# MW floor and per-player rounds (Appendix C)
assert round(16 / (8*3), 3) == 0.667  # 5-2-1 over 8 rounds
assert max(0.33, 3 / (4*3)) == 0.33   # 1-3-0 then drops → 0.25 raised
assert 9 / (5*3) == 0.60              # 3-2-0 incl. a bye, then drops

# GW (Appendix C)
assert 21 / (3*10) == 0.70            # 2-0, 2-1, 1-2, 2-0
assert max(0.33, 9 / (3*11)) == 0.33  # 1-2, 1-2, 0-2, 1-2 → 0.27 raised

# OMW, player 6-2-0 in an 8-round event. The denominators use each
# opponent's own rounds played: 1-3-1 played 5 rounds, 3-3-1 played 7.
mw = lambda pts, rounds: max(0.33, pts / (3 * rounds))
opps = [mw(12, 8), mw(21, 8), mw(4, 5), mw(10, 7),
        mw(18, 8), mw(16, 8), mw(13, 8), mw(19, 8)]
assert round(sum(opps) / 8, 2) == 0.62
# Same record, but round 1 was a bye: the bye is dropped from the mean.
assert round(sum(opps[1:]) / 7, 2) == 0.63
```

"""Catan pairing: the preliminary schedule (rule 1.1.6) and the game-4
serpentine (rule 1.1.7). Rules: https://catanevents.com/hosts/tournament-rules/.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from uuid import UUID

from typing_extensions import override

from ...core import Log
from ...interface import IPlayer, IPod, IRound
from ..common.pairing import CommonPairing
from .rules import structure_warnings

# Rule 1.1.6: the organizer makes "a schedule for three games".
PLAN_GAMES = 3

Table = list[UUID]
Schedule = list[list[Table]]  # games -> tables -> uids in play order


def _field(tour_round: IRound) -> frozenset[UUID]:
    """Players seated in a round's pods."""
    return frozenset(uid for pod in tour_round.pods for uid in pod._players)  # type: ignore[attr-defined]


def _history(rounds: Sequence[IRound]) -> Schedule:
    return [[list(pod._players) for pod in r.pods] for r in rounds]  # type: ignore[attr-defined]


def plan_seed(tour_uid: UUID, field: frozenset[UUID], first_game: int) -> int:
    """Stable seed from the tournament, the plan's first game and its field."""
    text = "|".join([str(tour_uid), str(first_game), *sorted(map(str, field))])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class PairingCatanPrelim(CommonPairing):
    """Result-blind preliminary pairing (rule 1.1.6).

    The first game of a plan schedules all the remaining preliminary games
    (games 1-3) at once, ignoring results: different opponents first, then
    rotating play order. Friends and family are not kept apart; the
    organizer swaps players by hand. Each round seats its own game from the
    plan, which is recomputed from a seed (tournament uid, the plan's first
    game and its field), so nothing extra is saved.

    A plan holds while the field stays the same and every round is seated
    as planned. When the field changes (a drop, a late player), or the
    organizer changes a round's seating by hand (a swap, pods filled by
    hand), the games after that change are re-planned for the players
    still in, with the actual earlier seating as history. A round after the
    preliminary schedule is planned on its own.

    Play order is planned with the tables: each player gets the positions
    they have held least often (exact counts, no win-rate weights).

    Pairing game 1 of a CatanRuleset event logs a warning when the player
    count or structure differs from the rules (rules.structure_warnings).
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (3, 4)
    RULESETS = ("CatanRuleset",)

    # Local-search budget: swap attempts per player per planned game.
    _STEPS_PER_PLAYER = 300

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        tour = tour_round.tour
        if tour_round.seq == 0 and tour.config.ruleset == "CatanRuleset":  # type: ignore[attr-defined]
            for msg in structure_warnings(tour):
                Log.log(msg, level=Log.Level.WARNING)
        byes = self.assign_byes(tour_round, players, pods)
        by_uid = {p.uid: p for p in players - byes}
        if any(len(pod) for pod in pods) or sum(p.cap for p in pods) != len(by_uid):
            # ponytail: pods pre-filled by hand get one greedy game, not a
            # plan; later rounds re-plan with it as history.
            self._greedy(tour_round, sorted(by_uid.values(), key=lambda p: str(p.uid)), pods)
            return players

        free = sorted(pods, key=lambda pod: -pod.cap)
        game = self.plan(tour_round, frozenset(by_uid), [pod.cap for pod in free])[0]
        for table, pod in zip(game, free):
            for uid in table:
                pod.add_player(by_uid[uid])  # type: ignore[arg-type]
        return players

    def plan(
        self, tour_round: IRound, field: frozenset[UUID], sizes: Sequence[int]
    ) -> Schedule:
        """The plan from this round to the end of the preliminary schedule.

        Its first game is this round's, with tables in `sizes` order. The
        plan starts at the earliest round whose field, and every later
        round's, equals `field`, and from which every earlier round was
        seated as the plan says; rounds before it are fixed history.
        """
        tour = tour_round.tour
        rounds = tour.rounds  # type: ignore[attr-defined]
        seq = tour_round.seq
        start = seq
        if seq < PLAN_GAMES:
            while start > 0 and _field(rounds[start - 1]) == field:
                start -= 1
        while True:
            rng = random.Random(plan_seed(tour.uid, field, start))  # type: ignore[attr-defined]
            schedule = self._schedule(
                rounds[start],
                sorted(field, key=str),
                sizes,
                max(PLAN_GAMES, seq + 1) - start,
                _history(rounds[:start]),
                rng,
            )
            seated = _history(rounds[start:seq])
            deviated = [
                k
                for k, (game, planned) in enumerate(zip(seated, schedule))
                if sorted(map(tuple, game)) != sorted(map(tuple, planned))
            ]
            if not deviated:
                return schedule[seq - start :]
            start += deviated[0] + 1

    def _schedule(
        self,
        tour_round: IRound,
        field: list[UUID],
        sizes: Sequence[int],
        n_games: int,
        history: Schedule,
        rng: random.Random,
    ) -> Schedule:
        """Plans n_games games, tables and play order, all together.

        Costs, as in CommonPairing.evaluate_pod: a pair's k-th meeting adds
        (k-1) ** rematch_penalty_exponent, and a player's k-th seat at a
        table smaller than the preferred pod size adds
        small_pod_penalty * (k-1). They outweigh play order (rule 1.1.6
        priority order), where a player's k-th time in the same position
        adds k-1. Hill climbing swaps two seats of one game whenever that
        does not raise the cost.
        """
        exponent = self._param(tour_round, "rematch_penalty_exponent")
        small_penalty = self._param(tour_round, "small_pod_penalty")
        preferred = tour_round.tour.config.pod_sizes[0]  # type: ignore[attr-defined]
        n = len(field)
        index = {uid: i for i, uid in enumerate(field)}
        meets = [[0] * n for _ in range(n)]
        small = [0] * n
        width = max([*sizes, *(len(t) for game in history for t in game)])
        held = [[0] * width for _ in range(n)]  # held[x][s]: times x played s-th

        def seat(table: Sequence[int], size: int, sign: int) -> None:
            for a, b in itertools.combinations(table, 2):
                meets[a][b] += sign
                meets[b][a] += sign
            if size < preferred:
                for a in table:
                    small[a] += sign

        for game in history:
            for table in game:
                known = [index[u] for u in table if u in index]
                seat(known, len(table), 1)
                for s, u in enumerate(table):
                    if u in index:
                        held[index[u]][s] += 1

        games: list[list[list[int]]] = []
        for _ in range(n_games):
            order = list(range(n))
            rng.shuffle(order)
            it = iter(order)
            game = [[next(it) for _ in range(size)] for size in sizes]
            for table, size in zip(game, sizes):
                seat(table, size, 1)
                for s, x in enumerate(table):
                    held[x][s] += 1
            games.append(game)

        def table_cost(x: int, others: Sequence[int], size: int) -> int:
            """Cost x adds by sitting with `others` at a table of `size`
            (x's current meetings with them already counted)."""
            row = meets[x]
            cost = sum((row[y] - 1) ** exponent for y in others if row[y] > 1)
            if size < preferred:
                cost += small_penalty * (small[x] - 1)
            return cost

        def swap_tables(A: list[int], ia: int, B: list[int], ib: int) -> int:
            """Swaps A[ia] and B[ib] between tables; returns the cost change."""
            a, b = A[ia], B[ib]
            A_rest, B_rest = A[:ia] + A[ia + 1 :], B[:ib] + B[ib + 1 :]
            before = table_cost(a, A_rest, len(A)) + table_cost(b, B_rest, len(B))
            seat(A, len(A), -1)
            seat(B, len(B), -1)
            A[ia], B[ib] = b, a
            seat(A, len(A), 1)
            seat(B, len(B), 1)
            return table_cost(a, B_rest, len(B)) + table_cost(b, A_rest, len(A)) - before

        def swap_positions(a: int, sa: int, b: int, sb: int) -> int:
            """Moves a from position sa to sb and b back; returns the cost change."""
            before = held[a][sa] + held[b][sb]
            held[a][sa] -= 1
            held[b][sb] -= 1
            held[a][sb] += 1
            held[b][sa] += 1
            return held[a][sb] + held[b][sa] - before

        # Tables outrank play order: no position gain pays for a repeat.
        weight = 100
        for _ in range(self._STEPS_PER_PLAYER * n * n_games):
            game = rng.choice(games)
            ta, tb = rng.randrange(len(game)), rng.randrange(len(game))
            A, B = game[ta], game[tb]
            ia, ib = rng.randrange(len(A)), rng.randrange(len(B))
            if ta == tb and ia == ib:
                continue
            a, b = A[ia], B[ib]
            d = swap_positions(a, ia, b, ib) if ia != ib else 0
            if ta != tb:
                d += weight * swap_tables(A, ia, B, ib)
            else:
                A[ia], A[ib] = b, a
            if d > 0:  # undo
                if ta != tb:
                    swap_tables(A, ia, B, ib)
                else:
                    A[ia], A[ib] = a, b
                if ia != ib:
                    swap_positions(b, ia, a, ib)

        return [[[field[x] for x in table] for table in game] for game in games]

    def _greedy(
        self, tour_round: IRound, players: list[IPlayer], pods: Sequence[IPod]
    ) -> None:
        """One game without a plan: CommonPairing.evaluate_pod, then play
        order by least-held position."""
        for p in players:
            scores = [self.evaluate_pod(p, pod, tour_round) for pod in pods]
            pods[scores.index(max(scores))].add_player(p)
        rounds = tour_round.tour.rounds[: tour_round.seq]  # type: ignore[attr-defined]
        seats: Counter[tuple[UUID, int]] = Counter()
        for game in _history(rounds):
            for table in game:
                seats.update((uid, i) for i, uid in enumerate(table))
        for pod in pods:
            pod._players[:] = least_held_order(pod._players, seats)  # type: ignore[attr-defined]


def least_held_order(
    table: Sequence[UUID], seats: Mapping[tuple[UUID, int], int]
) -> Table:
    """Play order giving each player the positions they held least often.

    seats[(uid, i)] counts how often uid played in position i (0 = first).
    Minimizes the summed counts of the assigned positions, then the largest
    single count; ties go to uid order.
    """
    table = sorted(table, key=str)
    best = min(
        itertools.permutations(table),
        key=lambda order: (
            sum(seats.get((uid, i), 0) for i, uid in enumerate(order)),
            max(seats.get((uid, i), 0) for i, uid in enumerate(order)),
        ),
    )
    return list(best)


class PairingCatanSerpentine(CommonPairing):
    """Game-4 seating for large events (rule 1.1.7).

    Ranks the players by standings, puts the 3-player tables first and
    snakes the ranks across the tables. Seat order is rank order: the order
    players pick color, chair and play order.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (3, 4)
    RULESETS = ("CatanRuleset",)

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        byes = self.assign_byes(tour_round, players, pods)
        prev = tour_round.tour.previous_round(tour_round)  # type: ignore[attr-defined]
        rank = {p: i for i, p in enumerate(tour_round.tour.get_standings(prev))}  # type: ignore[attr-defined]
        ranked = sorted(players - byes, key=lambda p: rank.get(p, len(rank)))
        tables = sorted(pods, key=lambda pod: pod.cap)  # 3-player tables first
        order = [
            pod
            for lap in range(max((pod.cap for pod in tables), default=0))
            for pod in (tables if lap % 2 == 0 else tables[::-1])
            if pod.cap - len(pod) > lap
        ]
        for p, pod in zip(ranked, order):
            pod.add_player(p)
        return players

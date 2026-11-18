from __future__ import annotations
from typing import Any

from ..common.pairing import CommonPairing
from ...interface import IPlayer, IPod, IRound


class Pairing1v1(CommonPairing):
    """Swiss pairing for 1v1 (2-player pod) Magic tournaments.

    A maximum-weight perfect matching (networkx), not Commander's greedy
    pod-fill: spec 4.1's rules (the bye to a fewest-byes, fewest-points
    player, no rematch when avoidable, fewest and smallest pair-downs)
    are exact priorities a greedy fill can't guarantee, but a matching
    can. For speed on large fields, round 1 pairs by seeded shuffle and
    later rounds match one score group at a time, falling back to the
    whole-field matching on a rematch or a pair-down past the next group.
    The shared CommonPairing base supplies rating helpers. This
    algorithm has no tunable parameters, so it ships no sidecar.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)

    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: list[IPod]
    ) -> set[IPlayer]:
        # Imported here, not at module level, so a missing networkx install
        # only breaks MTG Swiss pairing, not discovery of the whole mtg
        # package (matching.py/scoring.py are imported wholesale to find
        # every IS_COMPLETE class in them).
        import networkx as nx

        ratings = self.field_ratings(tour_round)
        tour = tour_round.tour
        standings = tour.get_standings(tour_round)  # pyright: ignore[attr-defined]

        anchored_pods = [pod for pod in pods if len(pod.players) == 1]
        anchored_players = [pod.players[0] for pod in anchored_pods]
        empty_pods = [pod for pod in pods if len(pod.players) == 0]
        free_players = list(players)

        capacity = 2 * len(empty_pods) + len(anchored_pods)
        n_byes = len(free_players) - capacity
        if n_byes < 0:
            raise ValueError(
                f"No valid 1v1 pairing: {len(free_players)} unpaired players "
                f"exceed the available pod capacity ({capacity})."
            )

        player_nodes = free_players + anchored_players
        free_set = set(free_players)
        anchored_set = set(anchored_players)
        participating = free_set | anchored_set
        ordered_players = [p for p in standings if p in participating]

        # g(p): p's score group - the index of its rating among the
        # distinct ratings of every player node, highest first.
        node_rating = {
            p: p.rating(tour_round, ratings) for p in player_nodes  # pyright: ignore[reportCallIssue]
        }
        distinct_ratings = sorted(set(node_rating.values()), reverse=True)
        group_index = {r: i for i, r in enumerate(distinct_ratings)}
        g = {p: group_index[node_rating[p]] for p in player_nodes}
        n_groups = len(distinct_ratings)

        rank_from_bottom = {
            p: len(ordered_players) - 1 - i for i, p in enumerate(ordered_players)
        }

        def met(a: IPlayer, b: IPlayer) -> int:
            return sum(
                1
                for pod in a.games(tour_round)
                if b in pod.players  # pyright: ignore[reportAttributeAccessIssue]
            )

        # Integer weights, so the priorities below are strictly
        # lexicographic (networkx.max_weight_matching is exact with ints):
        # (1) the bye to a fewest-byes player, then to the lowest score
        # group among them, (2) no rematch, (3) smallest total squared
        # score gap (pair-down as little as possible), (4) bye to the
        # lowest-ranked player, (5) random among equals.
        n_players = len(player_nodes)
        n_edges = (n_players + n_byes) // 2
        jitter_range = 1000
        unit = jitter_range * (n_edges + 1)
        gap_unit = n_byes * n_players + 1
        rematch_unit = gap_unit * (n_edges + 1) * (n_groups**2 + 1)

        rng = tour_round.rng("jitter")

        def pair_cost(a: IPlayer, b: IPlayer) -> int:
            return rematch_unit * met(a, b) + gap_unit * (g[a] - g[b]) ** 2

        # The bye outranks everything else: (a) bye count above the
        # minimum among free players, (b) score group below the highest
        # among the fewest-byes players. Each unit exceeds the sum of every
        # cost below it, so a rematch beats a bye for a non-candidate.
        # ponytail: loose bound (sum of all pair costs); fine at 1v1 sizes.
        pair_total = sum(
            pair_cost(a, b)
            for i, a in enumerate(ordered_players)
            for b in ordered_players[i + 1 :]
            if not (a in anchored_set and b in anchored_set)
        )
        pts_unit = pair_total + n_players + 1
        bye_unit = pts_unit * (n_groups + 1)
        min_byes = min((a.byes(tour_round) for a in free_players), default=0)
        g_cand = max(
            (g[a] for a in free_players if a.byes(tour_round) == min_byes),
            default=0,
        )

        def weight(cost: int) -> int:
            return -cost * unit + rng.randrange(jitter_range)

        bye_nodes = [f"__bye_{i}__" for i in range(n_byes)]
        # The graph's nodes are indices into `nodes`: max_weight_matching
        # hashes nodes O(P^3) times, and an int hashes much faster than a
        # player (whose hash is a Python method).
        nodes: list[Any] = [*ordered_players, *bye_nodes]
        n_ordered = len(ordered_players)
        expected = set(ordered_players) | set(bye_nodes)

        def solve() -> list[tuple[Any, Any]]:
            graph: Any = nx.Graph()
            graph.add_nodes_from(range(len(nodes)))
            for i, a in enumerate(ordered_players):
                for j in range(i + 1, n_ordered):
                    b = ordered_players[j]
                    if a in anchored_set and b in anchored_set:
                        continue
                    graph.add_edge(i, j, weight=weight(pair_cost(a, b)))
                if a in free_set:
                    for k in range(n_ordered, len(nodes)):
                        cost = (
                            bye_unit * (a.byes(tour_round) - min_byes)
                            + pts_unit * abs(g_cand - g[a])
                            + rank_from_bottom[a]
                        )
                        graph.add_edge(i, k, weight=weight(cost))
            return [
                (nodes[u], nodes[v])
                for u, v in nx.max_weight_matching(graph, maxcardinality=True)
            ]

        def solve_by_group() -> list[tuple[Any, Any]]:
            # Spec 4.1: the bye is fixed first (it outranks everything), then
            # each score group pairs inside itself, top down; an odd group
            # pairs one player down into the next one. Many small matchings
            # instead of one O(P^3) matching over the whole field.
            field = list(ordered_players)
            pairs: list[tuple[Any, Any]] = []
            if n_byes:
                bye_player = min(
                    (
                        a
                        for a in free_players
                        if a.byes(tour_round) == min_byes and g[a] == g_cand
                    ),
                    key=rank_from_bottom.__getitem__,
                )
                field.remove(bye_player)
                pairs.append((bye_player, bye_nodes[0]))
            groups: dict[int, list[IPlayer]] = {}
            for a in field:
                groups.setdefault(g[a], []).append(a)
            keys = sorted(groups)
            carry: list[IPlayer] = []
            for idx, k in enumerate(keys):
                pool = carry + groups[k]
                carry = []
                graph: Any = nx.Graph()
                graph.add_nodes_from(range(len(pool)))
                for i, a in enumerate(pool):
                    for j in range(i + 1, len(pool)):
                        graph.add_edge(
                            i, j, weight=weight(pair_cost(a, pool[j]))
                        )
                if len(pool) % 2:  # the player who pairs down
                    for i, a in enumerate(pool):
                        gap = (keys[idx + 1] - g[a]) ** 2
                        graph.add_edge(i, len(pool), weight=weight(gap_unit * gap))
                for u, v in nx.max_weight_matching(graph, maxcardinality=True):
                    if len(pool) in (u, v):
                        carry.append(pool[v if u == len(pool) else u])
                    else:
                        pairs.append((pool[u], pool[v]))
            return pairs

        simple = not anchored_pods and n_byes <= 1
        if tour_round.seq == 0 and n_groups == 1 and simple:
            # Round 1: one score group and no history, so every pairing is
            # equally good. The bye goes to the lowest-ranked player, as the
            # matching would; the rest pair at random.
            pool = list(ordered_players)
            matching: list[tuple[Any, Any]] = (
                [(pool.pop(), bye_nodes[0])] if n_byes else []
            )
            rng.shuffle(pool)
            matching += zip(pool[::2], pool[1::2])
        else:
            matching = solve_by_group() if simple else []
            # ponytail: group-wise is greedy top down. A rematch, or a
            # pair-down past the next group, may have a better fix that
            # only the full graph sees (spec 4.1), so that field solves
            # both. With every gap <= 1, one pair crosses each odd
            # boundary, which is already the minimum total squared gap.
            if not matching or any(
                met(u, v) or abs(g[u] - g[v]) > 1
                for u, v in matching
                if u in g and v in g
            ):
                matching = solve()
        if {n for pair in matching for n in pair} != expected:
            raise ValueError(
                "No valid 1v1 pairing: the field could not be fully matched."
            )

        bye_node_set = set(bye_nodes)
        new_pairs: list[tuple[IPlayer, IPlayer]] = []
        for u, v in matching:
            if u in bye_node_set or v in bye_node_set:
                bye_player = v if u in bye_node_set else u
                bye_player.set_result(tour_round, IPlayer.EResult.BYE)
            elif u in anchored_set or v in anchored_set:
                anchored_p, free_p = (u, v) if u in anchored_set else (v, u)
                pod = next(
                    p for p in anchored_pods if p.players and p.players[0] == anchored_p
                )
                pod.add_player(free_p)
            else:
                new_pairs.append((u, v))

        standings_index = {p: i for i, p in enumerate(standings)}
        new_pairs.sort(
            key=lambda pair: min(
                standings_index.get(pair[0], len(standings)),
                standings_index.get(pair[1], len(standings)),
            )
        )
        for pod, (u, v) in zip(empty_pods, new_pairs):
            better, worse = (
                (u, v)
                if standings_index.get(u, len(standings))
                < standings_index.get(v, len(standings))
                else (v, u)
            )
            pod.add_player(better)
            pod.add_player(worse)

        return players


def bracket_seed_order(n: int) -> list[int]:
    """Seeds in bracket order, e.g. 8 -> [1, 8, 4, 5, 2, 7, 3, 6] (MTR 10.4).

    Adjacent pairs at every power-of-two sub-block are the correct
    matchups for a standard single-elimination bracket: slots[0:2] is the
    first-round match "highest seed vs lowest seed" of the top half,
    slots[0:4] is that whole quarter's eventual final, and so on.
    """
    order = [1]
    while len(order) < n:
        m = 2 * len(order)
        order = [x for s in order for x in (s, m + 1 - s)]
    return order


class PairingBracket(CommonPairing):
    """Single-elimination playoff bracket (spec 4, MTR 10.4).

    Chosen automatically per playoff stage by Mtg1v1Ruleset.PLAYOFFS, never
    offered to the user (SELECTABLE = False) - so it ships no sidecar. The
    bracket is seeded once, at the cut, and never reseeded: every stage
    recomputes the same seed list from the final Swiss standings, filtered
    to players who were still active when the first playoff round was
    created (a player who dropped or was disabled before the cut does not
    occupy a bracket slot - see Round.disable_topcut).
    """

    IS_COMPLETE = True
    SELECTABLE = False
    SUPPORTED_POD_SIZES = (2,)

    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: list[IPod]
    ) -> set[IPlayer]:
        tour = tour_round.tour
        n = int(tour.config.top_cut)  # pyright: ignore[reportAttributeAccessIssue]
        s = tour_round.stage.value  # pyright: ignore[reportAttributeAccessIssue]

        final_swiss = tour.final_swiss_round  # pyright: ignore[reportAttributeAccessIssue]
        assert final_swiss is not None  # a playoff round always follows Swiss
        # final_swiss.active_players, not the first playoff round's: once
        # playoffs begin, final_swiss is never t.tour_round again, so
        # nothing can add a later drop to its _dropped set - it stays a
        # true snapshot of who was still active right when the cut was
        # made. The first playoff round's own _dropped, by contrast, keeps
        # growing every time someone drops while it is the current round
        # (e.g. after the QF is paired but before the SF), which would
        # otherwise shrink the seed list on a later PairingBracket call and
        # break the fixed bracket.
        eligible_at_cut = final_swiss.active_players  # pyright: ignore[reportAttributeAccessIssue]
        full_standings = tour.get_standings(final_swiss)  # pyright: ignore[reportAttributeAccessIssue]
        seeds = [p for p in full_standings if p in eligible_at_cut][:n]
        # A seed slot beyond the actual field (fewer players than top_cut)
        # has no player - treated as already eliminated, same as a dropped
        # seed: its bracket opponent gets a walkover bye below.
        slots = [
            seeds[i - 1] if i - 1 < len(seeds) else None
            for i in bracket_seed_order(n)
        ]
        seed_rank = {p: i for i, p in enumerate(seeds)}

        block = 2 * n // s
        half = block // 2
        matches: list[tuple[IPlayer, IPlayer] | IPlayer] = []
        for block_start in range(0, n, block):
            left = [
                p for p in slots[block_start : block_start + half] if p in players
            ]
            right = [
                p
                for p in slots[block_start + half : block_start + block]
                if p in players
            ]
            if len(left) > 1 or len(right) > 1:
                raise ValueError(
                    "No valid bracket pairing: more than one surviving "
                    "player on one side of a bracket block."
                )
            if left and right:
                a, b = left[0], right[0]
                # Higher seed (lower seed_rank) first - seat order is
                # informational only (spec 4: who chooses play/draw).
                matches.append((a, b) if seed_rank[a] < seed_rank[b] else (b, a))
            elif left or right:
                matches.append(left[0] if left else right[0])
            # Neither present (both eliminated or dropped): no match.

        pod_iter = iter(pods)
        used_pods: list[IPod] = []
        for match in matches:
            if isinstance(match, tuple):
                try:
                    pod = next(pod_iter)
                except StopIteration:
                    raise ValueError(
                        "No valid bracket pairing: more matches than pods."
                    ) from None
                used_pods.append(pod)
                a, b = match
                pod.add_player(a)
                pod.add_player(b)
            else:
                match.set_result(tour_round, IPlayer.EResult.BYE)

        for pod in pods:
            if pod not in used_pods:
                tour_round.remove_pod(pod)

        return players

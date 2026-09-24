from __future__ import annotations
import random
from typing import Any

from ..commander import matching as _commander_matching
from ...interface import IPlayer, IPod, IRound


class Pairing1v1(_commander_matching.CommonPairing):
    """Swiss pairing for 1v1 (2-player pod) Magic tournaments.

    A maximum-weight perfect matching (networkx), not Commander's greedy
    pod-fill: spec 4.1's rules (no rematch when avoidable, fewest and
    smallest pair-downs, the bye to the lowest-ranked player without one)
    are exact priorities a greedy fill can't guarantee, but a matching
    can. Subclasses CommonPairing directly (not commander's PairingDefault
    - see the CommonPairing import note in matching.py's own module
    docstring pattern): this algorithm doesn't reuse PairingDefault's sort
    key or pod-fit scoring, so it ships no sidecar of its own (no
    rematch_penalty_exponent / small_pod_penalty - those are
    PairingDefault-specific).
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)

    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: list[IPod]
    ) -> set[IPlayer]:
        # ponytail: complete graph, O(P^3); 256 players ~2.4 s. If large
        # events need it, keep only edges within +-2 score groups and fall
        # back to the full graph when the matching is not perfect.
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
        g_low = max((g[p] for p in free_players), default=0)

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
        # (1) no rematch and no second bye, (2) smallest total squared
        # score gap (pair-down as little as possible), (3) bye to the
        # lowest-ranked player without a bye, (4) random among equals.
        n_players = len(player_nodes)
        n_edges = (n_players + n_byes) // 2
        jitter_range = 1000
        unit = jitter_range * (n_edges + 1)
        gap_unit = n_byes * n_players + 1
        rematch_unit = gap_unit * (n_edges + 1) * (n_groups**2 + 1)

        def weight(cost: int) -> int:
            return -cost * unit + random.randrange(jitter_range)

        graph: Any = nx.Graph()
        bye_nodes = [f"__bye_{i}__" for i in range(n_byes)]
        graph.add_nodes_from(ordered_players)
        graph.add_nodes_from(bye_nodes)

        for i, a in enumerate(ordered_players):
            for b in ordered_players[i + 1 :]:
                if a in anchored_set and b in anchored_set:
                    continue
                cost = rematch_unit * met(a, b) + gap_unit * (g[a] - g[b]) ** 2
                graph.add_edge(a, b, weight=weight(cost))
            if a in free_set:
                for bye_node in bye_nodes:
                    cost = (
                        rematch_unit * a.byes(tour_round)
                        + gap_unit * (g_low - g[a]) ** 2
                        + rank_from_bottom[a]
                    )
                    graph.add_edge(a, bye_node, weight=weight(cost))

        matching = nx.max_weight_matching(graph, maxcardinality=True)
        matched = {node for pair in matching for node in pair}
        if matched != set(ordered_players) | set(bye_nodes):
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

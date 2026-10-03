from __future__ import annotations
from typing import Any, Callable
from collections.abc import Mapping, Sequence

from ...interface import IPlayer, IPod, IRound
from ..common.pairing import CommonPairing

from typing_extensions import override
import random


class PairingSnake(CommonPairing):
    IS_COMPLETE: bool = True
    SUPPORTED_POD_SIZES = (3, 4, 5)

    # Snake pods logic for 2nd tour_round
    # First bucket is players with most points and least unique opponents
    # Players are then distributed in buckets based on points and unique opponents
    # Players are then distributed in pods based on bucket order

    def snake_ranking(
        self,
        player: IPlayer,
        tour_round: IRound,
        ratings: Mapping[Any, float] | None = None,
    ) -> tuple[float, int]:
        """Helper method to get snake ranking for a player."""
        return (player.rating(tour_round, ratings), -len(player.played(tour_round)))

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: list[IPod]
    ) -> set[IPlayer]:
        prev_round: IRound = tour_round.tour.rounds[tour_round.seq - 1]
        ratings = self.field_ratings(tour_round)
        byes = self.assign_byes(tour_round, players, pods, ratings)
        active_players = tour_round.active_players - byes

        snake_ranking: Callable[[IPlayer], tuple[float, int]] = (
            lambda x: self.snake_ranking(x, tour_round, ratings)
        )

        # 1. Determine Buckets
        # Map: ranking_key -> List[Player]
        buckets: dict[tuple[float, int], list[IPlayer]] = {}
        for p in active_players:
            rank = snake_ranking(p)
            if rank not in buckets:
                buckets[rank] = []
            buckets[rank].append(p)

        # Sort bucket keys (best to worst)
        bucket_order = sorted(buckets.keys(), reverse=True)

        # Shuffle players within buckets for randomness
        for b in buckets.values():
            random.shuffle(b)

        # 2. Pre-process Candidates
        # Structure: bucket_key -> prev_pod_id -> List[Player]
        # This allows O(1) lookup of available players from a specific previous pod in a specific bucket.
        candidates: dict[tuple[float, int], dict[IPod | None, list[IPlayer]]] = {}

        # Also need a map to find which prev_pod a player was in
        player_prev_pod: dict[IPlayer, IPod | None] = {}

        for p in active_players:
            prev_pods = p.pods(prev_round)
            p_prev_pod = (
                prev_pods[-1] if prev_pods and isinstance(prev_pods[-1], IPod) else None
            )
            player_prev_pod[p] = p_prev_pod

            rank = snake_ranking(p)
            if rank not in candidates:
                candidates[rank] = {}
            if p_prev_pod not in candidates[rank]:
                candidates[rank][p_prev_pod] = []
            candidates[rank][p_prev_pod].append(p)

        # 3. State Tracking
        # forbidden_prev_pods[current_pod_index] = Set[prev_pod_id]
        # Tracks which previous pods are already represented in the current pod
        forbidden_prev_pods: list[set[IPod | None]] = [set() for _ in pods]

        # 4. Distribution Loop
        # We fill pods one seat at a time, iterating through pods in a round-robin fashion.
        # But wait, looking at the original logic, it tried to fill "bucket by bucket".
        # The original logic:
        # iterate buckets (best to worst)
        #   iterate players in bucket
        #     find valid pod (starting from pod_index 0)

        # Optimized Logic to match original intent (prioritize filling with best players):

        current_pod_idx = 0
        n_pods = len(pods)

        for bucket_key in bucket_order:
            # We must place ALL players in this bucket before moving to the next bucket.
            # But we can pick ANY player from this bucket that fits.

            # The 'bucket' list in original code was just a flat list of players.
            # Here we have them grouped by prev_pod in 'candidates[bucket_key]'.

            players_in_bucket_count = len(buckets[bucket_key])

            while players_in_bucket_count > 0:
                start_pod_idx = current_pod_idx
                placed = False

                # Try to place a player in the current_pod (or next ones)
                for i in range(n_pods):
                    pod_idx = (start_pod_idx + i) % n_pods
                    pod = pods[pod_idx]

                    if len(pod.players) >= pod.cap:
                        continue

                    # Find a player in this bucket whose prev_pod is NOT in forbidden_prev_pods[pod_idx]
                    # We iterate through available prev_pods in this bucket
                    found_prev_pod = None

                    # Optimization: Iterate through the keys of candidates[bucket_key]
                    # This is much smaller than iterating all players.
                    # We can also shuffle the keys to avoid bias if needed, but the players inside are already shuffled.
                    # To ensure randomness in *which* compatible group we pick, we can shuffle the keys or just iterate?
                    # Iterating keys is fine if we shuffled players. BUT if we always pick the first valid key,
                    # we might bias towards certain previous pods.
                    # Let's create a list of available prev_pods for this bucket and shuffle it?
                    # That might be too expensive to do every time.
                    # But the number of distinct prev_pods is small (N/4).

                    matches = list(candidates[bucket_key].keys())
                    # random.shuffle(matches) # Optional: Adds more randomness but costs time.

                    for prev_pod in matches:
                        if prev_pod not in forbidden_prev_pods[pod_idx]:
                            players_list = candidates[bucket_key][prev_pod]
                            if players_list:
                                # FOUND A MATCH
                                player = players_list.pop()
                                if not players_list:
                                    del candidates[bucket_key][prev_pod]

                                pod.add_player(player)
                                forbidden_prev_pods[pod_idx].add(prev_pod)
                                placed = True
                                players_in_bucket_count -= 1

                                # Update current_pod_idx to next one for fair distribution
                                current_pod_idx = (pod_idx + 1) % n_pods
                                break

                    if placed:
                        break

                if not placed:
                    # If we simply cannot place players respecting the constraint, we must relax it.
                    # The original code might have raised ValueError or implicitly relaxed?
                    # Original: "No pod can accept player" -> ValueError.
                    # BUT: Original code had a fallback? No...
                    # Actually, if capacity allows, we MUST place them.
                    # If we are here, it means for ALL available pods, all available players in this bucket create a collision.
                    # WE MUST FALLBACK to allowing collision.

                    # Fallback Strategy: Just pick the first available player for the first available pod.
                    # Find any pod with space
                    fallback_placed = False
                    for i in range(n_pods):
                        pod_idx = (current_pod_idx + i) % n_pods
                        pod = pods[pod_idx]
                        if len(pod.players) < pod.cap:
                            # Pick any player from this bucket
                            # Get first available group
                            if not candidates[bucket_key]:
                                raise ValueError(
                                    "Bucket logic error: count > 0 but no candidates."
                                )

                            prev_pod = next(iter(candidates[bucket_key]))
                            players_list = candidates[bucket_key][prev_pod]
                            player = players_list.pop()
                            if not players_list:
                                del candidates[bucket_key][prev_pod]

                            pod.add_player(player)
                            # We don't add to forbidden because it's a collision anyway
                            fallback_placed = True
                            players_in_bucket_count -= 1
                            current_pod_idx = (pod_idx + 1) % n_pods
                            break

                    if not fallback_placed:
                        raise ValueError(
                            "Critical failure: No pod has capacity left but players remain."
                        )

        return players


class PairingDefault(CommonPairing):
    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (3, 4, 5)

    def matching(
        self,
        player: IPlayer,
        tour_round: IRound,
        ratings: Mapping[Any, float] | None = None,
    ) -> tuple:
        return (
            -len(player.games(tour_round)),
            -len(player.played(tour_round)),
            player.rating(tour_round, ratings),
            player.opponent_pointrate(tour_round, ratings),
        )

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        # Compute the field ratings once and thread them through both sort
        # keys, so pairing does not recompute the whole field per player.
        ratings = self.field_ratings(tour_round)
        matching = lambda x: self.matching(x, tour_round, ratings)

        byes = self.assign_byes(tour_round, players, pods, ratings)

        active_players = players - byes

        assignment_order = sorted(active_players, key=matching, reverse=True)
        for i, p in enumerate(assignment_order):
            pod_scores = [self.evaluate_pod(p, pod, tour_round) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)
        return players


class PairingTopCut(CommonPairing):
    """Commander top-cut pairing for every stage of CommanderRuleset.PLAYOFFS.

    The cut size is the round's stage (Round.Stage value), never a user
    setting. The TOP_4 final seats all four finalists in one pod; larger
    cuts give the top seeds byes (_BYES) and snake-seat the rest.
    """

    IS_COMPLETE = True
    SELECTABLE = False  # top-cut pairing, chosen automatically by stage

    # Seeded byes per cut size.
    _BYES = {4: 0, 7: 3, 10: 2, 13: 1, 16: 0, 40: 8}

    # The per-cut classes this one replaced, still named in saved logs.
    ALIASES = (
        "PairingTop4", "PairingTop7", "PairingTop10",
        "PairingTop13", "PairingTop16", "PairingTop40",
    )

    @override
    def advance_topcut(self, tour_round: IRound, standings: list[IPlayer]) -> None:
        n_byes = self._BYES[tour_round.stage.value]  # pyright: ignore[reportAttributeAccessIssue]
        for p in standings[:n_byes]:
            p.set_result(tour_round, IPlayer.EResult.BYE)

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        if tour_round.stage.value == 4:  # pyright: ignore[reportAttributeAccessIssue]
            prev_round = tour_round.tour.previous_round(tour_round)  # pyright: ignore[reportAttributeAccessIssue]
            standings = tour_round.tour.get_standings(prev_round)
            for p in sorted(tour_round.active_players, key=standings.index):
                pods[0].add_player(p)
            return players

        standings = tour_round.tour.get_standings(tour_round)
        assignable_players = sorted(
            (tour_round.active_players - set(tour_round.byes)),
            key=standings.index,
        )
        n_pods = len(pods)
        for i, p in enumerate(assignable_players):
            pass_num = i // n_pods
            pos = i % n_pods
            pod_idx = pos if pass_num % 2 == 0 else (n_pods - 1 - pos)
            pods[pod_idx].add_player(p)
        return players

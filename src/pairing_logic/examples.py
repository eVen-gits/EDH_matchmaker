from __future__ import annotations
import time
from typing import Sequence, Callable

from ..interface import IPlayer, IPod, ITournament, IRound, IPairingLogic

from typing_extensions import override
import random
import sys
import numpy as np
from ..core import timeit


class CommonPairing(IPairingLogic):
    def __init__(self, name: str):
        super().__init__(name)

    def evaluate_pod(self, player: IPlayer, pod: IPod, tour_round: IRound) -> int:
        score = 0
        if len(pod) == pod.cap:
            return -sys.maxsize
        for p in pod.players:
            score -= player.played(tour_round).count(p) ** 2
        if pod.cap < player.tour.config.max_pod_size:
            for prev_pod in player.pods(tour_round):
                if isinstance(prev_pod, IPod):
                    score -= sum(
                        [
                            10
                            for _ in prev_pod.players
                            if prev_pod.cap < player.tour.config.max_pod_size
                        ]
                    )
        return score

    def make_pairings(self, players: list[IPlayer], pods: list[IPod]) -> set[IPlayer]:
        raise NotImplementedError(
            "PairingLogic.make_pairings not implemented - use subclass"
        )

    def bye_matching(self, player: IPlayer, tour_round: IRound) -> tuple:
        return (
            -len(player.games(tour_round)),
            player.rating(tour_round),
            -len(player.played(tour_round)),
        )

    def assign_byes(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        capacity = sum([pod.cap - len(pod.players) for pod in pods])
        n_byes = len(players) - capacity

        matching: Callable[[IPlayer], tuple] = lambda x: self.bye_matching(
            x, tour_round
        )
        player_matches = {p: matching(p) for p in players}
        keys: list[tuple] = sorted(set(player_matches.values()), reverse=True)

        buckets = [[p for p in players if player_matches[p] == k] for k in keys]

        byes = set()
        for b in buckets[::-1]:
            byes.update(random.sample(b, min(len(b), n_byes - len(byes))))
            if len(byes) >= n_byes:
                break
        pass
        for p in byes:
            p.set_result(tour_round, IPlayer.EResult.BYE)

        return byes


class PairingRandom(CommonPairing):
    IS_COMPLETE = True

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: list[IPod]
    ) -> set[IPlayer]:
        byes = self.assign_byes(tour_round, players, pods)
        active_players = list(players - byes)
        random.shuffle(active_players)

        player_index = 0
        for pod in pods:
            for _ in range(pod.cap - len(pod._players)):
                pod.add_player(active_players[player_index])
                player_index += 1

        return players


class PairingSnake(CommonPairing):
    IS_COMPLETE = True

    # Snake pods logic for 2nd tour_round
    # First bucket is players with most points and least unique opponents
    # Players are then distributed in buckets based on points and unique opponents
    # Players are then distributed in pods based on bucket order

    def snake_ranking(self, player: IPlayer, tour_round: IRound) -> tuple[float, int]:
        """Helper method to get snake ranking for a player."""
        return (player.rating(tour_round), -len(player.played(tour_round)))

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: list[IPod]
    ) -> set[IPlayer]:
        prev_round: IRound = tour_round.tour.rounds[tour_round.seq - 1]
        byes = self.assign_byes(tour_round, players, pods)
        active_players = tour_round.active_players - byes

        snake_ranking: Callable[[IPlayer], tuple[float, int]] = (
            lambda x: self.snake_ranking(x, tour_round)
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

    def matching(self, player: IPlayer, tour_round: IRound) -> tuple:
        return (
            -len(player.games(tour_round)),
            -len(player.played(tour_round)),
            player.rating(tour_round),
            player.opponent_pointrate(tour_round),
        )

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        # matching = lambda x: (
        #    -len(x.games(tour_round)),
        #    -len(x.played(tour_round)),
        #    x.rating(tour_round),
        #    x.opponent_pointrate(tour_round)
        # )
        # standings = tour_round.tour.get_standings(tour_round)
        matching = lambda x: self.matching(x, tour_round)

        byes = self.assign_byes(tour_round, players, pods)

        active_players = players - byes

        assignment_order = sorted(active_players, key=matching, reverse=True)
        for i, p in enumerate(assignment_order):
            pod_scores = [self.evaluate_pod(p, pod, tour_round) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)
        return players


class PairingTop4(CommonPairing):
    IS_COMPLETE = True

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        standings = tour_round.tour.get_standings(tour_round)
        assignable_players = sorted(
            tour_round.active_players, key=lambda x: standings.index(x)
        )

        # Distribute players across pods in snake order
        pod_index = 0
        forward = True

        for p in assignable_players:
            # Add player to current pod
            pods[pod_index].add_player(p)

            # Move to next pod
            if forward:
                pod_index += 1
                if pod_index >= len(pods):
                    pod_index = len(pods) - 1
                    forward = False
            else:
                pod_index -= 1
                if pod_index < 0:
                    pod_index = 0
                    forward = True

        return players


class PairingSemiCommon(CommonPairing):
    N_BYES = -1

    @classmethod
    def advance_topcut(cls, tour_round: IRound, standings: list[IPlayer]) -> None:
        byes = [standings[i] for i in range(cls.N_BYES)]

        for p in byes:
            p.set_result(tour_round, IPlayer.EResult.BYE)

    @staticmethod
    def make_pairings(
        n_byes: int, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        standings = tour_round.tour.get_standings(tour_round)

        assignable_players = sorted(
            (tour_round.active_players - set(tour_round.byes)),
            key=lambda x: standings.index(x),
        )

        n_pods = len(pods)
        for i, p in enumerate(assignable_players):
            pods[i % n_pods].add_player(p)
        return players


class PairingTop7(PairingSemiCommon):
    IS_COMPLETE = True
    N_BYES = 3

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        players = PairingSemiCommon.make_pairings(
            self.N_BYES, tour_round=tour_round, players=players, pods=pods
        )
        return players


class PairingTop10(PairingSemiCommon):
    IS_COMPLETE = True
    N_BYES = 2

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        """standings = tour_round.tour.get_standings(tour_round)

        p1 = standings[0]
        p2 = standings[1]
        p1.set_result(tour_round, IPlayer.EResult.BYE)
        p2.set_result(tour_round, IPlayer.EResult.BYE)

        assignable_players = sorted((tour_round.active_players - tour_round.byes), key=lambda x: standings.index(x))

        for i,p in enumerate(assignable_players):
            pods[i%2].add_player(p)

        return players"""
        players = PairingSemiCommon.make_pairings(
            self.N_BYES, tour_round=tour_round, players=players, pods=pods
        )

        return players


class PairingTop13(PairingSemiCommon):
    IS_COMPLETE = True
    N_BYES = 1

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        players = PairingSemiCommon.make_pairings(
            self.N_BYES, tour_round=tour_round, players=players, pods=pods
        )
        return players


class PairingTop16(PairingSemiCommon):
    IS_COMPLETE = True
    N_BYES = 0

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        players = PairingSemiCommon.make_pairings(
            self.N_BYES, tour_round=tour_round, players=players, pods=pods
        )
        return players


class PairingTop40(PairingSemiCommon):
    IS_COMPLETE = True
    N_BYES = 16

    @override
    def make_pairings(
        self, tour_round: IRound, players: set[IPlayer], pods: Sequence[IPod]
    ) -> set[IPlayer]:
        players = PairingSemiCommon.make_pairings(
            self.N_BYES, tour_round=tour_round, players=players, pods=pods
        )
        return players

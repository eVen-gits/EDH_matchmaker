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

    def evaluate_pod(self, player: IPlayer, pod:IPod, tour_round: IRound) -> int:
        score = 0
        if len(pod) == pod.cap:
            return -sys.maxsize
        for p in pod.players:
            score -= player.played(tour_round).count(p) ** 2
        if pod.cap < player.tour.config.max_pod_size:
            for prev_pod in player.pods(tour_round):
                if isinstance(prev_pod, IPod):
                    score -= sum([
                        10
                        for _
                        in prev_pod.players
                        if prev_pod.cap < player.tour.config.max_pod_size
                    ])
        return score

    def make_pairings(self, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        raise NotImplementedError('PairingLogic.make_pairings not implemented - use subclass')

    def bye_matching(self, player: IPlayer, tour_round: IRound) -> tuple:
        return (
            -len(player.games(tour_round)),
            player.rating(tour_round),
            -len(player.played(tour_round))
        )

    def assign_byes(self, tour_round: IRound, players: Sequence[IPlayer], pods: Sequence[IPod]) -> list[IPlayer]:
        capacity = sum([pod.cap - len(pod.players) for pod in pods])
        n_byes = len(players) - capacity

        matching: Callable[[IPlayer], tuple] = lambda x: self.bye_matching(x, tour_round)
        player_matches = {p: matching(p) for p in players}
        keys: list[tuple] = sorted(set(player_matches.values()), reverse=True)

        buckets = [
            [
                p for p in players
                if player_matches[p] == k
            ]
            for k in keys
        ]


        byes = []
        for b in buckets[::-1]:
            byes += random.sample(b, min(len(b), n_byes-len(byes)))
            if len(byes) >= n_byes:
                break
        pass
        for p in byes:
            p.set_result(tour_round, IPlayer.EResult.BYE)

        return byes

class PairingRandom(CommonPairing):
    IS_COMPLETE=True

    @override
    def make_pairings(self, tour_round: IRound, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        byes = self.assign_byes(tour_round, players, pods)
        active_players = [p for p in players if p not in byes]
        random.shuffle(active_players)

        player_index = 0
        for pod in pods:
            for _ in range(pod.cap - len(pod._players)):
                pod.add_player(active_players[player_index])
                player_index += 1

        return players

class PairingSnake(CommonPairing):
    IS_COMPLETE=True

    #Snake pods logic for 2nd tour_round
    #First bucket is players with most points and least unique opponents
    #Players are then distributed in buckets based on points and unique opponents
    #Players are then distributed in pods based on bucket order

    def optimize_bucket_assignments(self, tour_round: IRound, buckets: dict, pods: list[IPod]) -> None:
        """
        Optimize player assignments within each bucket by swapping players between pods
        to improve overall pod evaluation scores.
        """
        for bucket_key, bucket_players in buckets.items():
            if len(bucket_players) <= 1:
                continue  # Can't swap with only one player

            # Find which pods contain players from this bucket
            bucket_pods = []
            for pod in pods:
                pod_bucket_players = [p for p in pod.players if any(
                    self.snake_ranking(p, tour_round) == bucket_key
                )]
                if pod_bucket_players:
                    bucket_pods.append((pod, pod_bucket_players))

            if len(bucket_pods) <= 1:
                continue  # Can't swap if all players are in the same pod

            # Try to improve assignments by swapping players
            improved = True
            while improved:
                improved = False

                for i, (pod1, players1) in enumerate(bucket_pods):
                    for j, (pod2, players2) in enumerate(bucket_pods):
                        if i >= j:
                            continue  # Avoid duplicate comparisons

                        # Try swapping each pair of players
                        for p1 in players1:
                            for p2 in players2:
                                # Calculate current scores
                                current_score1 = self.evaluate_pod(p1, pod1, tour_round)
                                current_score2 = self.evaluate_pod(p2, pod2, tour_round)
                                current_total = current_score1 + current_score2

                                # Calculate scores after swap
                                # Temporarily remove players
                                pod1.remove_player(p1)
                                pod2.remove_player(p2)

                                # Calculate new scores
                                new_score1 = self.evaluate_pod(p2, pod1, tour_round)
                                new_score2 = self.evaluate_pod(p1, pod2, tour_round)
                                new_total = new_score1 + new_score2

                                # If swap improves total score, keep it
                                if new_total > current_total:
                                    # Add players to their new pods
                                    pod1.add_player(p2)
                                    pod2.add_player(p1)
                                    improved = True
                                    break
                                else:
                                    # Revert the swap
                                    pod1.add_player(p1)
                                    pod2.add_player(p2)

                            if improved:
                                break

                    if improved:
                        break

    def snake_ranking(self, player: IPlayer, tour_round: IRound) -> tuple[float, int]:
        """Helper method to get snake ranking for a player."""
        return (player.rating(tour_round), -len(player.played(tour_round)))

    def exclusory_pod_buckets(self, pods, prev_round, bucket_order, buckets):
        prev_pods = set(pod for pod in prev_round.pods)

        current_pods_filtered = [{key:[] for key in bucket_order} for _ in range(len(pods))]
        for i, pod in enumerate(pods):
            current_players_pods = set([ip.pods(prev_round)[-1] for ip in pod.players])
            for prev_pod in prev_pods:
                if prev_pod in current_players_pods:
                    continue
                #current_pods_filtered[i].append([
                #    p for p in prev_pod.players
                #])
                for key in bucket_order:
                    for p in buckets[key]:
                        if p.pods(prev_round)[-1] == prev_pod or not isinstance(p.pods(prev_round)[-1], IPod):
                            current_pods_filtered[i][key].append(p)




        #exclusory_pod_budckets = {
        #    i: {
        #        key: [
        #            p for p in players
        #            if p.pods(prev_round)[-1] != pod
        #            and snake_ranking(p) == key
        #        ]
        #        for key in bucket_order
        #    }
        #    for i, pod in enumerate(prev_round.pods)
        #}
        return current_pods_filtered

    @override
    def make_pairings(self, tour_round: IRound, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        prev_round: IRound = tour_round.tour.rounds[tour_round.seq-1]
        byes = self.assign_byes(tour_round, players, pods)
        active_players = [p for p in players if p not in byes]

        snake_ranking: Callable[[IPlayer], tuple[float, int]] = lambda x: self.snake_ranking(x, tour_round)
        active_players = sorted(active_players, key=snake_ranking, reverse=True)

        # Create buckets based on ranking
        bucket_order = sorted(
            list(set(
                [snake_ranking(p) for p in active_players]
            )), reverse=True)

        buckets = {
            k: [
                p for p in active_players
                if snake_ranking(p) == k
            ]
            for k in bucket_order
        }

        # Shuffle players within each bucket
        for b in buckets.values():
            random.shuffle(b)

        # Distribute players in snake order (always forward, restart from beginning)
        pod_index = 0

        #for bucket_key in bucket_order:
        while True:
            bucket_key = bucket_order[0]
            for bucket_key in bucket_order:
                if buckets[bucket_key]:
                    bucket = buckets[bucket_key]
                    break
            else:
                if any(len(b) > 0 for b in buckets.values()):
                    raise ValueError('Can not satisfy bucketing requirements')
                break


            # Reset pod_index to 0 for each new bucket
            pod_index = 0

            while bucket:
                exclusory_pod_buckets = self.exclusory_pod_buckets(pods, prev_round, bucket_order, buckets)
                # Try to add player to current pod
                current_pod = pods[pod_index]

                if len(current_pod.players) < current_pod.cap and len(exclusory_pod_buckets[pod_index][bucket_key]) > 0:
                    # Current pod has space, add player
                    #bucket.pop(0)
                    player = exclusory_pod_buckets[pod_index][bucket_key][0]
                    buckets[bucket_key].remove(player)
                    current_pod.add_player(player)
                else:
                    # Current pod is full, find next available pod
                    attempts = 0
                    while attempts < len(pods):
                        pod_index = (pod_index + 1) % len(pods)  # Move to next pod, wrap around
                        current_pod = pods[pod_index]

                        if len(current_pod.players) < current_pod.cap:
                            if len(exclusory_pod_buckets[pod_index][bucket_key]) > 0:
                                player = exclusory_pod_buckets[pod_index][bucket_key][0]
                                buckets[bucket_key].remove(player)
                                current_pod.add_player(player)
                                break
                            elif len(bucket) > 0:
                                # No player found in exclusory pod buckets, add player to current pod
                                #TODO: Add player to current pod, disregarding exclusory restriction
                                player = bucket[0]
                                buckets[bucket_key].remove(player)
                                current_pod.add_player(player)
                                break
                            else:
                                raise ValueError('No player found in bucket')

                        attempts += 1

                    if attempts >= len(pods):
                        # No pod can accept this player - this shouldn't happen if capacity is correct
                        raise ValueError(f'No pod can accept player {player.name}')

                # Move to next pod for next player
                pod_index = (pod_index + 1) % len(pods)
                pass

        # Optimize assignments within each bucket
        #self.optimize_bucket_assignments(tour_round, buckets, pods)

        return players

class PairingDefault(CommonPairing):
    IS_COMPLETE=True

    def matching(self, player: IPlayer, tour_round: IRound) -> tuple:
        return (
            -len(player.games(tour_round)),
            -len(player.played(tour_round)),
            player.rating(tour_round),
            player.opponent_pointrate(tour_round)
        )

    @override
    def make_pairings(self, tour_round: IRound, players: Sequence[IPlayer], pods: Sequence[IPod]) -> Sequence[IPlayer]:
        #matching = lambda x: (
        #    -len(x.games(tour_round)),
        #    -len(x.played(tour_round)),
        #    x.rating(tour_round),
        #    x.opponent_pointrate(tour_round)
        #)
        #standings = tour_round.tour.get_standings(tour_round)
        matching = lambda x: self.matching(x, tour_round)

        byes = self.assign_byes(tour_round, players, pods)

        active_players = [p for p in players if p not in byes]

        assignment_order = sorted(active_players, key=matching, reverse=True)
        for i, p in enumerate(assignment_order):
            pod_scores = [self.evaluate_pod(p, pod, tour_round) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)
        return players

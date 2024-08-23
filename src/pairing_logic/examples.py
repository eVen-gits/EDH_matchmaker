from __future__ import annotations

from ..interface import IPlayer, IPod, ITournament, IRound, IPairingLogic

from typing_extensions import override
import random


class PairingRandom(IPairingLogic):
    @override
    def make_pairings(self, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        random.shuffle(players)
        for pod in pods:
            for _ in range(pod.cap):
                pod.add_player(players.pop(0))

        return players

class PairingSnake(IPairingLogic):
    #Snake pods logic for 2nd round
    #First bucket is players with most points and least unique opponents
    #Players are then distributed in buckets based on points and unique opponents
    #Players are then distributed in pods based on bucket order
    #elif self.tour.TC.snake_pods and self.seq == 1:
    @override
    def make_pairings(self, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        pod_sizes = [pod.cap for pod in pods]
        bye_count = len(players) - sum(pod_sizes)
        snake_ranking = lambda x: (x.points, -x.unique_opponents)
        players = sorted(players, key=snake_ranking, reverse=True)
        bucket_order = sorted(
            list(set(
                [snake_ranking(p) for p in players]
            )), reverse=True)
        buckets = {
            k: [
                p for p in players
                if snake_ranking(p) == k
            ]
            for k in bucket_order
        }
        for b in buckets.values():
            random.shuffle(b)
        i = 0
        for order_idx, b in enumerate(bucket_order):
            if (
                order_idx == 0  # if not first bucket
                # and not same points as previous bucket
                or b[0] != bucket_order[order_idx-1][0]
            ):
                i = 0
            bucket = buckets[b]
            p = bucket[i]
            while len(bucket) > 0:
                #check if all pods full
                if sum(pod_sizes) == sum(len(pod_x.players) for pod_x in pods):
                    break
                ok = False

                if b == bucket_order[-1] and p in buckets[b][-1:-bye_count-1:-1]:
                    ok = True
                if sum(pod_sizes) == sum(len(pod_x.players) for pod_x in pods):
                    ok = True
                while not ok:
                    curr_pod = pods[i % len(pods)]
                    pod_evals = [p.evaluate_pod(curr_pod) for p in bucket]
                    index = pod_evals.index(max(pod_evals))
                    p = bucket[index]
                    ok = curr_pod.add_player(p)
                    if ok:
                        bucket.pop(index)
                    i += 1
        return players


class PairingDefault(IPairingLogic):
    @override
    def make_pairings(self, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        matching = lambda x: (
            -x.games_played,
            x.points,
            -x.unique_opponents,
            x.opponent_winrate
        )
        for p in sorted(random.sample(players, len(players)), key=matching, reverse=True):
            pod_scores = [p.evaluate_pod(pod) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)

        for pod in pods:
            pod.sort()
        return players

    #at this point, pods are created and filled with players
    #but seating order is not yet determined
    #swaps between pods need to be made first - your code here
    # Attempt to swap equivalent players between pods

    # Swapping equivalent players between pods to optimize seats
    '''if self.seq != 0:
        self.optimize_seatings()
        for pod in pods:
            pod.sort_players_by_avg_seat()
        pass'''
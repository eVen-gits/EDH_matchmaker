from __future__ import annotations
from typing import Sequence

from ..interface import IPlayer, IPod, ITournament, IRound, IPairingLogic

from typing_extensions import override
import random
import sys
import numpy as np

class CommonPairing(IPairingLogic):
    def __init__(self, name: str):
        super().__init__(name)

    def evaluate_pod(self, player: IPlayer, pod:IPod) -> int:
        score = 0
        if len(pod) == pod.cap:
            return -sys.maxsize
        for p in pod.players:
            score -= player.played.count(p) ** 2
        if pod.cap < player.tour.config.max_pod_size:
            for prev_pod in player.pods:
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

    def assign_byes(self, tour_round: IRound, players: Sequence[IPlayer], pods: Sequence[IPod]) -> list[IPlayer]:
        capacity = sum([pod.cap - len(pod.players) for pod in pods])
        n_byes = len(players) - capacity

        matching = lambda x: (
            -len(x.games),
            -len(x.played),
            x.rating(tour_round),
            x.opponent_winrate
        )
        buckets = [
            [
                p for p in players
                if matching(p) == k
            ]
            for k in sorted(set([matching(p) for p in players]), reverse=True)
        ]
        byes = []
        for b in buckets[::-1]:
            byes += random.sample(b, min(len(b), n_byes))
            if len(byes) >= n_byes:
                break

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

    @override
    def make_pairings(self, tour_round: IRound, players: list[IPlayer], pods: list[IPod]) -> list[IPlayer]:
        byes = self.assign_byes(tour_round, players, pods)
        active_players = [p for p in players if p not in byes]

        pod_sizes = [pod.cap for pod in pods]
        bye_count = len(players) - sum(pod_sizes)
        snake_ranking = lambda x: (x.rating(tour_round), -len(x.played))
        active_players = sorted(players, key=snake_ranking, reverse=True)
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
            j = 0
            bucket = buckets[b]
            p = bucket[j]
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
                    pod_evals = [self.evaluate_pod(p, curr_pod) for p in bucket]
                    index = pod_evals.index(max(pod_evals))
                    p = bucket[index]
                    ok = curr_pod.add_player(p)
                    if ok:
                        bucket.pop(index)
                    j += 1
                    i += 1
        return players

class PairingDefault(CommonPairing):
    IS_COMPLETE=True

    @override
    def make_pairings(self, tour_round: IRound, players: Sequence[IPlayer], pods: Sequence[IPod]) -> Sequence[IPlayer]:
        matching = lambda x: (
            -len(x.games),
            -len(x.played),
            x.rating(x.tour.tour_round),
            x.opponent_winrate
        )
        byes = self.assign_byes(tour_round, players, pods)

        active_players = [p for p in players if p not in byes]

        assignment_order = sorted(active_players, key=matching, reverse=True)
        for i, p in enumerate(assignment_order):
            pod_scores = [self.evaluate_pod(p, pod) for pod in pods]
            index = pod_scores.index(max(pod_scores))
            pods[index].add_player(p)

        return players

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from uuid import UUID

from ...interface import IRound, ITournament
from ..common.scoring import FixedPointsScoring


def dummy_score(scores: Sequence[int]) -> int:
    """The mean of scores, rounded half-up (rule 1.4.3: .5 up, .4 down).

    Python's round() rounds half to even, so this is done in integers.
    """
    n = len(scores)
    return (2 * sum(scores) + n) // (2 * n)


def vp_share(scores: Sequence[int], own: int) -> Fraction:
    """One game's VP share (rule 1.4.3): own VP over the table's VP total.

    A 3-player table adds the dummy fourth score to the total.
    """
    total = sum(scores) + (dummy_score(scores) if len(scores) == 3 else 0)
    return Fraction(own, total) if total else Fraction(0)


def finishing_places(scores: Mapping[Any, int]) -> dict[Any, int]:
    """Place per player by VP (1 = most); tied players share the better place."""
    return {u: 1 + sum(v > s for v in scores.values()) for u, s in scores.items()}


@dataclass
class CatanStats:
    """One player's tiebreaker inputs over the Swiss games played."""

    vp: int = 0  # total VP, capped per game
    vp_pct: Fraction = Fraction(0)  # exact sum of VP shares
    second: int = 0
    third: int = 0


def catan_stats(tour: ITournament, rounds: Iterable[IRound]) -> dict[UUID, CatanStats]:
    """Per-player VP (capped per game), VP%, 2nd and 3rd place counts over
    the games of `rounds` that have reported VP."""
    stats = {p.uid: CatanStats() for p in tour.players}
    for r in rounds:
        for pod in r.pods:
            game = pod.games[-1] if pod.games else None
            if game is None or game.scores is None:
                continue  # pending, or reported without VP
            # Looked up only for a game with VP, which only CatanRuleset has.
            cap = tour.ruleset._param(r, "victory_points")  # type: ignore[attr-defined]
            vp = dict(game.scores)
            values = list(vp.values())
            places = finishing_places(vp)
            for uid, v in vp.items():
                s = stats.setdefault(uid, CatanStats())
                s.vp += min(v, cap)
                s.vp_pct += vp_share(values, v)
                s.second += places[uid] == 2
                s.third += places[uid] == 3
    return stats


class ScoringCatan(FixedPointsScoring):
    """Wins and the official Catan tiebreakers (tournament rules 1.4.1-1.4.3).

    Rating is the win count (`win_points` per win; `draw_points` is unused,
    there are no draws). Equal ratings use, in order: total VP (at most
    `victory_points` per game), VP% (exact fractions), more 2nd places, more
    3rd places, then the core's stable UID order as the drawing of lots. A
    forced bye scores `bye_points` and adds no VP and no VP%.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (3, 4)
    RULESETS = ("CatanRuleset",)

    def stats(self, tour: ITournament, tour_round: IRound) -> dict[UUID, CatanStats]:
        """Per-player VP, VP%, 2nd and 3rd place counts through tour_round."""
        return catan_stats(tour, self._swiss_rounds_up_to(tour, tour_round))

    def standings_keys(
        self,
        tour: ITournament,
        tour_round: IRound,
        ratings: Mapping[Any, float],
    ) -> Mapping[UUID, tuple]:
        """Sort key, descending: (rating, VP, VP%, 2nds, 3rds, -uid)."""
        stats = self.stats(tour, tour_round)
        return {
            p.uid: (
                ratings.get(p.uid, 0),
                stats[p.uid].vp,
                stats[p.uid].vp_pct,
                stats[p.uid].second,
                stats[p.uid].third,
                -p.uid if isinstance(p.uid, int) else -int(p.uid.int),
            )
            for p in tour.players
        }

    def standings_columns(
        self, tour: ITournament, tour_round: IRound
    ) -> list[tuple[str, Mapping[UUID, str]]]:
        stats = self.stats(tour, tour_round)
        return [
            ("VP", {u: str(s.vp) for u, s in stats.items()}),
            ("VP%", {u: f"{float(s.vp_pct):.4f}" for u, s in stats.items()}),
            ("2nd", {u: str(s.second) for u, s in stats.items()}),
            ("3rd", {u: str(s.third) for u, s in stats.items()}),
        ]

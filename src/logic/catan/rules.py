from __future__ import annotations
import random
from collections.abc import Iterable, Sequence
from uuid import UUID

from ...core import Log, TournamentConfiguration
from ...interface import IGameResult, IPlayer, IPod, IRound, IRuleset, ITournament
from .scoring import catan_stats, finishing_places

# Rule 1.1.3: fewer players give no valid results.
MIN_PLAYERS = 12


class CatanConfiguration(TournamentConfiguration):
    """Catan's config fields, on top of the shared ones.

    Simulation only (they drive `random_report`). Fields and defaults: the
    sidecar `CatanConfiguration.params.yaml`.
    """

    seat_win_rates: Sequence[float]


def recommended_structure(n_players: int) -> tuple[int, int, dict[int, int]]:
    """Rule 1.1.5's structure for n_players: (n_rounds, top_cut, eliminations).

    eliminations maps a Swiss round's 0-based seq to its
    eliminate_below_wins. 12-28 players: 3 games, then a top-4 final (28
    follows 1.1.5, not 1.1.12's "12 to 27"). 29-79: 3 games, top-16
    semifinals, then the final. 80+: 4 games, players without a win are
    eliminated before game 4, then top-16 semifinals and the final.
    """
    if n_players <= 28:
        return 3, 4, {}
    if n_players < 80:
        return 3, 16, {}
    return 4, 16, {3: 1}


def structure_warnings(tour: ITournament) -> list[str]:
    """Warnings for a Catan event whose player count or configured structure
    differs from rules 1.1.3 and 1.1.5 (see recommended_structure)."""
    config = tour.config  # type: ignore[attr-defined]
    n = len(tour.players)
    out = []
    if n < MIN_PLAYERS:
        out.append(
            f"Catan tournament rule 1.1.3 needs at least {MIN_PLAYERS} players "
            f"for valid results; this event has {n}."
        )
    rounds, top_cut, elim = recommended_structure(n)
    actual_elim = {
        seq: v
        for seq, r in enumerate(config.pairing_rounds[: config.n_rounds])
        if (v := r.get("ruleset_params", {}).get("eliminate_below_wins", 0))
    }
    if (config.n_rounds, int(config.top_cut), actual_elim) != (rounds, top_cut, elim):
        out.append(
            f"Catan tournament rule 1.1.5 recommends for {n} players: "
            f"{_structure_str(rounds, top_cut, elim)}. This event is set to "
            f"{_structure_str(config.n_rounds, int(config.top_cut), actual_elim)}."
        )
    return out


def _structure_str(rounds: int, top_cut: int, elim: dict[int, int]) -> str:
    parts = [f"{rounds} games", f"top cut {top_cut}"]
    parts += [f"eliminate below {v} wins before game {seq + 1}" for seq, v in elim.items()]
    return ", ".join(parts)


class CatanRuleset(IRuleset):
    """Catan tournament rules (catanevents.com tournament rules, section 1).

    Every pod plays one game, ends at `victory_points` and has exactly one
    winner: there are no draws. The match report carries each player's
    victory points (`IGameResult.scores`). Save-format contracts:
    docs/tournament-log-spec.md, "`CatanRuleset`".

    Playoffs: top_cut 4 is a final of the top 4; top_cut 16 is serpentine
    semifinals of the top 16 (rule 1.1.10), then a final of the four
    semifinal winners (rule 1.1.12). Both reuse PairingTopCut by name.
    recommended_structure gives rule 1.1.5's structure for a player count.
    """

    IS_COMPLETE = True

    DEFAULT_POD_SIZES = (4, 3)
    ALLOWED_POD_SIZES = (3, 4)
    DEFAULT_SCORING_LOGIC = "ScoringCatan"
    # Play order is part of the pairing, not auto-balanced by core.
    SEAT_BALANCING = False
    # 3-player pods absorb remainders (rule 1.1.7). A forced bye scores
    # bye_points, never VP.
    DEFAULT_ALLOW_BYE = False
    # PairingTopCut is Commander's, looked up by name (no import): its TOP_16
    # snake is rule 1.1.10's semifinal seating, its TOP_4 one final pod.
    PLAYOFFS = {
        4: ((4, "PairingTopCut"),),
        16: ((16, "PairingTopCut"), (4, "PairingTopCut")),
    }
    CONFIG_CLASS = CatanConfiguration
    # (tournament uid, warnings) pairs already logged by warn_structure.
    _structure_warned: set[tuple[UUID, tuple[str, ...]]] = set()

    @classmethod
    def warn_structure(cls, tour_round: IRound) -> None:
        """Logs structure_warnings when game 1 of a CatanRuleset event is paired.

        PairingCatanPrelim and PairingCatanSerpentine call this, so it runs
        whichever of them pairs game 1. Each distinct set of warnings is
        logged once per tournament: re-pairing game 1 does not repeat it, a
        changed player count or structure does. A generic logic picked by
        hand for game 1 (e.g. PairingRandom) skips the check.
        """
        tour = tour_round.tour
        if tour_round.seq != 0 or tour.config.ruleset != cls.__name__:  # type: ignore[attr-defined]
            return
        msgs = tuple(structure_warnings(tour))
        key = (tour.uid, msgs)  # type: ignore[attr-defined]
        if not msgs or key in cls._structure_warned:
            return
        cls._structure_warned.add(key)
        for msg in msgs:
            Log.log(msg, level=Log.Level.WARNING)

    def validate_report(self, pod: IPod, games: Sequence[IGameResult]) -> None:
        """Raises ValueError unless games is a valid Catan report.

        One game; one winner; a VP score for every seated player and no
        others; the winner has at least `victory_points` (so the most VP);
        every other player is below `victory_points`.
        """
        name = pod.name  # type: ignore[attr-defined]
        if len(games) != 1:
            raise ValueError(
                f"{name}: Catan reports exactly one game per match, got {len(games)}."
            )
        game = games[0]
        seated = {p.uid for p in pod.players}
        if len(game.winners) != 1:
            raise ValueError(
                f"{name}: a Catan game has exactly one winner, got {len(game.winners)}."
            )
        (winner,) = game.winners
        if winner not in seated:
            raise ValueError(f"{name}: the winner must be seated in the pod.")
        if game.scores is None:
            raise ValueError(f"{name}: a Catan report needs victory points for every player.")
        vp = dict(game.scores)
        if len(vp) != len(game.scores) or set(vp) != seated:
            raise ValueError(
                f"{name}: victory points are needed for exactly the seated players."
            )
        if any(v < 0 for v in vp.values()):
            raise ValueError(f"{name}: victory points cannot be negative.")
        target = self._param(pod.tour_round, "victory_points")
        if vp[winner] < target:
            raise ValueError(
                f"{name}: the winner needs at least {target} victory points, "
                f"got {vp[winner]}."
            )
        for uid, v in vp.items():
            if uid != winner and v >= target:
                raise ValueError(
                    f"{name}: only the winner can have {target} or more victory points."
                )

    def match_winners(self, pod: IPod) -> frozenset[UUID]:
        return pod.games[-1].winners

    def report_from_winners(
        self, pod: IPod, winners: Iterable[UUID]
    ) -> list[IGameResult]:
        """Always refuses: the winner alone is not a Catan report.

        Victory points feed the tiebreakers, so report with
        Tournament.report_match and IGameResult.scores.
        """
        raise ValueError(
            f"{pod.name}: a Catan win needs victory points for every player; "  # type: ignore[attr-defined]
            "report the match with scores instead."
        )

    def random_report(self, pod: IPod) -> list[IGameResult]:
        """One game: the winner takes `victory_points`, the others get 2 up to
        one below it. Always a valid report."""
        config: CatanConfiguration = pod.tour_round.tour.config  # type: ignore[attr-defined]
        target = self._param(pod.tour_round, "victory_points")
        n = len(pod.players)
        winner = random.choices(
            pod.players, weights=list(config.seat_win_rates[:n])
        )[0]
        low = min(2, target - 1)
        scores = {
            p.uid: target if p is winner else random.randint(low, target - 1)
            for p in pod.players
        }
        return [IGameResult(frozenset({winner.uid}), tuple(scores.items()))]

    def swiss_pairing_logic(self, tour: ITournament, seq: int) -> str:
        """PairingCatanSerpentine on a round that eliminates players
        (`eliminate_below_wins` set for it, rule 1.1.7), else the
        preliminary schedule, PairingCatanPrelim (rule 1.1.6)."""
        rounds = tour.config.pairing_rounds  # type: ignore[attr-defined]
        overrides = rounds[seq].get("ruleset_params", {}) if seq < len(rounds) else {}
        if overrides.get("eliminate_below_wins", self.DEFAULT_PARAMS["eliminate_below_wins"]):
            return "PairingCatanSerpentine"
        return "PairingCatanPrelim"

    def swiss_eliminations(
        self, tour_round: IRound, standings: Sequence[IPlayer]
    ) -> set[IPlayer]:
        """Active players with fewer wins than this round's
        `eliminate_below_wins` (rule 1.1.5: 1 before game 4 of an 80+
        event). 0, the default, eliminates no one."""
        below = self._param(tour_round, "eliminate_below_wins")
        if not below:
            return set()
        earlier = tour_round.tour.rounds[: tour_round.seq]  # type: ignore[attr-defined]
        active = tour_round.active_players  # type: ignore[attr-defined]
        return {
            p
            for p in standings
            if p in active
            and sum(p.result(r) == IPlayer.EResult.WIN for r in earlier) < below
        }

    def playoff_standings(
        self, tour: ITournament, tour_round: IRound
    ) -> list[IPlayer] | None:
        """Rule 1.4.4: standings after a semifinal or final round.

        The round's players come first, grouped by finishing place at their
        table (after the semifinals: winners 1-4, 2nds 5-8, 3rds 9-12, 4ths
        13-16; after the final: places 1-4). Within a place, the 1.4.2
        tiebreakers (VP, VP%, 2nds, 3rds, lot) over the preliminary and
        playoff games so far decide. Everyone else keeps the previous
        round's order. Until every table of the round has reported, the
        previous round's standings stand.
        """
        base = tour.get_standings(tour.previous_round(tour_round))  # type: ignore[attr-defined]
        if not tour_round.done:  # type: ignore[attr-defined]
            return base
        place: dict[UUID, int] = {}
        for pod in tour_round.pods:
            if pod.games:
                game = pod.games[-1]
                vp = dict(game.scores or ((u, int(u in game.winners)) for u in pod._players))  # type: ignore[attr-defined]
                place.update(finishing_places(vp))
        stats = catan_stats(tour, tour.rounds[: tour_round.seq + 1])  # type: ignore[attr-defined]

        def key(uid: UUID) -> tuple:
            s = stats[uid]
            return (place[uid], -s.vp, -s.vp_pct, -s.second, -s.third, uid.int)

        by_uid = {p.uid: p for p in base}
        return [by_uid[u] for u in sorted(place, key=key)] + [
            p for p in base if p.uid not in place
        ]

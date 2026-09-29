"""Scoring selection controls standings through the tournament API."""
import pytest

from src.core import Tournament, TournamentAction, TournamentConfiguration

TournamentAction.LOGF = False


@pytest.mark.parametrize("logic", ["ScoringHareruya", "ScoringModifiedHareruya"])
def test_hareruya_points_only_and_switching(logic):
    t = Tournament(TournamentConfiguration(auto_export=False, scoring_logic=logic))
    t.add_player(["A", "B", "C", "D"])
    players = sorted(t.players, key=lambda p: p.uid.int)
    # Default scoring favors later seats, opposite to UID order here.
    t.new_round()
    t.manual_pod(players)
    t.report_draw(players)
    r = t.tour_round
    assert r is not None
    assert len(set(t.field_ratings(r).values())) == 1
    assert t.get_standings(r) == players
    assert t.get_standings(r) == players

    t.config.scoring_logic = "ScoringDefault"
    assert t.get_standings(r) == list(reversed(players))
    t.config.scoring_logic = logic
    assert t.get_standings(r) == players
    assert t.get_scoring_logic(logic).standings_columns(t, r) == []

    # Points outrank UID order, while tied losers remain in UID order.
    t.report_win(players[-1])
    assert t.get_standings(r) == [players[-1], *players[:-1]]
    saved = t.serialize()
    expected = [p.uid for p in t.get_standings(r)]
    Tournament.CACHE.clear()
    restored = Tournament.inflate(saved)
    assert [p.uid for p in restored.get_standings()] == expected

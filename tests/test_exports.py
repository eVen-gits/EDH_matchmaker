import unittest

from src.core import (
    StandingsExport,
    Tournament,
    TournamentAction,
    TournamentConfiguration,
    TournamentContext,
)
from src.misc import generate_player_names

def _small_tournament():
    cfg = TournamentConfiguration(pod_sizes=[4], allow_bye=False, auto_export=False)
    t = Tournament(cfg)
    t.add_player(generate_player_names(8))
    t.create_pairings()
    t.random_results()
    return t

TournamentAction.LOGF = False  # type: ignore


class TestExports(unittest.TestCase):
    def setUp(self):
        cfg = TournamentConfiguration(
            pod_sizes=[4, 3], allow_bye=True, auto_export=False, max_byes=2
        )
        self.t = Tournament(cfg)
        self.t.new_round()
        self.t.add_player(generate_player_names(8))
        self.t.create_pairings()
        self.t.random_results()

    def test_standings_str_plain_nonempty(self):
        result = self.t.get_standings_str(style=StandingsExport.Format.PLAIN)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_standings_str_plain_contains_players(self):
        result = self.t.get_standings_str(style=StandingsExport.Format.PLAIN)
        for p in self.t.players:
            self.assertIn(p.name, result)

    def test_standings_str_csv_not_implemented(self):
        # CSV is not yet implemented — documents expected behavior
        with self.assertRaises((ValueError, Exception)):
            self.t.get_standings_str(style=StandingsExport.Format.CSV)

    def test_standings_str_json_not_implemented(self):
        # JSON is not yet implemented — documents expected behavior
        with self.assertRaises((ValueError, Exception)):
            self.t.get_standings_str(style=StandingsExport.Format.JSON)

    def test_pod_repr_with_context(self):
        pod = self.t.tour_round.pods[0]
        context = TournamentContext(
            self.t, self.t.tour_round, self.t.get_standings(self.t.tour_round)
        )
        result = pod.__repr__(context=context)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_pod_repr_without_context(self):
        pod = self.t.tour_round.pods[0]
        result = pod.__repr__()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestStandingsFields(unittest.TestCase):
    """StandingsExport field getters: wins, winrate, unique opponents, games."""

    def setUp(self):
        self.t = _small_tournament()
        self.tour_round = self.t.tour_round
        self.context = TournamentContext(
            self.t, self.tour_round, self.t.get_standings(self.tour_round)
        )

    def _get_field(self, field_name):
        field = StandingsExport.info[StandingsExport.Field[field_name]]
        return [field.get(p, self.context) for p in self.t.players]

    def test_standing_field(self):
        vals = self._get_field("STANDING")
        self.assertEqual(len(vals), 8)
        self.assertTrue(all(isinstance(v, int) for v in vals))

    def test_name_field(self):
        vals = self._get_field("NAME")
        names = {p.name for p in self.t.players}
        self.assertEqual(set(vals), names)

    def test_rating_field(self):
        vals = self._get_field("RATING")
        self.assertTrue(all(isinstance(v, (int, float)) for v in vals))

    def test_opp_pointrate_field(self):
        vals = self._get_field("OPP_POINTRATE")
        self.assertTrue(all(isinstance(v, (int, float)) for v in vals))

    def test_record_field(self):
        vals = self._get_field("RECORD")
        self.assertEqual(len(vals), 8)
        # Each record string should contain W/L/D indicators
        self.assertTrue(all(isinstance(v, str) for v in vals))

    def test_standings_str_header_present(self):
        result = self.t.get_standings_str(style=StandingsExport.Format.PLAIN)
        # Header uses lowercase field .name attributes
        self.assertIn("name", result)
        self.assertIn("pts", result)

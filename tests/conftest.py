import pytest
from src.core import Tournament, TournamentConfiguration, TournamentAction
from src.misc import generate_player_names

TournamentAction.LOGF = False  # Disable file I/O in all tests


@pytest.fixture
def cfg():
    return TournamentConfiguration(
        pod_sizes=[4, 3],
        allow_bye=True,
        auto_export=False,
        max_byes=2,
        n_rounds=5,
    )


@pytest.fixture
def tournament(cfg):
    t = Tournament(cfg)
    t.new_round()
    return t


@pytest.fixture
def small_tournament(cfg):
    t = Tournament(cfg)
    t.new_round()
    t.add_player(generate_player_names(8))
    t.create_pairings()
    t.random_results()
    return t

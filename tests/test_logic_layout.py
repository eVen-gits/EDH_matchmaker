"""Game logic folders must remain independent of each other."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "game, forbidden",
    [
        ("mtg", "commander"),
        ("commander", "mtg"),
        ("common", "commander"),
        ("common", "mtg"),
    ],
)
def test_game_imports_do_not_load_other_game(game, forbidden):
    # A fresh interpreter also catches transitive and dynamic imports.
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import sys
for module in ('matching', 'scoring', 'rules'):
    try:
        importlib.import_module('src.logic.{game}.' + module)
    except ModuleNotFoundError as exc:
        if exc.name != 'src.logic.{game}.' + module:
            raise
assert not any(name.startswith('src.logic.{forbidden}.') for name in sys.modules)
""",
        ],
        check=True,
    )


"""Sidecars register methods; game modules must remain independent."""

import ast
import importlib
import inspect
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src import core
from src.core import Tournament, TournamentConfiguration
from src.interface import IPairingLogic, IRuleset, IScoringLogic
from src.logic.commander.matching import PairingDefault

LOGIC = Path(core.__file__).parent / "logic"


@pytest.mark.parametrize(
    "game, forbidden",
    [
        ("mtg", "commander"),
        ("commander", "mtg"),
        ("common", "commander"),
        ("common", "mtg"),
    ],
)
def test_game_imports_are_independent(game, forbidden):
    for path in (LOGIC / game).glob("*.py"):
        package = f"src.logic.{game}"
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = (
                    importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                    if node.level
                    else node.module or ""
                )
                names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            assert not any(
                name == f"src.logic.{forbidden}"
                or name.startswith(f"src.logic.{forbidden}.")
                for name in names
            ), path


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


def test_sidecars_and_discovered_classes_match():
    discovered = set()
    for filename, base in (
        ("matching.py", IPairingLogic),
        ("scoring.py", IScoringLogic),
        ("rules.py", IRuleset),
    ):
        cache = {}
        Tournament._discover_logic(filename, base, cache)
        for module_path in LOGIC.glob(f"*/{filename}"):
            module = importlib.import_module(
                f"src.logic.{module_path.parent.name}.{module_path.stem}"
            )
            for cls in vars(module).values():
                if (
                    isinstance(cls, type)
                    and cls.__module__ == module.__name__
                    and issubclass(cls, base)
                    and cls.IS_COMPLETE
                    and not inspect.isabstract(cls)
                ):
                    assert cls.__name__ in cache
        for name, method in cache.items():
            cls = type(method)
            path = Path(inspect.getfile(cls)).parent / f"{name}.params.yaml"
            assert path.is_file()
            discovered.add(path)
    configs = set()
    for path in LOGIC.glob("*/rules.py"):
        module = importlib.import_module(f"src.logic.{path.parent.name}.rules")
        for cls in vars(module).values():
            if (
                isinstance(cls, type)
                and cls.__module__ == module.__name__
                and issubclass(cls, TournamentConfiguration)
            ):
                configs.add(path.parent / f"{cls.__name__}.params.yaml")
    assert discovered == set(LOGIC.glob("*/*.params.yaml")) - configs
    assert LOGIC / "commander/PairingTop4.params.yaml" in discovered


def test_discovery_requires_own_sidecar_and_warns_about_drift(tmp_path, monkeypatch):
    game = tmp_path / "logic" / "testgame"
    game.mkdir(parents=True)
    (game / "matching.py").touch()
    module: Any = types.ModuleType("src.logic.testgame.matching")
    module.__file__ = str(game / "matching.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in ("Registered", "Missing", "Incomplete", "Abstract"):
        attrs: dict[str, Any] = {"__module__": module.__name__}
        parent: type = PairingDefault
        if name == "Incomplete":
            attrs["IS_COMPLETE"] = False
        elif name == "Abstract":
            parent = IPairingLogic
            attrs["IS_COMPLETE"] = True
        setattr(module, name, type(name, (parent,), attrs))
        if name != "Missing":
            (game / f"{name}.params.yaml").write_text("{}\n")
    (game / "Orphan.params.yaml").write_text("{}\n")
    module.Alias = PairingDefault
    (game / "Alias.params.yaml").write_text("{}\n")
    assert module.Missing.DEFAULT_PARAMS == PairingDefault.DEFAULT_PARAMS
    monkeypatch.setattr(core, "__file__", str(tmp_path / "core.py"))
    messages = []
    monkeypatch.setattr(
        core.Log, "log", lambda message, **kwargs: messages.append(message)
    )
    cache = {}
    Tournament._discover_logic("matching.py", IPairingLogic, cache)
    assert set(cache) == {"Registered"}
    assert any("Missing has no own sidecar" in message for message in messages)
    for name in ("Orphan", "Alias"):
        assert any(
            name in message and "no matching class" in message for message in messages
        )
    for name in ("Incomplete", "Abstract"):
        assert any(
            name in message and "complete concrete" in message for message in messages
        )

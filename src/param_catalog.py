"""Every parameter spec in one JSON document, for front ends.

Run ``python -m src.param_catalog`` to print it, or call `catalog()`. Shape::

    {
      "tournament": {<field>: <spec>, ...},        # TournamentConfiguration
      "games": {
        "<game>": {                                 # src/logic/<game>/
          "rulesets": {
            "<RulesetClass>": {
              "params": {<param>: <spec>},          # per-round ruleset params
              "config_fields": {<field>: <spec>},   # its CONFIG_CLASS's fields
              "defaults": {<field>: <value>}        # values of default_from
            }
          },
          "scoring": {"<ScoringClass>": {<param>: <spec>}},
          "pairing": {"<PairingClass>": {<param>: <spec>}}
        }
      }
    }

Each <spec> is a `ParamSpec` as a dict (see src/param_spec.py for its keys);
`visible_when` becomes ``{param: value}``. A spec's ``default_from:
"ruleset.X"`` is resolved per ruleset under that ruleset's "defaults".
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from .core import Tournament, TournamentConfiguration
from .param_spec import ParamSpec


def _spec_dict(specs: dict[str, ParamSpec]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, spec in specs.items():
        d = dataclasses.asdict(spec)
        if spec.choices is not None:
            d["choices"] = list(spec.choices)
        if spec.visible_when is not None:
            d["visible_when"] = dict([spec.visible_when])
        out[name] = d
    return out


def _game(obj: object) -> str:
    # src.logic.<game>.<module> -> <game>
    return type(obj).__module__.split(".")[2]


def catalog() -> dict[str, Any]:
    """Every tournament, game, ruleset, scoring and pairing parameter spec.

    Returns:
        A JSON-serializable dict; see the module docstring for its shape.
    """
    games: dict[str, dict[str, dict[str, Any]]] = {}

    def game(obj: object) -> dict[str, dict[str, Any]]:
        return games.setdefault(
            _game(obj), {"rulesets": {}, "scoring": {}, "pairing": {}}
        )

    for name in Tournament.ruleset_names():
        ruleset = Tournament.get_ruleset(name)
        config_cls = ruleset.CONFIG_CLASS or TournamentConfiguration
        game(ruleset)["rulesets"][name] = {
            "params": _spec_dict(ruleset.PARAM_SPEC),
            "config_fields": _spec_dict(
                getattr(config_cls, "GAME_PARAM_SPEC", {})
            ),
            "defaults": {
                field: list(value) if isinstance(value, tuple) else value
                for field, spec in TournamentConfiguration.PARAM_SPEC.items()
                if spec.default_from
                for value in [getattr(ruleset, spec.default_from.split(".", 1)[1])]
            },
        }
    for name in Tournament.scoring_logic_names():
        logic = Tournament.get_scoring_logic(name)
        game(logic)["scoring"][name] = _spec_dict(logic.PARAM_SPEC)
    Tournament.discover_pairing_logic()
    for name, logic in sorted(Tournament._pairing_logic_cache.items()):
        game(logic)["pairing"][name] = _spec_dict(logic.PARAM_SPEC)
    return {
        "tournament": _spec_dict(TournamentConfiguration.PARAM_SPEC),
        "games": games,
    }


if __name__ == "__main__":
    print(json.dumps(catalog(), indent=2))

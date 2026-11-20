"""Setup profiles: how a front end sets up an event of a ruleset.

Each ruleset declares its setup profile in a sidecar YAML file named
``<RulesetClass>.setup.yaml`` next to the module that defines the class, the
way ``<ClassName>.params.yaml`` declares parameters (see param_spec.py). The
profile groups formats under a category (e.g. MTG spans CommanderRuleset and
Mtg1v1Ruleset) and holds the event defaults a front end such as FairPair
shows: round timer, rounds, byes, points, report kind, table companion. The
desktop GUI ignores it. `param_catalog` exports it per ruleset.

Keys (required unless marked optional):

- ``category``: category id, e.g. ``mtg`` or ``catan``.
- ``formats``: format id -> display label.
- ``timer``: ``default_minutes`` (int, or null for untimed rounds with no
  round clock); optional ``phases`` (table phase timers) and ``turn``
  (turn-timer ``options`` and the ``default`` one).
- ``n_rounds``: Swiss rounds of a new event.
- ``structure`` (optional): player-count bands, ascending ``from_players``,
  each with ``n_rounds``, ``top_cut`` (a key of the ruleset's PLAYOFFS) and
  optional ``eliminations`` (``before_game``, 1-based, and ``below_wins``).
- ``valid_from_players`` (optional): fewest players for valid results.
- ``byes``: ``allowed``, ``required`` or ``forbidden``; must agree with the
  ruleset's BYES_REQUIRED and DEFAULT_ALLOW_BYE.
- ``seat_order``: ``organizer`` (may choose, e.g. snake draft) or ``engine``.
- ``points``: event defaults ``win``, ``draw``, ``bye`` and ``locked``.
- ``decklists``, ``rules_enforcement``, ``open_play``: whether offered.
- ``report``: ``winners``, ``best_of`` or ``scores``.
- ``companion``: ``tracker: life`` with ``start``, or ``tracker: vp`` with
  ``target``.
- ``confirmations``: 0, 1, 2 or ``all`` other players confirm a report.

The loader runs once per ruleset at class definition
(``IRuleset.__init_subclass__``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

from .param_spec import _sidecar_path

_REQUIRED = frozenset(
    {
        "category",
        "formats",
        "timer",
        "n_rounds",
        "byes",
        "seat_order",
        "points",
        "decklists",
        "rules_enforcement",
        "open_play",
        "report",
        "companion",
        "confirmations",
    }
)
_OPTIONAL = frozenset({"structure", "valid_from_players"})
_CHOICES: dict[str, tuple[Any, ...]] = {
    "byes": ("allowed", "required", "forbidden"),
    "seat_order": ("organizer", "engine"),
    "report": ("winners", "best_of", "scores"),
    "confirmations": (0, 1, 2, "all"),
}
_TRACKERS = {"life": "start", "vp": "target"}


def _pos_int(value: Any) -> bool:
    return type(value) is int and value > 0


def load_setup_profile(cls: type) -> dict[str, Any]:
    """Loads and validates a ruleset's setup profile.

    Args:
        cls: The ruleset class. Its PLAYOFFS, BYES_REQUIRED and
            DEFAULT_ALLOW_BYE are checked against the profile.

    Returns:
        The profile as plain data (JSON-serializable). Empty if the class
        (and its ancestors) ship no sidecar file.

    Raises:
        ValueError: If the sidecar is malformed or disagrees with the class.
            The message names the file and the offending key.
    """
    path = _sidecar_path(cls, ".setup.yaml")
    if path is None:
        return {}
    profile = yaml.safe_load(path.read_text()) or {}

    def fail(msg: str) -> ValueError:
        return ValueError(f"{path}: {msg}")

    if not isinstance(profile, dict):
        raise fail("top level must be a mapping.")
    if missing := _REQUIRED - profile.keys():
        raise fail(f"missing keys {sorted(missing)}.")
    if unknown := profile.keys() - _REQUIRED - _OPTIONAL:
        raise fail(f"unknown keys {sorted(unknown)}.")

    if not isinstance(profile["category"], str) or not profile["category"]:
        raise fail("'category' must be a non-empty string.")
    formats = profile["formats"]
    if not isinstance(formats, dict) or not formats or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in formats.items()
    ):
        raise fail("'formats' must be a non-empty mapping of id to label.")
    for key, choices in _CHOICES.items():
        if profile[key] not in choices:
            raise fail(f"'{key}' must be one of {list(choices)}, got {profile[key]!r}.")
    for key in ("decklists", "rules_enforcement", "open_play"):
        if type(profile[key]) is not bool:
            raise fail(f"'{key}' must be true or false.")
    if not _pos_int(profile["n_rounds"]):
        raise fail("'n_rounds' must be a positive int.")
    if "valid_from_players" in profile and not _pos_int(profile["valid_from_players"]):
        raise fail("'valid_from_players' must be a positive int.")

    if getattr(cls, "BYES_REQUIRED", False):
        byes = "required"
    elif getattr(cls, "DEFAULT_ALLOW_BYE", True):
        byes = "allowed"
    else:
        byes = "forbidden"
    if profile["byes"] != byes:
        raise fail(f"'byes' is {profile['byes']!r}, but the ruleset's byes are {byes!r}.")

    _check_timer(profile["timer"], fail)
    _check_points(profile["points"], fail)
    _check_companion(profile["companion"], fail)
    if "structure" in profile:
        _check_structure(profile["structure"], getattr(cls, "PLAYOFFS", {}), fail)
    return profile


def _check_timer(timer: Any, fail: Any) -> None:
    if not isinstance(timer, dict) or "default_minutes" not in timer:
        raise fail("'timer' must be a mapping with 'default_minutes'.")
    if unknown := timer.keys() - {"default_minutes", "phases", "turn"}:
        raise fail(f"'timer' has unknown keys {sorted(unknown)}.")
    minutes = timer["default_minutes"]
    if minutes is not None and not _pos_int(minutes):
        raise fail("'timer.default_minutes' must be a positive int or null.")
    if "phases" in timer and not isinstance(timer["phases"], dict):
        raise fail("'timer.phases' must be a mapping.")
    if "turn" in timer:
        turn = timer["turn"]
        if not isinstance(turn, dict) or not isinstance(turn.get("options"), dict):
            raise fail("'timer.turn' must be a mapping with 'options'.")
        if turn.get("default") not in turn["options"]:
            raise fail("'timer.turn.default' must name one of 'timer.turn.options'.")


def _check_points(points: Any, fail: Any) -> None:
    if not isinstance(points, dict) or points.keys() != {"win", "draw", "bye", "locked"}:
        raise fail("'points' must have exactly 'win', 'draw', 'bye' and 'locked'.")
    if not all(
        type(points[k]) in (int, float) for k in ("win", "draw", "bye")
    ) or type(points["locked"]) is not bool:
        raise fail("'points' win/draw/bye must be numbers and 'locked' true or false.")


def _check_companion(companion: Any, fail: Any) -> None:
    tracker = companion.get("tracker") if isinstance(companion, dict) else None
    if tracker not in _TRACKERS:
        raise fail(f"'companion.tracker' must be one of {list(_TRACKERS)}.")
    if companion.keys() != {"tracker", _TRACKERS[tracker]} or not _pos_int(
        companion[_TRACKERS[tracker]]
    ):
        raise fail(
            f"'companion' with tracker {tracker!r} needs exactly a positive int "
            f"'{_TRACKERS[tracker]}'."
        )


def _check_structure(bands: Any, playoffs: Mapping[int, Any], fail: Any) -> None:
    if not isinstance(bands, list) or not bands:
        raise fail("'structure' must be a non-empty list of bands.")
    previous = 0
    for band in bands:
        if not isinstance(band, dict) or not {"from_players", "n_rounds", "top_cut"} <= band.keys():
            raise fail("each 'structure' band needs 'from_players', 'n_rounds', 'top_cut'.")
        if not _pos_int(band["from_players"]) or band["from_players"] <= previous:
            raise fail("'structure' bands need ascending positive 'from_players'.")
        previous = band["from_players"]
        if not _pos_int(band["n_rounds"]):
            raise fail("a 'structure' band's 'n_rounds' must be a positive int.")
        if band["top_cut"] not in playoffs:
            raise fail(f"'structure' top_cut {band['top_cut']!r} is not in PLAYOFFS.")
        for elim in band.get("eliminations", []):
            if not (
                isinstance(elim, dict)
                and elim.keys() == {"before_game", "below_wins"}
                and _pos_int(elim["below_wins"])
                and _pos_int(elim["before_game"])
                and elim["before_game"] <= band["n_rounds"]
            ):
                raise fail(
                    "an elimination needs 'before_game' (1..n_rounds) and a "
                    "positive 'below_wins'."
                )

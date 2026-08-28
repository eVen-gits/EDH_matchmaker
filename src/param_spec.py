"""Parameter specifications for scoring and pairing algorithms.

Each algorithm declares its tunable parameters in a sidecar YAML file named
``<ClassName>.params.yaml`` next to the module that defines the class. The file
is the single source of truth for a parameter's name, default, type, range, and
human description. Code never hard-codes these values.

Example ``src/scoring_logic/ScoringHareruya.params.yaml``::

    wager_percent:
      default: 0.07
      type: float
      min: 0
      max: 1
      step: 0.01
      label: Wager %
      description: Fraction of a player's stack wagered into the pot each pod.

The loader is called once per class at definition time (see
``IScoringLogic.__init_subclass__`` / ``IPairingLogic.__init_subclass__``), so
there is no per-call file I/O.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Allowed `type:` names in a sidecar, mapped to the Python type a default and
# the min/max bounds must match.
_TYPES: dict[str, type] = {
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
}

# Allowed `widget:` hints. If omitted, one is inferred from the type (see
# _infer_widget). The GUI (run_ui.ParamForm) maps each to a Qt widget.
_WIDGETS = frozenset(
    {"spinbox", "doublespinbox", "checkbox", "lineedit", "combobox", "slider"}
)


def _infer_widget(type_name: str, has_choices: bool) -> str:
    if has_choices:
        return "combobox"
    return {
        "bool": "checkbox",
        "int": "spinbox",
        "float": "doublespinbox",
        "str": "lineedit",
    }[type_name]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter of an algorithm, loaded from a sidecar YAML file."""

    name: str
    type: str
    default: Any
    description: str
    label: str
    widget: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    # Combobox choices, as a tuple so the spec stays hashable/frozen.
    choices: tuple[Any, ...] | None = None
    # Display transform: the widget shows value * scale with `suffix`, and the
    # entered value is divided by scale on read (wager: scale 100, suffix "%").
    scale: float | None = None
    suffix: str | None = None
    # (param_name, value): show this field only while that param equals value.
    visible_when: tuple[str, Any] | None = None


def _sidecar_path(cls: type) -> Path | None:
    """Returns the ``<ClassName>.params.yaml`` beside the class's module.

    Walks the MRO so a subclass with no file of its own inherits the first
    ancestor that does (for example ScoringModifiedHareruya reuses
    ScoringHareruya's spec). Returns None if no file exists in the MRO.
    """
    for ancestor in cls.__mro__:
        module = sys.modules.get(ancestor.__module__)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        candidate = Path(module_file).parent / f"{ancestor.__name__}.params.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_param_spec(cls: type) -> dict[str, ParamSpec]:
    """Loads and validates the parameter spec for an algorithm class.

    Args:
        cls: The scoring or pairing logic class.

    Returns:
        An ordered mapping of parameter name to ParamSpec. Empty if the class
        (and its ancestors) ship no sidecar file.

    Raises:
        ValueError: If a sidecar file exists but is malformed. The message names
            the file and the offending parameter.
    """
    path = _sidecar_path(cls)
    if path is None:
        return {}

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping of parameter names.")

    specs: dict[str, ParamSpec] = {}
    for name, descriptor in raw.items():
        specs[name] = _build_spec(path, name, descriptor)

    # Post-pass: a visible_when can only reference a param defined in this file.
    for spec in specs.values():
        if spec.visible_when is not None and spec.visible_when[0] not in specs:
            raise ValueError(
                f"{path}: parameter '{spec.name}': visible_when references "
                f"unknown parameter '{spec.visible_when[0]}'."
            )
    return specs


def _build_spec(path: Path, name: str, descriptor: Any) -> ParamSpec:
    """Validates one descriptor mapping and returns a ParamSpec, or raises."""

    def fail(msg: str) -> ValueError:
        return ValueError(f"{path}: parameter '{name}': {msg}")

    if not isinstance(descriptor, dict):
        raise fail("descriptor must be a mapping.")
    if "default" not in descriptor:
        raise fail("missing required 'default'.")
    if "description" not in descriptor:
        raise fail("missing required 'description'.")

    default = descriptor["default"]

    type_name = descriptor.get("type")
    if type_name is None:
        # Infer from the default so `type:` is optional for obvious cases.
        for candidate, py in _TYPES.items():
            # bool is a subclass of int - check it before int wins.
            if type(default) is py:
                type_name = candidate
                break
        else:
            raise fail(f"cannot infer 'type' from default {default!r}; set it.")
    if type_name not in _TYPES:
        raise fail(f"unknown type '{type_name}'; use one of {sorted(_TYPES)}.")

    expected = _TYPES[type_name]
    # bool is a subclass of int, so guard both directions explicitly.
    if type(default) is not expected:
        raise fail(f"default {default!r} is not of type '{type_name}'.")

    low = descriptor.get("min")
    high = descriptor.get("max")
    for bound_name, bound in (("min", low), ("max", high)):
        if bound is not None and not isinstance(bound, (int, float)):
            raise fail(f"'{bound_name}' must be numeric.")
    if low is not None and high is not None and low > high:
        raise fail(f"min ({low}) is greater than max ({high}).")
    if low is not None and default < low:
        raise fail(f"default {default!r} is below min {low}.")
    if high is not None and default > high:
        raise fail(f"default {default!r} is above max {high}.")

    step = descriptor.get("step")
    if step is not None and not isinstance(step, (int, float)):
        raise fail("'step' must be numeric.")

    choices = descriptor.get("choices")
    if choices is not None:
        if not isinstance(choices, list) or not choices:
            raise fail("'choices' must be a non-empty list.")
        if default not in choices:
            raise fail(f"default {default!r} is not one of choices {choices}.")
        choices = tuple(choices)

    widget = descriptor.get("widget")
    if widget is None:
        widget = _infer_widget(type_name, choices is not None)
    elif widget not in _WIDGETS:
        raise fail(f"unknown widget '{widget}'; use one of {sorted(_WIDGETS)}.")

    scale = descriptor.get("scale")
    if scale is not None and not isinstance(scale, (int, float)):
        raise fail("'scale' must be numeric.")
    suffix = descriptor.get("suffix")

    visible_when = descriptor.get("visible_when")
    if visible_when is not None:
        if not isinstance(visible_when, dict) or len(visible_when) != 1:
            raise fail("'visible_when' must be a single {param: value} mapping.")
        visible_when = next(iter(visible_when.items()))  # (param_name, value)

    return ParamSpec(
        name=name,
        type=type_name,
        default=default,
        description=str(descriptor["description"]).strip(),
        label=str(descriptor.get("label", name)),
        widget=widget,
        min=low,
        max=high,
        step=step,
        choices=choices,
        scale=scale,
        suffix=str(suffix) if suffix is not None else None,
        visible_when=visible_when,
    )

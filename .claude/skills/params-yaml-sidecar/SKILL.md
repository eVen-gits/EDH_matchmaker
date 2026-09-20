---
name: params-yaml-sidecar
description: Use this skill whenever creating or editing a <ClassName>.params.yaml file for a pairing or scoring algorithm in the EDH Matchmaker repo — e.g. adding a parameter, changing a default/range/widget, or adding a new params.yaml alongside a new algorithm class. Trigger on requests like "add a param to ScoringHareruya", "change the default weighting", "make this a slider instead of a spinbox", "add a min/max to this parameter", or any edit to a *.params.yaml file, even without the user naming the file directly.
---

# Writing or editing a params.yaml sidecar

Every configurable pairing/scoring algorithm parameter is defined in a YAML
file next to its class, not in Python. `src/param_spec.py` loads and
validates this file at class-definition time — a mistake here breaks
**import**, not just the GUI, so re-run the tests (or just import the
module) after editing to catch errors immediately.

## File location and naming

The file must be named exactly `<ClassName>.params.yaml` and live in the
same directory as the module that defines the class (e.g.
`src/logic/commander/ScoringHareruya.params.yaml` next to
`src/logic/commander/scoring.py`).

A subclass that doesn't ship its own sidecar automatically inherits its
parent's — the loader walks the class MRO to find one. Only add a new file
when the subclass's parameters actually differ from the parent's; don't
copy a sidecar unchanged just because the class is new.

## Field contract

Each top-level key is a parameter name. Required per-parameter keys:

- `default` — the parameter's default value.
- `description` — shown in the GUI.

Optional keys:

- `type` — one of `float`, `int`, `bool`, `str`. Inferred from `default`'s
  Python type if omitted (bool is checked before int, since bool is a
  subtype of int in Python).
- `min` / `max` — numeric bounds. Validated so `min <= default <= max`.
- `step` — numeric increment for spinbox-style widgets.
- `label` — display name in the GUI; defaults to the parameter's key.
- `choices` — a non-empty list of allowed values. `default` must be a
  member. Setting this forces `widget: combobox` regardless of `type`.
- `widget` — one of `spinbox`, `doublespinbox`, `checkbox`, `lineedit`,
  `combobox`, `slider`. Inferred from `type` if omitted.
- `scale` / `suffix` — a display transform, e.g. storing a fraction
  internally (`0.1`) but showing it as a percent (`10%`) in the GUI.
- `visible_when` — a single `{other_param: value}` mapping that makes this
  field's GUI widget conditionally visible. `other_param` must be another
  parameter defined in the *same* file.

## Rules that matter

- **This file is the single source of truth for the parameter.** Never
  hardcode a conflicting default in the Python class, and never restate its
  values (defaults, ranges, descriptions) in `CLAUDE.md` or other repo docs
  — those docs are supposed to point at this file, not duplicate it, so a
  future edit here doesn't silently leave the docs wrong.
- Validation happens in `__init_subclass__` (`src/interface.py`) the moment
  the class is defined — a malformed sidecar (bad type, out-of-range
  default, dangling `visible_when` reference) raises `ValueError` at import
  time, which will break every test and the whole app, not just this one
  algorithm's GUI panel.
- If you're adding a parameter to a *new* algorithm class rather than
  editing an existing one, do it in the same commit as the class itself —
  see the `add-pairing-scoring-algorithm` skill for that full workflow.

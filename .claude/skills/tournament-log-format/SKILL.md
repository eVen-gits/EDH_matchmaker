---
name: tournament-log-format
description: Use this skill whenever touching tournament save/load or persistence code in the EDH Matchmaker repo — TournamentAction persistence, the Log class, StandingsExport/PodsExport, or anything that reads or writes files under logs/. Trigger on requests like "add a field to the saved tournament state", "change how results get logged", "fix a bug in loading old tournament files", "add a new scoring formula's persisted data", or any change to serialization/deserialization code, even if the user doesn't mention the spec doc by name.
---

# Working with the tournament log format

`docs/tournament-log-spec.md` is the authoritative specification for the
save/load JSON format (currently format version `1.1`) and for scoring
formulas that get persisted (e.g. Hareruya wagering). **Read it before
changing any persistence behavior** — it documents exact field semantics
that aren't always obvious from the code alone.

## What the format is

- It's a **full snapshot**, not an event log — each save fully represents
  current tournament state, it doesn't append deltas.
- Writes must be atomic: write to a temp file, then rename over the target.
  Never write the target file directly. If you're adding or changing a
  write path, preserve this — a partial write on crash/interrupt should
  never corrupt a previously-good log file.

## When you change something here

- **If your change alters the save format** — a new field, a changed
  field's meaning, new persisted state for a new algorithm — **update
  `docs/tournament-log-spec.md` in the same commit.** The doc is the
  contract external tools rely on; letting it drift from the code defeats
  its purpose.
- **If the change is backward-incompatible**, bump the format version field
  and document what changed for readers of older files.
- **Never copy the spec's concrete values** (field names, formula
  derivations) into `CLAUDE.md`, `CONTRIBUTING.md`, or docstrings elsewhere
  — reference `docs/tournament-log-spec.md` by path instead, so there's
  exactly one place these values can go stale.
- Check `tests/test_log_schema.py` and `tests/test_serialization.py` for
  existing coverage patterns and extend them rather than writing
  parallel/duplicate test infrastructure.

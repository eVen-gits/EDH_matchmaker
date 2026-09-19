# Contributing to EDH Matchmaker

First off, thanks for taking the time to contribute!

## How we work

1. **Fork the repository**: create your own fork of the project.
2. **Create a branch**: create a branch for your feature or bugfix
   (`git checkout -b feature/amazing-feature`).
3. **Commit your changes** (see Commit messages below).
4. **Push to the branch**: push to your fork
   (`git push origin feature/amazing-feature`).
5. **Open a pull request**: submit a pull request to the main repository.

### Development setup

1. Install Python 3.10+
2. Install dependencies: `pip install -r requirements.txt`
3. Install the git hooks: `pre-commit install`

### Running tests

```bash
PYTHONPATH=. pytest
```

`pytest.ini` skips slow and performance tests by default. Run them with
`pytest -m slow` or `pytest tests/test_performance.py -m performance`.

### Lint & type checking

- `pre-commit run --all-files` runs the same hygiene checks CI expects
  (trailing whitespace, YAML syntax, merge conflict markers) plus `pyright`.
- `pyright` alone: basic mode, config in `pyrightconfig.json`. Prefer
  `# pyright: ignore` or `cast()` over disabling rules globally.
- **Pyright is advisory, not blocking, for now.** There's a backlog of
  ~245 pre-existing errors (it was never wired into CI or pre-commit before
  this hook existed) — the CI job and pre-commit hook both report without
  failing. Please don't add to the backlog in new code, and feel free to
  fix errors incidentally in files you're already touching. Once the
  backlog's cleared, flip `continue-on-error` off in
  `.github/workflows/typecheck.yml` and drop the `|| true` in
  `.pre-commit-config.yaml`.

## Commit messages

No tool enforces this — it's a convention, not a gate. Match the existing
history: an imperative subject (`Add exhaustive standings-formatting
tests`, not `Added` or `Adds`), optionally prefixed with a type when it aids
scanning (`docs:`, `fix:`, `CI:`). Keep the subject line short; put the
"why" in the body if it isn't obvious from the diff.

## Judgment standards (no tool enforces these)

Rules that need human/agent judgment rather than mechanical enforcement.
See `CONTEXT.md` for the terms referenced here.

- **Auto-discovery, not hand-wiring.** A new scoring or pairing algorithm
  needs a class (`IS_COMPLETE = True`) plus a `<ClassName>.params.yaml`
  sidecar if it has parameters — never a core or GUI change to register it.
- **The params sidecar is the source of truth**, not a hardcoded default in
  the class. If you add or change a parameter, update the sidecar in the
  same commit.
- **`TournamentAction.LOGF = False`** must be set at the top of any test
  module that builds a `Tournament`, or the test writes a real log file
  under `logs/` as a side effect.
- **`docs/tournament-log-spec.md` and the `*.params.yaml` sidecars are
  authoritative** for save format, scoring formulas, and algorithm
  parameters. `CLAUDE.md` deliberately doesn't restate their values — they
  drift. Point at the file, don't copy the number.
- **`SUPPORTED_POD_SIZES`** must be set deliberately on any pairing
  algorithm that can't handle every pod size — leaving it at the `None`
  default on an algorithm that actually can't handle 3-player pods is a
  silent correctness bug, not just a GUI omission.

## Docstrings

Google style (required by MkDocs `mkdocstrings`).

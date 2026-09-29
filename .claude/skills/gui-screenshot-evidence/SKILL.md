---
name: gui-screenshot-evidence
description: Use this skill when a GUI bug fix needs screenshot evidence, a before/after PNG pair, or a headless capture of an EDH Matchmaker window, dialog, widget, or message box. Trigger on requests to reproduce a visible UI bug or attach GUI screenshots to a PR.
---

# Capture GUI evidence

Run Qt headless with `QT_QPA_PLATFORM=offscreen` set before importing PyQt6 or `run_ui`. Keep a live `QApplication` reference. Build the real `run_ui.MainWindow(tournament)` or the relevant dialog as the GUI tests do (`tests/test_ui_load_players.py`, `tests/test_exports.py`, `tests/test_ui_mtg.py`). Disable tournament log writes for standalone captures.

This example was run headless and produced a non-empty PNG:

```bash
QT_QPA_PLATFORM=offscreen python - <<'PY'
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from src.core import Tournament, TournamentAction, TournamentConfiguration
import run_ui

TournamentAction.LOGF = False
app = QApplication.instance() or QApplication([])
t = Tournament(TournamentConfiguration(auto_export=False))
t.add_player(['Alice', 'Bob'])
window = run_ui.MainWindow(t)
out = Path('evidence/player-list-after.png')
out.parent.mkdir(exist_ok=True)
assert window.ui.lv_players.grab().save(str(out))
assert out.stat().st_size > 0
PY
```

Capture the smallest widget that shows the change (`window.ui.lv_players.grab()`), or use `window.grab()` for the full window and `dlg.grab()` for a dialog. To capture a modal message box, patch `QMessageBox.warning` (or the called modal method) with a callback that grabs the message-box widget before it closes; do not wait for `exec()` in an offscreen session. Save with `grab().save(path)` and check its return value and file size.

Put committed screenshots in `evidence/` as descriptive `.png` files, like `open-missing.png` and `standings-before-round.png`. For a bug fix, capture the failing state before changing the code and the corrected state after, using matching `-before.png` and `-after.png` names. Keep temporary verification PNGs outside the repository or remove them before committing. Inspect each image to confirm the relevant UI state is visible.

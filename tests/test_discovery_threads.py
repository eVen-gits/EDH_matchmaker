"""Logic discovery must be safe when a fresh process gets overlapping lookups.

A server running the engine in a thread pool (e.g. a FastAPI wrapper) can get
its first requests at the same time, right after a restart. A lookup that
starts while another thread is still discovering must not see a half-filled
cache.
"""

import importlib
import threading
import time
import unittest
from unittest import mock

from src.core import Tournament, TournamentAction

TournamentAction.LOGF = False  # type: ignore

# Each module import is slowed to this, so the second lookup reliably starts
# while the first is between two game directories.
IMPORT_DELAY = 0.1


class TestOverlappingDiscovery(unittest.TestCase):
    CACHES = ("_ruleset_cache", "_pairing_logic_cache", "_scoring_logic_cache")

    def setUp(self) -> None:
        # Fresh-process state: empty caches. Restore the originals afterwards.
        for attr in self.CACHES:
            self.addCleanup(setattr, Tournament, attr, getattr(Tournament, attr))
            setattr(Tournament, attr, {})
        real_import = importlib.import_module

        def slow_import(name, *args, **kwargs):
            time.sleep(IMPORT_DELAY)
            return real_import(name, *args, **kwargs)

        patcher = mock.patch("src.core.importlib.import_module", slow_import)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _overlap(self, lookup):
        """Runs lookup in two threads, the second starting mid-discovery."""
        errors: list[Exception] = []

        def run():
            try:
                lookup()
            except Exception as e:  # collected, asserted below
                errors.append(e)

        first = threading.Thread(target=run)
        second = threading.Thread(target=run)
        first.start()
        time.sleep(IMPORT_DELAY * 1.5)  # first dir cached, next import in progress
        second.start()
        first.join()
        second.join()
        self.assertEqual(errors, [])

    def test_ruleset(self):
        self._overlap(lambda: Tournament.get_ruleset("Mtg1v1Ruleset"))
        self.assertEqual(Tournament.ruleset_names(), ["CommanderRuleset", "Mtg1v1Ruleset"])

    def test_pairing_logic(self):
        self._overlap(lambda: Tournament.get_pairing_logic("Pairing1v1"))
        self.assertIn("PairingSnake", Tournament._pairing_logic_cache)

    def test_scoring_logic(self):
        self._overlap(lambda: Tournament.get_scoring_logic("Scoring1v1"))
        self.assertIn("ScoringDefault", Tournament.scoring_logic_names())


if __name__ == "__main__":
    unittest.main()

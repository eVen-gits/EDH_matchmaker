from __future__ import annotations
from ..commander import matching as _commander_matching


class Pairing1v1(_commander_matching.PairingDefault):
    """Swiss pairing for 1v1 (2-player pod) Magic tournaments.

    Commander's PairingDefault sort key (fewest games played, fewest repeat
    opponents, rating, opponent pointrate) and greedy pod-fit scoring are
    generic Swiss-pairing logic, not multiplayer-specific, so 1v1 reuses it
    as-is - only the supported pod size differs. Ships its own
    Pairing1v1.params.yaml to override games_to_win to 2 (best-of-3 per the
    Magic Tournament Rules), re-declaring PairingDefault.params.yaml's other
    parameters unchanged since the sidecar lookup doesn't merge across
    ancestors.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)

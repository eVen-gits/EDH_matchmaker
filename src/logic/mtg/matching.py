from __future__ import annotations
from ..commander import matching as _commander_matching


class Pairing1v1(_commander_matching.PairingDefault):
    """Swiss pairing for 1v1 (2-player pod) Magic tournaments.

    Commander's PairingDefault sort key (fewest games played, fewest repeat
    opponents, rating, opponent pointrate) and greedy pod-fit scoring are
    generic Swiss-pairing logic, not multiplayer-specific, so 1v1 reuses it
    as-is - only the supported pod size differs. Ships no sidecar of its
    own, so it inherits commander/PairingDefault.params.yaml's parameters.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)

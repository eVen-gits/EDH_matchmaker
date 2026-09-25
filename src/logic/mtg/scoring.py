from __future__ import annotations
from ..commander import scoring as _commander_scoring


class Scoring1v1(_commander_scoring.ScoringDefault):
    """Match points for 1v1 Magic tournaments.

    Magic Tournament Rules Appendix C ("Match Points"): 3 points for a
    match win, 1 for a draw, 0 for a loss; a bye counts as an automatic
    2-0 win (3 points). This is exactly ScoringDefault's win/draw/bye-point
    formula (src/logic/commander/scoring.py) - only the point values
    differ, MTR-correct by default instead of Commander's, set in this
    class's own Scoring1v1.params.yaml sidecar.
    """

    IS_COMPLETE = True
    SUPPORTED_POD_SIZES = (2,)

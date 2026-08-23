"""ffopt -- a fantasy football draft optimizer.

Recommends who to take next, given your draft slot and who is already gone.
Adapts to any league's scoring and roster shape, and to how the room in front of
you is actually drafting rather than how an average draft usually goes.

    from ffopt import make_league, build_board, DraftState, Recommender

    league = make_league(n_teams=12, scoring="half_ppr")
    board = build_board(league)
    state = DraftState(league=league, my_team=4)   # 0-based: the 5th slot
    rec = Recommender(board)

    print(rec.format(rec.recommend(state), state))
"""

from .data import load_history, score_history
from .draft import DraftState, Roster
from .engine import build_board, find_players
from .league import (
    PPR,
    HALF_PPR,
    STANDARD,
    Bonus,
    League,
    Scoring,
    Slot,
    make_league,
    standard_roster,
)
from .opponents import OpponentModel
from .projections import ProjectionModel
from .recommend import Recommendation, Recommender
from .simulate import DraftSimulator
from .valuation import ValueBoard

__all__ = [
    "Bonus",
    "DraftSimulator",
    "DraftState",
    "HALF_PPR",
    "League",
    "OpponentModel",
    "PPR",
    "ProjectionModel",
    "Recommendation",
    "Recommender",
    "Roster",
    "STANDARD",
    "Scoring",
    "Slot",
    "ValueBoard",
    "build_board",
    "find_players",
    "load_history",
    "make_league",
    "score_history",
    "standard_roster",
]

__version__ = "0.1.0"

"""One-call setup: history in, ready-to-draft board out.

Fitting the projection model takes a little while, so the result is cached to
disk keyed by the league's scoring and shape.  A second run against the same
league starts instantly.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .data import CACHE_DIR, load_history
from .league import League
from .projections import ProjectionModel
from .valuation import ValueBoard


def _league_key(league: League, season: int) -> str:
    payload = {
        "season": season,
        "teams": league.n_teams,
        "bench": league.bench,
        "te_premium": league.te_premium,
        "scoring": sorted(league.scoring.points_per.items()),
        "slots": [(s.name, sorted(s.eligible), s.count) for s in league.starters],
        "positions": list(league.positions),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def build_board(
    league: League,
    *,
    season: Optional[int] = None,
    seasons: Optional[Sequence[int]] = None,
    refresh: bool = False,
    history: Optional[pd.DataFrame] = None,
) -> ValueBoard:
    """Load history, fit projections, and build the value board for ``league``."""
    if history is None:
        history = load_history(seasons)
    if season is None:
        season = int(history["season"].max()) + 1

    cache = CACHE_DIR / f"projections_{_league_key(league, season)}.parquet"
    if cache.exists() and not refresh:
        projections = pd.read_parquet(cache)
    else:
        model = ProjectionModel.fit(history, league, through_season=season - 1)
        projections = model.project(history, season)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        projections.to_parquet(cache, index=False)

    return ValueBoard.build(projections, league)


# --------------------------------------------------------------------------
# Finding players by the names humans actually type
# --------------------------------------------------------------------------


@dataclass
class Match:
    player_id: str
    player_name: str
    position: str
    score: float


def find_players(
    board: pd.DataFrame, query: str, *, limit: int = 5, cutoff: float = 0.55
) -> List[Match]:
    """Fuzzy-match a typed name against the board.

    Draft rooms move fast and nobody types "Amon-Ra St. Brown" correctly, so
    substring hits are preferred and everything else falls back to edit
    distance.
    """
    query = query.strip().lower()
    if not query:
        return []

    names = board["player_name"].astype(str)
    lowered = names.str.lower()

    exact = board[lowered == query]
    if len(exact) == 1:
        row = exact.iloc[0]
        return [Match(str(row["player_id"]), str(row["player_name"]), str(row["position"]), 1.0)]

    contains = board[lowered.str.contains(query, regex=False)]
    matches = [
        Match(str(r["player_id"]), str(r["player_name"]), str(r["position"]), 0.9)
        for _, r in contains.head(limit).iterrows()
    ]
    if matches:
        return matches

    close = difflib.get_close_matches(query, lowered.tolist(), n=limit, cutoff=cutoff)
    for name in close:
        row = board[lowered == name].iloc[0]
        matches.append(
            Match(
                str(row["player_id"]),
                str(row["player_name"]),
                str(row["position"]),
                difflib.SequenceMatcher(None, query, name).ratio(),
            )
        )
    return matches

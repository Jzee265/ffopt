"""Export projections to JSON for the standalone HTML draft board.

The page has to stay league-agnostic, which means it cannot be handed a list of
fantasy points -- points depend on scoring, and scoring is chosen in the page.
So what gets exported is the *projected stat line* plus *last season's actual
stat line*, and the page scores both and blends them exactly as
``ProjectionModel`` does (0.6 model / 0.4 last season).

That way switching a league from standard to PPR, or adding a superflex slot,
genuinely reshuffles the board in the browser rather than re-ranking a frozen
list.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffopt.data import STAT_COLUMNS, load_history
from ffopt.league import make_league
from ffopt.projections import ProjectionModel

# Stats the browser needs in order to score a line under any ruleset.
EXPORT_STATS = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_2pt_conversions",
    "fumbles_lost",
    "special_teams_tds",
]


def main() -> None:
    history = load_history()
    season = int(history["season"].max()) + 1

    # Scoring only affects the model through the recency blend, which the page
    # redoes itself -- so any league works here for fitting the stat lines.
    league = make_league()
    model = ProjectionModel.fit(history, league, through_season=season - 1)
    projected = model._project_core(history[history["season"] < season], season)

    last = history[history["season"] == season - 1].set_index("player_id")

    players = []
    for _, row in projected.iterrows():
        pid = str(row["player_id"])
        games = float(row["projected_games"])
        if games <= 0:
            continue

        line = {s: round(float(row.get(s, 0.0)), 2) for s in EXPORT_STATS}
        if pid in last.index:
            prev = last.loc[pid]
            if isinstance(prev, pd.DataFrame):
                prev = prev.iloc[0]
            prior = {s: round(float(prev.get(s, 0.0)), 2) for s in EXPORT_STATS}
            prior_games = float(prev.get("games", 0.0))
        else:
            prior, prior_games = None, 0.0

        players.append(
            {
                "id": pid,
                "name": str(row["player_name"]),
                "pos": str(row["position"]),
                "team": str(row.get("team", "") or ""),
                "age": None if pd.isna(row.get("age")) else round(float(row["age"]), 1),
                "g": round(games, 2),
                "s": line,
                "p": prior,
                "pg": round(prior_games, 2),
            }
        )

    # Drop players who cannot matter in any league, to keep the file small.
    def rough_points(entry: dict) -> float:
        s = entry["s"]
        return (
            s["passing_yards"] * 0.04
            + s["passing_tds"] * 4
            + s["rushing_yards"] * 0.1
            + s["rushing_tds"] * 6
            + s["receiving_yards"] * 0.1
            + s["receiving_tds"] * 6
            + s["receptions"] * 0.5
        )

    players.sort(key=rough_points, reverse=True)
    players = [p for p in players if rough_points(p) > 15][:500]

    payload = {
        "season": season,
        "blend": model.recency_blend,
        "generated_from": f"{int(history['season'].min())}-{int(history['season'].max())}",
        "players": players,
    }

    out = Path(__file__).parent / "board_data.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    size = out.stat().st_size / 1024
    print(f"wrote {len(players)} players to {out.name} ({size:.0f} KB), season {season}")


if __name__ == "__main__":
    main()

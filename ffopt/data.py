"""Historical data ingestion.

Pulls season-level raw statistics from nflverse (the open data project behind
nflfastR), joins them to roster data for age, and normalises columns into the
vocabulary that :mod:`ffopt.league` knows how to score.

Raw stats -- not fantasy points -- are what gets stored.  That is deliberate:
scoring is a property of the league, so points must be computed at valuation
time or the whole thing stops being league-agnostic.

Everything is cached to parquet on disk, so the network is touched once.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

CACHE_DIR = Path(
    os.environ.get("FFOPT_CACHE", Path(__file__).resolve().parent.parent / "cache")
)

#: nflverse player stats begin in 1999.
FIRST_SEASON = 1999

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_reg_{season}.parquet"
)

#: nflverse column -> our stat vocabulary.  Anything unmapped is dropped.
_COLUMN_MAP = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_interceptions": "interceptions",
    "passing_2pt_conversions": "passing_2pt_conversions",
    "carries": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "rushing_2pt_conversions": "rushing_2pt_conversions",
    "receptions": "receptions",
    "targets": "targets",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "receiving_2pt_conversions": "receiving_2pt_conversions",
    "special_teams_tds": "special_teams_tds",
    "fumbles_lost_total": "fumbles_lost",
    # Kicking, so K leagues work without a separate code path.
    "fg_made_0_19": "fg_made_0_19",
    "fg_made_20_29": "fg_made_20_29",
    "fg_made_30_39": "fg_made_30_39",
    "fg_made_40_49": "fg_made_40_49",
    "fg_made_50_59": "fg_made_50_59",
    "fg_made_60_": "fg_made_60_plus",
    "fg_missed": "fg_missed",
    "pat_made": "pat_made",
    "pat_missed": "pat_missed",
}

STAT_COLUMNS: tuple = tuple(_COLUMN_MAP.values())

ID_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "team",
    "season",
    "age",
    "years_exp",
    "games",
)


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def latest_available_season(probe_from: Optional[int] = None) -> int:
    """Most recent season with published stats, found by probing backwards."""
    import datetime as _dt

    import requests

    year = probe_from or _dt.date.today().year
    for candidate in range(year, year - 4, -1):
        try:
            resp = requests.head(
                _STATS_URL.format(season=candidate), allow_redirects=True, timeout=20
            )
            if resp.status_code == 200:
                return candidate
        except Exception:
            continue
    raise RuntimeError("could not determine the latest available NFL season")


def load_history(
    seasons: Optional[Sequence[int]] = None,
    *,
    refresh: bool = False,
    positions: Optional[Iterable[str]] = FANTASY_POSITIONS,
) -> pd.DataFrame:
    """One row per player-season of raw fantasy-relevant statistics.

    Columns: :data:`ID_COLUMNS` plus every entry in :data:`STAT_COLUMNS`.
    """
    if seasons is None:
        seasons = list(range(FIRST_SEASON, latest_available_season() + 1))
    seasons = sorted({int(s) for s in seasons})

    cache = _cache_path(f"history_{seasons[0]}_{seasons[-1]}.parquet")
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
    else:
        df = _download_history(seasons)
        df.to_parquet(cache, index=False)

    if positions is not None:
        df = df[df["position"].isin(set(positions))]
    return df.reset_index(drop=True)


def _download_history(seasons: Sequence[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for season in seasons:
        raw = pd.read_parquet(_STATS_URL.format(season=season))
        available = {c: v for c, v in _COLUMN_MAP.items() if c in raw.columns}
        keep = ["player_id", "player_display_name", "position", "recent_team", "season", "games"]
        keep = [c for c in keep if c in raw.columns]
        block = raw[keep + list(available)].rename(columns=available)
        block = block.rename(
            columns={"player_display_name": "player_name", "recent_team": "team"}
        )
        frames.append(block)

    stats = pd.concat(frames, ignore_index=True)
    stats = stats[stats["position"].notna()]

    ages = _download_ages(seasons)
    merged = stats.merge(ages, on=["player_id", "season"], how="left")

    for col in STAT_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["games"] = pd.to_numeric(merged.get("games"), errors="coerce").fillna(0.0)
    merged["age"] = pd.to_numeric(merged.get("age"), errors="coerce")
    merged["years_exp"] = pd.to_numeric(merged.get("years_exp"), errors="coerce")

    # Some players lack a roster row; back-fill experience from first appearance.
    first_seen = merged.groupby("player_id")["season"].transform("min")
    merged["years_exp"] = merged["years_exp"].fillna(merged["season"] - first_seen)

    ordered = [c for c in ID_COLUMNS if c in merged.columns] + list(STAT_COLUMNS)
    return (
        merged[ordered]
        .sort_values(["season", "player_id"])
        .reset_index(drop=True)
    )


def _download_ages(seasons: Sequence[int]) -> pd.DataFrame:
    """Age and experience per player-season, from nflverse rosters."""
    import nfl_data_py as nfl

    try:
        rosters = nfl.import_seasonal_rosters(list(seasons))
    except Exception:
        return pd.DataFrame(columns=["player_id", "season", "age", "years_exp"])

    cols = [c for c in ("player_id", "season", "age", "years_exp") if c in rosters.columns]
    return (
        rosters[cols]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id", "season"], keep="first")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_history(history: pd.DataFrame, league) -> pd.DataFrame:
    """Attach ``fantasy_points`` to a history frame under a league's scoring.

    Applied per position so that position-conditional rules (a TE premium, for
    instance) are respected.
    """
    out = history.copy().reset_index(drop=True)
    points = np.zeros(len(out), dtype=float)

    for position, idx in out.groupby("position").groups.items():
        rules = league.scoring_for(str(position))
        block = out.loc[idx]
        subtotal = np.zeros(len(block), dtype=float)

        for stat, per_unit in rules.points_per.items():
            if stat in block.columns:
                subtotal += per_unit * block[stat].to_numpy(dtype=float)

        if rules.bonuses:
            games = block["games"].to_numpy(dtype=float)
            games = np.where(games > 0, games, league.games)
            for bonus in rules.bonuses:
                if bonus.stat not in block.columns:
                    continue
                value = block[bonus.stat].to_numpy(dtype=float)
                if bonus.per_game:
                    clears = (value / games) >= bonus.threshold
                    subtotal += np.where(clears, bonus.points * games, 0.0)
                else:
                    subtotal += np.where(value >= bonus.threshold, bonus.points, 0.0)

        points[np.asarray(idx)] = subtotal

    out["fantasy_points"] = points
    out["points_per_game"] = np.where(
        out["games"] > 0, out["fantasy_points"] / out["games"], 0.0
    )
    return out

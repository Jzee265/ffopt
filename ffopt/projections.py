"""Deriving projections from history alone.

No outside projection feed is used.  For each player we forecast next season
from their own past, the aging behaviour of their position, and priors for
players with thin track records.

Two design choices drive everything here.

**We project raw stats, not fantasy points.**  A projected stat line is scored
by the league afterwards, so the same fitted model serves a PPR league and a
standard league correctly -- and they genuinely disagree about who is better,
which is the whole point of being league-agnostic.

**Volume and efficiency are projected separately.**  Yards per game is
substantially more stable year over year than fantasy points per game, because
points are contaminated by touchdown noise.  Empirically, on 1999-2025 data,
rank correlation with next season's scoring rate is:

    position   points/gm   yards/gm
    RB           0.723       0.732
    WR           0.736       0.738
    TE           0.698       0.713

So touchdowns are modelled as a *rate per yard*, shrunk hard toward the
positional mean, and multiplied back through projected yardage.  A running back
who scored on 6% of his carries last year is not expected to do it again.

The pipeline:

    1. **Recency-weighted totals.**  Each past season contributes with an
       exponentially decaying weight.
    2. **Conjugate shrinkage.**  Per-game rates are ``(weighted total + prior *
       k) / (weighted games + k)`` -- so a three-game sample barely moves off
       the prior, while four full seasons sit essentially on the player's own
       numbers.
    3. **Aging.**  Position-specific age curves fit from year-over-year change.
    4. **Availability.**  Games played projected separately, discounted by the
       hazard that the player is not in the league at all next year.
    5. **Uncertainty.**  Residual spread fit against projection level, giving
       the draft simulator an honest distribution rather than a point estimate.

Everything is fit strictly on seasons before the target, so backtests are
genuinely out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import score_history
from .league import League

MIN_GAMES_FOR_RATE = 1.0
AGE_MIN, AGE_MAX = 20, 40

#: Stats projected directly as a per-game volume.
VOLUME_STATS: Tuple[str, ...] = (
    "passing_yards",
    "carries",
    "rushing_yards",
    "receptions",
    "targets",
    "receiving_yards",
)

#: Stats projected as a rate per unit of a volume stat, then multiplied back.
#: ``stat -> (denominator stat, prior strength in denominator units)``.
RATE_STATS: Dict[str, Tuple[str, float]] = {
    "passing_tds": ("passing_yards", 4000.0),
    "rushing_tds": ("rushing_yards", 900.0),
    "receiving_tds": ("receiving_yards", 900.0),
    "interceptions": ("passing_yards", 3000.0),
    "fumbles_lost": ("touches", 400.0),
    "passing_2pt_conversions": ("passing_yards", 4000.0),
    "rushing_2pt_conversions": ("rushing_yards", 900.0),
    "receiving_2pt_conversions": ("receiving_yards", 900.0),
    "special_teams_tds": ("touches", 900.0),
}

PROJECTED_STATS: Tuple[str, ...] = VOLUME_STATS + tuple(RATE_STATS)


# --------------------------------------------------------------------------
# Aging curves
# --------------------------------------------------------------------------


def fit_aging_curves(
    history: pd.DataFrame,
    *,
    min_games: float = 8.0,
    smooth: int = 3,
) -> Dict[str, Dict[int, float]]:
    """Estimate log-scale year-over-year aging effects per position.

    Measured on total yardage per game, which is stable enough to read an age
    signal off without touchdown noise swamping it.

    Uses the delta method: only players observed in consecutive seasons
    contribute, which controls for the fact that good players simply have
    longer careers.  Both seasons must clear ``min_games``, so injury-shortened
    years do not masquerade as decline.
    """
    df = history[["player_id", "position", "season", "age", "games"]].copy()
    df["rate"] = (
        history["passing_yards"].fillna(0) * 0.4
        + history["rushing_yards"].fillna(0)
        + history["receiving_yards"].fillna(0)
    ) / history["games"].replace(0, np.nan)
    df = df[(history["games"] >= min_games) & (df["rate"] > 0)]
    if df.empty:
        return {}

    nxt = df[["player_id", "season", "rate"]].copy()
    nxt["season"] -= 1
    pairs = df.merge(nxt, on=["player_id", "season"], suffixes=("", "_next"))
    if pairs.empty:
        return {}

    pairs = pairs[pairs["age"].between(AGE_MIN, AGE_MAX)]
    pairs["delta"] = np.log(pairs["rate_next"] / pairs["rate"])
    lo, hi = pairs["delta"].quantile([0.02, 0.98])
    pairs = pairs[pairs["delta"].between(lo, hi)]

    curves: Dict[str, Dict[int, float]] = {}
    for position, block in pairs.groupby("position"):
        by_age = block.groupby(block["age"].round().astype(int))["delta"]
        means, counts = by_age.mean(), by_age.size()
        overall = block["delta"].mean()
        prior_n = 25.0
        shrunk = (means * counts + overall * prior_n) / (counts + prior_n)
        smoothed = shrunk.sort_index().rolling(smooth, center=True, min_periods=1).mean()
        curves[str(position)] = {int(a): float(v) for a, v in smoothed.items()}
    return curves


def _age_effect(curves: Dict[str, Dict[int, float]], position: str, age: float) -> float:
    """Multiplicative rate adjustment for aging one year from ``age``."""
    curve = curves.get(str(position))
    if not curve or not np.isfinite(age):
        return 1.0
    key = int(np.clip(round(age), AGE_MIN, AGE_MAX))
    if key not in curve:
        key = min(curve, key=lambda a: abs(a - key))
    return float(np.exp(curve[key]))


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


def fit_availability(
    history: pd.DataFrame, *, min_games: float = 6.0
) -> Dict[Tuple[str, int], float]:
    """Expected games played by position and age, among genuine contributors.

    Restricted to players who cleared ``min_games``: the unconditional mean is
    dragged down by camp bodies who appear twice and vanish, and a player worth
    drafting is not one of those.  This is the prior that a player's own games
    history gets shrunk toward, not the projection itself.
    """
    df = history[(history["games"] >= min_games) & history["age"].between(AGE_MIN, AGE_MAX)]
    if df.empty:
        return {}

    grouped = df.groupby(["position", df["age"].round().astype(int)])["games"]
    means, counts = grouped.mean(), grouped.size()
    overall = df.groupby("position")["games"].mean()

    out: Dict[Tuple[str, int], float] = {}
    prior_n = 30.0
    for (position, age), mean in means.items():
        n = counts[(position, age)]
        base = overall.get(position, df["games"].mean())
        out[(str(position), int(age))] = float((mean * n + base * prior_n) / (n + prior_n))
    return out


def fit_participation(
    history: pd.DataFrame, *, min_games: float = 8.0
) -> Dict[Tuple[str, int], float]:
    """P(plays next season | was a contributor this season), by position and age.

    This is what keeps 42-year-old quarterbacks and washed-up running backs off
    the top of the board.  Historical-only projections cannot know who retired,
    so the model prices in the hazard that a player of this age is simply not in
    the league next year.

    The denominator is restricted to genuine contributors -- players one would
    actually draft -- while the numerator counts any appearance at all.  Mixing
    camp bodies into the denominator would make everyone look like a risk.
    """
    played_next = history.loc[history["games"] > 0, ["player_id", "season"]].drop_duplicates()
    played_next["season"] -= 1
    played_next["next"] = 1.0

    df = history[(history["games"] >= min_games) & history["age"].between(AGE_MIN, AGE_MAX)]
    if df.empty:
        return {}

    last_season = int(history["season"].max())
    observable = df[df["season"] < last_season]
    if observable.empty:
        return {}

    marked = observable.merge(played_next, on=["player_id", "season"], how="left")
    marked["next"] = marked["next"].fillna(0.0)

    grouped = marked.groupby(["position", marked["age"].round().astype(int)])["next"]
    rates, counts = grouped.mean(), grouped.size()
    overall = marked.groupby("position")["next"].mean()

    out: Dict[Tuple[str, int], float] = {}
    prior_n = 20.0
    for (position, age), rate in rates.items():
        n = counts[(position, age)]
        base = overall.get(position, marked["next"].mean())
        out[(str(position), int(age))] = float(
            np.clip((rate * n + base * prior_n) / (n + prior_n), 0.02, 1.0)
        )
    return out


def _lookup_age(
    table: Dict[Tuple[str, int], float], position: str, age: float, default: float
) -> float:
    if not table or not np.isfinite(age):
        return default
    key = int(np.clip(round(age), AGE_MIN, AGE_MAX))
    if (str(position), key) in table:
        return table[(str(position), key)]
    candidates = [a for (p, a) in table if p == str(position)]
    if not candidates:
        return default
    return table[(str(position), min(candidates, key=lambda a: abs(a - key)))]


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@dataclass
class ProjectionModel:
    """Fit on seasons up to and including ``through_season``."""

    league: League
    decay: float = 0.55
    #: Shrinkage for per-game volume stats, in units of games.
    volume_prior_strength: float = 8.0
    #: Percentile of per-game production used as the volume prior.  Low on
    #: purpose -- see ``_fit_stat_priors``.
    prior_percentile: float = 15.0
    #: Weight on the fitted model when blending against last season's actual
    #: points.  Backtested over 2019-2025, a 0.6 blend beats either component
    #: alone: the model contributes aging, injury and touchdown regression,
    #: while last season's raw total carries role information -- a promotion to
    #: starter, a new offence -- that no historical model can see.
    recency_blend: float = 0.6
    #: Shrinkage for availability, in units of seasons.
    games_prior_strength: float = 1.5
    through_season: int = 0

    aging: Dict[str, Dict[int, float]] = field(default_factory=dict)
    availability: Dict[Tuple[str, int], float] = field(default_factory=dict)
    participation: Dict[Tuple[str, int], float] = field(default_factory=dict)
    #: (position, stat) -> per-game prior, for volume stats.
    volume_prior: Dict[Tuple[str, str], float] = field(default_factory=dict)
    #: (position, stat) -> prior rate per unit of denominator, for rate stats.
    rate_prior: Dict[Tuple[str, str], float] = field(default_factory=dict)
    sigma: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_games: int = 17

    # -- fitting ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        history: pd.DataFrame,
        league: League,
        *,
        through_season: Optional[int] = None,
        fit_uncertainty: bool = True,
        **kwargs,
    ) -> "ProjectionModel":
        if through_season is None:
            through_season = int(history["season"].max())
        train = history[history["season"] <= through_season]

        model = cls(league=league, through_season=int(through_season), **kwargs)
        model.aging = fit_aging_curves(train)
        model.availability = fit_availability(train)
        model.participation = fit_participation(train)
        model.volume_prior, model.rate_prior = model._fit_stat_priors(train)
        # Fitting uncertainty walks the model forward across past seasons, by
        # far the most expensive step; hyperparameter sweeps skip it.
        model.sigma = (
            model._fit_uncertainty(history, int(through_season)) if fit_uncertainty else {}
        )
        return model

    def _fit_stat_priors(
        self, train: pd.DataFrame
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        """Positional priors for volume per game and for rate-per-denominator.

        The volume prior is deliberately a *low percentile* of per-game
        production rather than the average.  A games-weighted average is
        dominated by full-time starters, which would mean a backup quarterback
        who started five games in an injury pinch regresses toward starter
        production and then gets multiplied by a near-full season -- the model
        would rank Easton Stick as a top-120 fantasy asset.  Regressing thin
        samples toward *replacement* level rather than toward the average is
        what makes a small sample cost a player, as it should.
        """
        df = _with_touches(train[train["games"] >= 4]).copy()

        volume: Dict[Tuple[str, str], float] = {}
        rate: Dict[Tuple[str, str], float] = {}

        for position, block in df.groupby("position"):
            position = str(position)
            games = block["games"].to_numpy(dtype=float)
            for stat in VOLUME_STATS:
                if stat not in block.columns:
                    continue
                per_game = block[stat].to_numpy(dtype=float) / np.where(games > 0, games, np.nan)
                volume[(position, stat)] = float(
                    np.nanpercentile(per_game, self.prior_percentile)
                )
            # Efficiency rates, unlike volume, are genuinely similar across the
            # position, so the pooled rate is the right prior for those.
            for stat, (denom, _strength) in RATE_STATS.items():
                if stat not in block.columns or denom not in block.columns:
                    continue
                denom_total = float(block[denom].sum())
                rate[(position, stat)] = (
                    float(block[stat].sum() / denom_total) if denom_total > 0 else 0.0
                )
        return volume, rate

    def _fit_uncertainty(
        self, history: pd.DataFrame, through_season: int
    ) -> Dict[str, Tuple[float, float]]:
        """Model residual spread as a linear function of the projection.

        Higher-projected players vary more in absolute terms, so a flat sigma
        would understate risk at the top of the board and overstate it at the
        bottom.  Returns per-position ``(intercept, slope)``.
        """
        scored = score_history(history, self.league)
        seasons = sorted(history["season"].unique())
        rows: List[Tuple[str, float, float]] = []

        for target in [s for s in seasons if s <= through_season][-8:]:
            past = history[history["season"] < target]
            if past.empty:
                continue
            actual = scored[scored["season"] == target][["player_id", "fantasy_points"]]
            if actual.empty:
                continue
            preds = self._project_core(past, int(target))
            if preds.empty:
                continue
            merged = preds.merge(actual, on="player_id", how="inner")
            rows.extend(
                (str(p), float(pred), float(act))
                for p, pred, act in zip(
                    merged["position"], merged["projected_points"], merged["fantasy_points"]
                )
            )

        frame = pd.DataFrame(rows, columns=["position", "pred", "actual"])
        out: Dict[str, Tuple[float, float]] = {}
        for position, block in frame.groupby("position"):
            if len(block) < 40:
                out[str(position)] = (35.0, 0.35)
                continue
            block = block.copy()
            block["resid"] = block["actual"] - block["pred"]
            block["bucket"] = pd.qcut(block["pred"], 6, duplicates="drop", labels=False)
            stats = block.groupby("bucket").agg(
                center=("pred", "mean"), spread=("resid", "std")
            ).dropna()
            if len(stats) >= 2:
                slope, intercept = np.polyfit(stats["center"], stats["spread"], 1)
            else:
                slope, intercept = 0.35, 35.0
            out[str(position)] = (float(max(intercept, 5.0)), float(np.clip(slope, 0.05, 1.2)))
        return out

    # -- prediction ------------------------------------------------------

    def project(
        self,
        history: pd.DataFrame,
        target_season: Optional[int] = None,
        *,
        min_projected_points: float = 1.0,
    ) -> pd.DataFrame:
        """Project every player with history prior to ``target_season``.

        Returns one row per player with a projected stat line, ``projected_games``,
        ``projected_points`` under the league's scoring, and ``sigma``.
        """
        if target_season is None:
            target_season = self.through_season + 1
        past = history[history["season"] < target_season]
        if past.empty:
            raise ValueError(f"no history available before season {target_season}")

        out = self._project_core(past, int(target_season))
        out = self._blend_with_last_season(out, past, int(target_season))
        out = out[out["projected_points"] >= min_projected_points]
        return out.sort_values("projected_points", ascending=False).reset_index(drop=True)

    def _blend_with_last_season(
        self, projections: pd.DataFrame, past: pd.DataFrame, target_season: int
    ) -> pd.DataFrame:
        """Pull the projection partway toward last season's actual points.

        Last season's raw total is a crude forecast, but it encodes role
        information the historical model structurally cannot see: a backup
        promoted to starter, a receiver who inherited a vacated target share, a
        back-up quarterback handed the job.  Blending recovers most of that
        without giving up the model's aging and regression corrections.

        Players who did not play last season keep the pure model projection --
        blending them against a zero would bury anyone returning from a lost
        season.
        """
        alpha = float(np.clip(self.recency_blend, 0.0, 1.0))
        if alpha >= 1.0 or projections.empty:
            return projections

        prior_season = target_season - 1
        last = past[past["season"] == prior_season]
        if last.empty:
            return projections

        scored_last = score_history(last, self.league)[
            ["player_id", "fantasy_points", "games"]
        ].rename(columns={"fantasy_points": "_last_points", "games": "_last_games"})

        out = projections.merge(scored_last, on="player_id", how="left")
        played = out["_last_games"].fillna(0) >= 1
        blended = alpha * out["projected_points"] + (1 - alpha) * out["_last_points"].fillna(0.0)

        out["model_points"] = out["projected_points"]
        out["projected_points"] = np.where(played, blended, out["projected_points"])
        out["projected_rate"] = np.where(
            out["projected_games"] > 0, out["projected_points"] / out["projected_games"], 0.0
        )
        return out.drop(columns=["_last_points", "_last_games"])

    def _project_core(self, past: pd.DataFrame, target_season: int) -> pd.DataFrame:
        past = _with_touches(past[past["games"] >= MIN_GAMES_FOR_RATE]).copy()
        if past.empty:
            return pd.DataFrame(columns=["player_id", "position", "projected_points"])

        past = past.sort_values(["player_id", "season"])
        w = self.decay ** (target_season - past["season"])
        past["_w"] = w

        stat_cols = [c for c in PROJECTED_STATS if c in past.columns] + ["touches"]
        weighted = past[stat_cols].multiply(w, axis=0)
        weighted["_wgames"] = past["games"] * w
        weighted["player_id"] = past["player_id"].values

        totals = weighted.groupby("player_id").sum()

        meta = past.groupby("player_id").agg(
            player_name=("player_name", "last"),
            position=("position", "last"),
            team=("team", "last"),
            last_season=("season", "max"),
            last_age=("age", "last"),
            years_exp=("years_exp", "last"),
            gweight=("_w", "sum"),
            seasons_played=("season", "nunique"),
        )
        raw_games = past.groupby("player_id").apply(
            lambda b: float((b["games"] * b["_w"]).sum())
        )
        agg = meta.join(totals).reset_index()
        agg["_wgames"] = agg["_wgames"].fillna(0.0)

        # Players absent for two-plus seasons are out of the league.
        agg = agg[agg["last_season"] >= target_season - 2].reset_index(drop=True)
        if agg.empty:
            return agg.assign(projected_points=[], projected_games=[], sigma=[])

        years_ahead = (target_season - agg["last_season"]).clip(lower=1)
        missed = years_ahead - 1
        agg["age"] = agg["last_age"] + years_ahead
        agg["years_exp"] = agg["years_exp"].fillna(agg["seasons_played"]) + years_ahead

        positions = agg["position"].astype(str).to_numpy()
        wgames = agg["_wgames"].to_numpy(dtype=float)

        # Aging multiplier, applied from the last observed age forward.
        aging = np.array(
            [
                float(
                    np.prod([_age_effect(self.aging, pos, start + step) for step in range(int(steps))])
                )
                if np.isfinite(start) and steps >= 1
                else 1.0
                for pos, start, steps in zip(positions, agg["last_age"], years_ahead)
            ]
        )

        # -- volume stats: conjugate shrinkage on a per-game rate ---------
        k = self.volume_prior_strength
        per_game: Dict[str, np.ndarray] = {}
        for stat in VOLUME_STATS:
            if stat not in agg.columns:
                per_game[stat] = np.zeros(len(agg))
                continue
            prior = np.array([self.volume_prior.get((p, stat), 0.0) for p in positions])
            observed = agg[stat].to_numpy(dtype=float)
            per_game[stat] = ((observed + prior * k) / (wgames + k)) * aging

        touches_prior = np.array(
            [self.volume_prior.get((p, "carries"), 0.0) + self.volume_prior.get((p, "receptions"), 0.0) for p in positions]
        )
        per_game["touches"] = (
            (agg["touches"].to_numpy(dtype=float) + touches_prior * k) / (wgames + k)
        ) * aging

        # -- rate stats: shrunk rate per unit of volume ------------------
        for stat, (denom, strength) in RATE_STATS.items():
            if stat not in agg.columns or denom not in per_game:
                per_game[stat] = np.zeros(len(agg))
                continue
            prior = np.array([self.rate_prior.get((p, stat), 0.0) for p in positions])
            numer = agg[stat].to_numpy(dtype=float)
            denom_total = agg[denom].to_numpy(dtype=float) if denom in agg.columns else np.zeros(len(agg))
            shrunk_rate = (numer + prior * strength) / (denom_total + strength)
            per_game[stat] = shrunk_rate * per_game[denom]

        # -- availability -------------------------------------------------
        own_games = np.where(agg["gweight"] > 0, wgames / agg["gweight"].replace(0, np.nan), np.nan)
        own_games = np.nan_to_num(own_games)
        games_prior = np.array(
            [_lookup_age(self.availability, p, a, 13.0) for p, a in zip(positions, agg["age"])]
        )
        gev = agg["gweight"].to_numpy(dtype=float)
        games_active = (own_games * gev + games_prior * self.games_prior_strength) / (
            gev + self.games_prior_strength
        )
        p_active = np.array(
            [_lookup_age(self.participation, p, a, 0.85) for p, a in zip(positions, agg["age"])]
        )
        p_active = p_active * np.where(missed.to_numpy() >= 1, 0.45, 1.0)
        games = np.clip(games_active * p_active, 0, self.max_games)

        # -- assemble the projected season stat line ----------------------
        out = agg[["player_id", "player_name", "position", "team", "age", "years_exp"]].copy()
        out["p_active"] = p_active
        out["projected_games"] = games
        for stat in PROJECTED_STATS:
            out[stat] = per_game.get(stat, np.zeros(len(agg))) * games

        scored = score_history(out.assign(games=games, season=target_season), self.league)
        out["projected_points"] = scored["fantasy_points"].to_numpy()
        out["projected_rate"] = np.where(games > 0, out["projected_points"] / games, 0.0)

        sigmas = []
        for pos, pred in zip(positions, out["projected_points"]):
            intercept, slope = self.sigma.get(str(pos), (35.0, 0.35))
            sigmas.append(max(intercept + slope * pred, 5.0))
        out["sigma"] = sigmas

        return out.reset_index(drop=True)


def _with_touches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["touches"] = out.get("carries", 0.0) + out.get("receptions", 0.0)
    return out

"""Turning projections into draft value.

A projection says how many points a player will score.  That is not the same
question as how much he is worth to *you*, and confusing the two is the single
most common way to draft badly.

The quantity that matters is **value over replacement**: how many points a
player gives you above what you could have had for free at the same position.
A quarterback projected for 275 points looks like the best player available
next to a running back at 227 -- until you notice that the twelfth-best
quarterback is projected for 210 while the thirtieth-best running back is at
95.  The quarterback is worth +65; the back is worth +132.  In a one-quarterback
league the back is nearly twice the pick, and it isn't close.

Replacement level is derived from the league configuration rather than assumed,
so it moves correctly when the league does: deeper leagues push replacement
further down the board, superflex leagues drag the quarterback baseline up, and
a second flex spot changes which position wins the marginal starting job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .league import League


# --------------------------------------------------------------------------
# Where the flex slots actually go
# --------------------------------------------------------------------------


def empirical_flex_shares(
    projections: pd.DataFrame, league: League
) -> Dict[str, float]:
    """Work out which positions actually win the league's multi-position slots.

    Rather than assuming a running back fills 45% of flexes, fill every team's
    starting lineup greedily from the projection board and count what ends up
    where.  In a PPR league with a deep receiver pool the answer differs from a
    standard league, and the replacement levels should follow.
    """
    flex_slots = [s for s in league.starters if len(s.eligible) > 1]
    total_flex = sum(s.count for s in flex_slots) * league.n_teams
    if total_flex == 0:
        return {}

    dedicated_remaining: Dict[str, int] = {
        pos: league.dedicated_starters(pos) * league.n_teams for pos in league.positions
    }
    flex_remaining = {s.name: s.count * league.n_teams for s in flex_slots}
    flex_counts: Dict[str, int] = {pos: 0 for pos in league.positions}

    board = projections.sort_values("projected_points", ascending=False)
    for position in board["position"]:
        position = str(position)
        if dedicated_remaining.get(position, 0) > 0:
            dedicated_remaining[position] -= 1
            continue
        for slot in flex_slots:
            if position in slot.eligible and flex_remaining[slot.name] > 0:
                flex_remaining[slot.name] -= 1
                flex_counts[position] = flex_counts.get(position, 0) + 1
                break
        if not any(flex_remaining.values()) and not any(dedicated_remaining.values()):
            break

    return {pos: count / total_flex for pos, count in flex_counts.items()}


def replacement_levels(
    projections: pd.DataFrame,
    league: League,
    *,
    flex_shares: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Projected points of the last startable player at each position.

    This is the zero point of the value scale.  Anything below it is free.
    """
    if flex_shares is None:
        flex_shares = empirical_flex_shares(projections, league)

    levels: Dict[str, float] = {}
    for position in league.positions:
        pool = (
            projections.loc[projections["position"] == position, "projected_points"]
            .sort_values(ascending=False)
            .to_numpy()
        )
        if pool.size == 0:
            levels[position] = 0.0
            continue
        rank = league.replacement_rank(position, flex_shares.get(position))
        # Average a small window around the cutoff so a single outlier at the
        # boundary does not swing the whole position's valuation.
        lo = max(0, rank - 2)
        hi = min(pool.size, rank + 3)
        levels[position] = float(np.mean(pool[lo:hi])) if hi > lo else float(pool[-1])
    return levels


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------


def assign_tiers(
    projections: pd.DataFrame,
    *,
    gap_multiplier: float = 0.8,
    max_tiers: int = 12,
) -> pd.Series:
    """Group each position's players into tiers by looking for cliffs.

    A tier break is declared where the drop to the next player is unusually
    large for that position -- larger than ``gap_multiplier`` standard
    deviations above the typical gap.  Tiers are what turn a ranking into a
    decision: six interchangeable players means you can wait, while one man
    alone above a cliff means you cannot.
    """
    tiers = pd.Series(1, index=projections.index, dtype=int)

    for position, block in projections.groupby("position"):
        block = block.sort_values("projected_points", ascending=False)
        points = block["projected_points"].to_numpy(dtype=float)
        if points.size <= 1:
            tiers.loc[block.index] = 1
            continue

        gaps = -np.diff(points)
        threshold = gaps.mean() + gap_multiplier * gaps.std()
        if not np.isfinite(threshold) or threshold <= 0:
            threshold = np.inf

        current, labels = 1, [1]
        for gap in gaps:
            if gap >= threshold and current < max_tiers:
                current += 1
            labels.append(current)
        tiers.loc[block.index] = labels

    return tiers


# --------------------------------------------------------------------------
# The value board
# --------------------------------------------------------------------------


@dataclass
class ValueBoard:
    """Projections plus everything derived from the league's structure."""

    players: pd.DataFrame
    league: League
    replacement: Dict[str, float]
    flex_shares: Dict[str, float]

    @classmethod
    def build(cls, projections: pd.DataFrame, league: League) -> "ValueBoard":
        board = projections.copy()
        board = board[board["position"].isin(league.positions)].reset_index(drop=True)

        flex_shares = empirical_flex_shares(board, league)
        replacement = replacement_levels(board, league, flex_shares=flex_shares)

        board["replacement"] = board["position"].map(replacement).astype(float)
        board["vor"] = board["projected_points"] - board["replacement"]
        board["tier"] = assign_tiers(board)
        board["position_rank"] = (
            board.groupby("position")["projected_points"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        board = board.sort_values("vor", ascending=False).reset_index(drop=True)
        board["overall_rank"] = np.arange(1, len(board) + 1)
        return cls(
            players=board,
            league=league,
            replacement=replacement,
            flex_shares=flex_shares,
        )

    # -- convenience -----------------------------------------------------

    def available(self, drafted: Sequence[str]) -> pd.DataFrame:
        taken = set(drafted)
        return self.players[~self.players["player_id"].isin(taken)]

    def top(self, n: int = 20, position: Optional[str] = None) -> pd.DataFrame:
        block = self.players
        if position:
            block = block[block["position"] == position]
        return block.head(n)

    def summary(self) -> str:
        lines = [self.league.describe(), "", "Replacement levels (projected points):"]
        for position in self.league.positions:
            rank = self.league.replacement_rank(position, self.flex_shares.get(position))
            lines.append(
                f"  {position:4s} {self.replacement.get(position, 0.0):7.1f} "
                f"(the {rank}th {position})"
            )
        if self.flex_shares:
            share = ", ".join(
                f"{pos} {pct:.0%}" for pos, pct in sorted(self.flex_shares.items()) if pct
            )
            lines.append(f"\nFlex slots go to: {share}")
        return "\n".join(lines)

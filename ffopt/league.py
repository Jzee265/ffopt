"""League configuration.

Everything about a league that the optimizer needs to know lives here. The
design goal is that *any* league can be expressed without touching the rest of
the codebase:

  * Scoring is a dict from raw stat category -> points per unit, plus optional
    threshold bonuses.  A league that awards 6 points for passing TDs, 0.5 per
    reception and a 3-point bonus for a 100-yard rushing game is just a
    different dict, not a different code path.
  * Roster slots are a list of (name, eligible positions, count).  FLEX,
    superflex, 2QB, TE-premium, WR/RB-only flexes and IDP all fall out of that
    representation for free.

Nothing downstream hardcodes "QB/RB/WR/TE".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

#: Raw stat columns the projection layer produces.  Scoring rules are keyed on
#: these names.  Anything not mentioned in a league's scoring dict is worth 0.
STAT_CATEGORIES: Tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class Bonus:
    """A threshold bonus, e.g. +3 points for a 300-yard passing game.

    ``per_game`` bonuses are evaluated against per-game production and then
    multiplied by games played, which is the honest way to handle them when
    all we project is a season total.  ``per_season`` bonuses are evaluated
    once against the season total.
    """

    stat: str
    threshold: float
    points: float
    per_game: bool = True


@dataclass(frozen=True)
class Scoring:
    """Points awarded per unit of each raw stat."""

    points_per: Dict[str, float] = field(default_factory=dict)
    bonuses: Tuple[Bonus, ...] = ()

    def score(self, stats: Dict[str, float], games: float = 17.0) -> float:
        """Convert a line of raw season stats into fantasy points."""
        total = 0.0
        for stat, per_unit in self.points_per.items():
            total += per_unit * float(stats.get(stat, 0.0) or 0.0)

        for bonus in self.bonuses:
            value = float(stats.get(bonus.stat, 0.0) or 0.0)
            if bonus.per_game:
                if games <= 0:
                    continue
                # Expected number of games clearing the threshold, approximated
                # from the per-game average.  Crude, but unbiased enough for
                # draft-time valuation and far better than ignoring bonuses.
                per_game_value = value / games
                if per_game_value >= bonus.threshold:
                    total += bonus.points * games
            elif value >= bonus.threshold:
                total += bonus.points

        return total

    def with_overrides(self, **points_per: float) -> "Scoring":
        merged = dict(self.points_per)
        merged.update(points_per)
        return Scoring(points_per=merged, bonuses=self.bonuses)


def _base_points(reception_points: float) -> Dict[str, float]:
    return {
        "passing_yards": 0.04,
        "passing_tds": 4.0,
        "interceptions": -2.0,
        "passing_2pt_conversions": 2.0,
        "rushing_yards": 0.1,
        "rushing_tds": 6.0,
        "rushing_2pt_conversions": 2.0,
        "receptions": reception_points,
        "receiving_yards": 0.1,
        "receiving_tds": 6.0,
        "receiving_2pt_conversions": 2.0,
        "fumbles_lost": -2.0,
        "special_teams_tds": 6.0,
    }


STANDARD = Scoring(points_per=_base_points(0.0))
HALF_PPR = Scoring(points_per=_base_points(0.5))
PPR = Scoring(points_per=_base_points(1.0))

#: Common presets, addressable by name from the CLI.
SCORING_PRESETS: Dict[str, Scoring] = {
    "standard": STANDARD,
    "half_ppr": HALF_PPR,
    "ppr": PPR,
}


# --------------------------------------------------------------------------
# Roster
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """A startable roster position.

    ``eligible`` is the set of player positions that may fill it.  A FLEX is
    just ``{"RB", "WR", "TE"}``; a superflex adds ``"QB"``.
    """

    name: str
    eligible: FrozenSet[str]
    count: int = 1

    @staticmethod
    def of(name: str, eligible: Iterable[str], count: int = 1) -> "Slot":
        return Slot(name=name, eligible=frozenset(eligible), count=count)


def standard_roster(
    *,
    qb: int = 1,
    rb: int = 2,
    wr: int = 2,
    te: int = 1,
    flex: int = 1,
    superflex: int = 0,
    k: int = 1,
    dst: int = 1,
) -> Tuple[Slot, ...]:
    """Build the usual roster shape without spelling out every slot."""
    slots: List[Slot] = []
    if qb:
        slots.append(Slot.of("QB", ["QB"], qb))
    if rb:
        slots.append(Slot.of("RB", ["RB"], rb))
    if wr:
        slots.append(Slot.of("WR", ["WR"], wr))
    if te:
        slots.append(Slot.of("TE", ["TE"], te))
    if flex:
        slots.append(Slot.of("FLEX", ["RB", "WR", "TE"], flex))
    if superflex:
        slots.append(Slot.of("SUPERFLEX", ["QB", "RB", "WR", "TE"], superflex))
    if k:
        slots.append(Slot.of("K", ["K"], k))
    if dst:
        slots.append(Slot.of("DST", ["DST"], dst))
    return tuple(slots)


# --------------------------------------------------------------------------
# League
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class League:
    """A complete description of the league being drafted."""

    n_teams: int = 12
    scoring: Scoring = HALF_PPR
    starters: Tuple[Slot, ...] = field(default_factory=standard_roster)
    bench: int = 6
    #: Positions the optimizer is allowed to consider at all.
    positions: Tuple[str, ...] = ("QB", "RB", "WR", "TE")
    #: Number of games in the fantasy regular season; used to scale bonuses and
    #: to weight availability.
    games: int = 17
    #: Extra points per reception awarded to tight ends only.  Kept separate
    #: from ``scoring`` because it is position-conditional.
    te_premium: float = 0.0

    def scoring_for(self, position: str) -> Scoring:
        """Scoring rules as they apply to one position."""
        if position == "TE" and self.te_premium:
            base = self.scoring.points_per.get("receptions", 0.0)
            return self.scoring.with_overrides(receptions=base + self.te_premium)
        return self.scoring

    # -- derived ---------------------------------------------------------

    @property
    def roster_size(self) -> int:
        return self.n_starter_slots + self.bench

    @property
    def n_starter_slots(self) -> int:
        return sum(slot.count for slot in self.starters)

    @property
    def total_picks(self) -> int:
        return self.n_teams * self.roster_size

    def slots_for(self, position: str) -> Tuple[Slot, ...]:
        return tuple(s for s in self.starters if position in s.eligible)

    def dedicated_starters(self, position: str) -> int:
        """Slots that *only* this position can fill, per team."""
        return sum(
            s.count for s in self.starters if s.eligible == frozenset({position})
        )

    def flex_slots_for(self, position: str) -> int:
        """Multi-position slots this position competes for, per team."""
        return sum(
            s.count
            for s in self.starters
            if position in s.eligible and len(s.eligible) > 1
        )

    def replacement_rank(self, position: str, flex_share: Optional[float] = None) -> int:
        """The rank at which a position becomes freely available.

        This is the backbone of Value Over Replacement.  A position's
        replacement level sits just past the last player at that position who
        will realistically be starting somewhere in the league.  Dedicated
        slots contribute fully; flex slots contribute in proportion to how
        often that position actually wins the flex.
        """
        dedicated = self.dedicated_starters(position) * self.n_teams
        flex = self.flex_slots_for(position) * self.n_teams
        if flex_share is None:
            flex_share = self.default_flex_share(position)
        return max(1, int(round(dedicated + flex * flex_share)))

    def default_flex_share(self, position: str) -> float:
        """Prior on how often a position wins a flex slot it is eligible for.

        Overridden empirically by the valuation layer once projections exist;
        these are only the cold-start values.
        """
        return {"RB": 0.45, "WR": 0.45, "TE": 0.10, "QB": 0.90}.get(position, 0.0)

    def describe(self) -> str:
        slot_desc = ", ".join(
            f"{s.count}x{s.name}" if s.count > 1 else s.name for s in self.starters
        )
        return (
            f"{self.n_teams}-team | {slot_desc} | {self.bench} bench "
            f"| {self.roster_size} roster spots | {self.total_picks} total picks"
        )


# --------------------------------------------------------------------------
# Convenience constructors
# --------------------------------------------------------------------------


def make_league(
    n_teams: int = 12,
    scoring: str | Scoring = "half_ppr",
    *,
    bench: int = 6,
    include_k_dst: bool = False,
    superflex: bool = False,
    te_premium: float = 0.0,
    **roster_kwargs,
) -> League:
    """Build a League from plain arguments.

    ``te_premium`` adds extra points per reception for tight ends only; that is
    handled by the projection layer, which is why it is stored on the league
    rather than folded into the scoring dict.
    """
    if isinstance(scoring, str):
        try:
            rules = SCORING_PRESETS[scoring]
        except KeyError:
            raise ValueError(
                f"unknown scoring preset {scoring!r}; "
                f"choose from {sorted(SCORING_PRESETS)} or pass a Scoring object"
            ) from None
    else:
        rules = scoring

    starters = standard_roster(
        superflex=1 if superflex else 0,
        k=1 if include_k_dst else 0,
        dst=1 if include_k_dst else 0,
        **roster_kwargs,
    )
    positions: Tuple[str, ...] = ("QB", "RB", "WR", "TE")
    if include_k_dst:
        positions = positions + ("K", "DST")

    return League(
        n_teams=n_teams,
        scoring=rules,
        starters=starters,
        bench=bench,
        positions=positions,
        te_premium=te_premium,
    )

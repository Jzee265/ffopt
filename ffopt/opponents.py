"""Modelling what the other eleven people in the room will actually do.

Most draft tools assume opponents follow average draft position.  That
assumption is comfortable and wrong.  Real drafts have personality: one year
quarterbacks fly off the board in the third round, the next year they last
until the tenth; somebody always reaches two rounds early for their own team's
running back; a run on tight ends starts because two people panicked.

So this model does not assume a fixed ordering.  It starts from a loose,
value-shaped prior and then **re-fits itself to the room from the picks it has
actually seen**.  Three forces combine:

*Value.*  Most picks come from near the top of the value board.  How near is
controlled by a noise temperature -- a low temperature means a disciplined room,
a high one means chaos.

*Roster need.*  A team that has already started two quarterbacks will almost
never take a third, and a team with no tight end in the twelfth round is about
to take one.  Need is computed from each team's actual roster against the
league's starting requirements, so it adapts to league shape for free.

*Live positional drift.*  For each position we compare how often it has *really*
been drafted against how often the value model expected it to be.  If
quarterbacks are going twice as fast as value alone predicts, the multiplier for
quarterbacks rises toward 2 and the simulator starts expecting them to keep
going early.  Recent picks are weighted more heavily than old ones, so a run
that starts three picks ago registers immediately -- and fades if it was a blip.

That last piece is the whole answer to "people draft differently every year."
The model does not need to know which year it is.  It reads the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .draft import DraftState, Roster
from .league import League


@dataclass
class OpponentModel:
    """A behavioural model of the other teams, refitted as the draft unfolds."""

    league: League

    #: Softmax temperature over value rank.  Larger = more erratic drafters.
    #: ~4 reproduces the amount of reaching seen in real home leagues.
    temperature: float = 4.0
    #: How fast the influence of a pick decays as it recedes into the past.
    #: 0.93 gives a run of three picks real weight while letting it fade.
    drift_decay: float = 0.93
    #: Smoothing on the positional drift multiplier.  Higher = slower to
    #: believe a run is real.
    drift_smoothing: float = 4.0
    #: How far the drift multiplier is allowed to move the board.
    drift_cap: float = 3.5
    #: Strength of roster-need pressure.
    need_weight: float = 1.6
    #: How many players at a position a team will stockpile beyond its starting
    #: requirement before effectively refusing more.
    depth_tolerance: int = 2

    #: Running weighted tallies, updated by :meth:`observe`.
    observed: Dict[str, float] = field(default_factory=dict)
    expected: Dict[str, float] = field(default_factory=dict)
    _picks_seen: int = 0

    # ------------------------------------------------------------------
    # Learning from the draft in progress
    # ------------------------------------------------------------------

    def observe(self, position: str, expected_probs: Dict[str, float]) -> None:
        """Record one pick, together with what the model had expected.

        ``expected_probs`` is the position distribution the model predicted for
        this pick.  Comparing what happened against what was expected is what
        isolates *behavioural* drift from the ordinary fact that good players go
        early.
        """
        decay = self.drift_decay
        for key in set(self.observed) | set(self.expected) | set(expected_probs) | {position}:
            self.observed[key] = self.observed.get(key, 0.0) * decay
            self.expected[key] = self.expected.get(key, 0.0) * decay

        self.observed[position] = self.observed.get(position, 0.0) + 1.0
        for pos, prob in expected_probs.items():
            self.expected[pos] = self.expected.get(pos, 0.0) + float(prob)
        self._picks_seen += 1

    def drift_multipliers(self) -> Dict[str, float]:
        """How much faster or slower each position is going than value predicts.

        1.0 means the position is moving exactly as the value board implies;
        2.0 means it is going twice as fast -- a run.
        """
        out: Dict[str, float] = {}
        a = self.drift_smoothing
        for position in self.league.positions:
            observed = self.observed.get(position, 0.0)
            expected = self.expected.get(position, 0.0)
            multiplier = (observed + a) / (expected + a)
            out[position] = float(np.clip(multiplier, 1.0 / self.drift_cap, self.drift_cap))
        return out

    def reset(self) -> None:
        self.observed.clear()
        self.expected.clear()
        self._picks_seen = 0

    def snapshot(self) -> "OpponentModel":
        """A copy that can be advanced inside a simulation without side effects."""
        clone = OpponentModel(
            league=self.league,
            temperature=self.temperature,
            drift_decay=self.drift_decay,
            drift_smoothing=self.drift_smoothing,
            drift_cap=self.drift_cap,
            need_weight=self.need_weight,
            depth_tolerance=self.depth_tolerance,
        )
        clone.observed = dict(self.observed)
        clone.expected = dict(self.expected)
        clone._picks_seen = self._picks_seen
        return clone

    # ------------------------------------------------------------------
    # Predicting the next pick
    # ------------------------------------------------------------------

    def need_multipliers(self, roster: Roster, picks_left: int) -> Dict[str, float]:
        """Roster-need pressure for one team, by position.

        Early on, need barely matters -- everyone takes the best player.  As a
        team's remaining picks dwindle relative to its unfilled starting slots,
        need takes over, which is exactly the behaviour real drafters show.
        """
        counts = roster.counts()
        out: Dict[str, float] = {}

        for position in self.league.positions:
            need = roster.starter_need(position)
            if picks_left > 0 and need > 0:
                urgency = min(1.0, need / max(picks_left, 1))
                multiplier = 1.0 + self.need_weight * urgency
            else:
                multiplier = 1.0

            # Teams stop taking a position once they are well past needing it.
            capacity = (
                sum(
                    slot.count
                    for slot in self.league.starters
                    if position in slot.eligible
                )
                + self.depth_tolerance
            )
            held = counts.get(position, 0)
            if held >= capacity:
                multiplier *= 0.04
            elif held >= capacity - 1:
                multiplier *= 0.4

            out[position] = multiplier
        return out

    def pick_probabilities(
        self,
        available: pd.DataFrame,
        roster: Roster,
        picks_left: int,
        *,
        value_column: str = "vor",
    ) -> np.ndarray:
        """Probability that each available player is taken with the next pick."""
        if available.empty:
            return np.array([])

        # Value component: a softmax over position on the value board.  Rank is
        # used rather than raw value so the model behaves the same in a
        # high-scoring league as in a low-scoring one.
        ranks = np.arange(len(available), dtype=float)
        logits = -ranks / max(self.temperature, 1e-6)

        drift = self.drift_multipliers()
        need = self.need_multipliers(roster, picks_left)
        positions = available["position"].astype(str).to_numpy()

        adjust = np.array(
            [drift.get(p, 1.0) * need.get(p, 1.0) for p in positions], dtype=float
        )
        logits = logits + np.log(np.maximum(adjust, 1e-9))

        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        return weights / total if total > 0 else np.full(len(available), 1.0 / len(available))

    def position_probabilities(
        self, available: pd.DataFrame, probs: np.ndarray
    ) -> Dict[str, float]:
        """Collapse per-player probabilities into a per-position distribution."""
        out: Dict[str, float] = {}
        if len(probs) == 0:
            return out
        for position, prob in zip(available["position"].astype(str), probs):
            out[position] = out.get(position, 0.0) + float(prob)
        return out

    def sample_pick(
        self,
        available: pd.DataFrame,
        roster: Roster,
        picks_left: int,
        rng: np.random.Generator,
    ) -> int:
        """Index (positional, into ``available``) of the player taken."""
        probs = self.pick_probabilities(available, roster, picks_left)
        if len(probs) == 0:
            raise ValueError("no players available")
        return int(rng.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def describe(self) -> str:
        if self._picks_seen == 0:
            return "No picks observed yet; using the value-based prior."
        drift = self.drift_multipliers()
        parts = []
        for position, multiplier in sorted(drift.items(), key=lambda kv: -kv[1]):
            if multiplier >= 1.25:
                parts.append(f"{position} going {multiplier:.1f}x faster than value implies")
            elif multiplier <= 0.8:
                parts.append(f"{position} sliding ({multiplier:.1f}x)")
        if not parts:
            return "The room is drafting close to value; no strong positional drift."
        return "; ".join(parts)


def fit_temperature_from_picks(
    board: pd.DataFrame,
    picks: Sequence[str],
    *,
    candidates: Sequence[float] = (1.5, 2.5, 4.0, 6.0, 9.0, 14.0),
) -> float:
    """Estimate how disciplined this room is from the picks made so far.

    Chooses the temperature whose implied likelihood best explains where on the
    value board the observed picks actually came from.  A room that keeps taking
    the consensus best player scores a low temperature; a room full of reaches
    scores a high one.
    """
    if not picks:
        return 4.0

    order = {pid: i for i, pid in enumerate(board["player_id"])}
    ranks: List[int] = []
    taken: set = set()
    for player_id in picks:
        if player_id not in order:
            continue
        # Rank among players still available when the pick was made.
        absolute = order[player_id]
        ranks.append(absolute - sum(1 for t in taken if order.get(t, 10**9) < absolute))
        taken.add(player_id)

    if not ranks:
        return 4.0

    pool = len(board)
    best, best_ll = 4.0, -np.inf
    for temperature in candidates:
        weights = np.exp(-np.arange(pool) / temperature)
        log_norm = np.log(weights.sum())
        ll = sum(-r / temperature - log_norm for r in ranks)
        if ll > best_ll:
            best, best_ll = temperature, ll
    return float(best)

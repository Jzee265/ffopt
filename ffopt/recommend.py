"""Turning simulation output into a short list of picks, with reasons.

The output is deliberately a handful of players rather than one, because a draft
recommendation is not a fact -- it is a judgement under uncertainty, and the gap
between the top options is usually smaller than the error bars.  Three or four
names with the reasoning attached lets a human apply the things the model cannot
see: that a player is hurt, that the rookie everyone likes is not in the data at
all, that you would simply rather not own a particular quarterback.

Each recommendation carries the two numbers that justify it -- what the pick is
worth, and whether the player would still be there next time -- plus a plain
sentence explaining which of those is doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .draft import DraftState
from .league import League
from .opponents import OpponentModel
from .simulate import DraftSimulator
from .valuation import ValueBoard


@dataclass
class Recommendation:
    player_id: str
    player_name: str
    position: str
    team: str
    tier: int
    position_rank: int
    projected_points: float
    vor: float
    survival: float
    roster_value: float
    #: Expected starting-lineup value lost by passing on this player.
    edge: float
    reason: str

    def line(self) -> str:
        return (
            f"{self.player_name:<24s} {self.position:<3s} "
            f"{self.position + str(self.position_rank):<6s} "
            f"proj {self.projected_points:6.1f}  VOR {self.vor:6.1f}  "
            f"survives {self.survival:5.0%}  edge {self.edge:+6.1f}"
        )


class Recommender:
    """Puts the pieces together: value board, opponent model, simulator."""

    def __init__(
        self,
        board: ValueBoard,
        *,
        opponents: Optional[OpponentModel] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.board = board
        self.league = board.league
        self.opponents = opponents or OpponentModel(league=board.league)
        self.seed = seed

    # ------------------------------------------------------------------

    def recommend(
        self,
        state: DraftState,
        *,
        n: int = 4,
        n_candidates: int = 12,
        n_sims: int = 250,
        survival_sims: int = 400,
    ) -> List[Recommendation]:
        """Rank the best picks available to me right now."""
        available = self.board.available(state.drafted_ids)
        if available.empty:
            return []

        simulator = DraftSimulator(
            self.board.players, self.league, self.opponents, seed=self.seed
        )

        survival = simulator.survival_probabilities(state, n_sims=survival_sims)

        # Only the top of the board is worth simulating; the fortieth-best
        # player is never the answer, and simulation is the expensive step.
        candidates = self._shortlist(self._usable(available, state), n_candidates)
        results = simulator.evaluate(
            state, candidates["player_id"].tolist(), n_sims=n_sims
        )
        if not results:
            return []

        best_value = max(value for value, _ in results.values())

        recommendations: List[Recommendation] = []
        for _, row in candidates.iterrows():
            player_id = str(row["player_id"])
            if player_id not in results:
                continue
            value, _sd = results[player_id]
            surv = float(survival.get(player_id, 0.0))
            recommendations.append(
                Recommendation(
                    player_id=player_id,
                    player_name=str(row["player_name"]),
                    position=str(row["position"]),
                    team=str(row.get("team", "")),
                    tier=int(row.get("tier", 1)),
                    position_rank=int(row.get("position_rank", 0)),
                    projected_points=float(row["projected_points"]),
                    vor=float(row["vor"]),
                    survival=surv,
                    roster_value=value,
                    edge=value - best_value,
                    reason=self._explain(row, surv, available, state),
                )
            )

        recommendations.sort(key=lambda r: -r.roster_value)
        return recommendations[:n]

    # ------------------------------------------------------------------

    def _usable(self, available: pd.DataFrame, state: DraftState) -> pd.DataFrame:
        """Drop positions my roster genuinely cannot use another of.

        A player only scores for you from a starting slot, so beyond one spare
        body per startable slot the next one at that position is worth nothing
        at all.  In a one-quarterback league that means a third quarterback is
        never the pick, however good he is -- and a recommender that suggests
        one is not one anybody will trust.
        """
        counts = state.my_roster().counts()
        blocked = {
            position
            for position in self.league.positions
            if counts.get(position, 0)
            >= sum(s.count for s in self.league.starters if position in s.eligible) + 1
        }
        if not blocked:
            return available
        usable = available[~available["position"].isin(blocked)]
        # Never hand back an empty board, however odd the roster.
        return usable if not usable.empty else available

    def _shortlist(self, available: pd.DataFrame, n_candidates: int) -> pd.DataFrame:
        """Candidates worth simulating: the best few overall and per position.

        Including each position's best player guarantees the simulator gets a
        chance to tell us that the top tight end is worth more than the fourth
        receiver, which a pure VOR cut would sometimes hide.
        """
        top = available.head(max(n_candidates - len(self.league.positions), 3))
        per_position = (
            available.sort_values("vor", ascending=False)
            .groupby("position", as_index=False)
            .head(1)
        )
        return (
            pd.concat([top, per_position])
            .drop_duplicates(subset="player_id")
            .head(n_candidates)
            .reset_index(drop=True)
        )

    def _explain(
        self,
        row: pd.Series,
        survival: float,
        available: pd.DataFrame,
        state: DraftState,
    ) -> str:
        """One sentence on why this player is in the list."""
        position = str(row["position"])
        same_tier = available[
            (available["position"] == position) & (available["tier"] == row["tier"])
        ]
        gap = state.picks_between_my_turns()
        reasons: List[str] = []

        if len(same_tier) <= 1:
            reasons.append(f"last {position} in tier {int(row['tier'])}")
        elif len(same_tier) <= 3:
            reasons.append(f"only {len(same_tier)} left in this {position} tier")

        if survival < 0.25:
            reasons.append(f"almost certainly gone by your next pick ({survival:.0%} survival)")
        elif survival > 0.7:
            reasons.append(f"likely still there next turn ({survival:.0%}) — you can wait")

        need = state.my_roster().starter_need(position)
        if need > 0 and state.my_roster().size >= self.league.n_starter_slots - 3:
            reasons.append(f"you still need {need} starting {position}")

        drift = self.opponents.drift_multipliers().get(position, 1.0)
        if drift >= 1.4:
            reasons.append(f"{position} run under way ({drift:.1f}x normal pace)")
        elif drift <= 0.7:
            reasons.append(f"{position}s are sliding in this room")

        if not reasons:
            reasons.append(f"best value on the board ({row['vor']:.0f} over replacement)")
        return "; ".join(reasons)

    # ------------------------------------------------------------------

    def observe_pick(self, state: DraftState, player_id: str) -> None:
        """Feed a completed pick into the opponent model.

        Called for every pick, mine and theirs, so the model's sense of the room
        stays current.
        """
        row = self.board.players[self.board.players["player_id"] == player_id]
        if row.empty:
            return
        position = str(row.iloc[0]["position"])

        available = self.board.available(state.drafted_ids)
        team = state.team_on_clock()
        if team is None or available.empty:
            return
        roster = state.roster(team)
        picks_left = self.league.roster_size - roster.size
        probs = self.opponents.pick_probabilities(available, roster, picks_left)
        expected = self.opponents.position_probabilities(available, probs)
        self.opponents.observe(position, expected)

    def format(self, recommendations: Sequence[Recommendation], state: DraftState) -> str:
        """Render the shortlist for a terminal."""
        if not recommendations:
            return "No players available."

        my_pick = state.next_pick_for(state.my_team)
        if my_pick is None:
            return "Your draft is complete."

        rnd, slot = state.round_and_slot(my_pick)
        gap = state.picks_between_my_turns()
        header = f"Your pick — round {rnd}, pick {slot} (overall {my_pick + 1})"
        if not state.is_my_turn():
            header += f", {my_pick - state.pick_number} picks away"
        if gap:
            header += f"\nYou pick again {gap} picks later"

        lines = [header, "=" * max(len(l) for l in header.split("\n")), ""]
        for i, rec in enumerate(recommendations, 1):
            marker = "*" if i == 1 else " "
            lines.append(f"{marker} {i}. {rec.line()}")
            lines.append(f"       {rec.reason}")
            lines.append("")

        room = self.opponents.describe()
        lines.append(f"Room read: {room}")
        return "\n".join(lines)

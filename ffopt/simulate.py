"""Playing the rest of the draft out, thousands of times.

Value over replacement answers "who is worth the most?"  It does not answer the
question a drafter actually faces, which is "who should I take *now*, given that
I pick again in nineteen picks and half these players will be gone?"

A player you could still get next round is not really a decision you have to
make now.  The way to find out who will still be there is to simulate: run the
remaining draft forward many times using the opponent model, and count.

That yields the two numbers a recommendation rests on:

*Survival probability* -- the chance a player is still available at your next
pick.  A player at 95% costs nothing to pass on.

*Expected roster value* -- the value of the starting lineup you end up with if
you take this player now and keep drafting sensibly.  This is the real
objective, because it prices in the whole chain: taking the tight end now means
missing the receiver, but also means not needing a tight end in round nine.  It
is why the answer is sometimes not the highest-VOR player on the board.

Everything in the simulation loop runs on integer-coded NumPy arrays rather than
DataFrames.  That is not premature optimisation -- a recommendation involves on
the order of a quarter of a million simulated pick decisions, and this has to
return between picks while a live draft clock is running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .draft import DraftState
from .league import League
from .opponents import OpponentModel

#: Beyond the top few dozen available players the softmax mass is negligible
#: (exp(-80/4) is about 1e-9), so the sampling pool is truncated for speed.
DEFAULT_POOL = 80


class DraftSimulator:
    """Monte Carlo over the remainder of the draft."""

    def __init__(
        self,
        board: pd.DataFrame,
        league: League,
        opponents: OpponentModel,
        *,
        seed: Optional[int] = None,
        value_column: str = "vor",
        pool_size: int = DEFAULT_POOL,
    ) -> None:
        self.league = league
        self.opponents = opponents
        self.pool_size = pool_size
        self.rng = np.random.default_rng(seed)

        # The board must be in descending value order: the opponent model's
        # softmax is defined over board rank, and the simulator relies on
        # available-index order matching that ranking.
        self.board = board.sort_values(value_column, ascending=False).reset_index(drop=True)
        self._index = {pid: i for i, pid in enumerate(self.board["player_id"])}

        self.positions: Tuple[str, ...] = tuple(league.positions)
        self._pos_id = {p: i for i, p in enumerate(self.positions)}
        self._codes = np.array(
            [self._pos_id.get(str(p), 0) for p in self.board["position"]], dtype=np.int64
        )

        self._values = self.board[value_column].to_numpy(dtype=float)
        self._points = self.board["projected_points"].to_numpy(dtype=float)
        self._replacement = self._points - self._values
        self._sigma = (
            self.board["sigma"].to_numpy(dtype=float)
            if "sigma" in self.board.columns
            else np.zeros(len(self.board))
        )

        n_pos = len(self.positions)
        self._dedicated = np.array(
            [league.dedicated_starters(p) for p in self.positions], dtype=np.int64
        )
        self._flex_total = sum(
            s.count for s in league.starters if len(s.eligible) > 1
        )
        self._flex_ok = np.array(
            [
                any(p in s.eligible for s in league.starters if len(s.eligible) > 1)
                for p in self.positions
            ],
            dtype=bool,
        )
        self._capacity = np.array(
            [
                sum(s.count for s in league.starters if p in s.eligible)
                for p in self.positions
            ],
            dtype=np.int64,
        )
        self._n_pos = n_pos

        # Slot table for fast lineup valuation: (eligibility mask, count),
        # most-restrictive first so dedicated slots claim their players before
        # flexes take the leftovers.
        slots = sorted(league.starters, key=lambda s: len(s.eligible))
        self._slots: List[Tuple[np.ndarray, int]] = [
            (
                np.array([p in s.eligible for p in self.positions], dtype=bool),
                int(s.count),
            )
            for s in slots
        ]

    # ------------------------------------------------------------------
    # Fast mirrors of the opponent model
    # ------------------------------------------------------------------

    def _need_multipliers(self, counts: np.ndarray, picks_left: int) -> np.ndarray:
        """Roster-need pressure, mirroring :meth:`OpponentModel.need_multipliers`.

        Open starting slots are derived from counts arithmetically instead of by
        greedy slot-filling, which is equivalent for the shapes leagues actually
        use and vastly faster.
        """
        open_dedicated = np.maximum(self._dedicated - counts, 0)
        surplus = int(np.sum(np.maximum(counts - self._dedicated, 0) * self._flex_ok))
        open_flex = max(self._flex_total - surplus, 0)
        need = open_dedicated + open_flex * self._flex_ok

        mult = np.ones(self._n_pos, dtype=float)
        if picks_left > 0:
            urgency = np.minimum(1.0, need / max(picks_left, 1))
            mult = 1.0 + self.opponents.need_weight * urgency
            mult = np.where(need > 0, mult, 1.0)

        limit = self._capacity + self.opponents.depth_tolerance
        mult = np.where(counts >= limit, mult * 0.04, mult)
        mult = np.where(counts == limit - 1, mult * 0.4, mult)
        return mult

    def _lineup_full(self, counts: np.ndarray) -> bool:
        """Can a complete starting lineup already be fielded from these counts?

        Once mine can be, nothing later in the draft can change my lineup value:
        every remaining pick of mine is a bench player, and bench players are
        worth zero here.  Cutting the simulation short at that point is exact,
        not an approximation, and removes roughly the back half of every run.
        """
        if np.any(counts < self._dedicated):
            return False
        surplus = int(np.sum(np.maximum(counts - self._dedicated, 0) * self._flex_ok))
        return surplus >= self._flex_total

    def _drift_multipliers(self, observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
        a = self.opponents.drift_smoothing
        cap = self.opponents.drift_cap
        return np.clip((observed + a) / (expected + a), 1.0 / cap, cap)

    def _pick_probs(
        self,
        pool: np.ndarray,
        counts: np.ndarray,
        picks_left: int,
        drift: np.ndarray,
    ) -> np.ndarray:
        """Softmax over board rank, tilted by positional drift and roster need."""
        codes = self._codes[pool]
        logits = -np.arange(pool.size, dtype=float) / max(self.opponents.temperature, 1e-6)
        logits = logits + np.log(np.maximum(drift[codes] * self._need_multipliers(counts, picks_left)[codes], 1e-9))
        logits -= logits.max()
        weights = np.exp(logits)
        total = weights.sum()
        return weights / total if total > 0 else np.full(pool.size, 1.0 / pool.size)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _initial(self, state: DraftState) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        mask = np.ones(len(self.board), dtype=bool)
        counts = np.zeros((self.league.n_teams, self._n_pos), dtype=np.int64)
        mine: List[int] = []

        for team, player_id, position in state.picks:
            idx = self._index.get(player_id)
            if idx is not None:
                mask[idx] = False
            code = self._pos_id.get(str(position))
            if code is not None:
                counts[team, code] += 1
            if team == state.my_team and idx is not None:
                mine.append(idx)
        return mask, counts, mine

    def _drift_state(self) -> Tuple[np.ndarray, np.ndarray]:
        observed = np.array(
            [self.opponents.observed.get(p, 0.0) for p in self.positions], dtype=float
        )
        expected = np.array(
            [self.opponents.expected.get(p, 0.0) for p in self.positions], dtype=float
        )
        return observed, expected

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def lineup_value(self, indices: Sequence[int], values: np.ndarray) -> float:
        """Value of the best starting lineup from a set of drafted players.

        Bench players contribute nothing directly.  That is the right call: a
        bench player's worth is insurance against injury, which is already
        reflected in the fact that projections are discounted for missed games.
        """
        if not indices:
            return 0.0
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(-values[idx])
        idx = idx[order]
        codes = self._codes[idx]
        used = np.zeros(idx.size, dtype=bool)

        total = 0.0
        for eligible, count in self._slots:
            if count <= 0:
                continue
            ok = eligible[codes] & ~used
            take = np.flatnonzero(ok)[:count]
            used[take] = True
            total += float(values[idx[take]].sum())
        return total

    # ------------------------------------------------------------------
    # The simulation loop
    # ------------------------------------------------------------------

    def _run(
        self,
        state: DraftState,
        forced: Optional[int],
        realized: np.ndarray,
        *,
        stop_at: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, np.ndarray]:
        """One draft forward.  Returns my lineup value and the availability mask.

        ``rng`` may be supplied so the *same* stream of opponent randomness can
        be replayed while only my own first pick changes -- see :meth:`evaluate`.
        """
        rng = self.rng if rng is None else rng
        mask, counts, mine = self._initial(state)
        # Availability is carried as a value-ordered index array and spliced on
        # each pick.  Rebuilding it from the mask on every pick was the single
        # hottest line in the profile.
        avail = np.flatnonzero(mask)

        observed, expected = self._drift_state()
        pick_order = state.order
        pick_no = state.pick_number
        limit = len(pick_order) if stop_at is None else stop_at
        decay = self.opponents.drift_decay
        forced_used = forced is None

        while pick_no < limit and avail.size:
            team = pick_order[pick_no]
            pool = avail[: self.pool_size]

            if team == state.my_team and not forced_used:
                where = np.flatnonzero(avail == forced)
                if where.size == 0:
                    break
                slot = int(where[0])
                forced_used = True
            elif team == state.my_team:
                # I keep drafting sensibly: best realized value, discounted once
                # a position is already well covered.
                codes = self._codes[pool]
                scores = realized[pool].copy()
                scores[counts[team][codes] >= (self._capacity[codes] + 1)] *= 0.15
                slot = int(np.argmax(scores))
            else:
                picks_left = self.league.roster_size - int(counts[team].sum())
                drift = self._drift_multipliers(observed, expected)
                probs = self._pick_probs(pool, counts[team], picks_left, drift)
                slot = int(rng.choice(probs.size, p=probs))
                observed *= decay
                expected *= decay
                observed[self._codes[int(pool[slot])]] += 1.0
                np.add.at(expected, self._codes[pool], probs)

            chosen = int(avail[slot])
            avail = np.delete(avail, slot)
            mask[chosen] = False
            counts[team, self._codes[chosen]] += 1
            if team == state.my_team:
                mine.append(chosen)
                if stop_at is None and self._lineup_full(counts[team]):
                    break
            pick_no += 1

        return self.lineup_value(mine, realized), mask

    def _realized(self, noisy: bool) -> np.ndarray:
        if not noisy:
            return self._values
        draws = self._points + self.rng.normal(0.0, self._sigma)
        return np.maximum(draws, 0.0) - self._replacement

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def survival_probabilities(
        self, state: DraftState, *, n_sims: int = 300
    ) -> pd.Series:
        """P(each available player survives to my next turn), by player id."""
        mask0, _, _ = self._initial(state)
        target = state.next_pick_for(state.my_team, after=state.pick_number)
        ids = self.board.loc[mask0, "player_id"]

        if target is None:
            return pd.Series(1.0, index=ids, dtype=float)

        counts = np.zeros(len(self.board), dtype=float)
        for _ in range(n_sims):
            _, mask = self._run(state, None, self._values, stop_at=target)
            counts += mask
        return pd.Series((counts / n_sims)[mask0.nonzero()[0]], index=ids, dtype=float)

    def _seeds(self, n_sims: int) -> np.ndarray:
        return self.rng.integers(0, 2**63 - 1, size=n_sims)

    def evaluate(
        self,
        state: DraftState,
        candidates: Sequence[str],
        *,
        n_sims: int = 120,
    ) -> Dict[str, Tuple[float, float]]:
        """Expected final starting-lineup value for each candidate pick.

        Two things keep this comparison honest at a tractable number of runs.

        First, **common random numbers**: every candidate faces the identical
        stream of opponent behaviour, so the comparison isolates the only thing
        that differs -- my own pick.  Without pairing, each candidate meets a
        differently-behaved room and that noise swamps the real differences,
        which are small.

        Second, players are valued at their **projections rather than sampled
        outcomes**.  Sampling a season for each player sounds more principled,
        but the candidate's own draw does not cancel between candidates: with a
        standard deviation near 90 points per player, that one term alone puts a
        ±10 point error bar on a comparison whose real differences are about
        that size.  Since the quantity being estimated is an *expectation* over
        seasons anyway, and the projections already are those expectations,
        sampling them only adds variance.  The uncertainty that actually changes
        the decision is over who will still be on the board -- and that is
        simulated in full.
        """
        seeds = self._seeds(n_sims)
        values_by_player: Dict[str, Tuple[float, float]] = {}

        for player_id in candidates:
            idx = self._index.get(player_id)
            if idx is None:
                continue
            outcomes = np.array(
                [
                    self._run(
                        state, idx, self._values, rng=np.random.default_rng(int(seed))
                    )[0]
                    for seed in seeds
                ],
                dtype=float,
            )
            values_by_player[player_id] = (float(outcomes.mean()), float(outcomes.std()))
        return values_by_player

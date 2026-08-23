"""Does it actually draft better?

The honest test of a draft tool is not whether its projections are accurate --
it is whether following it wins leagues.  So the backtest runs whole simulated
drafts in a past season, using only information available before that season,
and then scores each team on what its players *actually* did.

Every strategy drafts against the same room of opponents, from the same board,
in the same draft slot, on the same random seed.  The only thing that varies is
the strategy in one seat.  Points are the value of the best starting lineup the
team could field, which is what a fantasy season really rewards.

Strategies compared:

``optimizer``   the full thing: VOR, opponent modelling, simulation
``vor``         take the highest value over replacement, no simulation
``best_points`` take the highest projected total -- the classic mistake, which
                drafts three quarterbacks by round five
``last_year``   take whoever scored most last season
``adp_ish``     draft near the top of the board but reach a little, the way a
                reasonable human following a cheat sheet actually does
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import score_history
from .draft import DraftState
from .league import League
from .opponents import OpponentModel
from .projections import ProjectionModel
from .recommend import Recommender
from .simulate import DraftSimulator
from .valuation import ValueBoard

STRATEGIES = ("optimizer", "vor", "best_points", "last_year", "adp_ish")


@dataclass
class BacktestResult:
    season: int
    slot: int
    strategy: str
    lineup_points: float
    roster: List[str]


def _pick_by_column(
    board: pd.DataFrame, available: np.ndarray, column: str, counts: Dict[str, int], league: League
) -> int:
    """Greedy pick on one column, refusing to stockpile a position."""
    order = np.argsort(-board[column].to_numpy(dtype=float)[available])
    for j in order:
        idx = int(available[j])
        position = str(board.iloc[idx]["position"])
        capacity = sum(s.count for s in league.starters if position in s.eligible) + 1
        if counts.get(position, 0) < capacity:
            return idx
    return int(available[order[0]])


def run_backtest(
    history: pd.DataFrame,
    league: League,
    season: int,
    *,
    slots: Sequence[int] = (0, 3, 7, 11),
    strategies: Sequence[str] = STRATEGIES,
    seed: int = 0,
    n_sims: int = 40,
    survival_sims: int = 120,
    board: Optional[ValueBoard] = None,
) -> pd.DataFrame:
    """Draft ``season`` many times over and score the results on real outcomes.

    Pass ``board`` to reuse a previously built one; fitting the projection model
    is by far the slowest step and is identical across seeds and slots.
    """
    if board is None:
        train = history[history["season"] < season]
        model = ProjectionModel.fit(train, league, through_season=season - 1)
        projections = model.project(train, season)
        board = ValueBoard.build(projections, league)

    # What actually happened, for scoring.
    actual = score_history(history[history["season"] == season], league)
    actual_points = dict(zip(actual["player_id"], actual["fantasy_points"]))

    # Last season's points, for the naive strategy.
    previous = score_history(history[history["season"] == season - 1], league)
    last_year = dict(zip(previous["player_id"], previous["fantasy_points"]))

    players = board.players.copy()
    players["actual"] = players["player_id"].map(actual_points).fillna(0.0)
    players["last_year"] = players["player_id"].map(last_year).fillna(0.0)
    # A human with a cheat sheet: mostly by the board, but reaching a couple of
    # slots here and there.  Without the noise this is identical to ``vor`` and
    # tells us nothing.
    noise = np.random.default_rng(seed).normal(0.0, 4.0, size=len(players))
    players["adp_ish"] = -(players["overall_rank"].astype(float) + noise)

    rows: List[BacktestResult] = []
    for slot in slots:
        for strategy in strategies:
            rows.append(
                _run_one(players, board, league, season, slot, strategy, seed, n_sims, survival_sims)
            )
    return pd.DataFrame([r.__dict__ for r in rows])


def _run_one(
    players: pd.DataFrame,
    board: ValueBoard,
    league: League,
    season: int,
    slot: int,
    strategy: str,
    seed: int,
    n_sims: int,
    survival_sims: int,
) -> BacktestResult:
    state = DraftState(league=league, my_team=slot)
    opponents = OpponentModel(league=league)
    recommender = Recommender(board, opponents=opponents, seed=seed)
    simulator = DraftSimulator(players, league, opponents, seed=seed)

    # Opponents share one random stream across strategies, so the room behaves
    # identically no matter what the tested seat does.
    room_rng = np.random.default_rng(seed + 1000 * slot)
    ordered = players.reset_index(drop=True)
    index = {pid: i for i, pid in enumerate(ordered["player_id"])}
    codes = ordered["position"].astype(str).to_numpy()

    mask = np.ones(len(ordered), dtype=bool)
    counts: Dict[int, Dict[str, int]] = {t: {} for t in range(league.n_teams)}

    while not state.complete:
        team = state.team_on_clock()
        available = np.flatnonzero(mask)
        if available.size == 0:
            break

        if team == slot:
            idx = _strategy_pick(
                strategy,
                ordered,
                available,
                counts[team],
                league,
                state,
                recommender,
                n_sims,
                survival_sims,
            )
        else:
            picks_left = league.roster_size - sum(counts[team].values())
            from .draft import Roster

            roster = Roster(league=league, player_positions=_expand(counts[team]))
            sub = ordered.iloc[available]
            probs = opponents.pick_probabilities(sub, roster, picks_left)
            choice = int(room_rng.choice(probs.size, p=probs))
            idx = int(available[choice])
            opponents.observe(
                codes[idx], opponents.position_probabilities(sub, probs)
            )

        position = str(codes[idx])
        mask[idx] = False
        counts[team][position] = counts[team].get(position, 0) + 1
        state.record(str(ordered.iloc[idx]["player_id"]), position, team=team)

    my_ids = [pid for t, pid, _ in state.picks if t == slot]
    my_idx = [index[p] for p in my_ids if p in index]
    points = simulator.lineup_value(my_idx, ordered["actual"].to_numpy(dtype=float))
    names = [str(ordered.iloc[i]["player_name"]) for i in my_idx]
    return BacktestResult(season, slot, strategy, float(points), names)


def _expand(counts: Dict[str, int]) -> List[str]:
    return [pos for pos, n in counts.items() for _ in range(n)]


def _strategy_pick(
    strategy: str,
    ordered: pd.DataFrame,
    available: np.ndarray,
    counts: Dict[str, int],
    league: League,
    state: DraftState,
    recommender: Recommender,
    n_sims: int,
    survival_sims: int,
) -> int:
    if strategy == "optimizer":
        recs = recommender.recommend(
            state, n=1, n_sims=n_sims, survival_sims=survival_sims
        )
        if recs:
            match = ordered.index[ordered["player_id"] == recs[0].player_id]
            if len(match):
                return int(match[0])
        return _pick_by_column(ordered, available, "vor", counts, league)

    column = {
        "vor": "vor",
        "best_points": "projected_points",
        "last_year": "last_year",
        "adp_ish": "adp_ish",
    }[strategy]
    return _pick_by_column(ordered, available, column, counts, league)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean starting-lineup points by strategy, plus margin over the field."""
    table = (
        results.groupby("strategy")["lineup_points"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    table["vs_vor"] = table["mean"] - table["mean"].get("vor", np.nan)
    return table.round(1)

"""Tests for ffopt.

The interesting properties here are structural: replacement level has to move
when the league changes, the opponent model has to notice a run, and the
simulator has to agree with a hand-computed lineup.  Those are what these check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffopt.draft import DraftState, Roster, snake_order
from ffopt.league import Bonus, League, Scoring, Slot, make_league, standard_roster
from ffopt.opponents import OpponentModel
from ffopt.projections import ProjectionModel, fit_aging_curves, fit_participation
from ffopt.simulate import DraftSimulator
from ffopt.valuation import ValueBoard, assign_tiers, empirical_flex_shares


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def board_frame() -> pd.DataFrame:
    """A synthetic board with a deliberate shape.

    Running backs fall off a cliff after eight; quarterbacks score more in raw
    points but are flat, which is exactly the configuration where value over
    replacement should disagree with raw projections.
    """
    rows = []
    for i in range(40):
        rows.append(("RB", f"rb{i}", 260 - (8 * i if i < 8 else 64 + 3 * (i - 8))))
    for i in range(40):
        rows.append(("WR", f"wr{i}", 230 - 5 * i))
    for i in range(24):
        rows.append(("QB", f"qb{i}", 330 - 6 * i))
    for i in range(24):
        rows.append(("TE", f"te{i}", 180 - 9 * i))

    frame = pd.DataFrame(rows, columns=["position", "player_id", "projected_points"])
    frame["player_name"] = frame["player_id"].str.upper()
    frame["team"] = "XX"
    frame["sigma"] = 30.0
    frame["projected_games"] = 15.0
    return frame


@pytest.fixture
def league() -> League:
    return make_league(n_teams=12, scoring="half_ppr")


# --------------------------------------------------------------------------
# League configuration
# --------------------------------------------------------------------------


def test_roster_size_and_picks(league: League) -> None:
    assert league.n_starter_slots == 7  # QB, 2RB, 2WR, TE, FLEX
    assert league.roster_size == 13
    assert league.total_picks == 156


def test_scoring_respects_ppr_variants() -> None:
    line = {"receptions": 100, "receiving_yards": 1200, "receiving_tds": 8}
    standard = make_league(scoring="standard").scoring.score(line)
    half = make_league(scoring="half_ppr").scoring.score(line)
    full = make_league(scoring="ppr").scoring.score(line)
    assert full - half == pytest.approx(50.0)
    assert half - standard == pytest.approx(50.0)


def test_te_premium_applies_only_to_tight_ends() -> None:
    league = make_league(scoring="half_ppr", te_premium=0.5)
    assert league.scoring_for("TE").points_per["receptions"] == pytest.approx(1.0)
    assert league.scoring_for("WR").points_per["receptions"] == pytest.approx(0.5)


def test_threshold_bonus_is_scored() -> None:
    scoring = Scoring(
        points_per={"rushing_yards": 0.1},
        bonuses=(Bonus(stat="rushing_yards", threshold=100.0, points=3.0),),
    )
    # 1700 yards over 17 games is exactly 100 per game, so the bonus applies.
    assert scoring.score({"rushing_yards": 1700}, games=17) == pytest.approx(170 + 51)
    assert scoring.score({"rushing_yards": 850}, games=17) == pytest.approx(85.0)


def test_superflex_raises_quarterback_replacement(league: League) -> None:
    superflex = make_league(n_teams=12, superflex=True)
    assert superflex.replacement_rank("QB") > league.replacement_rank("QB")


def test_more_teams_pushes_replacement_deeper() -> None:
    small = make_league(n_teams=8)
    large = make_league(n_teams=14)
    for position in ("RB", "WR", "TE"):
        assert large.replacement_rank(position) > small.replacement_rank(position)


# --------------------------------------------------------------------------
# Valuation
# --------------------------------------------------------------------------


def test_vor_reorders_against_raw_points(board_frame: pd.DataFrame, league: League) -> None:
    """The headline claim: raw points say QB, value over replacement says RB."""
    board = ValueBoard.build(board_frame, league)
    assert board_frame.nlargest(1, "projected_points").iloc[0]["position"] == "QB"
    assert board.players.iloc[0]["position"] == "RB"


def test_replacement_level_tracks_league_size(board_frame: pd.DataFrame) -> None:
    small = ValueBoard.build(board_frame, make_league(n_teams=8))
    large = ValueBoard.build(board_frame, make_league(n_teams=14))
    # Deeper leagues reach further down the board, so replacement is worse.
    assert large.replacement["RB"] < small.replacement["RB"]


def test_superflex_lifts_quarterback_value(board_frame: pd.DataFrame) -> None:
    single = ValueBoard.build(board_frame, make_league(n_teams=12))
    superflex = ValueBoard.build(board_frame, make_league(n_teams=12, superflex=True))

    def best_qb_rank(board: ValueBoard) -> int:
        qbs = board.players[board.players["position"] == "QB"]
        return int(qbs.iloc[0]["overall_rank"])

    assert best_qb_rank(superflex) < best_qb_rank(single)


def test_flex_shares_sum_within_one(board_frame: pd.DataFrame, league: League) -> None:
    shares = empirical_flex_shares(board_frame, league)
    assert 0.99 <= sum(shares.values()) <= 1.01
    assert shares["TE"] < shares["RB"] + shares["WR"]


def test_tiers_break_at_a_cliff() -> None:
    frame = pd.DataFrame(
        {
            "position": ["RB"] * 6,
            "player_id": [f"r{i}" for i in range(6)],
            # A 60-point chasm between the third and fourth player.
            "projected_points": [200.0, 197.0, 194.0, 134.0, 131.0, 128.0],
        }
    )
    tiers = assign_tiers(frame)
    assert tiers.iloc[0] == tiers.iloc[2]
    assert tiers.iloc[3] > tiers.iloc[2]
    assert tiers.iloc[3] == tiers.iloc[5]


# --------------------------------------------------------------------------
# Draft bookkeeping
# --------------------------------------------------------------------------


def test_snake_order_reverses_each_round() -> None:
    order = snake_order(4, 3)
    assert order[:4] == [0, 1, 2, 3]
    assert order[4:8] == [3, 2, 1, 0]
    assert order[8:] == [0, 1, 2, 3]


def test_turn_gap_at_the_ends_of_the_round(league: League) -> None:
    first = DraftState(league=league, my_team=0)
    middle = DraftState(league=league, my_team=5)
    # The team at the turn waits longest between picks.
    assert first.picks_between_my_turns() > middle.picks_between_my_turns()


def test_open_starter_slots_fills_dedicated_before_flex(league: League) -> None:
    roster = Roster(league=league, player_positions=["RB", "RB", "RB"])
    open_slots = roster.open_starter_slots()
    assert open_slots["RB"] == 0
    assert open_slots["FLEX"] == 0  # the third back takes the flex
    assert open_slots["WR"] == 2


def test_starter_need_counts_flex_eligibility(league: League) -> None:
    roster = Roster(league=league, player_positions=["QB", "RB", "RB", "WR", "WR"])
    assert roster.starter_need("TE") == 2  # the TE slot and the open flex
    assert roster.starter_need("QB") == 0


def test_rejects_duplicate_and_out_of_range(league: League) -> None:
    state = DraftState(league=league, my_team=0)
    state.record("rb0", "RB")
    with pytest.raises(ValueError):
        state.record("rb0", "RB")
    with pytest.raises(ValueError):
        DraftState(league=league, my_team=99)


def test_undo_restores_previous_state(league: League) -> None:
    state = DraftState(league=league, my_team=0)
    state.record("rb0", "RB")
    state.record("wr0", "WR")
    assert state.undo() == (1, "wr0", "WR")
    assert state.drafted_ids == ["rb0"]


# --------------------------------------------------------------------------
# Opponent model
# --------------------------------------------------------------------------


def test_drift_starts_neutral(league: League) -> None:
    model = OpponentModel(league=league)
    assert all(value == pytest.approx(1.0) for value in model.drift_multipliers().values())


def test_model_detects_a_positional_run(league: League) -> None:
    """The core adaptive claim: a run should register as one."""
    model = OpponentModel(league=league)
    expected = {"QB": 0.1, "RB": 0.4, "WR": 0.4, "TE": 0.1}
    for _ in range(6):
        model.observe("QB", expected)

    drift = model.drift_multipliers()
    assert drift["QB"] > 1.5
    assert drift["RB"] < 1.0
    assert "run" in model.describe() or "faster" in model.describe()


def test_drift_fades_as_a_run_recedes(league: League) -> None:
    model = OpponentModel(league=league, drift_decay=0.8)
    expected = {"QB": 0.1, "RB": 0.45, "WR": 0.45}
    for _ in range(5):
        model.observe("QB", expected)
    peak = model.drift_multipliers()["QB"]
    for _ in range(15):
        model.observe("RB", expected)
    assert model.drift_multipliers()["QB"] < peak


def test_need_pressure_rises_as_picks_run_out(league: League) -> None:
    model = OpponentModel(league=league)
    empty = Roster(league=league, player_positions=[])
    early = model.need_multipliers(empty, picks_left=13)["TE"]
    late = model.need_multipliers(empty, picks_left=2)["TE"]
    assert late > early


def test_teams_stop_taking_a_saturated_position(league: League) -> None:
    model = OpponentModel(league=league)
    stacked = Roster(league=league, player_positions=["QB", "QB", "QB"])
    assert model.need_multipliers(stacked, picks_left=5)["QB"] < 0.1


def test_probabilities_are_a_distribution(board_frame: pd.DataFrame, league: League) -> None:
    board = ValueBoard.build(board_frame, league)
    model = OpponentModel(league=league)
    probs = model.pick_probabilities(
        board.players, Roster(league=league), picks_left=13
    )
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0).all()
    # A disciplined room takes near the top of the board.
    assert probs[0] > probs[50]


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def test_lineup_value_ignores_the_bench(board_frame: pd.DataFrame, league: League) -> None:
    board = ValueBoard.build(board_frame, league)
    simulator = DraftSimulator(board.players, league, OpponentModel(league=league), seed=0)
    values = np.zeros(len(board.players))

    codes = board.players["position"].to_numpy()
    rb_indices = [int(i) for i in np.flatnonzero(codes == "RB")[:5]]
    for rank, idx in enumerate(rb_indices):
        values[idx] = 100.0 - rank

    # Five backs, but only RB, RB and FLEX can start: 100 + 99 + 98.
    assert simulator.lineup_value(rb_indices, values) == pytest.approx(297.0)


def test_lineup_value_prefers_the_best_eligible(board_frame: pd.DataFrame, league: League) -> None:
    board = ValueBoard.build(board_frame, league)
    simulator = DraftSimulator(board.players, league, OpponentModel(league=league), seed=0)
    values = np.zeros(len(board.players))
    codes = board.players["position"].to_numpy()
    qb = int(np.flatnonzero(codes == "QB")[0])
    rb = int(np.flatnonzero(codes == "RB")[0])
    values[qb], values[rb] = 50.0, 80.0
    assert simulator.lineup_value([qb, rb], values) == pytest.approx(130.0)


def test_survival_probability_falls_with_board_position(
    board_frame: pd.DataFrame, league: League
) -> None:
    board = ValueBoard.build(board_frame, league)
    state = DraftState(league=league, my_team=0)
    simulator = DraftSimulator(board.players, league, OpponentModel(league=league), seed=3)
    survival = simulator.survival_probabilities(state, n_sims=120)

    top = board.players.iloc[0]["player_id"]
    deep = board.players.iloc[60]["player_id"]
    assert survival[top] < survival[deep]
    assert survival[deep] > 0.5


def test_early_stop_matches_full_simulation(
    board_frame: pd.DataFrame, league: League
) -> None:
    """Cutting the run short once my lineup is full must not change the answer."""
    board = ValueBoard.build(board_frame, league)
    state = DraftState(league=league, my_team=3)
    opponents = OpponentModel(league=league)

    simulator = DraftSimulator(board.players, league, opponents, seed=11)
    candidate = board.players.iloc[0]["player_id"]
    idx = simulator._index[candidate]

    rng_a = np.random.default_rng(99)
    value_with_stop, _ = simulator._run(state, idx, simulator._values, rng=rng_a)

    # Same stream, but forced to play every pick out.
    simulator_no_stop = DraftSimulator(board.players, league, opponents, seed=11)
    simulator_no_stop._lineup_full = lambda counts: False  # type: ignore[assignment]
    rng_b = np.random.default_rng(99)
    value_full, _ = simulator_no_stop._run(state, idx, simulator._values, rng=rng_b)

    assert value_with_stop == pytest.approx(value_full)


def test_simulation_respects_already_drafted_players(
    board_frame: pd.DataFrame, league: League
) -> None:
    board = ValueBoard.build(board_frame, league)
    state = DraftState(league=league, my_team=0)
    gone = board.players.iloc[0]["player_id"]
    state.record(gone, str(board.players.iloc[0]["position"]))

    simulator = DraftSimulator(board.players, league, OpponentModel(league=league), seed=5)
    survival = simulator.survival_probabilities(state, n_sims=20)
    assert gone not in survival.index


# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------


def _synthetic_history(seasons: int = 8, players: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(players):
        position = ["QB", "RB", "WR", "TE"][pid % 4]
        talent = rng.uniform(0.5, 1.5)
        for season in range(2015, 2015 + seasons):
            age = 23 + (season - 2015)
            rows.append(
                {
                    "player_id": f"p{pid}",
                    "player_name": f"Player {pid}",
                    "position": position,
                    "team": "XX",
                    "season": season,
                    "age": age,
                    "years_exp": season - 2015,
                    "games": 16,
                    "passing_yards": 4000 * talent if position == "QB" else 0.0,
                    "passing_tds": 28 * talent if position == "QB" else 0.0,
                    "interceptions": 12.0 if position == "QB" else 0.0,
                    "carries": 240 * talent if position == "RB" else 0.0,
                    "rushing_yards": 1100 * talent if position == "RB" else 0.0,
                    "rushing_tds": 8 * talent if position == "RB" else 0.0,
                    "receptions": 80 * talent if position in ("WR", "TE") else 0.0,
                    "targets": 120 * talent if position in ("WR", "TE") else 0.0,
                    "receiving_yards": 1000 * talent if position in ("WR", "TE") else 0.0,
                    "receiving_tds": 7 * talent if position in ("WR", "TE") else 0.0,
                    "fumbles_lost": 1.0,
                    "passing_2pt_conversions": 0.0,
                    "rushing_2pt_conversions": 0.0,
                    "receiving_2pt_conversions": 0.0,
                    "special_teams_tds": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_projection_model_runs_and_ranks_by_talent(league: League) -> None:
    history = _synthetic_history()
    model = ProjectionModel.fit(history, league, through_season=2021, fit_uncertainty=False)
    projections = model.project(history, 2022)

    assert not projections.empty
    assert projections["projected_points"].is_monotonic_decreasing
    assert (projections["projected_games"] <= 17).all()
    assert (projections["sigma"] > 0).all()


def test_projection_is_league_aware(league: League) -> None:
    """The same fitted stat line must value differently under different scoring."""
    history = _synthetic_history()
    standard = make_league(scoring="standard")
    ppr = make_league(scoring="ppr")

    proj_std = ProjectionModel.fit(
        history, standard, through_season=2021, fit_uncertainty=False
    ).project(history, 2022)
    proj_ppr = ProjectionModel.fit(
        history, ppr, through_season=2021, fit_uncertainty=False
    ).project(history, 2022)

    wr = proj_std[proj_std["position"] == "WR"].iloc[0]["player_id"]
    std_points = float(proj_std.loc[proj_std["player_id"] == wr, "projected_points"].iloc[0])
    ppr_points = float(proj_ppr.loc[proj_ppr["player_id"] == wr, "projected_points"].iloc[0])
    assert ppr_points > std_points


def test_participation_declines_with_age() -> None:
    """Retirement hazard is what keeps 40-year-olds off the top of the board."""
    rows = []
    rng = np.random.default_rng(1)
    for pid in range(400):
        start_age = int(rng.integers(23, 33))
        # Careers end sooner the older a player is.
        for offset in range(6):
            age = start_age + offset
            if rng.random() > max(0.05, 1.15 - age / 30):
                break
            rows.append(
                {
                    "player_id": f"p{pid}",
                    "position": "RB",
                    "season": 2010 + offset,
                    "age": age,
                    "games": 16,
                }
            )
    history = pd.DataFrame(rows)
    table = fit_participation(history)
    young = np.mean([v for (p, a), v in table.items() if a <= 26])
    old = np.mean([v for (p, a), v in table.items() if a >= 32])
    assert young > old


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_recommendation_pipeline(board_frame: pd.DataFrame, league: League) -> None:
    from ffopt.recommend import Recommender

    board = ValueBoard.build(board_frame, league)
    state = DraftState(league=league, my_team=2)
    recommender = Recommender(board, seed=7)

    # Play out the two picks ahead of mine.
    for _, row in board.players.head(2).iterrows():
        recommender.observe_pick(state, str(row["player_id"]))
        state.record(str(row["player_id"]), str(row["position"]))

    recs = recommender.recommend(state, n=3, n_sims=25, survival_sims=60)
    assert len(recs) == 3
    assert all(0.0 <= r.survival <= 1.0 for r in recs)
    assert recs[0].edge == pytest.approx(0.0)
    assert all(r.edge <= 0 for r in recs)
    assert all(r.reason for r in recs)
    # Nobody already drafted may be recommended.
    assert not set(r.player_id for r in recs) & set(state.drafted_ids)


def test_recommendations_avoid_a_saturated_position(
    board_frame: pd.DataFrame, league: League
) -> None:
    """After filling every quarterback slot, the model should stop suggesting them."""
    from ffopt.recommend import Recommender

    board = ValueBoard.build(board_frame, league)
    state = DraftState(league=league, my_team=0)
    recommender = Recommender(board, seed=4)

    qbs = board.players[board.players["position"] == "QB"].head(2)
    for _, row in qbs.iterrows():
        state.record(str(row["player_id"]), "QB", team=0)

    recs = recommender.recommend(state, n=4, n_sims=25, survival_sims=60)
    assert all(r.position != "QB" for r in recs)

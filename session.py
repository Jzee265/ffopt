"""Stateful driver so a draft can be run one command at a time.

The CLI in ffopt.cli owns an interactive terminal loop.  Here the draft state
lives in a JSON file between commands instead, so each pick is a separate
invocation:

    python session.py init --teams 12 --scoring half_ppr --slot 5
    python session.py pick "Bijan Robinson" "Ja'Marr Chase" ...
    python session.py rec
    python session.py board RB 15
    python session.py undo
    python session.py roster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

from ffopt.draft import DraftState
from ffopt.engine import build_board, find_players
from ffopt.league import make_league
from ffopt.opponents import OpponentModel, fit_temperature_from_picks
from ffopt.recommend import Recommender

STATE = Path(__file__).parent / "draft_state.json"


def load_config() -> dict:
    if not STATE.exists():
        sys.exit("no draft in progress — run `init` first")
    return json.loads(STATE.read_text())


def save(config: dict) -> None:
    STATE.write_text(json.dumps(config, indent=2))


def rebuild(config: dict):
    """Reconstruct board, draft state and opponent model from the saved picks."""
    league = make_league(
        n_teams=config["teams"],
        scoring=config["scoring"],
        bench=config["bench"],
        superflex=config.get("superflex", False),
        te_premium=config.get("te_premium", 0.0),
        qb=config.get("qb", 1),
        rb=config.get("rb", 2),
        wr=config.get("wr", 2),
        te=config.get("te", 1),
        flex=config.get("flex", 1),
    )
    board = build_board(league, season=config.get("season"))
    state = DraftState(league=league, my_team=config["slot"] - 1)
    recommender = Recommender(board, opponents=OpponentModel(league=league), seed=7)

    # Replay so the opponent model sees the draft unfold exactly as it happened.
    # An entry is either a board player id, or a dict describing someone the
    # projections cannot see -- a rookie.
    for entry in config["picks"]:
        if isinstance(entry, dict):
            recommender.observe_pick(state, entry["id"], position=entry["position"])
            state.record(entry["id"], entry["position"])
        else:
            recommender.observe_pick(state, entry)
            row = board.players[board.players["player_id"] == entry].iloc[0]
            state.record(entry, str(row["position"]))

    real = [e for e in config["picks"] if not isinstance(e, dict)]
    if len(real) >= 8:
        recommender.opponents.temperature = fit_temperature_from_picks(board.players, real)
    return league, board, state, recommender


def show_status(state, board, recommender) -> None:
    rnd, slot = state.round_and_slot()
    team = state.team_on_clock()
    who = "YOU" if team == state.my_team else f"team {team + 1}"
    print(f"On the clock: {rnd}.{slot:02d} — {who}   ({state.pick_number} picks made)")


def cmd_init(args) -> None:
    config = {
        "teams": args.teams,
        "scoring": args.scoring,
        "bench": args.bench,
        "slot": args.slot,
        "superflex": args.superflex,
        "te_premium": args.te_premium,
        "qb": args.qb,
        "rb": args.rb,
        "wr": args.wr,
        "te": args.te,
        "flex": args.flex,
        "season": args.season,
        "picks": [],
    }
    save(config)
    league, board, state, rec = rebuild(config)
    print(league.describe())
    print(f"Your slot: {args.slot}\n")
    print(board.summary())


def cmd_pick(args) -> None:
    config = load_config()
    league, board, state, rec = rebuild(config)

    for name in args.names:
        available = board.available(state.drafted_ids)
        matches = find_players(available, name)
        if not matches:
            print(f"  ?? no match for {name!r} — skipped")
            continue
        if len(matches) > 1 and matches[0].score < 1.0:
            options = ", ".join(f"{m.player_name} ({m.position})" for m in matches[:4])
            print(f"  ?? {name!r} is ambiguous: {options} — skipped")
            continue

        match = matches[0]
        team = state.team_on_clock()
        rnd, slot = state.round_and_slot()
        rec.observe_pick(state, match.player_id)
        state.record(match.player_id, match.position)
        config["picks"].append(match.player_id)
        who = "YOU" if team == state.my_team else f"team {team + 1}"
        print(f"  {rnd}.{slot:02d} {who:8s} {match.player_name} ({match.position})")

    save(config)
    print()
    show_status(state, board, rec)


def cmd_rec(args) -> None:
    config = load_config()
    league, board, state, rec = rebuild(config)
    recs = rec.recommend(state, n=args.n, n_sims=args.sims, survival_sims=args.survival)
    print(rec.format(recs, state))


def cmd_board(args) -> None:
    config = load_config()
    league, board, state, rec = rebuild(config)
    available = board.available(state.drafted_ids)
    if args.position:
        available = available[available["position"] == args.position.upper()]
    cols = ["player_name", "position", "position_rank", "tier", "projected_points", "vor"]
    print(available.head(args.count)[cols].round(1).to_string(index=False))


def cmd_roster(args) -> None:
    config = load_config()
    league, board, state, rec = rebuild(config)
    which = args.team - 1 if args.team else state.my_team
    picks = [(p, pos) for t, p, pos in state.picks if t == which]
    if not picks:
        print("  empty")
        return
    total = 0.0
    for pid, pos in picks:
        row = board.players[board.players["player_id"] == pid].iloc[0]
        total += float(row["projected_points"])
        print(f"  {pos:3s} {row['player_name']:<24s} {row['projected_points']:6.1f}")
    open_slots = {k: v for k, v in state.roster(which).open_starter_slots().items() if v}
    print(f"\n  still needs: {open_slots or 'nothing — lineup is full'}")


def cmd_rookie(args) -> None:
    """Log a pick the projections cannot see, keeping the draft order honest.

    Historical-only projections have no rookies in them.  Skipping such a pick
    would shift every later pick onto the wrong team, so it is recorded as a
    placeholder: it consumes a slot and feeds the opponent model a position,
    without removing anyone real from the board.
    """
    config = load_config()
    league, board, state, rec = rebuild(config)

    entry = {
        "id": f"rookie:{args.name.lower().replace(' ', '_')}",
        "name": args.name,
        "position": args.position.upper(),
    }
    team = state.team_on_clock()
    rnd, slot = state.round_and_slot()
    rec.observe_pick(state, entry["id"], position=entry["position"])
    state.record(entry["id"], entry["position"])
    config["picks"].append(entry)
    save(config)

    who = "YOU" if team == state.my_team else f"team {team + 1}"
    print(f"  {rnd}.{slot:02d} {who:8s} {args.name} ({entry['position']}) [rookie — not projected]")
    print()
    show_status(state, board, rec)


def cmd_auto(args) -> None:
    """Advance the draft by letting the opponent model make the next N picks.

    ``--bias`` multiplies one position's odds, which is how a run gets staged:
    the point of the exercise is to watch the model notice it.
    """
    import numpy as np

    config = load_config()
    league, board, state, rec = rebuild(config)
    rng = np.random.default_rng(args.seed)

    for _ in range(args.count):
        if state.complete:
            break
        team = state.team_on_clock()
        if team == state.my_team and not args.include_me:
            print("  (stopping — it's your turn)")
            break

        available = board.available(state.drafted_ids)
        roster = state.roster(team)
        picks_left = league.roster_size - roster.size
        probs = rec.opponents.pick_probabilities(available, roster, picks_left)

        if args.bias:
            boost = np.where(
                available["position"].to_numpy() == args.bias.upper(), args.bias_factor, 1.0
            )
            probs = probs * boost
            probs = probs / probs.sum()

        choice = int(rng.choice(len(probs), p=probs))
        row = available.iloc[choice]
        rnd, slot = state.round_and_slot()
        rec.observe_pick(state, str(row["player_id"]))
        state.record(str(row["player_id"]), str(row["position"]))
        config["picks"].append(str(row["player_id"]))
        print(f"  {rnd}.{slot:02d} team {team + 1:<3d} {row['player_name']:<24s} ({row['position']})")

    save(config)
    print()
    show_status(state, board, rec)


def cmd_undo(args) -> None:
    config = load_config()
    if not config["picks"]:
        sys.exit("nothing to undo")
    removed = config["picks"].pop()
    save(config)
    league, board, state, rec = rebuild(config)
    row = board.players[board.players["player_id"] == removed]
    name = row.iloc[0]["player_name"] if len(row) else removed
    print(f"  removed {name}")
    show_status(state, board, rec)


def main() -> None:
    parser = argparse.ArgumentParser(prog="session")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--teams", type=int, default=12)
    p.add_argument("--scoring", default="half_ppr")
    p.add_argument("--slot", type=int, required=True)
    p.add_argument("--bench", type=int, default=6)
    p.add_argument("--qb", type=int, default=1)
    p.add_argument("--rb", type=int, default=2)
    p.add_argument("--wr", type=int, default=2)
    p.add_argument("--te", type=int, default=1)
    p.add_argument("--flex", type=int, default=1)
    p.add_argument("--superflex", action="store_true")
    p.add_argument("--te-premium", type=float, default=0.0)
    p.add_argument("--season", type=int, default=None)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("pick")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("rec")
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--sims", type=int, default=80)
    p.add_argument("--survival", type=int, default=250)
    p.set_defaults(func=cmd_rec)

    p = sub.add_parser("board")
    p.add_argument("position", nargs="?", default=None)
    p.add_argument("count", nargs="?", type=int, default=15)
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("roster")
    p.add_argument("team", nargs="?", type=int, default=None)
    p.set_defaults(func=cmd_roster)

    p = sub.add_parser("rookie")
    p.add_argument("name")
    p.add_argument("position")
    p.set_defaults(func=cmd_rookie)

    p = sub.add_parser("auto")
    p.add_argument("count", type=int, default=1, nargs="?")
    p.add_argument("--bias", default=None, help="position to over-draft, staging a run")
    p.add_argument("--bias-factor", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-me", action="store_true")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("undo")
    p.set_defaults(func=cmd_undo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

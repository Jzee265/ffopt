"""Interactive draft assistant.

Run it, tell it your draft slot, then type each player's name as they come off
the board.  When your turn arrives it prints a shortlist.

    python -m ffopt.cli --teams 12 --scoring half_ppr --slot 5
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import pandas as pd

from .draft import DraftState
from .engine import build_board, find_players
from .league import SCORING_PRESETS, League, make_league
from .opponents import OpponentModel, fit_temperature_from_picks
from .recommend import Recommender

HELP = """
Commands
  <name>            record the next pick (whoever is on the clock)
  undo              take back the last pick
  board [POS] [n]   show the top of the value board
  roster [team]     show a roster (default: yours)
  need              show what your starting lineup still needs
  room              what the opponent model has learned about this draft
  picks             list the picks so far
  help              this text
  quit              exit
""".strip()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffopt", description="Fantasy football draft optimizer"
    )
    parser.add_argument("--teams", type=int, default=12, help="number of teams")
    parser.add_argument(
        "--scoring", default="half_ppr", choices=sorted(SCORING_PRESETS), help="scoring preset"
    )
    parser.add_argument("--slot", type=int, required=True, help="your draft position (1-based)")
    parser.add_argument("--bench", type=int, default=6, help="bench spots")
    parser.add_argument("--qb", type=int, default=1)
    parser.add_argument("--rb", type=int, default=2)
    parser.add_argument("--wr", type=int, default=2)
    parser.add_argument("--te", type=int, default=1)
    parser.add_argument("--flex", type=int, default=1)
    parser.add_argument("--superflex", action="store_true", help="add a superflex slot")
    parser.add_argument(
        "--te-premium", type=float, default=0.0, help="extra points per TE reception"
    )
    parser.add_argument("--linear", action="store_true", help="non-snake draft order")
    parser.add_argument("--season", type=int, default=None, help="season being drafted")
    parser.add_argument("--sims", type=int, default=80, help="simulations per candidate")
    parser.add_argument("--show", type=int, default=4, help="how many players to recommend")
    parser.add_argument("--refresh", action="store_true", help="refetch and refit")
    return parser.parse_args(argv)


def build_league(args: argparse.Namespace) -> League:
    return make_league(
        n_teams=args.teams,
        scoring=args.scoring,
        bench=args.bench,
        superflex=args.superflex,
        te_premium=args.te_premium,
        qb=args.qb,
        rb=args.rb,
        wr=args.wr,
        te=args.te,
        flex=args.flex,
    )


def resolve(board: pd.DataFrame, query: str) -> Optional[str]:
    """Turn typed text into a player id, asking if it is ambiguous."""
    matches = find_players(board, query)
    if not matches:
        print(f"  no player found matching {query!r}")
        return None
    if len(matches) == 1 or matches[0].score >= 1.0:
        return matches[0].player_id

    print("  which one?")
    for i, match in enumerate(matches, 1):
        print(f"    {i}. {match.player_name} ({match.position})")
    try:
        choice = input("  number (or blank to cancel): ").strip()
    except EOFError:
        return None
    if not choice.isdigit() or not 1 <= int(choice) <= len(matches):
        return None
    return matches[int(choice) - 1].player_id


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    league = build_league(args)

    if not 1 <= args.slot <= league.n_teams:
        print(f"--slot must be between 1 and {league.n_teams}", file=sys.stderr)
        return 2

    print(league.describe())
    print("Loading history and fitting projections...")
    board = build_board(league, season=args.season, refresh=args.refresh)
    print(board.summary())
    print()
    print(HELP)
    print()

    state = DraftState(league=league, my_team=args.slot - 1, snake=not args.linear)
    recommender = Recommender(board, opponents=OpponentModel(league=league))
    players = board.players

    def show_recommendations() -> None:
        recs = recommender.recommend(state, n=args.show, n_sims=args.sims)
        print()
        print(recommender.format(recs, state))
        print()

    if state.is_my_turn():
        show_recommendations()

    while not state.complete:
        team = state.team_on_clock()
        rnd, slot = state.round_and_slot()
        marker = "YOU" if team == state.my_team else f"team {team + 1}"
        try:
            raw = input(f"[{rnd}.{slot:02d} {marker}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        command, _, rest = raw.partition(" ")
        command = command.lower()

        if command in {"quit", "exit", "q"}:
            break
        if command in {"help", "?"}:
            print(HELP)
            continue
        if command == "undo":
            removed = state.undo()
            if removed is None:
                print("  nothing to undo")
            else:
                name = players.loc[players["player_id"] == removed[1], "player_name"]
                print(f"  removed {name.iloc[0] if len(name) else removed[1]}")
                # The opponent model is a running tally; rebuild it from scratch
                # so an undo cannot leave it believing in a run that never was.
                recommender.opponents.reset()
                replay = DraftState(
                    league=league, my_team=state.my_team, snake=state.snake
                )
                for t, pid, pos in state.picks:
                    recommender.observe_pick(replay, pid)
                    replay.record(pid, pos, team=t)
            if state.is_my_turn():
                show_recommendations()
            continue
        if command == "board":
            parts = rest.split()
            position = parts[0].upper() if parts and parts[0].isalpha() else None
            count = int(parts[-1]) if parts and parts[-1].isdigit() else 15
            available = board.available(state.drafted_ids)
            if position:
                available = available[available["position"] == position]
            cols = ["player_name", "position", "position_rank", "tier", "projected_points", "vor"]
            print(available.head(count)[cols].to_string(index=False))
            continue
        if command == "roster":
            which = int(rest) - 1 if rest.strip().isdigit() else state.my_team
            drafted = [(p, pos) for t, p, pos in state.picks if t == which]
            if not drafted:
                print("  empty")
                continue
            for pid, pos in drafted:
                row = players[players["player_id"] == pid]
                name = row.iloc[0]["player_name"] if len(row) else pid
                points = row.iloc[0]["projected_points"] if len(row) else float("nan")
                print(f"  {pos:3s} {name:<24s} {points:6.1f}")
            continue
        if command == "need":
            roster = state.my_roster()
            open_slots = {k: v for k, v in roster.open_starter_slots().items() if v > 0}
            print(f"  open starting slots: {open_slots or 'none — lineup is full'}")
            continue
        if command == "room":
            print(f"  {recommender.opponents.describe()}")
            temp = fit_temperature_from_picks(board.players, state.drafted_ids)
            print(f"  discipline estimate: temperature {temp:.1f} (lower = more by-the-book)")
            recommender.opponents.temperature = temp
            continue
        if command == "picks":
            for i, (t, pid, pos) in enumerate(state.picks, 1):
                row = players[players["player_id"] == pid]
                name = row.iloc[0]["player_name"] if len(row) else pid
                r, s = state.round_and_slot(i - 1)
                print(f"  {r}.{s:02d} team {t + 1:<3d} {pos:3s} {name}")
            continue

        player_id = resolve(board.available(state.drafted_ids), raw)
        if player_id is None:
            continue

        row = players[players["player_id"] == player_id].iloc[0]
        recommender.observe_pick(state, player_id)
        try:
            state.record(player_id, str(row["position"]))
        except ValueError as exc:
            print(f"  {exc}")
            continue
        print(f"  {row['player_name']} ({row['position']}) to {marker}")

        if state.is_my_turn():
            show_recommendations()

    print("\nYour roster:")
    for pid, pos in [(p, pos) for t, p, pos in state.picks if t == state.my_team]:
        row = players[players["player_id"] == pid]
        name = row.iloc[0]["player_name"] if len(row) else pid
        print(f"  {pos:3s} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

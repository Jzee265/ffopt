"""Draft bookkeeping: pick order, rosters, and whose turn it is."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .league import League, Slot


def snake_order(n_teams: int, rounds: int) -> List[int]:
    """Team index (0-based) on the clock for each overall pick.

    Snake drafts reverse every round, which is why a pick at the turn is worth
    thinking about differently: at 1.01 you wait 22 picks for your next choice,
    at 1.06 you wait 12.
    """
    order: List[int] = []
    for rnd in range(rounds):
        teams = range(n_teams) if rnd % 2 == 0 else reversed(range(n_teams))
        order.extend(teams)
    return order


def linear_order(n_teams: int, rounds: int) -> List[int]:
    """Pick order for leagues that do not snake."""
    return [team for _ in range(rounds) for team in range(n_teams)]


@dataclass
class Roster:
    """One team's players, and what starting slots they still need."""

    league: League
    player_positions: List[str] = field(default_factory=list)

    def add(self, position: str) -> None:
        self.player_positions.append(position)

    @property
    def size(self) -> int:
        return len(self.player_positions)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for position in self.player_positions:
            out[position] = out.get(position, 0) + 1
        return out

    def open_starter_slots(self) -> Dict[str, int]:
        """Starting slots this roster has not yet filled, by slot name.

        Filled greedily: dedicated slots first, then flexes, which matches how
        a lineup would actually be set.
        """
        remaining = {slot.name: slot.count for slot in self.league.starters}
        pool = list(self.player_positions)

        for slot in self.league.starters:
            if len(slot.eligible) > 1:
                continue
            for position in list(pool):
                if remaining[slot.name] <= 0:
                    break
                if position in slot.eligible:
                    remaining[slot.name] -= 1
                    pool.remove(position)

        for slot in self.league.starters:
            if len(slot.eligible) == 1:
                continue
            for position in list(pool):
                if remaining[slot.name] <= 0:
                    break
                if position in slot.eligible:
                    remaining[slot.name] -= 1
                    pool.remove(position)

        return remaining

    def starter_need(self, position: str) -> int:
        """How many unfilled starting slots this position could still fill."""
        open_slots = self.open_starter_slots()
        return sum(
            open_slots.get(slot.name, 0)
            for slot in self.league.starters
            if position in slot.eligible
        )


@dataclass
class DraftState:
    """The draft as it stands: who has picked, who is on the clock, what is left."""

    league: League
    my_team: int = 0
    snake: bool = True
    picks: List[Tuple[int, str, str]] = field(default_factory=list)  # (team, player_id, position)

    def __post_init__(self) -> None:
        if not 0 <= self.my_team < self.league.n_teams:
            raise ValueError(
                f"draft position must be between 1 and {self.league.n_teams}"
            )

    # -- order -----------------------------------------------------------

    @property
    def order(self) -> List[int]:
        maker = snake_order if self.snake else linear_order
        return maker(self.league.n_teams, self.league.roster_size)

    @property
    def pick_number(self) -> int:
        """Zero-based index of the pick currently on the clock."""
        return len(self.picks)

    @property
    def complete(self) -> bool:
        return self.pick_number >= self.league.total_picks

    def team_on_clock(self) -> Optional[int]:
        if self.complete:
            return None
        return self.order[self.pick_number]

    def is_my_turn(self) -> bool:
        return self.team_on_clock() == self.my_team

    def my_pick_numbers(self) -> List[int]:
        return [i for i, team in enumerate(self.order) if team == self.my_team]

    def next_pick_for(self, team: int, after: Optional[int] = None) -> Optional[int]:
        start = self.pick_number if after is None else after + 1
        for i in range(start, len(self.order)):
            if self.order[i] == team:
                return i
        return None

    def picks_until_my_turn(self) -> Optional[int]:
        nxt = self.next_pick_for(self.my_team)
        return None if nxt is None else nxt - self.pick_number

    def picks_between_my_turns(self) -> Optional[int]:
        """How many picks elapse between this turn and my following one.

        This is the number that decides whether a player can wait.
        """
        first = self.next_pick_for(self.my_team)
        if first is None:
            return None
        second = self.next_pick_for(self.my_team, after=first)
        return None if second is None else second - first

    def round_and_slot(self, pick_number: Optional[int] = None) -> Tuple[int, int]:
        n = self.pick_number if pick_number is None else pick_number
        return n // self.league.n_teams + 1, n % self.league.n_teams + 1

    # -- mutation --------------------------------------------------------

    def record(self, player_id: str, position: str, team: Optional[int] = None) -> None:
        if self.complete:
            raise ValueError("the draft is already complete")
        if any(p[1] == player_id for p in self.picks):
            raise ValueError(f"{player_id} has already been drafted")
        self.picks.append(
            (self.team_on_clock() if team is None else team, player_id, str(position))
        )

    def undo(self) -> Optional[Tuple[int, str, str]]:
        return self.picks.pop() if self.picks else None

    # -- views -----------------------------------------------------------

    @property
    def drafted_ids(self) -> List[str]:
        return [player_id for _, player_id, _ in self.picks]

    def roster(self, team: int) -> Roster:
        roster = Roster(league=self.league)
        for t, _, position in self.picks:
            if t == team:
                roster.add(position)
        return roster

    def rosters(self) -> Dict[int, Roster]:
        return {team: self.roster(team) for team in range(self.league.n_teams)}

    def my_roster(self) -> Roster:
        return self.roster(self.my_team)

    def recent_positions(self, n: int) -> List[str]:
        return [position for _, _, position in self.picks[-n:]]

    def copy(self) -> "DraftState":
        return DraftState(
            league=self.league,
            my_team=self.my_team,
            snake=self.snake,
            picks=list(self.picks),
        )

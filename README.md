# ffopt — fantasy football draft optimizer

Tell it your draft slot and who has been taken. It tells you who to take next,
and why.

It works for any league — team count, roster shape and scoring are all
configuration, not assumptions — and it adapts to how the room in front of you
is *actually* drafting rather than how an average draft usually goes.

```bash
python -m ffopt.cli --teams 12 --scoring half_ppr --slot 5
```

```
Your pick — round 1, pick 5 (overall 5)
You pick again 15 picks later
=======================================

* 1. Puka Nacua              WR  WR1    proj  209.9  VOR  110.3  survives    0%  edge   +0.0
       last WR in tier 1; almost certainly gone by your next pick (0% survival)

  2. Jaxon Smith-Njigba      WR  WR2    proj  201.3  VOR  101.8  survives    4%  edge   -8.0
       last WR in tier 2; almost certainly gone by your next pick (4% survival)

  3. De'Von Achane           RB  RB5    proj  195.1  VOR   98.3  survives    1%  edge  -11.3
       only 2 left in this RB tier

Room read: QB going 1.8x faster than value implies
```

Type each name as it comes off the board. When your turn arrives, you get a
shortlist. `help` lists the other commands (`board`, `roster`, `need`, `room`,
`undo`, `picks`).

---

## The browser version

`draft_board.html` is the whole thing as a single file. Double-click it — no
Python, no install, works offline, and it runs fine on a phone.

The projections are baked in, but **scoring and roster shape are not**. The page
carries each player's projected *stat line* and re-scores it under whatever
league you set up, so switching from standard to PPR or adding a superflex slot
genuinely reshuffles the board rather than re-ranking a frozen list. The VOR
maths, tiering, adaptive opponent model and Monte Carlo are all ported to
JavaScript and produce the same numbers as the Python engine, about seven times
faster (roughly 1.2 seconds per recommendation).

Type a name and press Enter to log each pick. The matcher ranks surnames above
first names, because that is how picks get called out in a room — typing
"chase" gives you Ja'Marr Chase, not Chase Brown. Rookies, which the
projections structurally cannot see, get logged as placeholders via **Add
rookie**: they hold their place in the draft order and feed the run detector a
position, without pretending to be projected.

There is no browser storage. Instead **Draft log** gives you a plain text list
of picks you can copy, paste into another device, or reload after a refresh —
readable by a human and portable in a way `localStorage` is not.

To rebuild it after a new season lands:

```bash
python export_board.py    # refit projections, write board_data.json
python build_html.py      # inline everything into draft_board.html
```

`session.py` is a third option: the same engine driven one command at a time
(`init`, `pick`, `rec`, `board`, `roster`, `rookie`, `undo`), useful when you
want the real Python engine without an interactive terminal loop.

---

## The idea

Three questions have to be answered in order, and most tools stop after the
first.

**1. How many points will he score?** — `projections.py`

Projections are built from scratch out of 27 seasons of play-by-play-derived
statistics (1999–2025, via nflverse). Nothing is hand-entered and no paid feed
is used.

Two things make the projections better than "look at last year":

*Raw stats are projected, then scored.* The model forecasts passing yards,
receptions, carries and the rest — not fantasy points. The league scores that
stat line afterwards. This is what makes the whole thing genuinely
league-agnostic: a PPR league and a standard league disagree about who the best
receiver is, and they should.

*Volume and efficiency are separated.* Yards per game is more stable year over
year than fantasy points per game, because points are contaminated by touchdown
luck. So touchdowns are modelled as a rate per yard, regressed hard toward the
positional mean, and multiplied back through projected yardage. Measured on
1999–2025:

| position | points/gm | yards/gm |
|----------|-----------|----------|
| RB       | 0.723     | 0.732    |
| WR       | 0.736     | 0.738    |
| TE       | 0.698     | 0.713    |

On top of that sit an age curve fit per position, a games-played forecast, and a
retirement hazard — the probability that a player of this age simply is not in
the league next year, which is what keeps 40-year-old quarterbacks off the top
of the board.

The final number is a 60/40 blend of the model and last season's actual points.
Backtested over 2019–2025, that blend beats either component alone: the model
contributes aging and touchdown regression, while last season's raw total
carries role information — a promotion to starter, a vacated target share — that
no purely historical model can see.

**2. What is he worth to me?** — `valuation.py`

Points are not value. A quarterback projected for 275 looks like the best player
available next to a running back at 227 — until you notice the twelfth-best
quarterback is at 210 while the thirtieth-best back is at 95. The quarterback is
worth +65 over what you could have had for free; the back is worth +132.

Replacement level is *derived* from your league rather than assumed. The
optimizer fills every team's starting lineup from the board to see which
positions actually win the flex slots, then sets each position's baseline
accordingly. Deeper leagues push replacement down the board; superflex drags the
quarterback baseline up; PPR shifts it toward receivers. Tiers come from finding
the cliffs — the drops big enough to mean the next player is a real step down.

**3. Who will still be there next time?** — `opponents.py`, `simulate.py`

This is the part that answers "people draft differently every year."

Most tools assume opponents follow average draft position. Real drafts have
personality: quarterbacks fly off the board in the third round one year and last
until the tenth the next; somebody always reaches for their own team's running
back. So the opponent model starts from a loose value-shaped prior and **re-fits
itself to the room from the picks it has actually seen**. For each position it
compares how often it has really been drafted against how often value alone
predicted — if quarterbacks are going twice as fast as expected, the multiplier
climbs toward 2 and the simulator starts expecting them to keep going early.
Recent picks weigh more than old ones, so a run registers within a pick or two
and fades if it was a blip. The model never needs to know which year it is. It
reads the room.

That model then plays the remaining draft forward a few hundred times, which
produces the two numbers a recommendation rests on:

- **survival** — the chance the player is still there at your next pick. At 95%,
  passing on him costs nothing.
- **expected roster value** — the value of the starting lineup you finish with
  if you take him now and keep drafting sensibly. This is the real objective,
  because it prices in the whole chain: taking the tight end now means missing
  the receiver, but also means not needing a tight end in round nine.

It returns three or four names rather than one on purpose. The gap between the
top options is usually smaller than the error bars, and you know things the
model does not — that a player is hurt, that the rookie everyone likes is not in
the data at all.

---

## Does it work?

The honest test of a draft tool is not projection accuracy — it is whether
following it wins. So `backtest.py` runs complete drafts in past seasons using
only information available beforehand, then scores each team on what its players
*actually* did. Every strategy drafts from the same board, in the same slot,
against the same room, on the same seed. Only the one seat differs.

**1,360 full drafts** — 2022–2025, three league types (12-team half-PPR, 10-team
PPR, 12-team superflex), every draft slot, two seeds:

| strategy | starting-lineup points | vs VOR | p |
|---|---|---|---|
| **optimizer** | **1562.7** | **+25.4** | **0.015** |
| vor | 1537.3 | — | — |
| adp_ish | 1527.8 | −9.5 | 0.41 |
| best_points | 1500.6 | −36.7 | 0.010 |
| last_year | 1382.7 | −154.6 | <0.0001 |

Read it in three parts.

**Value over replacement is the big win.** Drafting off last season's points
costs 155 points a season against VOR — that gap dwarfs everything else here,
and it is the single most valuable thing in the repository. Chasing raw
projected totals costs 37.

**The simulation layer adds a real but modest edge.** +25.4 points over plain
VOR, 95% CI [+5.2, +45.6], winning 54% of head-to-head drafts. Statistically
significant, and consistent in direction across all three league types (+18.8,
+23.2, +33.1) and all four seasons (+20.7 to +29.1) — but it is an edge, not a
transformation. Anyone claiming a draft model doubles your win rate is selling
something.

**The superflex result is the most encouraging.** The optimizer's margin is
largest (+33.1) in the league type that departs furthest from convention, which
is what you would hope for from a system that derives replacement levels from
league structure instead of inheriting them from a cheat sheet.

---

## As a library

```python
from ffopt import make_league, build_board, DraftState, Recommender

league = make_league(n_teams=12, scoring="half_ppr")
board = build_board(league)                      # loads, fits, caches

state = DraftState(league=league, my_team=4)     # 0-based: the 5th slot
rec = Recommender(board)

for player_id in picks_so_far:                   # feed the room as it happens
    rec.observe_pick(state, player_id)
    state.record(player_id, position_of(player_id))

print(rec.format(rec.recommend(state), state))
```

Any league shape can be expressed:

```python
from ffopt import League, Scoring, Slot, Bonus

league = League(
    n_teams=14,
    scoring=Scoring(
        points_per={"passing_yards": 0.04, "passing_tds": 6.0, "receptions": 1.0, ...},
        bonuses=(Bonus(stat="rushing_yards", threshold=100.0, points=3.0),),
    ),
    starters=(
        Slot.of("QB", ["QB"]),
        Slot.of("RB", ["RB"], 2),
        Slot.of("WR", ["WR"], 3),
        Slot.of("TE", ["TE"]),
        Slot.of("SUPERFLEX", ["QB", "RB", "WR", "TE"]),
    ),
    bench=7,
)
```

Nothing downstream hardcodes QB/RB/WR/TE — replacement levels, the opponent
model's sense of roster need, and lineup valuation all read the slot table.

---

## Layout

| file | what it does |
|------|--------------|
| `league.py` | scoring rules and roster shape |
| `data.py` | nflverse ingestion, cached to parquet |
| `projections.py` | historical projections, aging, availability |
| `valuation.py` | replacement levels, VOR, tiers |
| `draft.py` | pick order, rosters, whose turn it is |
| `opponents.py` | the adaptive model of the room |
| `simulate.py` | Monte Carlo over the rest of the draft |
| `recommend.py` | the shortlist, with reasoning |
| `backtest.py` | strategy comparison on real seasons |
| `cli.py` | the live draft interface |

Run the tests with `python -m pytest tests/`.

---

## What it cannot do

Worth being blunt about, because these are structural rather than bugs to be
fixed later.

**Rookies are invisible.** They have no NFL history, so a purely historical
model cannot see them at all. They will not appear on the board. In practice
this means you should be ready to override the recommendation in the rounds
where rookies go — and it is the single biggest reason the output is a shortlist
rather than one name.

**It cannot see this offseason.** Team changes, coaching changes, holdouts,
training-camp injuries, a receiver whose target competition just left — none of
it is in the data. The 60/40 blend toward last season's points recovers some
role information, but not events that happened after the last snap of 2025.

**Projections are good, not great.** Over 2019–2025 the top 120 players by
projection averaged 163.6 actual points, against 161.2 for ranking by last
season alone and 192.6 for a perfect oracle. So the model beats the naive
baseline by a nose, and both are a long way from clairvoyant. Fantasy football
is mostly variance. The value here is in the valuation and draft-flow layers,
which turn merely-decent projections into good decisions — that is exactly what
the backtest above shows, and it is why effort went there rather than into
chasing another point of projection accuracy.

**Kickers and defenses are not modelled.** The scoring vocabulary includes
kicking stats and the roster code handles K/DST slots, but no projections are
built for them. Draft them last, as everyone does anyway.

---

## Data

Season statistics come from
[nflverse](https://github.com/nflverse/nflverse-data), read directly from its
release parquet files and cached locally on first use.

Note that `nfl_data_py.import_seasonal_data()` points at a deprecated release
path that 404s for 2025 onward; `data.py` reads the current `stats_player`
release instead. If a season suddenly disappears, check whether nflverse renamed
the release tag before assuming the data is gone.

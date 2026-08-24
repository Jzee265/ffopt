// ---------------------------------------------------------------------------
// ffopt engine, ported from the Python package.
//
// Everything the Python version does at draft time happens here: scoring a
// projected stat line under the league's rules, deriving replacement levels
// from the roster shape, adaptive opponent modelling, and Monte Carlo over the
// remaining draft.
//
// Projections themselves are NOT recomputed here -- the fitted stat lines are
// baked into the page. But scoring is, which is what keeps the board honest
// when you change the league: switching to PPR genuinely reshuffles it.
// ---------------------------------------------------------------------------

const FFOPT = (() => {
  'use strict';

  const POSITIONS = ['QB', 'RB', 'WR', 'TE'];

  // -- scoring --------------------------------------------------------------

  function basePoints(reception) {
    return {
      passing_yards: 0.04, passing_tds: 4, interceptions: -2,
      passing_2pt_conversions: 2,
      rushing_yards: 0.1, rushing_tds: 6, rushing_2pt_conversions: 2,
      receptions: reception, receiving_yards: 0.1, receiving_tds: 6,
      receiving_2pt_conversions: 2,
      fumbles_lost: -2, special_teams_tds: 6,
    };
  }

  const PRESETS = {
    standard: basePoints(0),
    half_ppr: basePoints(0.5),
    ppr: basePoints(1),
  };

  function scoreLine(line, rules) {
    if (!line) return 0;
    let total = 0;
    for (const stat in rules) total += rules[stat] * (line[stat] || 0);
    return total;
  }

  // Position-conditional scoring, so a TE premium applies only to tight ends.
  function rulesFor(league, pos) {
    const rules = Object.assign({}, league.scoring);
    if (pos === 'TE' && league.tePremium) {
      rules.receptions = (rules.receptions || 0) + league.tePremium;
    }
    return rules;
  }

  // -- league ---------------------------------------------------------------

  function makeLeague(cfg) {
    const slots = [];
    const add = (name, eligible, count) => {
      if (count > 0) slots.push({ name, eligible, count });
    };
    add('QB', ['QB'], cfg.qb);
    add('RB', ['RB'], cfg.rb);
    add('WR', ['WR'], cfg.wr);
    add('TE', ['TE'], cfg.te);
    add('FLEX', ['RB', 'WR', 'TE'], cfg.flex);
    add('SUPERFLEX', ['QB', 'RB', 'WR', 'TE'], cfg.superflex);

    const starters = slots.reduce((n, s) => n + s.count, 0);
    return {
      teams: cfg.teams,
      bench: cfg.bench,
      scoring: typeof cfg.scoring === 'string' ? PRESETS[cfg.scoring] : cfg.scoring,
      tePremium: cfg.tePremium || 0,
      starters: slots,
      nStarters: starters,
      rosterSize: starters + cfg.bench,
      totalPicks: (starters + cfg.bench) * cfg.teams,
      positions: POSITIONS,
      snake: cfg.snake !== false,
    };
  }

  const dedicatedFor = (league, pos) =>
    league.starters.filter((s) => s.eligible.length === 1 && s.eligible[0] === pos)
      .reduce((n, s) => n + s.count, 0);

  const flexSlotsFor = (league, pos) =>
    league.starters.filter((s) => s.eligible.length > 1 && s.eligible.includes(pos))
      .reduce((n, s) => n + s.count, 0);

  const capacityFor = (league, pos) =>
    league.starters.filter((s) => s.eligible.includes(pos))
      .reduce((n, s) => n + s.count, 0);

  // -- the board ------------------------------------------------------------

  // Blend the fitted projection with last season's actual points, exactly as
  // ProjectionModel does. Both halves are scored under the *current* league,
  // which is what keeps this league-agnostic rather than re-ranking a frozen
  // list of points.
  function buildBoard(players, league, blend) {
    const scored = [];
    for (const p of players) {
      const rules = rulesFor(league, p.pos);
      const model = scoreLine(p.s, rules);
      let points = model;
      if (p.p && p.pg >= 1) {
        points = blend * model + (1 - blend) * scoreLine(p.p, rules);
      }
      if (points < 1) continue;
      scored.push({
        id: p.id, name: p.name, pos: p.pos, team: p.team, age: p.age,
        games: p.g, points, modelPoints: model,
      });
    }

    scored.sort((a, b) => b.points - a.points);
    const flexShares = empiricalFlexShares(scored, league);
    const replacement = replacementLevels(scored, league, flexShares);

    for (const row of scored) {
      row.replacement = replacement[row.pos] || 0;
      row.vor = row.points - row.replacement;
    }
    scored.sort((a, b) => b.vor - a.vor);

    const posCount = {};
    scored.forEach((row, i) => {
      row.overallRank = i + 1;
      posCount[row.pos] = (posCount[row.pos] || 0) + 1;
      row.posRank = posCount[row.pos];
    });

    assignTiers(scored);
    return { players: scored, league, replacement, flexShares };
  }

  // Which positions actually win the multi-position slots? Rather than assume,
  // fill every team's lineup greedily from the board and count.
  function empiricalFlexShares(scored, league) {
    const flexSlots = league.starters.filter((s) => s.eligible.length > 1);
    const totalFlex = flexSlots.reduce((n, s) => n + s.count, 0) * league.teams;
    if (!totalFlex) return {};

    const dedicated = {}, flexLeft = {}, counts = {};
    for (const pos of league.positions) {
      dedicated[pos] = dedicatedFor(league, pos) * league.teams;
      counts[pos] = 0;
    }
    for (const s of flexSlots) flexLeft[s.name] = s.count * league.teams;

    const byPoints = scored.slice().sort((a, b) => b.points - a.points);
    for (const row of byPoints) {
      if (dedicated[row.pos] > 0) { dedicated[row.pos] -= 1; continue; }
      for (const s of flexSlots) {
        if (s.eligible.includes(row.pos) && flexLeft[s.name] > 0) {
          flexLeft[s.name] -= 1;
          counts[row.pos] += 1;
          break;
        }
      }
    }
    const shares = {};
    for (const pos in counts) shares[pos] = counts[pos] / totalFlex;
    return shares;
  }

  function replacementRank(league, pos, flexShare) {
    const dedicated = dedicatedFor(league, pos) * league.teams;
    const flex = flexSlotsFor(league, pos) * league.teams;
    const share = flexShare === undefined
      ? ({ RB: 0.45, WR: 0.45, TE: 0.1, QB: 0.9 }[pos] || 0)
      : flexShare;
    return Math.max(1, Math.round(dedicated + flex * share));
  }

  function replacementLevels(scored, league, flexShares) {
    const levels = {};
    for (const pos of league.positions) {
      const pool = scored.filter((r) => r.pos === pos)
        .map((r) => r.points).sort((a, b) => b - a);
      if (!pool.length) { levels[pos] = 0; continue; }
      const rank = replacementRank(league, pos, flexShares[pos]);
      // Average a window around the cutoff so one outlier at the boundary
      // does not swing the whole position.
      const lo = Math.max(0, rank - 2), hi = Math.min(pool.length, rank + 3);
      const slice = pool.slice(lo, hi);
      levels[pos] = slice.length
        ? slice.reduce((a, b) => a + b, 0) / slice.length
        : pool[pool.length - 1];
    }
    return levels;
  }

  // Tier breaks sit where the drop to the next player is unusually large for
  // that position. Tiers turn a ranking into a decision.
  function assignTiers(scored, gapMultiplier = 0.8, maxTiers = 12) {
    for (const pos of POSITIONS) {
      const block = scored.filter((r) => r.pos === pos)
        .sort((a, b) => b.points - a.points);
      if (block.length <= 1) { block.forEach((r) => (r.tier = 1)); continue; }

      const gaps = [];
      for (let i = 1; i < block.length; i++) gaps.push(block[i - 1].points - block[i].points);
      const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
      const sd = Math.sqrt(gaps.reduce((a, g) => a + (g - mean) ** 2, 0) / gaps.length);
      const threshold = mean + gapMultiplier * sd;

      let tier = 1;
      block[0].tier = 1;
      for (let i = 1; i < block.length; i++) {
        if (gaps[i - 1] >= threshold && tier < maxTiers) tier += 1;
        block[i].tier = tier;
      }
    }
  }

  // -- draft state ----------------------------------------------------------

  function pickOrder(league) {
    const order = [];
    for (let r = 0; r < league.rosterSize; r++) {
      for (let t = 0; t < league.teams; t++) {
        order.push(league.snake && r % 2 === 1 ? league.teams - 1 - t : t);
      }
    }
    return order;
  }

  // Open starting slots, filled greedily: dedicated first, then flexes.
  function openSlots(league, positions) {
    const remaining = {};
    for (const s of league.starters) remaining[s.name] = s.count;
    const pool = positions.slice();

    for (const pass of [1, 2]) {
      for (const s of league.starters) {
        const isFlex = s.eligible.length > 1;
        if ((pass === 1) === isFlex) continue;
        for (let i = pool.length - 1; i >= 0; i--) {
          if (remaining[s.name] <= 0) break;
          if (s.eligible.includes(pool[i])) {
            remaining[s.name] -= 1;
            pool.splice(i, 1);
          }
        }
      }
    }
    return remaining;
  }

  function starterNeed(league, positions, pos) {
    const open = openSlots(league, positions);
    return league.starters
      .filter((s) => s.eligible.includes(pos))
      .reduce((n, s) => n + (open[s.name] || 0), 0);
  }

  // -- opponent model -------------------------------------------------------
  //
  // Not an ADP assumption. It starts from a value-shaped prior and re-fits from
  // the picks actually seen: for each position it compares how often it has
  // really gone against how often value alone predicted, weighted toward recent
  // picks. A run registers within a pick or two, and fades if it was a blip.

  function makeOpponents(league, opts = {}) {
    return {
      league,
      temperature: opts.temperature || 4.0,
      driftDecay: 0.93,
      driftSmoothing: 4.0,
      driftCap: 3.5,
      needWeight: 1.6,
      depthTolerance: 2,
      observed: {},
      expected: {},
      seen: 0,
    };
  }

  function observe(model, pos, expectedProbs) {
    const d = model.driftDecay;
    const keys = new Set([
      ...Object.keys(model.observed), ...Object.keys(model.expected),
      ...Object.keys(expectedProbs), pos,
    ]);
    for (const k of keys) {
      model.observed[k] = (model.observed[k] || 0) * d;
      model.expected[k] = (model.expected[k] || 0) * d;
    }
    model.observed[pos] = (model.observed[pos] || 0) + 1;
    for (const p in expectedProbs) model.expected[p] = (model.expected[p] || 0) + expectedProbs[p];
    model.seen += 1;
  }

  function driftMultipliers(model) {
    const out = {}, a = model.driftSmoothing, cap = model.driftCap;
    for (const pos of model.league.positions) {
      const m = ((model.observed[pos] || 0) + a) / ((model.expected[pos] || 0) + a);
      out[pos] = Math.min(cap, Math.max(1 / cap, m));
    }
    return out;
  }

  function cloneOpponents(model) {
    const c = makeOpponents(model.league, { temperature: model.temperature });
    c.observed = Object.assign({}, model.observed);
    c.expected = Object.assign({}, model.expected);
    c.seen = model.seen;
    return c;
  }

  function needMultipliers(model, counts, picksLeft) {
    const league = model.league, out = {};
    const open = {};
    let surplus = 0, flexTotal = 0;
    for (const s of league.starters) if (s.eligible.length > 1) flexTotal += s.count;

    for (const pos of league.positions) {
      const ded = dedicatedFor(league, pos);
      open[pos] = Math.max(ded - (counts[pos] || 0), 0);
      if (flexSlotsFor(league, pos) > 0) surplus += Math.max((counts[pos] || 0) - ded, 0);
    }
    const openFlex = Math.max(flexTotal - surplus, 0);

    for (const pos of league.positions) {
      const need = open[pos] + (flexSlotsFor(league, pos) > 0 ? openFlex : 0);
      let m = 1;
      if (picksLeft > 0 && need > 0) {
        m = 1 + model.needWeight * Math.min(1, need / Math.max(picksLeft, 1));
      }
      const limit = capacityFor(league, pos) + model.depthTolerance;
      const held = counts[pos] || 0;
      if (held >= limit) m *= 0.04;
      else if (held === limit - 1) m *= 0.4;
      out[pos] = m;
    }
    return out;
  }

  // Softmax over board rank, tilted by drift and roster need.
  function pickProbs(model, pool, counts, picksLeft, drift) {
    const need = needMultipliers(model, counts, picksLeft);
    const logits = new Float64Array(pool.length);
    let max = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      const pos = pool[i].pos;
      const adj = (drift[pos] || 1) * (need[pos] || 1);
      logits[i] = -i / model.temperature + Math.log(Math.max(adj, 1e-9));
      if (logits[i] > max) max = logits[i];
    }
    let sum = 0;
    for (let i = 0; i < pool.length; i++) { logits[i] = Math.exp(logits[i] - max); sum += logits[i]; }
    for (let i = 0; i < pool.length; i++) logits[i] /= sum || 1;
    return logits;
  }

  // -- simulation -----------------------------------------------------------

  // Seeded RNG so candidates can be compared under identical opponent
  // behaviour (common random numbers). Without pairing, the noise between runs
  // swamps the real differences, which are small.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sample(probs, rand) {
    let r = rand(), acc = 0;
    for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r <= acc) return i; }
    return probs.length - 1;
  }

  // Best startable lineup. Bench players contribute nothing: their worth is
  // insurance, already priced into games-played discounting.
  function lineupValue(league, picks, valueOf) {
    const sorted = picks.slice().sort((a, b) => valueOf(b) - valueOf(a));
    const used = new Array(sorted.length).fill(false);
    const slots = league.starters.slice().sort((a, b) => a.eligible.length - b.eligible.length);
    let total = 0;
    for (const s of slots) {
      let filled = 0;
      for (let i = 0; i < sorted.length && filled < s.count; i++) {
        if (used[i] || !s.eligible.includes(sorted[i].pos)) continue;
        used[i] = true; total += valueOf(sorted[i]); filled += 1;
      }
    }
    return total;
  }

  const POOL_SIZE = 60;

  function lineupFull(league, counts) {
    let surplus = 0, flexTotal = 0;
    for (const s of league.starters) if (s.eligible.length > 1) flexTotal += s.count;
    for (const pos of league.positions) {
      const ded = dedicatedFor(league, pos);
      if ((counts[pos] || 0) < ded) return false;
      if (flexSlotsFor(league, pos) > 0) surplus += Math.max((counts[pos] || 0) - ded, 0);
    }
    return surplus >= flexTotal;
  }

  // One draft played forward. Returns my lineup value and who survived.
  function runDraft(ctx, forcedId, rand, stopAt) {
    const { board, league, state } = ctx;
    const taken = new Set(state.picks.map((p) => p.id));
    const avail = board.players.filter((p) => !taken.has(p.id));

    const counts = {};
    for (let t = 0; t < league.teams; t++) counts[t] = {};
    const rosters = {};
    for (let t = 0; t < league.teams; t++) rosters[t] = [];
    for (const p of state.picks) {
      counts[p.team][p.pos] = (counts[p.team][p.pos] || 0) + 1;
      rosters[p.team].push(p);
    }

    const opponents = cloneOpponents(ctx.opponents);
    const order = ctx.order;
    let pickNo = state.picks.length;
    const limit = stopAt === undefined ? order.length : stopAt;
    let forcedUsed = forcedId === null;

    while (pickNo < limit && avail.length) {
      const team = order[pickNo];
      const pool = avail.slice(0, POOL_SIZE);
      let slot;

      if (team === state.mySlot && !forcedUsed) {
        slot = avail.findIndex((p) => p.id === forcedId);
        if (slot < 0) break;
        forcedUsed = true;
      } else if (team === state.mySlot) {
        // I keep drafting sensibly: best value, discounted once a position is
        // already well covered.
        let best = 0, bestScore = -Infinity;
        for (let i = 0; i < pool.length; i++) {
          let v = pool[i].vor;
          if ((counts[team][pool[i].pos] || 0) >= capacityFor(league, pool[i].pos) + 1) v *= 0.15;
          if (v > bestScore) { bestScore = v; best = i; }
        }
        slot = best;
      } else {
        const picksLeft = league.rosterSize - (rosters[team].length);
        const drift = driftMultipliers(opponents);
        const probs = pickProbs(opponents, pool, counts[team], picksLeft, drift);
        slot = sample(probs, rand);
        const expected = {};
        for (let i = 0; i < pool.length; i++) {
          expected[pool[i].pos] = (expected[pool[i].pos] || 0) + probs[i];
        }
        observe(opponents, pool[slot].pos, expected);
      }

      const chosen = avail[slot];
      avail.splice(slot, 1);
      counts[team][chosen.pos] = (counts[team][chosen.pos] || 0) + 1;
      rosters[team].push(chosen);
      pickNo += 1;

      // Once my lineup can be fielded, nothing later changes its value. Exact,
      // not an approximation -- and it removes the back half of every run.
      if (team === state.mySlot && stopAt === undefined && lineupFull(league, counts[team])) break;
    }

    return {
      value: lineupValue(league, rosters[state.mySlot], (p) => p.vor || 0),
      survivors: new Set(avail.map((p) => p.id)),
    };
  }

  function nextPickFor(order, team, after) {
    for (let i = after; i < order.length; i++) if (order[i] === team) return i;
    return null;
  }

  function survivalProbabilities(ctx, nSims) {
    const { order, state, board } = ctx;
    const taken = new Set(state.picks.map((p) => p.id));
    const counts = new Map();
    // Seed every available player at zero. A player who is taken in *every*
    // simulation never lands in the survivor set at all, so without this an
    // absent key is indistinguishable from "not simulated" -- and the top
    // player on the board would report 100% survival instead of 0%.
    for (const p of board.players) if (!taken.has(p.id)) counts.set(p.id, 0);

    const target = nextPickFor(order, state.mySlot, state.picks.length + 1);
    if (target === null) {
      for (const k of counts.keys()) counts.set(k, 1);
      return counts;
    }

    for (let i = 0; i < nSims; i++) {
      const { survivors } = runDraft(ctx, null, mulberry32(1000 + i), target);
      for (const id of survivors) {
        if (counts.has(id)) counts.set(id, counts.get(id) + 1);
      }
    }
    for (const [k, v] of counts) counts.set(k, v / nSims);
    return counts;
  }

  function evaluate(ctx, candidates, nSims) {
    const out = new Map();
    for (const cand of candidates) {
      let total = 0;
      for (let i = 0; i < nSims; i++) {
        // Same seed per simulation index across every candidate: the only
        // thing that differs between them is my own pick.
        total += runDraft(ctx, cand.id, mulberry32(7000 + i), undefined).value;
      }
      out.set(cand.id, total / nSims);
    }
    return out;
  }

  // -- recommendations ------------------------------------------------------

  function usable(league, avail, myPositions) {
    const counts = {};
    for (const pos of myPositions) counts[pos] = (counts[pos] || 0) + 1;
    const blocked = new Set(
      league.positions.filter((p) => (counts[p] || 0) >= capacityFor(league, p) + 1)
    );
    if (!blocked.size) return avail;
    const kept = avail.filter((p) => !blocked.has(p.pos));
    return kept.length ? kept : avail;
  }

  function shortlist(league, avail, n) {
    const top = avail.slice(0, Math.max(n - league.positions.length, 3));
    const seen = new Set(top.map((p) => p.id));
    for (const pos of league.positions) {
      const best = avail.find((p) => p.pos === pos);
      if (best && !seen.has(best.id)) { top.push(best); seen.add(best.id); }
    }
    return top.slice(0, n);
  }

  function explain(ctx, row, survival, avail) {
    const { league, state } = ctx;
    const reasons = [];
    const sameTier = avail.filter((p) => p.pos === row.pos && p.tier === row.tier);

    if (sameTier.length <= 1) reasons.push(`last ${row.pos} in tier ${row.tier}`);
    else if (sameTier.length <= 3) reasons.push(`only ${sameTier.length} left in this ${row.pos} tier`);

    if (survival < 0.25) reasons.push(`almost certainly gone by your next pick (${Math.round(survival * 100)}% survival)`);
    else if (survival > 0.7) reasons.push(`likely still there next turn (${Math.round(survival * 100)}%) — you can wait`);

    const mine = state.picks.filter((p) => p.team === state.mySlot).map((p) => p.pos);
    if (mine.length >= league.nStarters - 3) {
      const open = openSlots(league, mine);
      const dedicated = league.starters
        .filter((s) => s.eligible.length === 1 && s.eligible[0] === row.pos)
        .reduce((n, s) => n + (open[s.name] || 0), 0);
      const sharedNames = league.starters
        .filter((s) => s.eligible.length > 1 && s.eligible.includes(row.pos) && (open[s.name] || 0) > 0)
        .map((s) => s.name);
      if (dedicated > 0) reasons.push(`you still need ${dedicated} starting ${row.pos}`);
      else if (sharedNames.length) reasons.push(`would fill your open ${sharedNames.join('/')}`);
    }

    const drift = driftMultipliers(ctx.opponents)[row.pos] || 1;
    if (drift >= 1.4) reasons.push(`${row.pos} run under way (${drift.toFixed(1)}x normal pace)`);
    else if (drift <= 0.7) reasons.push(`${row.pos}s are sliding in this room`);

    if (!reasons.length) reasons.push(`best value on the board (${Math.round(row.vor)} over replacement)`);
    return reasons.join('; ');
  }

  function recommend(ctx, opts = {}) {
    const n = opts.n || 5;
    const nSims = opts.sims || 120;
    const survivalSims = opts.survivalSims || 200;

    const taken = new Set(ctx.state.picks.map((p) => p.id));
    const avail = ctx.board.players.filter((p) => !taken.has(p.id));
    if (!avail.length) return [];

    const survival = survivalProbabilities(ctx, survivalSims);
    const myPositions = ctx.state.picks
      .filter((p) => p.team === ctx.state.mySlot).map((p) => p.pos);
    const cands = shortlist(ctx.league, usable(ctx.league, avail, myPositions), opts.candidates || 12);
    const values = evaluate(ctx, cands, nSims);

    let best = -Infinity;
    for (const v of values.values()) if (v > best) best = v;

    const out = cands.map((row) => {
      const surv = survival.has(row.id) ? survival.get(row.id) : 0;
      return {
        ...row,
        survival: surv,
        rosterValue: values.get(row.id),
        edge: values.get(row.id) - best,
        reason: explain(ctx, row, surv, avail),
      };
    });
    out.sort((a, b) => b.rosterValue - a.rosterValue);
    return out.slice(0, n);
  }

  function roomRead(model) {
    if (!model.seen) return 'No picks observed yet; using the value-based prior.';
    const drift = driftMultipliers(model);
    const parts = [];
    for (const pos of Object.keys(drift).sort((a, b) => drift[b] - drift[a])) {
      if (drift[pos] >= 1.25) parts.push(`${pos} going ${drift[pos].toFixed(1)}x faster than value implies`);
      else if (drift[pos] <= 0.8) parts.push(`${pos} sliding (${drift[pos].toFixed(1)}x)`);
    }
    return parts.length ? parts.join('; ')
      : 'The room is drafting close to value; no strong positional drift.';
  }

  // How disciplined is this room? Fit from where on the board picks came from.
  function fitTemperature(board, pickIds) {
    if (pickIds.length < 8) return 4.0;
    const order = new Map(board.players.map((p, i) => [p.id, i]));
    const ranks = [];
    const taken = new Set();
    for (const id of pickIds) {
      if (!order.has(id)) continue;
      const abs = order.get(id);
      let ahead = 0;
      for (const t of taken) if ((order.get(t) ?? 1e9) < abs) ahead += 1;
      ranks.push(abs - ahead);
      taken.add(id);
    }
    if (!ranks.length) return 4.0;

    let best = 4.0, bestLL = -Infinity;
    for (const temp of [1.5, 2.5, 4.0, 6.0, 9.0, 14.0]) {
      let norm = 0;
      for (let i = 0; i < board.players.length; i++) norm += Math.exp(-i / temp);
      const logNorm = Math.log(norm);
      let ll = 0;
      for (const r of ranks) ll += -r / temp - logNorm;
      if (ll > bestLL) { bestLL = ll; best = temp; }
    }
    return best;
  }

  return {
    PRESETS, POSITIONS, makeLeague, buildBoard, pickOrder, openSlots, starterNeed,
    makeOpponents, observe, driftMultipliers, pickProbs, cloneOpponents,
    recommend, roomRead, fitTemperature, lineupValue, capacityFor,
    replacementRank, nextPickFor,
  };
})();

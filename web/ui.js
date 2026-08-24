(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const DATA = window.__BOARD__;

  const state = {
    cfg: {
      teams: 12, scoring: 'ppr', bench: 6,
      qb: 1, rb: 2, wr: 2, te: 1, flex: 1, superflex: 0,
      tePremium: 0, snake: true,
    },
    mySlot: 10,          // zero-based
    picks: [],           // {id, name, pos, team, rookie?}
    league: null,
    board: null,
    order: null,
    opponents: null,
    recs: null,
    busy: false,
  };

  // -- setup ----------------------------------------------------------------

  function readConfig() {
    const num = (id, fallback) => {
      const v = parseInt($(id).value, 10);
      return Number.isFinite(v) ? v : fallback;
    };
    state.cfg = {
      teams: num('cfgTeams', 12),
      bench: num('cfgBench', 6),
      qb: num('cfgQB', 1), rb: num('cfgRB', 2), wr: num('cfgWR', 2),
      te: num('cfgTE', 1), flex: num('cfgFLEX', 1), superflex: num('cfgSF', 0),
      scoring: $('cfgScoring').value,
      tePremium: parseFloat($('cfgTEP').value) || 0,
      snake: $('cfgSnake').value === 'snake',
    };
    const slot = num('cfgSlot', 1);
    state.mySlot = Math.min(Math.max(slot, 1), state.cfg.teams) - 1;
    $('cfgSlot').value = state.mySlot + 1;
  }

  function rebuild() {
    readConfig();
    state.league = FFOPT.makeLeague(state.cfg);
    state.board = FFOPT.buildBoard(DATA.players, state.league, DATA.blend);
    state.order = FFOPT.pickOrder(state.league);

    // Replay every pick so the opponent model sees the draft as it unfolded.
    state.opponents = FFOPT.makeOpponents(state.league);
    const byId = new Map(state.board.players.map((p) => [p.id, p]));
    const replay = [];
    for (const pick of state.picks) {
      const taken = new Set(replay.map((p) => p.id));
      const avail = state.board.players.filter((p) => !taken.has(p.id));
      const team = state.order[replay.length];
      const counts = {};
      for (const p of replay) if (p.team === team) counts[p.pos] = (counts[p.pos] || 0) + 1;
      const picksLeft = state.league.rosterSize - replay.filter((p) => p.team === team).length;
      const pool = avail.slice(0, 60);
      const probs = FFOPT.pickProbs(
        state.opponents, pool, counts, picksLeft, FFOPT.driftMultipliers(state.opponents)
      );
      const expected = {};
      for (let i = 0; i < pool.length; i++) expected[pool[i].pos] = (expected[pool[i].pos] || 0) + probs[i];
      FFOPT.observe(state.opponents, pick.pos, expected);
      replay.push(pick);
    }

    const realIds = state.picks.filter((p) => !p.rookie).map((p) => p.id);
    if (realIds.length >= 8) {
      state.opponents.temperature = FFOPT.fitTemperature(state.board, realIds);
    }

    // Keep pick metadata in sync with the current scoring.
    for (const pick of state.picks) {
      const row = byId.get(pick.id);
      if (row) { pick.pos = row.pos; pick.name = row.name; }
    }
    state.recs = null;
    render();
  }

  // -- draft mechanics ------------------------------------------------------

  const teamOnClock = () => state.order[state.picks.length];
  const isMyTurn = () => teamOnClock() === state.mySlot;
  const complete = () => state.picks.length >= state.league.totalPicks;

  function roundSlot(n) {
    return [Math.floor(n / state.league.teams) + 1, (n % state.league.teams) + 1];
  }

  function recordPick(row, rookie) {
    if (complete()) return;
    state.picks.push({
      id: row.id, name: row.name, pos: row.pos,
      team: teamOnClock(), rookie: !!rookie,
    });
    rebuild();
  }

  function undo() {
    if (!state.picks.length) return;
    state.picks.pop();
    rebuild();
  }

  // -- rendering ------------------------------------------------------------

  function render() {
    renderStatus();
    renderBoard();
    renderRoster();
    renderLog();
    renderRecs();
    renderLeagueInfo();
  }

  function renderStatus() {
    const box = $('status');
    if (complete()) {
      box.innerHTML = '<div class="turn done">Draft complete</div>';
      return;
    }
    const team = teamOnClock();
    const [rnd, slot] = roundSlot(state.picks.length);
    const mine = team === state.mySlot;
    const nextMine = FFOPT.nextPickFor(state.order, state.mySlot, state.picks.length);
    const away = nextMine === null ? null : nextMine - state.picks.length;

    box.innerHTML = `
      <div class="turn ${mine ? 'mine' : ''}">
        <span class="rnd">${rnd}.${String(slot).padStart(2, '0')}</span>
        <span class="who">${mine ? 'Your pick' : 'Team ' + (team + 1)}</span>
        ${!mine && away !== null ? `<span class="away">your turn in ${away}</span>` : ''}
      </div>`;
  }

  function renderLeagueInfo() {
    const b = state.board;
    const parts = state.league.starters
      .map((s) => (s.count > 1 ? `${s.count}×${s.name}` : s.name)).join(', ');
    const repl = FFOPT.POSITIONS.map((p) => {
      const rank = FFOPT.replacementRank(state.league, p, b.flexShares[p]);
      return `<span><b>${p}</b> ${b.replacement[p].toFixed(0)} <i>(${rank}th)</i></span>`;
    }).join('');
    const flex = Object.entries(b.flexShares)
      .filter(([, v]) => v > 0.01)
      .sort((a, b2) => b2[1] - a[1])
      .map(([p, v]) => `${p} ${Math.round(v * 100)}%`).join(', ');

    $('leagueInfo').innerHTML = `
      <div class="li-row">${state.league.teams} teams · ${parts} · ${state.league.bench} bench</div>
      <div class="li-row repl">Replacement level: ${repl}</div>
      ${flex ? `<div class="li-row muted">Flex slots go to ${flex}</div>` : ''}`;
  }

  let boardFilter = '';
  let boardPos = '';

  function renderBoard() {
    const taken = new Set(state.picks.map((p) => p.id));
    let rows = state.board.players.filter((p) => !taken.has(p.id));
    if (boardPos) rows = rows.filter((p) => p.pos === boardPos);
    if (boardFilter) {
      const q = boardFilter.toLowerCase();
      rows = rows.filter((p) => p.name.toLowerCase().includes(q));
    }
    const shown = rows.slice(0, 60);

    $('boardBody').innerHTML = shown.map((p) => `
      <tr data-id="${p.id}">
        <td class="num">${p.overallRank}</td>
        <td class="nm">${p.name}</td>
        <td><span class="pos ${p.pos}">${p.pos}${p.posRank}</span></td>
        <td class="num">${p.tier}</td>
        <td class="num">${p.points.toFixed(0)}</td>
        <td class="num strong">${p.vor.toFixed(0)}</td>
        <td><button class="mini" data-pick="${p.id}">draft</button></td>
      </tr>`).join('');

    $('boardCount').textContent = `${rows.length} available`;
  }

  function renderRoster() {
    const mine = state.picks.filter((p) => p.team === state.mySlot);
    const byId = new Map(state.board.players.map((p) => [p.id, p]));
    if (!mine.length) {
      $('rosterBody').innerHTML = '<div class="empty">No picks yet</div>';
    } else {
      $('rosterBody').innerHTML = mine.map((p) => {
        const row = byId.get(p.id);
        return `<div class="rrow">
          <span class="pos ${p.pos}">${p.pos}</span>
          <span class="rn">${p.name}</span>
          <span class="rp">${row ? row.points.toFixed(0) : '—'}</span>
        </div>`;
      }).join('');
    }
    const open = FFOPT.openSlots(state.league, mine.map((p) => p.pos));
    const needs = Object.entries(open).filter(([, v]) => v > 0)
      .map(([k, v]) => (v > 1 ? `${v}×${k}` : k)).join(', ');
    $('needs').textContent = needs ? `Still needs: ${needs}` : 'Starting lineup is full';
  }

  function renderLog() {
    if (!state.picks.length) {
      $('logBody').innerHTML = '<div class="empty">No picks recorded</div>';
      return;
    }
    $('logBody').innerHTML = state.picks.map((p, i) => {
      const [r, s] = roundSlot(i);
      const mine = p.team === state.mySlot;
      return `<div class="lrow ${mine ? 'mine' : ''}">
        <span class="lp">${r}.${String(s).padStart(2, '0')}</span>
        <span class="lt">${mine ? 'YOU' : 'T' + (p.team + 1)}</span>
        <span class="pos ${p.pos}">${p.pos}</span>
        <span class="ln">${p.name}${p.rookie ? ' <i>rookie</i>' : ''}</span>
      </div>`;
    }).reverse().join('');
  }

  function renderRecs() {
    const box = $('recBody');
    if (complete()) { box.innerHTML = '<div class="empty">Draft complete</div>'; return; }
    if (state.busy) { box.innerHTML = '<div class="empty">Simulating…</div>'; return; }
    if (!isMyTurn()) {
      box.innerHTML = '<div class="empty">Record picks until your turn, or press <b>Recommend</b> to look ahead.</div>';
      return;
    }
    if (!state.recs) { box.innerHTML = '<div class="empty">Press <b>Recommend</b>.</div>'; return; }

    box.innerHTML = state.recs.map((r, i) => `
      <div class="rec ${i === 0 ? 'top' : ''}">
        <div class="rec-head">
          <span class="rank">${i + 1}</span>
          <span class="rname">${r.name}</span>
          <span class="pos ${r.pos}">${r.pos}${r.posRank}</span>
          <button class="mini" data-pick="${r.id}">draft</button>
        </div>
        <div class="rec-nums">
          <span>proj <b>${r.points.toFixed(0)}</b></span>
          <span>VOR <b>${r.vor.toFixed(0)}</b></span>
          <span>survives <b>${Math.round(r.survival * 100)}%</b></span>
          <span>edge <b>${r.edge >= 0 ? '+' : ''}${r.edge.toFixed(1)}</b></span>
        </div>
        <div class="rec-why">${r.reason}</div>
      </div>`).join('');
    $('roomRead').textContent = FFOPT.roomRead(state.opponents);
  }

  function runRecommend() {
    state.busy = true;
    renderRecs();
    // Yield so the "Simulating…" paint lands before the main thread blocks.
    setTimeout(() => {
      const ctx = {
        board: state.board, league: state.league, order: state.order,
        opponents: state.opponents,
        state: { picks: state.picks, mySlot: state.mySlot },
      };
      state.recs = FFOPT.recommend(ctx, {
        n: 5,
        sims: parseInt($('cfgSims').value, 10) || 120,
        survivalSims: 200,
      });
      state.busy = false;
      renderRecs();
      $('roomRead').textContent = FFOPT.roomRead(state.opponents);
    }, 20);
  }

  // -- quick entry ----------------------------------------------------------

  // Draft rooms call picks by surname — "Chase!", "Nacua!" — so a surname hit
  // has to outrank a first-name hit. Prefix-matching alone answers "chase" with
  // Chase Brown rather than Ja'Marr Chase, which is exactly the kind of mistake
  // that is hard to notice and expensive to undo. Within each tier of match
  // quality, the more valuable player comes first, since that is overwhelmingly
  // the one being called out.
  function matches(query) {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const taken = new Set(state.picks.map((p) => p.id));
    const avail = state.board.players.filter((p) => !taken.has(p.id));

    const exact = avail.filter((p) => p.name.toLowerCase() === q);
    if (exact.length === 1) return exact;

    const norm = (s) => s.toLowerCase().replace(/[.'’]/g, '');
    const nq = norm(q);
    const rank = (p) => {
      const name = norm(p.name);
      const parts = name.split(/\s+/);
      const surname = parts.length > 1 ? parts.slice(1).join(' ') : parts[0];
      if (name === nq) return 0;
      if (surname === nq) return 1;
      if (surname.startsWith(nq)) return 2;
      if (name.startsWith(nq)) return 3;
      if (parts.some((part) => part.startsWith(nq))) return 4;
      if (name.includes(nq)) return 5;
      return 99;
    };

    return avail
      .map((p) => ({ p, r: rank(p) }))
      .filter((x) => x.r < 99)
      .sort((a, b) => (a.r - b.r) || (b.p.vor - a.p.vor))
      .slice(0, 8)
      .map((x) => x.p);
  }

  let suggestList = [];

  function renderSuggest(list) {
    suggestList = list;
    const box = $('suggest');
    if (!list.length) { box.innerHTML = ''; box.classList.remove('open'); return; }
    if (suggestIndex >= list.length) suggestIndex = 0;
    box.classList.add('open');
    box.innerHTML = list.map((p, i) => `
      <div class="sug ${i === suggestIndex ? 'sel' : ''}" data-pick="${p.id}">
        <span class="pos ${p.pos}">${p.pos}${p.posRank}</span>
        <span class="sn">${p.name}</span>
        <span class="sv">${p.vor.toFixed(0)}</span>
      </div>`).join('');
  }

  let suggestIndex = 0;

  function submitEntry() {
    const input = $('entry');
    const list = matches(input.value);
    if (!list.length) {
      flash('No match — use “Add rookie” for a player with no NFL history.');
      return;
    }
    const chosen = list[Math.min(suggestIndex, list.length - 1)];
    recordPick(chosen, false);
    // Say what was actually recorded. A mis-matched name is easy to miss and
    // expensive to find later, and Undo is one key away.
    flash(`Recorded ${chosen.name} (${chosen.pos}) — press Undo if that's wrong`);
    input.value = '';
    suggestIndex = 0;
    renderSuggest([]);
    input.focus();
  }

  function flash(message) {
    const el = $('flash');
    el.textContent = message;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 3200);
  }

  // -- draft log import / export -------------------------------------------
  //
  // Deliberately not browser storage: a plain text log can be copied between
  // devices, pasted back after a refresh, and read by a human.

  function exportLog() {
    return state.picks.map((p) => (p.rookie ? `*${p.name}|${p.pos}` : p.name)).join('\n');
  }

  function importLog(text) {
    state.picks = [];
    rebuild();
    const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
    let failed = 0;
    for (const line of lines) {
      if (line.startsWith('*')) {
        const [name, pos] = line.slice(1).split('|');
        recordRookieDirect(name, (pos || 'RB').toUpperCase());
        continue;
      }
      const list = matches(line);
      if (!list.length) { failed += 1; continue; }
      state.picks.push({
        id: list[0].id, name: list[0].name, pos: list[0].pos,
        team: state.order[state.picks.length], rookie: false,
      });
    }
    rebuild();
    flash(failed ? `Loaded, ${failed} name(s) not recognised` : `Loaded ${state.picks.length} picks`);
  }

  function recordRookieDirect(name, pos) {
    state.picks.push({
      id: `rookie:${name.toLowerCase().replace(/\s+/g, '_')}:${state.picks.length}`,
      name, pos, team: state.order[state.picks.length], rookie: true,
    });
  }

  // -- wiring ---------------------------------------------------------------

  function init() {
    $('season').textContent = DATA.season;
    $('dataRange').textContent = DATA.generated_from;

    ['cfgTeams', 'cfgBench', 'cfgQB', 'cfgRB', 'cfgWR', 'cfgTE', 'cfgFLEX',
      'cfgSF', 'cfgScoring', 'cfgTEP', 'cfgSlot', 'cfgSnake'].forEach((id) => {
      $(id).addEventListener('change', () => {
        // Changing the roster shape mid-draft would invalidate the pick order.
        if (state.picks.length && ['cfgTeams', 'cfgSnake'].includes(id)) {
          if (!confirm('Changing team count or draft type resets the draft. Continue?')) {
            rebuild(); return;
          }
          state.picks = [];
        }
        rebuild();
      });
    });

    $('entry').addEventListener('input', (e) => {
      suggestIndex = 0;
      renderSuggest(matches(e.target.value));
    });
    $('entry').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); submitEntry(); return; }
      if (e.key === 'Escape') { renderSuggest([]); return; }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (!suggestList.length) return;
        e.preventDefault();
        suggestIndex = (suggestIndex + (e.key === 'ArrowDown' ? 1 : -1) + suggestList.length)
          % suggestList.length;
        renderSuggest(suggestList);
      }
    });

    document.body.addEventListener('click', (e) => {
      const pickBtn = e.target.closest('[data-pick]');
      if (pickBtn) {
        const id = pickBtn.getAttribute('data-pick');
        const row = state.board.players.find((p) => p.id === id);
        if (row) {
          recordPick(row, false);
          $('entry').value = '';
          renderSuggest([]);
        }
        return;
      }
      if (!e.target.closest('#entryWrap')) renderSuggest([]);
    });

    $('btnRec').addEventListener('click', runRecommend);
    $('btnUndo').addEventListener('click', undo);
    $('btnReset').addEventListener('click', () => {
      if (state.picks.length && !confirm('Clear all picks?')) return;
      state.picks = []; rebuild();
    });

    $('btnRookie').addEventListener('click', () => {
      const name = $('entry').value.trim();
      if (!name) { flash('Type the name first, then press Add rookie.'); return; }
      const pos = prompt(`Position for ${name}? (QB, RB, WR, TE)`, 'RB');
      if (!pos) return;
      const clean = pos.trim().toUpperCase();
      if (!FFOPT.POSITIONS.includes(clean)) { flash('Position must be QB, RB, WR or TE.'); return; }
      recordRookieDirect(name, clean);
      $('entry').value = '';
      renderSuggest([]);
      rebuild();
    });

    $('boardSearch').addEventListener('input', (e) => { boardFilter = e.target.value; renderBoard(); });
    document.querySelectorAll('[data-posfilter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        boardPos = btn.getAttribute('data-posfilter');
        document.querySelectorAll('[data-posfilter]').forEach((b) => b.classList.remove('on'));
        btn.classList.add('on');
        renderBoard();
      });
    });

    $('btnLog').addEventListener('click', () => {
      $('logText').value = exportLog();
      $('logModal').classList.add('open');
    });
    $('btnLogClose').addEventListener('click', () => $('logModal').classList.remove('open'));
    $('btnLogLoad').addEventListener('click', () => {
      importLog($('logText').value);
      $('logModal').classList.remove('open');
    });
    $('btnLogCopy').addEventListener('click', () => {
      $('logText').select();
      try { document.execCommand('copy'); flash('Copied'); } catch (err) { flash('Select and copy manually'); }
    });

    document.querySelectorAll('[data-preset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const preset = btn.getAttribute('data-preset');
        const map = {
          std12: { teams: 12, scoring: 'half_ppr', sf: 0 },
          ppr12: { teams: 12, scoring: 'ppr', sf: 0 },
          ppr10: { teams: 10, scoring: 'ppr', sf: 0 },
          sf12: { teams: 12, scoring: 'half_ppr', sf: 1 },
        }[preset];
        if (!map) return;
        if (state.picks.length && !confirm('Switching preset resets the draft. Continue?')) return;
        state.picks = [];
        $('cfgTeams').value = map.teams;
        $('cfgScoring').value = map.scoring;
        $('cfgSF').value = map.sf;
        rebuild();
      });
    });

    rebuild();
    $('entry').focus();
  }

  document.addEventListener('DOMContentLoaded', init);
})();

"use strict";

const D = window.LOTTO;
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};

/* draw row: [round,date,n1..n6,bonus,winners,amountEok] */
const numsOf = (r) => r.slice(2, 8);
const bonusOf = (r) => r[8];
const winnersOf = (r) => r[9];
const amtOf = (r) => r[10];
const oddOf = (r) => numsOf(r).filter((n) => n % 2 === 1).length;
const oeOf = (r) => `${oddOf(r)}:${6 - oddOf(r)}`;
const ballClass = (n) => (n <= 10 ? "c1" : n <= 20 ? "c2" : n <= 30 ? "c3" : n <= 40 ? "c4" : "c5");
const ballsHtml = (nums, fixed = [], cls = "") => {
  const f = new Set(fixed);
  return (
    '<div class="balls">' +
    nums.map((n) => `<div class="ball ${cls} ${ballClass(n)}${f.has(n) ? " fixed" : ""}">${n}</div>`).join("") +
    "</div>"
  );
};
const lerpHex = (a, b, t) => {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
};

/* ── SVG 차트 ─────────────────────────── */
function barSVG(items, { maxY, fmt = (v) => v } = {}) {
  const W = 420, H = 240, pl = 38, pr = 10, pt = 12, pb = 40;
  const iw = W - pl - pr, ih = H - pt - pb;
  const max = maxY || Math.max(...items.map((i) => i.value), 1);
  const bw = (iw / items.length) * 0.62;
  const gap = (iw / items.length) * 0.38;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const y = pt + (ih * k) / 4;
    const val = max * (1 - k / 4);
    g += `<line x1="${pl}" y1="${y.toFixed(1)}" x2="${W - pr}" y2="${y.toFixed(1)}" stroke="#eef1f5"/>`;
    g += `<text x="${pl - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#9aa3b2">${fmt(Math.round(val))}</text>`;
  }
  let bars = "";
  items.forEach((it, idx) => {
    const x = pl + idx * (iw / items.length) + gap / 2;
    const h = (it.value / max) * ih;
    const y = pt + ih - h;
    const col = it.color || "#60a5fa";
    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(h, 0).toFixed(1)}" rx="3" fill="${col}"><title>${it.label}: ${it.value}</title></rect>`;
    bars += `<text x="${(x + bw / 2).toFixed(1)}" y="${H - pb + 14}" text-anchor="middle" font-size="10" fill="#6b7280">${it.label}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${g}${bars}</svg>`;
}

function lineSVG(points, { fmt = (v) => v } = {}) {
  const W = 460, H = 240, pl = 40, pr = 12, pt = 12, pb = 34;
  const iw = W - pl - pr, ih = H - pt - pb;
  if (points.length === 0) return `<svg viewBox="0 0 ${W} ${H}"></svg>`;
  const xs = points.map((p) => p.x);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const maxY = Math.max(...points.map((p) => p.y), 1);
  const sx = (x) => pl + (maxX === minX ? iw / 2 : ((x - minX) / (maxX - minX)) * iw);
  const sy = (y) => pt + ih - (y / maxY) * ih;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const y = pt + (ih * k) / 4;
    g += `<line x1="${pl}" y1="${y.toFixed(1)}" x2="${W - pr}" y2="${y.toFixed(1)}" stroke="#eef1f5"/>`;
    g += `<text x="${pl - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#9aa3b2">${Math.round(maxY * (1 - k / 4))}</text>`;
  }
  const line = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  const area = `M${sx(minX).toFixed(1)},${(pt + ih).toFixed(1)} ` +
    points.map((p) => `L${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ") +
    ` L${sx(maxX).toFixed(1)},${(pt + ih).toFixed(1)} Z`;
  // x 라벨 ~5개
  const ticks = 5;
  let xl = "";
  for (let i = 0; i < ticks; i++) {
    const p = points[Math.round((points.length - 1) * (i / (ticks - 1)))];
    if (!p) continue;
    xl += `<text x="${sx(p.x).toFixed(1)}" y="${H - pb + 16}" text-anchor="middle" font-size="9.5" fill="#6b7280">${p.x}회</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${g}<path d="${area}" fill="#dbeafe" opacity="0.7"/><path d="${line}" fill="none" stroke="#2563eb" stroke-width="2" stroke-linejoin="round"/>${xl}</svg>`;
}

function donutSVG(slices, opts = {}) {
  const W = 420, H = 220, cx = 120, cy = 110, r = 82, ir = 50;
  const total = slices.reduce((s, x) => s + x.value, 0) || 1;
  let a0 = -Math.PI / 2;
  let arcs = "";
  slices.forEach((s) => {
    const a1 = a0 + (s.value / total) * Math.PI * 2;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const xi0 = cx + ir * Math.cos(a1), yi0 = cy + ir * Math.sin(a1);
    const xi1 = cx + ir * Math.cos(a0), yi1 = cy + ir * Math.sin(a0);
    arcs += `<path d="M${x0.toFixed(1)},${y0.toFixed(1)} A${r},${r} 0 ${large} 1 ${x1.toFixed(1)},${y1.toFixed(1)} L${xi0.toFixed(1)},${yi0.toFixed(1)} A${ir},${ir} 0 ${large} 0 ${xi1.toFixed(1)},${yi1.toFixed(1)} Z" fill="${s.color}"><title>${s.label}: ${s.value} (${Math.round((s.value / total) * 100)}%)</title></path>`;
    a0 = a1;
  });
  let legend = "";
  slices.forEach((s, i) => {
    const y = 34 + i * 32;
    const pct = Math.round((s.value / total) * 100);
    legend += `<rect x="250" y="${y - 10}" width="12" height="12" rx="3" fill="${s.color}"/>`;
    legend += `<text x="270" y="${y}" font-size="12" fill="#374151">${s.label}</text>`;
    legend += `<text x="408" y="${y}" font-size="12" fill="#6b7280" text-anchor="end">${pct}%</text>`;
  });
  const centerBig = (opts.centerTop ?? total).toLocaleString();
  const centerSub = opts.centerSub ?? "합계";
  const center =
    `<text x="${cx}" y="${cy - 2}" text-anchor="middle" font-size="24" font-weight="800" fill="#1f2937">${centerBig}</text>` +
    `<text x="${cx}" y="${cy + 18}" text-anchor="middle" font-size="11" fill="#6b7280">${centerSub}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img">${arcs}${center}${legend}</svg>`;
}

/* ── 대시보드 상태/필터 ───────────────── */
const state = {
  range: 200,
  oe: "all",
  include: null,
  tableSearch: "",
  sort: { key: "round", dir: -1 },
};

function filteredDraws() {
  let rows = D.draws; // 회차 오름차순
  if (state.range > 0) rows = rows.slice(-state.range);
  return rows.filter((r) => {
    if (state.oe !== "all" && oeOf(r) !== state.oe) return false;
    if (state.include != null && !numsOf(r).includes(state.include)) return false;
    return true;
  });
}

const GBANDS = [
  { label: "1~10 (G1)", lo: 1, hi: 10, color: "#3b82f6" },
  { label: "11~20 (G2)", lo: 11, hi: 20, color: "#60a5fa" },
  { label: "21~30 (G3)", lo: 21, hi: 30, color: "#93c5fd" },
  { label: "31~40 (G4)", lo: 31, hi: 40, color: "#34d399" },
  { label: "41~45 (G5)", lo: 41, hi: 45, color: "#fbbf24" },
];

function renderKPIs(rows) {
  const n = rows.length;
  const winners = rows.map(winnersOf);
  const avgW = n ? winners.reduce((a, b) => a + b, 0) / n : 0;
  const maxW = n ? Math.max(...winners) : 0;
  const avgAmt = n ? rows.reduce((a, r) => a + amtOf(r), 0) / n : 0;
  // 핫넘버
  const freq = numberFreq(rows);
  let hot = 1, hotC = -1;
  for (let k = 1; k <= 45; k++) if (freq[k] > hotC) { hotC = freq[k]; hot = k; }
  const cards = [
    { big: `${n}<small>회</small>`, lbl: "분석 회차" },
    { big: `${avgW.toFixed(1)}<small>명</small>`, lbl: "평균 1등 당첨자" },
    { big: `${avgAmt.toFixed(1)}<small>억</small>`, lbl: "평균 1인당 당첨금" },
    { big: `${maxW}<small>명</small>`, lbl: "최다 당첨자(회차)" },
    { big: `${hot}<small>번</small>`, lbl: `최다 출현 (${hotC}회)` },
  ];
  $("#kpis").innerHTML = cards
    .map((c) => `<div class="kpi"><div class="big">${c.big}</div><div class="lbl">${c.lbl}</div></div>`)
    .join("");
}

function numberFreq(rows) {
  const f = {};
  for (let k = 1; k <= 45; k++) f[k] = 0;
  rows.forEach((r) => numsOf(r).forEach((n) => f[n]++));
  return f;
}

function renderHeatmap(rows) {
  const f = numberFreq(rows);
  const vals = Object.values(f);
  const min = Math.min(...vals), max = Math.max(...vals, 1);
  let html = "";
  for (let n = 1; n <= 45; n++) {
    const t = max === min ? 0.5 : (f[n] - min) / (max - min);
    const bg = lerpHex("#dbe7ff", "#1d4ed8", t);
    const fg = t > 0.55 ? "#fff" : "#0b2a6b";
    const active = state.include === n ? " active" : "";
    html += `<div class="hm-cell${active}" data-n="${n}" style="background:${bg};color:${fg}"><div class="n">${n}</div><div class="c">${f[n]}</div></div>`;
  }
  $("#heatmap").innerHTML = html;
  $$(".hm-cell").forEach((c) =>
    c.addEventListener("click", () => {
      const n = Number(c.dataset.n);
      state.include = state.include === n ? null : n;
      $("#fInclude").value = state.include || "";
      renderDashboard();
    })
  );
  $("#legendClear").hidden = state.include == null;
}

function renderCharts(rows) {
  // 홀짝 막대
  const oeOrder = ["0:6", "1:5", "2:4", "3:3", "4:2", "5:1", "6:0"];
  const oeCount = {};
  oeOrder.forEach((k) => (oeCount[k] = 0));
  rows.forEach((r) => (oeCount[oeOf(r)] = (oeCount[oeOf(r)] || 0) + 1));
  const oeMax = Math.max(...Object.values(oeCount), 1);
  $("#chartOE").innerHTML = barSVG(
    oeOrder.map((k) => ({ label: k, value: oeCount[k], color: oeCount[k] === oeMax ? "#2563eb" : "#93c5fd" }))
  );

  // 회차별 당첨자 라인
  const pts = rows.map((r) => ({ x: r[0], y: winnersOf(r) }));
  $("#chartLine").innerHTML = lineSVG(pts);

  // 번호대 도넛
  const band = GBANDS.map((b) => ({ label: b.label, color: b.color, value: 0 }));
  rows.forEach((r) =>
    numsOf(r).forEach((n) => {
      const idx = GBANDS.findIndex((b) => n >= b.lo && n <= b.hi);
      if (idx >= 0) band[idx].value++;
    })
  );
  $("#chartDonut").innerHTML = donutSVG(band, { centerSub: "총 출현 번호" });
}

function renderTable(rows) {
  const q = state.tableSearch.trim();
  let shown = rows;
  if (q) {
    shown = rows.filter((r) =>
      `${r[0]}회 ${r[1]} ${numsOf(r).join(" ")} 보너스${bonusOf(r)}`.includes(q)
    );
  }
  const { key, dir } = state.sort;
  const getKey = {
    round: (r) => r[0],
    date: (r) => r[1],
    sum: (r) => numsOf(r).reduce((total, number) => total + number, 0),
    winners: winnersOf,
    amount: amtOf,
  }[key];
  const sorted = [...shown].sort((a, b) => {
    const va = getKey(a), vb = getKey(b);
    return va < vb ? -dir : va > vb ? dir : 0;
  });
  const body = sorted
    .map((r) => {
      const total = numsOf(r).reduce((sum, number) => sum + number, 0);
      return (
        `<tr><td><b>${r[0]}</b>회</td><td>${r[1]}</td>` +
        `<td>${ballsHtml(numsOf(r))}</td>` +
        `<td>${total}</td>` +
        `<td>${oeOf(r)}</td><td>${winnersOf(r)}명</td><td class="amt">${amtOf(r)}억</td></tr>`
      );
    })
    .join("");
  $("#drawTbody").innerHTML = body;
  $("#tableCount").textContent = q ? `${sorted.length} / ${rows.length}회` : `총 ${rows.length}회`;
  $$("#drawTable thead th[data-sort]").forEach((th) => {
    th.classList.toggle("sorted", th.dataset.sort === key);
  });
}

function renderNotes() {
  $("#notes").innerHTML =
    `데이터: 동행복권 공식 당첨 결과 ${D.meta.firstRound}~${D.meta.lastRound}회, ${D.meta.count}회.<br>` +
    `필터는 표·차트·KPI·히트맵에 함께 적용됩니다.<br>` +
    `후보 생성 모델: ${D.prediction.name}, 다음 대상 ${D.prediction.nextRound}회.<br>` +
    `모든 단일 조합의 1등 확률은 1/8,145,060으로 같습니다.`;
}

function renderDashboard() {
  const rows = filteredDraws();
  renderKPIs(rows);
  renderHeatmap(rows);
  renderCharts(rows);
  renderTable(rows);
  $("#legendClear").hidden = state.include == null;
}

/* 필터 이벤트 */
function bindFilters() {
  // 홀짝 옵션 채우기
  ["0:6", "1:5", "2:4", "3:3", "4:2", "5:1", "6:0"].forEach((k) => {
    $("#fOE").appendChild(new Option(`${k}`, k));
  });
  $("#fRange").addEventListener("change", (e) => { state.range = Number(e.target.value); renderDashboard(); });
  $("#fOE").addEventListener("change", (e) => { state.oe = e.target.value; renderDashboard(); });
  $("#fInclude").addEventListener("input", (e) => {
    const v = parseInt(e.target.value, 10);
    state.include = Number.isInteger(v) && v >= 1 && v <= 45 ? v : null;
    renderDashboard();
  });
  $("#tableSearch").addEventListener("input", (e) => {
    state.tableSearch = e.target.value;
    renderTable(filteredDraws());
  });
  $("#fReset").addEventListener("click", () => {
    Object.assign(state, { range: 200, oe: "all", include: null, tableSearch: "" });
    $("#fRange").value = "200"; $("#fOE").value = "all";
    $("#fInclude").value = ""; $("#tableSearch").value = "";
    renderDashboard();
  });
  $("#legendClear").addEventListener("click", () => {
    state.include = null; $("#fInclude").value = ""; renderDashboard();
  });
  $$("#drawTable thead th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const k = th.dataset.sort;
      if (state.sort.key === k) state.sort.dir *= -1;
      else state.sort = { key: k, dir: k === "date" || k === "round" ? -1 : -1 };
      renderTable(filteredDraws());
    });
  });
}

/* ── 상단 내비 ────────────────────────── */
$$(".topnav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".topnav button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    $$(".tab").forEach((s) => (s.hidden = s.id !== `tab-${tab}`));
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

/* ── 번호 생성 ────────────────────────── */
function parseNumberList(raw) {
  if (!raw.trim()) return [];
  const out = [];
  for (const part of raw.split(/[,\s]+/)) {
    if (!part) continue;
    const value = parseInt(part, 10);
    if (!Number.isInteger(value) || value < 1 || value > 45) throw new Error(`잘못된 번호: "${part}"`);
    if (!out.includes(value)) out.push(value);
  }
  return out;
}
function randomInt(max) {
  if (window.crypto?.getRandomValues) {
    const values = new Uint32Array(1);
    const limit = Math.floor(0x100000000 / max) * max;
    do window.crypto.getRandomValues(values); while (values[0] >= limit);
    return values[0] % max;
  }
  return Math.floor(Math.random() * max);
}
function sample(arr, count) {
  const copy = [...arr];
  for (let i = 0; i < count; i++) {
    const j = i + randomInt(copy.length - i); [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, count);
}
const consecutiveCount = (numbers) => numbers.slice(1).filter((number, i) => number - numbers[i] === 1).length;
function matchesScenario(numbers, scenario) {
  if (!scenario) return true;
  const low = numbers.filter((number) => number <= 10).length;
  const high = numbers.filter((number) => number >= 31).length;
  const total = numbers.reduce((sum, number) => sum + number, 0);
  const checks = [
    scenario.min_low == null || low >= scenario.min_low, scenario.max_low == null || low <= scenario.max_low,
    scenario.min_high == null || high >= scenario.min_high, scenario.max_high == null || high <= scenario.max_high,
    scenario.min_sum == null || total >= scenario.min_sum, scenario.max_sum == null || total <= scenario.max_sum,
    scenario.min_first == null || numbers[0] >= scenario.min_first, scenario.max_first == null || numbers[0] <= scenario.max_first,
    scenario.min_consecutive == null || consecutiveCount(numbers) >= scenario.min_consecutive,
    scenario.max_consecutive == null || consecutiveCount(numbers) <= scenario.max_consecutive,
  ];
  return checks.every(Boolean);
}
function scoreParts(numbers) {
  let pair = 0;
  for (let i = 0; i < numbers.length; i++) for (let j = i + 1; j < numbers.length; j++) {
    pair += D.prediction.pairScores[`${numbers[i]}-${numbers[j]}`] || 0;
  }
  pair /= 15;
  const cycle = numbers.reduce((sum, number) => sum + (D.prediction.cycleScores[String(number)] || 0), 0) / 6;
  const oe = `OE=${numbers.filter((number) => number % 2).length}:${numbers.filter((number) => number % 2 === 0).length}`;
  const penalty = D.prediction.rareOePatterns.includes(D.prediction.previousOe) && D.prediction.rareOePatterns.includes(oe)
    ? D.prediction.rareAfterRarePenalty : 0;
  const total = pair * D.prediction.weights.pair + cycle * D.prediction.weights.cycle - penalty;
  return { pair, cycle, penalty, total };
}
function generateCombos({ count, poolSize, scenarioName, noConsec, fix, exclude }) {
  const fixed = new Set(fix), excluded = new Set(exclude);
  if (fix.length > 6) throw new Error("고정 번호는 최대 6개입니다.");
  if (fix.some((number) => excluded.has(number))) throw new Error("고정 번호와 제외 번호가 겹칩니다.");
  const available = Array.from({ length: 45 }, (_, i) => i + 1).filter((number) => !fixed.has(number) && !excluded.has(number));
  const needed = 6 - fix.length;
  if (needed > available.length) throw new Error("제외수가 너무 많아 조합을 만들 수 없습니다.");
  const scenario = D.prediction.scenarios.find((item) => item.name === scenarioName);
  const history = new Set(D.draws.map((row) => numsOf(row).join(",")));
  const found = new Map();
  const maxAttempts = Math.max(poolSize * 30, 20000);
  for (let attempt = 0; attempt < maxAttempts && found.size < poolSize; attempt++) {
    const numbers = [...fix, ...sample(available, needed)].sort((a, b) => a - b);
    const key = numbers.join(",");
    if (found.has(key) || history.has(key)) continue;
    if (noConsec && consecutiveCount(numbers) > 0) continue;
    if (!matchesScenario(numbers, scenario)) continue;
    found.set(key, { numbers, parts: scoreParts(numbers) });
  }
  if (!found.size) throw new Error("조건을 만족하는 조합을 찾지 못했습니다.");
  return [...found.values()].sort((a, b) => b.parts.total - a.parts.total).slice(0, count);
}
function renderCombos(combos, fix, poolSize) {
  const wrap = $("#genResults"); wrap.innerHTML = "";
  const card = el("div", "card");
  card.appendChild(el("h3", null, `${poolSize.toLocaleString()}개 후보 중 상위 ${combos.length}개`));
  combos.forEach(({ numbers, parts }, index) => {
    const row = el("div", "combo");
    row.innerHTML = ballsHtml(numbers, fix, "lg") +
      `<div class="tag"><b>${index + 1}위 · ${parts.total.toFixed(6)}</b><br>페어 ${parts.pair.toFixed(4)} · 주기 ${parts.cycle.toFixed(4)}${parts.penalty ? `<br>홀짝 감점 -${parts.penalty.toFixed(2)}` : ""}</div>`;
    card.appendChild(row);
  });
  card.appendChild(el("p", "muted", "점수는 후보 풀 안의 정렬 기준입니다. 조합별 1등 확률은 모두 1/8,145,060으로 동일합니다."));
  wrap.appendChild(card);
}

/* ── 모델 검증 ────────────────────────── */
function renderBacktest() {
  const validation = D.validation;
  const rows = validation.variants.map((item) => `<tr><td>${item.label}</td><td>${item.averagePercentile}%</td><td>${item.medianPercentile}%</td></tr>`).join("");
  $("#btTable").innerHTML = `<table class="bt-table"><thead><tr><th>모델</th><th>평균 백분위</th><th>중앙 백분위</th></tr></thead><tbody>${rows}</tbody></table>`;
  $("#btCompare").innerHTML = `<b>${validation.rounds} · ${validation.count}회 검증</b><br>현재 모델의 Top 100,000 진입: ${validation.top100kHits}회. 백분위 개선만으로 당첨 예측력이 입증된 것은 아닙니다.`;
  const weights = D.prediction.weights;
  $("#btStability").innerHTML =
    `<table class="bt-table"><tbody><tr><td>페어 평균</td><td>× ${weights.pair}</td></tr><tr><td>출현주기 평균</td><td>× ${weights.cycle}</td></tr><tr><td>희귀 홀짝 연속</td><td>− ${D.prediction.rareAfterRarePenalty}</td></tr></tbody></table>` +
    `<p class="muted" style="margin-top:10px">최근 ${D.prediction.recentWindow}회 페어 빈도와 전체 빈도를 결합합니다. 조건부 패턴과 개별 번호 빈도 점수의 가중치는 0입니다.</p>`;
}

/* ── 인사이트 ─────────────────────────── */
function renderInsights() {
  const labelMap = { good: "검증된 신호", warn: "한계 / 주의", info: "참고" };
  $("#insightCards").innerHTML = D.insights
    .map((it) => `<div class="card insight ${it.tone}"><span class="badge">${labelMap[it.tone] || "참고"}</span><h3>${it.title}</h3><p>${it.body}</p></div>`)
    .join("");
}

/* ── 이벤트 바인딩 (추천/QP/조회) ─────── */
function bindActions() {
  $("#genBtn").addEventListener("click", () => {
    const errBox = $("#genError");
    errBox.hidden = true;
    try {
      const fix = parseNumberList($("#genFix").value);
      const exclude = parseNumberList($("#genExclude").value);
      const poolSize = parseInt($("#genPool").value, 10);
      const combos = generateCombos({
        count: Math.max(1, Math.min(20, parseInt($("#genCount").value, 10) || 6)),
        poolSize,
        scenarioName: $("#genScenario").value,
        noConsec: $("#genNoConsec").checked,
        fix, exclude,
      });
      renderCombos(combos, fix, poolSize);
    } catch (e) { errBox.textContent = e.message; errBox.hidden = false; }
  });
  $("#qpBtn").addEventListener("click", () => {
    const pool = Array.from({ length: 45 }, (_, i) => i + 1);
    const nums = sample(pool, 6).sort((a, b) => a - b);
    $("#qpResult").innerHTML = `<div class="combo" style="margin-top:12px">${ballsHtml(nums, [], "lg")}<div class="tag">완전 무작위</div></div>`;
  });
  $("#lookupBtn").addEventListener("click", () => {
    const r = parseInt($("#lookupRound").value, 10);
    const box = $("#lookupResult");
    const row = D.draws.find((d) => d[0] === r);
    if (!row) { box.innerHTML = `<p class="error">${D.meta.firstRound}~${D.meta.lastRound}회에서 ${r}회를 찾을 수 없습니다.</p>`; return; }
    box.innerHTML =
      `<div class="combo" style="margin-top:8px">${ballsHtml(numsOf(row), [], "lg")}<div class="ball lg bonus">+${bonusOf(row)}</div>` +
      `<div class="tag">${row[1]}<br>합계 <b>${numsOf(row).reduce((sum, number) => sum + number, 0)}</b></div></div>` +
      `<p class="muted">${r}회 · 1등 ${winnersOf(row)}명 · 1인당 ${amtOf(row)}억</p>`;
  });
}

/* ── 초기화 ───────────────────────────── */
function init() {
  $("#updated").textContent = `업데이트: ${D.meta.latest.date} · ${D.meta.latest.round}회`;
  $("#generatorTitle").textContent = `${D.prediction.nextRound}회 후보 생성`;
  D.prediction.scenarios.forEach((scenario) => {
    $("#genScenario").appendChild(new Option(scenario.description, scenario.name));
  });
  bindFilters();
  bindActions();
  renderDashboard();
  renderBacktest();
  renderInsights();
  $("#genBtn").click();
}

if (window.LOTTO) init();
else document.body.insertAdjacentHTML("afterbegin", "<p style='padding:20px'>데이터(data.js)를 불러오지 못했습니다.</p>");

if ("serviceWorker" in navigator && location.protocol !== "file:" && !location.search.includes("nosw")) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}

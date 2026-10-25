const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`);
let state = null, chart = null, evidence = {};

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 7000);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

async function loadHealth() {
  try {
    const h = await api("/api/health");
    $("#health").innerHTML = h.gemini_key
      ? `Gemma 4 · ${esc(h.deep_model)} + ${esc(h.fast_model)}`
      : `<span class="bad">GEMINI_API_KEY not set</span>`;
  } catch { /* server still starting */ }
}

async function loadWatchlist() {
  const ul = $("#watchlist");
  ul.innerHTML = `<li class="muted">Scanning…</li>`;
  try {
    const { items } = await api(`/api/watchlist?window=${$("#window").value}`);
    ul.innerHTML = items.length ? "" : `<li class="muted">No market data.</li>`;
    for (const it of items) {
      const li = document.createElement("li");
      li.className = "item"; li.dataset.t = it.ticker;
      li.innerHTML = `<span>${esc(it.ticker)}</span><span class="${it.change < 0 ? "down" : "up"}">${pct(it.change)}</span>`;
      li.onclick = () => { $("#ticker").value = it.ticker; analyze(it.ticker); };
      ul.appendChild(li);
    }
  } catch (e) { ul.innerHTML = `<li class="muted">Watchlist unavailable: ${esc(e.message)}</li>`; }
}

async function analyze(ticker) {
  ticker = ticker.trim().toUpperCase(); if (!ticker) return;
  document.querySelectorAll(".side li.item").forEach((li) => li.classList.toggle("on", li.dataset.t === ticker));
  const btn = $("#search button"); btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Reading`;
  try {
    state = await api(`/api/analyze/${encodeURIComponent(ticker)}?window=${$("#window").value}&replay=${$("#replay").checked}`);
    render();
  } catch (e) {
    toast(`Could not analyze ${ticker}: ${e.message}. Tick "Replay" to use a saved run.`);
  } finally { btn.disabled = false; btn.textContent = "Analyze"; }
}

function render() {
  const s = state.summary, d = state.detector;
  evidence = Object.fromEntries(state.evidence.map((e) => [e.id, e]));
  evidence.C1 = { id: "C1", text: "The price and volume chart image sent to Gemma." };
  $("#empty").hidden = true; $("#stock").hidden = false;
  $("#s-name").textContent = `${state.name} (${state.ticker})`;
  $("#s-meta").textContent = `${s.window}-session window ${s.start_date} → ${s.end_date} · benchmark ${s.bench_name} ${pct(s.bench_return)}${state.replayed ? " · replayed snapshot" : ""}`;
  $("#s-price").textContent = s.price.toFixed(2);
  $("#s-change").innerHTML = `<span class="${s.window_return < 0 ? "down" : "up"}">${pct(s.window_return)} in ${s.window} sessions</span>`;
  drawChart(); renderHeadlines(); renderDetector(); renderFacts();
  $("#investigation").hidden = !state.investigation;
  if (state.investigation) renderInvestigation();
  renderTrace();
}

function drawChart() {
  const { dates, close } = state.series, w = state.summary.window;
  chart?.destroy(); chart = null;
  if (typeof Chart === "undefined") { // CDN unreachable: show the server-rendered PNG instead
    const c = $("#chart"); const img = document.createElement("img");
    img.id = "chart"; img.style.width = "100%"; img.style.borderRadius = "6px";
    img.src = `data:image/png;base64,${state.chart_png_b64}`; c.replaceWith(img); return;
  }
  if ($("#chart").tagName !== "CANVAS") $("#chart").replaceWith(Object.assign(document.createElement("canvas"), { id: "chart", height: 190 }));
  chart = new Chart($("#chart"), {
    type: "line",
    data: { labels: dates, datasets: [
      { data: close, borderColor: "#4fa3ff", borderWidth: 2, pointRadius: 0, tension: .15 },
      { data: close.map((v, i) => (i >= close.length - 1 - w ? v : null)), borderColor: "#ff5c5c", borderWidth: 3, pointRadius: 0, fill: false },
    ] },
    options: { plugins: { legend: { display: false } }, interaction: { intersect: false, mode: "index" },
      scales: { x: { ticks: { color: "#8a95a5", maxTicksLimit: 6 }, grid: { color: "#1f2731" } },
                y: { ticks: { color: "#8a95a5" }, grid: { color: "#1f2731" } } } },
  });
}

function renderHeadlines() {
  const heads = state.evidence.filter((e) => e.kind === "headline");
  const d = state.detector, n = Math.max(1, d.total);
  $("#h-count").textContent = `${heads.length} · ${d.positive} positive · ${d.negative} negative · ${d.neutral} neutral (labelled by Gemma 4)`;
  $("#sentbar").innerHTML = ["positive", "neutral", "negative"]
    .map((k) => `<div style="width:${(100 * d[k]) / n}%;background:var(--${k.slice(0, 3)})" title="${d[k]} ${k}"></div>`).join("");
  $("#headlines").innerHTML = heads.map((e) => {
    const h = e.data, sen = e.sentiment || {};
    return `<li id="ev-${e.id}"><span class="eid">${e.id}</span>
      <span class="badge ${sen.sentiment}" title="${esc(sen.reason)}">${esc(sen.sentiment)}</span>
      <div class="htitle"><a href="${esc(h.url || "#")}" target="_blank" rel="noopener">${esc(h.title)}</a>
      <div class="hmeta">${esc(h.publisher)} · ${esc((h.published || "").slice(0, 10))}${sen.reason ? " · " + esc(sen.reason) : ""}</div></div></li>`;
  }).join("") || `<li class="muted">No recent headlines found.</li>`;
}

function renderFacts() {
  $("#pricefacts").innerHTML = state.evidence.filter((e) => e.kind === "price")
    .map((e) => `<li id="ev-${e.id}"><span class="eid">${e.id}</span><span>${esc(e.text)}</span></li>`).join("");
}

function renderDetector() {
  const d = state.detector, card = $("#detector-card");
  card.classList.toggle("alarm", d.active);
  const c = d.checks, th = d.thresholds;
  $("#detector").innerHTML = `
    <div class="det-state ${d.active ? "on" : "off"}">${d.active ? "⚠ Contradiction detected" : "No contradiction"}</div>
    <div>${esc(d.message)}</div>
    ${d.active ? `<div class="meter"><div style="width:${d.strength}%"></div></div><div class="muted">Strength ${d.strength}/100</div>` : ""}
    <ul class="checks">
      <li class="${c.price_fell ? "ok" : ""}">Price fell ≥ ${Math.abs(th.drop * 100)}% (${pct(state.summary.window_return)})</li>
      <li class="${c.mostly_positive ? "ok" : ""}">≥ ${th.positive_share * 100}% of headlines positive (${Math.round(d.positive_share * 100)}%)</li>
      <li class="${c.enough_headlines ? "ok" : ""}">At least ${th.min_headlines} headlines (${d.total})</li>
    </ul>
    <button class="btn investigate" id="investigate" ${d.active ? "" : "disabled"} title="${d.active ? "" : "Investigate unlocks when a contradiction is detected"}">Investigate</button>`;
  $("#investigate").onclick = investigate;
}

async function investigate() {
  const btn = $("#investigate");
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Gemma 4 is reasoning over the evidence and chart…`;
  try {
    state = await api(`/api/investigate/${encodeURIComponent(state.ticker)}?replay=${$("#replay").checked}`, { method: "POST" });
    $("#investigation").hidden = false;
    renderInvestigation(); renderTrace();
    $("#investigation").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { toast(`Investigation failed: ${e.message}`); }
  finally { btn.disabled = false; btn.textContent = "Investigate again"; }
}

function cite(id, valid) {
  return `<span class="cite ${valid ? "" : "bad"}" data-ev="${esc(id)}">${esc(id)}</span>`;
}

function renderInvestigation() {
  const inv = state.investigation, hs = inv.hypotheses;
  $("#seen-chart").src = `data:image/png;base64,${state.chart_png_b64}`;
  $("#chart-reading").textContent = inv.chart_reading || "—";
  const counts = hs.reduce((a, h) => ((a[h.status] = (a[h.status] || 0) + 1), a), {});
  $("#inv-summary").textContent = `${hs.length} competing explanations · ${counts.supported || 0} supported · ${counts.partial || 0} partial · ${counts.insufficient || 0} flagged${inv.judge_ok ? "" : " · judge unavailable, deterministic checks only"}`;
  const label = { supported: "Supported", partial: "Partially supported", insufficient: "Insufficient support" };
  const color = { supported: "var(--pos)", partial: "var(--warn)", insufficient: "var(--neg)" };
  $("#hypotheses").innerHTML = hs.map((h, i) => `
    <article class="hyp ${h.status}" style="animation-delay:${i * 120}ms">
      <span class="status ${h.status}">${label[h.status]}</span>
      <h5>${esc(h.id)}. ${esc(h.title)}</h5>
      <p class="mech">${esc(h.mechanism)}</p>
      ${h.flagged ? `<div class="flagbox"><strong>Evidence checker flag:</strong> ${esc(h.flag_reasons.join("; "))}.</div>` : ""}
      <div class="score">Support score ${h.support_score.toFixed(2)} · Gemma's own confidence ${Number(h.model_confidence ?? 0).toFixed(2)} · backed by ${h.supporting_ids.length ? h.supporting_ids.map((x) => cite(x, true)).join("") : "nothing"}
        <div class="scorebar"><div style="width:${Math.max(0, h.support_score) * 100}%;background:${color[h.status]}"></div></div></div>
      <ul class="claims">${h.claims.map((c) => `
        <li class="claim"><span class="v ${c.verdict}">${c.verdict.replace("_", " ")}</span>${esc(c.text)}
          ${c.cited.map((id) => cite(id, c.valid_ids.includes(id))).join("")}
          ${c.quote ? `<span class="quote">“${esc(c.quote)}”</span>` : ""}
          ${c.reasons.length ? `<ul class="reasons">${c.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
        </li>`).join("")}</ul>
      ${h.would_confirm || h.would_refute ? `<div class="tests">${h.would_confirm ? `<b>Would confirm:</b> ${esc(h.would_confirm)}<br>` : ""}${h.would_refute ? `<b>Would refute:</b> ${esc(h.would_refute)}` : ""}</div>` : ""}
    </article>`).join("");
}

function renderTrace() {
  const t = state.investigation?.trace || state.trace || [];
  $("#trace").innerHTML = `<tr><th>Step</th><th>Model</th><th>Thinking</th><th>Image</th><th>Latency</th><th>Tokens in/out</th></tr>` +
    t.map((r) => `<tr><td>${esc(r.purpose)}</td><td>${esc(r.model || "—")}</td><td>${esc(r.thinking_level || "—")}</td>
      <td>${r.image_attached ? "chart.png" : "—"}</td><td>${r.latency_s ?? "—"}s</td>
      <td>${r.error ? `<span class="down">${esc(r.error)}</span>` : `${r.input_tokens ?? "?"} / ${r.output_tokens ?? "?"}`}</td></tr>`).join("");
}

// hover a citation chip -> show the evidence and highlight its row
document.addEventListener("mouseover", (e) => {
  const c = e.target.closest(".cite"); const tip = $("#tooltip");
  document.querySelectorAll("li.hl").forEach((li) => li.classList.remove("hl"));
  if (!c) { tip.hidden = true; return; }
  const ev = evidence[c.dataset.ev];
  tip.innerHTML = ev ? `<b>${esc(ev.id)}</b> ${esc(ev.text)}` : `<b>${esc(c.dataset.ev)}</b> <span class="down">does not exist in the evidence set</span>`;
  const r = c.getBoundingClientRect();
  tip.style.left = `${Math.min(r.left, innerWidth - 400)}px`; tip.style.top = `${r.bottom + 6}px`; tip.hidden = false;
  document.getElementById(`ev-${c.dataset.ev}`)?.classList.add("hl");
});

$("#search").onsubmit = (e) => { e.preventDefault(); analyze($("#ticker").value); };
$("#refresh").onclick = loadWatchlist;
$("#window").onchange = loadWatchlist;
loadHealth(); loadWatchlist();

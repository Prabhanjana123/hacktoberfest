const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`);
let state = null, chart = null, evidence = {};
let qpState = null, qpChart = null;

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

// ---------------------------------------------------------------------------
// QuantProof Lab — constrained strategy research
// ---------------------------------------------------------------------------
function setMode(mode) {
  const quant = mode === "quantproof";
  const chat = mode === "chat";
  const stockpulse = !quant && !chat;
  document.body.classList.toggle("quantproof-mode", quant);
  document.body.classList.toggle("chat-mode", chat);
  $("#tab-stockpulse").classList.toggle("active", stockpulse);
  $("#tab-quantproof").classList.toggle("active", quant);
  $("#tab-chat").classList.toggle("active", chat);
  $("#search").hidden = !stockpulse;
  $("#empty").hidden = !stockpulse || Boolean(state);
  $("#stock").hidden = !stockpulse || !state;
  $("#quantproof-view").hidden = !quant;
  $("#chat-view").hidden = !chat;
}

function qpPct(value) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(2)}%`;
}
function qpNum(value, digits = 2) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
}
function qpMetric(value) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(2);
}

function qpMetricTable(result) {
  const m = result.metrics;
  const rows = [
    ["Total return", "total_return", "pct"],
    ["Annualised return", "annualized_return", "pct"],
    ["Annualised volatility", "annualized_volatility", "pct"],
    ["Sharpe (risk-free = 0)", "sharpe", "num"],
    ["Maximum drawdown", "max_drawdown", "pct"],
    ["Market exposure", "exposure_pct", "exposure"],
    ["Observations", "observations", "integer"],
    ["Position changes", "position_changes", "integer"],
  ];
  const cells = (obj, key, type) => {
    const v = obj?.[key];
    if (type === "pct") return qpPct(v);
    if (type === "exposure") return v == null ? "—" : `${qpNum(v, 1)}%`;
    if (type === "integer") return v == null ? "—" : String(v);
    return qpNum(v, 2);
  };
  return `<thead><tr><th>Metric</th><th>In-sample</th><th>Out-of-sample</th><th>Buy & hold OOS</th><th>Strategy OOS at 2× costs</th></tr></thead><tbody>${rows.map(([label, key, type]) =>
    `<tr><td>${esc(label)}</td><td>${cells(m.strategy.in_sample, key, type)}</td><td>${cells(m.strategy.out_of_sample, key, type)}</td><td>${cells(m.buy_hold.out_of_sample, key, type)}</td><td>${cells(m.double_costs_out_of_sample, key, type)}</td></tr>`
  ).join("")}</tbody>`;
}

function renderQpChart(curve) {
  if (qpChart) { qpChart.destroy(); qpChart = null; }
  const canvas = $("#qp-equity-chart");
  if (!canvas || typeof Chart === "undefined") {
    if (canvas) canvas.insertAdjacentHTML("afterend", `<p class="muted small">Chart library unavailable. See the metrics and sensitivity table below.</p>`);
    return;
  }
  qpChart = new Chart(canvas, {
    type: "line",
    data: { labels: curve.map(x => (x.date || "").slice(0, 10)), datasets: [
      { label: "Strategy after costs", data: curve.map(x => x.strategy_equity), borderColor: "#4fa3ff", borderWidth: 2, pointRadius: 0, tension: .08 },
      { label: "Buy and hold", data: curve.map(x => x.buy_hold_equity), borderColor: "#ffb547", borderWidth: 1.5, pointRadius: 0, tension: .08 },
    ] },
    options: { responsive: true, interaction: { intersect: false, mode: "index" },
      plugins: { legend: { labels: { color: "#e6e9ee" } } },
      scales: { x: { ticks: { color: "#8a95a5", maxTicksLimit: 6 }, grid: { color: "#1f2731" } },
        y: { ticks: { color: "#8a95a5" }, grid: { color: "#1f2731" } } } },
  });
}

function renderQuantProof(result) {
  qpState = result;
  $("#qp-results").hidden = false;
  $("#qp-data-meta").textContent = result.data_quality
    ? `${result.data_quality.rows_used} bars · ${result.data_quality.start_date.slice(0, 10)} → ${result.data_quality.end_date.slice(0, 10)} · ${result.filename}`
    : result.filename || "";
  $("#qp-unsupported").hidden = Boolean(result.supported);
  if (!result.supported) {
    if (qpChart) { qpChart.destroy(); qpChart = null; }
    $("#qp-unsupported").textContent = `This hypothesis could not be mapped to a supported strategy: ${result.reason}. Supported templates: ${(result.supported_templates || []).join(", ")}.`;
    $("#qp-summary").innerHTML = `<strong>Backtest not run.</strong> QuantProof will not force an unsupported hypothesis into an unrelated strategy template.`;
    $("#qp-spec").innerHTML = "—";
    $("#qp-robustness").innerHTML = "—";
    $("#qp-metrics").innerHTML = "";
    $("#qp-sensitivity").innerHTML = "";
    $("#qp-regimes").innerHTML = "";
    $("#qp-regime-note").textContent = "";
    $("#qp-report").innerHTML = "";
    return;
  }
  const spec = result.strategy_spec, m = result.metrics, r = result.robustness, report = result.report || {};
  const perf = m.strategy;
  const perfStatus = r.performance.status.replaceAll("_", " ");
  const aiWarnings = Array.isArray(result.ai_status?.warnings) ? result.ai_status.warnings : [];
  $("#qp-summary").innerHTML = `
    <div class="qp-result-top"><div><div class="eyebrow">RESEARCH STATUS</div><h3>${esc(perfStatus)}</h3></div><span class="qp-pill">${esc(spec.strategy_type)}</span></div>
    <p>${esc(r.performance.message)}</p>
    <p class="muted small">Train/test boundary: ${esc(m.split_date.slice(0, 10))}. Out-of-sample interval: ${esc(m.test_start.slice(0, 10))} to ${esc(m.test_end.slice(0, 10))}. Performance on unseen bars is informative but not proof of future performance.</p>
    ${esc(report.headline || "")}
    ${aiWarnings.map(w => `<p class="qp-warning small">AI availability note: ${esc(w)}</p>`).join("")}`;
  $("#qp-spec").innerHTML = `
    <h3>${esc(spec.strategy_title)}</h3>
    <p>${esc(spec.hypothesis_interpretation || result.hypothesis)}</p>
    <dl class="qp-dl"><dt>Template</dt><dd>${esc(spec.strategy_type)}</dd><dt>Lookback</dt><dd>${esc(spec.lookback)} bars</dd><dt>Threshold</dt><dd>${esc(spec.threshold_bps)} bps</dd><dt>Cost assumption</dt><dd>${esc(m.cost_bps_one_way)} bps per position change</dd><dt>Frequency</dt><dd>${esc(result.data_quality.frequency_estimate)}</dd></dl>
    <h5>Assumptions</h5><ul>${(spec.assumptions || []).map(x => `<li>${esc(x)}</li>`).join("") || "<li>See the fixed strategy definition.</li>"}</ul>
    <h5>Model limitations</h5><ul>${(spec.limitations || []).map(x => `<li>${esc(x)}</li>`).join("") || "<li>Long-only; simplified cost model.</li>"}</ul>`;
  $("#qp-robustness").innerHTML = [
    ["Out-of-sample", r.performance], ["Transaction costs", r.transaction_costs], ["Parameter sensitivity", r.parameters]
  ].map(([label, item]) => `<div class="qp-diagnostic"><div><strong>${esc(label)}</strong><span class="qp-pill ${item.status.includes("deterioration") || item.status.includes("loss") || item.status.includes("sensitive") ? "warn" : ""}">${esc(item.status.replaceAll("_", " "))}</span></div><p>${esc(item.message)}</p></div>`).join("") + `<p class="muted small">${esc(r.warning)}</p>`;
  $("#qp-metrics").innerHTML = qpMetricTable(result);
  const regimes = m.market_regime_analysis || {};
  if (regimes.groups && regimes.groups.length) {
    $("#qp-regimes").innerHTML = `<thead><tr><th>OOS volatility regime</th><th>Bars</th><th>Strategy mean (bps/bar)</th><th>Buy & hold mean (bps/bar)</th><th>Positive strategy bars</th><th>Exposure</th></tr></thead><tbody>${regimes.groups.map(x => `<tr><td>${esc(x.regime)}</td><td>${x.oos_bars}</td><td>${qpNum(x.mean_strategy_return_bps_per_bar, 2)}</td><td>${qpNum(x.mean_buy_hold_return_bps_per_bar, 2)}</td><td>${x.positive_strategy_bar_pct == null ? "—" : `${qpNum(x.positive_strategy_bar_pct, 1)}%`}</td><td>${x.market_exposure_pct == null ? "—" : `${qpNum(x.market_exposure_pct, 1)}%`}</td></tr>`).join("")}</tbody>`;
    $("#qp-regime-note").textContent = `${regimes.method || ""} ${regimes.warning || regimes.reason || ""}`;
  } else {
    $("#qp-regimes").innerHTML = "";
    $("#qp-regime-note").textContent = regimes.reason || "Not enough in-sample data to define volatility regimes.";
  }
  $("#qp-sensitivity").innerHTML = `<thead><tr><th>Lookback</th><th>Threshold (bps)</th><th>OOS return</th><th>OOS Sharpe</th><th>OOS max drawdown</th></tr></thead><tbody>${result.parameter_sensitivity.map(x => `<tr class="${x.is_reference ? "qp-reference" : ""}"><td>${x.lookback}</td><td>${qpNum(x.threshold_bps, 1)}</td><td>${qpPct(x.oos_total_return)}</td><td>${qpNum(x.oos_sharpe)}</td><td>${qpPct(x.oos_max_drawdown)}</td></tr>`).join("")}</tbody>`;
  $("#qp-report").innerHTML = `<p>${esc(report.assessment || "")}</p>
    ${Array.isArray(report.key_findings) && report.key_findings.length ? `<h5>Key findings</h5><ul>${report.key_findings.map(x => `<li><strong>${esc(x.title || "Finding")}:</strong> ${esc(x.detail || "")}</li>`).join("")}</ul>` : ""}
    ${Array.isArray(report.caveats) && report.caveats.length ? `<h5>Caveats</h5><ul>${report.caveats.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    ${Array.isArray(report.next_experiments) && report.next_experiments.length ? `<h5>Next experiments</h5><ul>${report.next_experiments.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    <p class="muted small">${esc(report.not_financial_advice ? "Research analysis only; not financial advice." : "Research prototype; do not use as a trading recommendation.")}</p>`;
  const trace = result.trace || [];
  $("#qp-trace").innerHTML = `<thead><tr><th>Purpose</th><th>Model</th><th>Thinking</th><th>Chart image</th><th>Latency</th></tr></thead><tbody>${trace.map(t => `<tr><td>${esc(t.purpose || "—")}</td><td>${esc(t.model || "—")}</td><td>${esc(t.thinking_level || "—")}</td><td>${t.image_attached ? "Yes" : "No"}</td><td>${t.error ? esc(t.error) : `${esc(t.latency_s ?? "—")}s`}</td></tr>`).join("")}</tbody>`;
  renderQpChart(result.equity_curve || []);
}

async function runQuantProof() {
  const btn = $("#qp-run"), status = $("#qp-status"), file = $("#qp-file").files[0];
  const hypothesis = $("#qp-hypothesis").value.trim();
  if (!hypothesis || hypothesis.length < 10) { toast("Describe one testable trading hypothesis (at least 10 characters)."); return; }
  if (!file) { toast("Choose an OHLCV CSV file first."); return; }
  if (!file.name.toLowerCase().endsWith(".csv")) { toast("Please choose a .csv file."); return; }
  if (file.size > 2_500_000) { toast("CSV is too large for this demo (maximum 2.5 MB). Try a smaller sample."); return; }
  const cost = Number($("#qp-cost").value);
  if (!Number.isFinite(cost) || cost < 0 || cost > 100) { toast("Transaction cost must be between 0 and 100 bps."); return; }
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Gemma is mapping the hypothesis; Python is testing it…`;
  status.textContent = "Parsing data, mapping a constrained strategy, running cost/OOS/parameter tests, then reviewing the results.";
  $("#qp-results").hidden = true;
  try {
    const csvText = await file.text();
    const result = await api("/api/quantproof/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hypothesis, csv_text: csvText, filename: file.name,
        transaction_cost_bps: cost, train_fraction: Number($("#qp-split").value),
      }),
    });
    renderQuantProof(result);
    if (!result.supported) {
      status.textContent = "Hypothesis mapping stopped safely; revise the hypothesis or choose a supported strategy family.";
    } else {
      const mappingFallback = result.ai_status?.mapping_source === "explicit_hypothesis_fallback";
      const reportFallback = result.ai_status?.report_source === "deterministic_fallback";
      const textOnlyReview = result.ai_status?.report_source === "gemma_text_only_fallback";
      if (mappingFallback && reportFallback) {
        status.textContent = "Backtest complete using your explicit strategy settings. Gemma API was unavailable for mapping and review; the report is a deterministic fallback.";
      } else if (mappingFallback) {
        status.textContent = "Backtest complete using your explicit strategy settings. Gemma mapping failed; check the disclosed fallback notice.";
      } else if (reportFallback) {
        status.textContent = "Backtest complete. Both Gemma review attempts failed; deterministic results and fallback commentary are shown.";
      } else if (textOnlyReview) {
        status.textContent = "Research complete. Gemma's multimodal review failed, so a text-only Gemma 4 review was used.";
      } else {
        status.textContent = "Research complete. Numbers are calculated in Python; Gemma provides the interpretation.";
      }
    }
  } catch (e) {
    status.textContent = `Research failed: ${e.message}`;
    toast(`QuantProof failed: ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = "Map hypothesis & run research";
  }
}


// ---------------------------------------------------------------------------
// Gemma Chat — conversation is held in browser memory, not server-side storage
// ---------------------------------------------------------------------------
let chatHistory = [];

function chatMarkdown(text) {
  let value = String(text ?? "");
  const blocks = [];
  value = value.replace(/```(?:[a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)```/g, (_, code) => {
    const token = `CHATCODE${blocks.length}TOKEN`;
    blocks.push(`<pre><code>${esc(code)}</code></pre>`);
    return token;
  });
  value = esc(value);
  value = value.replace(/^###\s+(.+)$/gm, "<h4>$1</h4>");
  value = value.replace(/^##\s+(.+)$/gm, "<h3>$1</h3>");
  value = value.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  value = value.replace(/^[-*]\s+(.+)$/gm, "• $1");
  value = value.replace(/\n/g, "<br>");
  blocks.forEach((html, i) => { value = value.replace(`CHATCODE${i}TOKEN`, html); });
  return value;
}

function renderChatMessages() {
  const host = $("#chat-messages");
  if (!host) return;
  host.replaceChildren();
  const messages = chatHistory.length ? chatHistory : [{
    role: "assistant",
    content: "Hi! I'm StockPulse Chat, powered by Gemma 4. I can help you reason through quantitative research, backtesting, market evidence, statistics, and the code in this project.\n\nAsk a question below, or choose a suggested prompt on the left."
  }];
  for (const item of messages) {
    const row = document.createElement("article");
    row.className = `chat-message ${item.role === "user" ? "user" : "assistant"}${item.error ? " error" : ""}`;
    const header = document.createElement("div");
    header.className = "chat-message-meta";
    const label = document.createElement("span");
    label.textContent = item.role === "user" ? "You" : "StockPulse · Gemma 4";
    header.appendChild(label);
    if (item.model) {
      const model = document.createElement("span");
      model.textContent = item.model;
      header.appendChild(model);
    }
    row.appendChild(header);
    const body = document.createElement("div");
    body.className = "chat-message-body";
    body.innerHTML = chatMarkdown(item.content);
    row.appendChild(body);
    if (item.warning) {
      const note = document.createElement("p");
      note.className = "muted small chat-warning";
      note.textContent = item.warning;
      row.appendChild(note);
    }
    host.appendChild(row);
  }
  host.scrollTop = host.scrollHeight;
}

function buildChatContext() {
  const chunks = [];
  if (state) {
    const s = state.summary || {};
    chunks.push("CURRENT STOCKPULSE ANALYSIS");
    chunks.push(JSON.stringify({
      ticker: state.ticker, company: state.name, summary: s,
      contradiction_detector: state.detector,
      headlines: (state.evidence || []).filter(e => String(e.id || "").startsWith("H")).slice(0, 10).map(e => ({
        id: e.id, text: e.text, sentiment: e.sentiment?.sentiment, publisher: e.publisher, published: e.published, url: e.url
      })),
      existing_investigation: state.investigation ? {
        summary: state.investigation.summary,
        hypotheses: (state.investigation.hypotheses || []).slice(0, 4).map(h => ({
          title: h.title, status: h.status, support_score: h.support_score, claims: (h.claims || []).slice(0, 5).map(c => ({text:c.text, verdict:c.verdict, reasons:c.reasons}))
        }))
      } : null
    }));
  }
  if (qpState && qpState.supported) {
    const m = qpState.metrics || {};
    chunks.push("CURRENT QUANTPROOF RESEARCH");
    chunks.push(JSON.stringify({
      hypothesis: qpState.hypothesis,
      filename: qpState.filename,
      data_quality: qpState.data_quality,
      strategy_spec: qpState.strategy_spec,
      metrics: {
        split_date: m.split_date, test_start: m.test_start, test_end: m.test_end,
        strategy: m.strategy, buy_hold: m.buy_hold, double_costs_out_of_sample: m.double_costs_out_of_sample
      },
      robustness: qpState.robustness,
      report: qpState.report
    }));
  }
  return chunks.join("\n\n").slice(0, 5800);
}

async function sendChatMessage(event) {
  if (event) event.preventDefault();
  const input = $("#chat-input"), btn = $("#chat-send"), pending = $("#chat-pending");
  const content = input.value.trim();
  if (!content) return;
  if (content.length > 4000) { toast("Please keep each message under 4,000 characters."); return; }
  chatHistory.push({ role: "user", content });
  input.value = "";
  renderChatMessages();
  btn.disabled = true;
  pending.hidden = false;
  $("#chat-status").textContent = "Sending message to Gemma 4…";
  try {
    const context = $("#chat-use-context").checked ? buildChatContext() : "";
    const messages = chatHistory.filter(m => !m.error).slice(-12).map(m => ({ role: m.role, content: m.content }));
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, context, depth: $("#chat-depth").value })
    });
    chatHistory.push({
      role: "assistant", content: result.reply, model: result.model,
      warning: result.warning || ""
    });
    $("#chat-status").textContent = `Answered by ${result.model || "Gemma 4"}${result.used_fallback ? " · alternate model fallback" : ""}.`;
  } catch (e) {
    chatHistory.push({ role: "assistant", error: true, content: `I couldn't complete that request. ${e.message}\n\nCheck that GEMINI_API_KEY is set, and inspect the server terminal for details. Your earlier messages remain in this browser tab.` });
    $("#chat-status").textContent = "Request failed · see the message below.";
  } finally {
    pending.hidden = true;
    btn.disabled = false;
    renderChatMessages();
    input.focus();
  }
}

function clearChat() {
  chatHistory = [];
  $("#chat-status").textContent = "Conversation cleared. Chat messages are not saved on the server.";
  renderChatMessages();
  $("#chat-input").focus();
}

$("#tab-stockpulse").onclick = () => setMode("stockpulse");
$("#tab-quantproof").onclick = () => setMode("quantproof");
$("#tab-chat").onclick = () => setMode("chat");
$("#chat-form").onsubmit = sendChatMessage;
$("#chat-clear").onclick = clearChat;
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
});
document.querySelectorAll("[data-chat-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    $("#chat-input").value = button.dataset.chatPrompt || "";
    $("#chat-input").focus();
  });
});
renderChatMessages();
$("#qp-run").onclick = runQuantProof;
$("#search").onsubmit = (e) => { e.preventDefault(); analyze($("#ticker").value); };
$("#refresh").onclick = loadWatchlist;
$("#window").onchange = loadWatchlist;
setMode("stockpulse");
loadHealth(); loadWatchlist();

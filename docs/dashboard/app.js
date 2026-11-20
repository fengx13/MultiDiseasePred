// UI for the static MultiDiseasePred dashboard (port of the Streamlit app).
"use strict";

const EXAMPLE = {
  age: 67, gender: 0, triage_acuity: 2, triage_heartrate: 122, triage_resprate: 24,
  triage_o2sat: 90, triage_sbp: 92, triage_dbp: 55, n_ed_30d: 1, n_ed_365d: 3,
  chiefcom_abdominal_pain: 1, cci_Pulmonary: 0, cci_Renal: 1, eci_FluidsLytes: 1, eci_HTN1: 1,
};
const VITAL_RANGES = {
  triage_temperature: [36.1, 37.2], triage_heartrate: [60, 100], triage_resprate: [12, 20],
  triage_o2sat: [95, 100], triage_sbp: [90, 140], triage_dbp: [60, 90],
};
const LIFT_MAX = 3.0, LIFT_MOD = 0.7, LIFT_HIGH = 1.5;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let M, FM, tab = "summary";

function defaultFor(c) { return c in EXAMPLE ? EXAMPLE[c] : FM[c].default; }

function widget(c) {
  const m = FM[c], d = defaultFor(c), id = "f_" + c;
  let ctl;
  if (m.type === "numeric") {
    ctl = `<input type="number" id="${id}" value="${d}" min="${m.min}" max="${m.max}" step="${m.step}">`;
  } else {
    const ch = m.type === "binary" ? [["No", 0], ["Yes", 1]] : m.choices;
    ctl = `<div class="seg">${ch.map(([l, v]) =>
      `<label><input type="radio" name="${id}" value="${v}" ${v === d ? "checked" : ""}><span>${esc(l)}</span></label>`).join("")}</div>`;
  }
  return `<div class="row"><label class="lab" for="${id}">${esc(m.label)}</label>${ctl}</div>`;
}

function readInputs() {
  const u = {};
  for (const c of M.meta.feat_cols) {
    const m = FM[c];
    if (m.type === "numeric") {
      const el = $("f_" + c), v = parseFloat(el.value);
      u[c] = Number.isFinite(v) ? Math.min(Math.max(v, m.min), m.max) : null;
    } else {
      const el = document.querySelector(`input[name="f_${c}"]:checked`);
      u[c] = el ? +el.value : null;
    }
  }
  return u;
}

function band(l) {
  if (l >= 1.5) return ["High Risk", "#d64545"];
  if (l >= 0.7) return ["Moderate Risk", "#e3a008"];
  return ["Low Risk", "#3f9d5a"];
}

// Semicircular gauge on the log-lift axis, probability in the centre.
function gauge(p, lift) {
  const [label, color] = band(lift);
  const cx = 150, cy = 148, r = 100, sw = 26;
  const ang = v => Math.PI * (1 - Math.max(0, Math.min(v, LIFT_MAX)) / LIFT_MAX);
  const pt = (a, rr) => [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
  const arc = (v0, v1, rr, col, w) => {
    const [x0, y0] = pt(ang(v0), rr), [x1, y1] = pt(ang(v1), rr);
    return `<path d="M${x0} ${y0} A${rr} ${rr} 0 0 1 ${x1} ${y1}" fill="none" stroke="${col}" stroke-width="${w}"/>`;
  };
  const ticks = [[0, "0"], [LIFT_MOD, "0.7"], [LIFT_HIGH, "1.5"], [LIFT_MAX, "3+"]].map(([v, t]) => {
    const [x, y] = pt(ang(v), r + sw / 2 + 12);
    return `<text x="${x}" y="${y}" font-size="11" text-anchor="middle" fill="#9aa6b5" style="fill:#9aa6b5">${t}</text>`;
  }).join("");
  const lc = Math.max(0, Math.min(lift, LIFT_MAX));
  return `<svg viewBox="0 0 300 214" width="100%" style="max-width:340px;display:block;margin:0 auto" role="img" aria-label="${label}, probability ${(p * 100).toFixed(1)}%">
    ${arc(0, LIFT_MOD, r, "#dff0e2", sw)}${arc(LIFT_MOD, LIFT_HIGH, r, "#fdf0cf", sw)}${arc(LIFT_HIGH, LIFT_MAX, r, "#f7dcdc", sw)}
    ${lc > 0.001 ? arc(0, lc, r, color, sw * 0.32) : ""}${ticks}
    <text x="${cx}" y="${cy - 14}" font-size="30" font-weight="700" text-anchor="middle" style="fill:${color}">${(p * 100).toFixed(1)}%</text>
    <text x="${cx}" y="${cy + 4}" font-size="11" text-anchor="middle" style="fill:#8a97a6">probability</text>
    <text x="${cx}" y="${cy + 52}" font-size="16" font-weight="700" text-anchor="middle" style="fill:${color}">${label}  ·  lift ${lift >= 0 ? "+" : ""}${lift.toFixed(1)}</text>
  </svg>`;
}

function liftColor(l) { return l >= 1.5 ? "#d64545" : l >= 0.7 ? "#e9a23b" : l >= 0.2 ? "#c9cf52" : "#69b06b"; }

// Horizontal bars: length = log-lift, label = probability and lift.
function topRisks(idx, probs, lift) {
  const names = idx.map(i => M.meta.outcome_display[M.meta.outcomes[i]]);
  const widths = idx.map(i => Math.max(lift[i], 0));
  const xmax = Math.max(...widths, 1) * 1.45;
  const L = 150, W = 300, rowH = 34, H = 26 + rowH * idx.length;
  const sx = v => L + (v + 0.15) / (xmax + 0.15) * W;
  let s = `<svg viewBox="0 0 ${L + W + 20} ${H}" width="100%" role="img" aria-label="Top predicted risks">`;
  s += `<line x1="${sx(0)}" x2="${sx(0)}" y1="16" y2="${H}" stroke="#8a97a6" stroke-dasharray="4 3" stroke-width="1.4"/>`;
  s += `<text x="${sx(0)}" y="11" font-size="10" text-anchor="middle" style="fill:#8a97a6">baseline (prevalence)</text>`;
  idx.forEach((i, k) => {
    const y = 22 + k * rowH, w = sx(widths[k]) - sx(0);
    s += `<text x="${L - 8}" y="${y + 15}" font-size="13" text-anchor="end">${esc(names[k])}</text>`;
    s += `<rect x="${sx(0)}" y="${y + 3}" width="${Math.max(w, 0)}" height="${rowH - 10}" rx="3" fill="${liftColor(lift[i])}"/>`;
    s += `<text x="${sx(0) + Math.max(w, 0) + 6}" y="${y + 15}" font-size="11">${(probs[i] * 100).toFixed(0)}%  ·  lift ${lift[i] >= 0 ? "+" : ""}${lift[i].toFixed(1)}</text>`;
  });
  return s + "</svg>";
}

function humanValue(v, c) {
  const m = FM[c];
  if (m.type === "binary") return v >= 0.5 ? "Yes" : "No";
  if (m.type === "categorical") { const f = m.choices.find(([, code]) => Math.abs(code - v) < 1e-6); if (f) return f[0]; }
  return Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(2);
}

function attrChart(rows) {
  const mx = Math.max(...rows.map(r => Math.abs(r.a)), 1e-9);
  const L = 230, W = 300, rowH = 30, H = 8 + rowH * rows.length, mid = L + W / 2;
  let s = `<svg viewBox="0 0 ${L + W} ${H}" width="100%" role="img" aria-label="Feature attributions">`;
  s += `<line x1="${mid}" x2="${mid}" y1="0" y2="${H}" stroke="#9aa6b5" stroke-width="1"/>`;
  rows.forEach((r, k) => {
    const y = 4 + k * rowH, w = Math.abs(r.a) / mx * (W / 2 - 46), x = r.a >= 0 ? mid : mid - w;
    s += `<text x="${L - 8}" y="${y + 17}" font-size="12" text-anchor="end">${esc(r.label)}</text>`;
    s += `<rect x="${x}" y="${y + 5}" width="${w}" height="${rowH - 12}" rx="3" fill="${r.a >= 0 ? "#d64545" : "#2f6fb0"}"/>`;
    s += `<text x="${r.a >= 0 ? mid + w + 4 : mid - w - 4}" y="${y + 17}" font-size="11" text-anchor="${r.a >= 0 ? "start" : "end"}">${r.a >= 0 ? "+" : ""}${r.a.toFixed(2)}</text>`;
  });
  return s + "</svg>";
}

function render() {
  const meta = M.meta, inputs = readInputs();
  const thr = +$("thr").value; $("thv").textContent = thr.toFixed(2);
  const sel = meta.outcomes.map((o, i) => i).filter(i => $("o_" + i).checked);
  if (!sel.length) { $("tabs").innerHTML = ""; $("view").innerHTML = `<p class="warn">Select at least one outcome in the sidebar.</p>`; return; }
  const r = predict(M, inputs), probs = r.cal;
  const lift = probs.map((p, i) => Math.log(Math.min(Math.max(p, 1e-6), 1 - 1e-6) / Math.max(meta.baseline_prevalence[meta.outcomes[i]] ?? 0.1, 1e-6)));
  const ranked = sel.slice().sort((a, b) => lift[b] - lift[a]);
  const RAW = Object.fromEntries(meta.feat_cols.map((c, i) => [c, r.raw[i]]));
  if (tab !== "summary" && !ranked.includes(+tab)) tab = "summary";

  $("tabs").innerHTML = [["summary", "Summary"], ...ranked.map(i => [String(i), meta.outcome_display[meta.outcomes[i]]])]
    .map(([k, l]) => `<button data-k="${k}" class="${String(tab) === k ? "on" : ""}">${esc(l)}</button>`).join("");
  $("tabs").querySelectorAll("button").forEach(b => b.onclick = () => { tab = b.dataset.k === "summary" ? "summary" : +b.dataset.k; render(); });

  if (tab === "summary") {
    const top = ranked[0];
    const vrow = (c, unit) => {
      const v = RAW[c], [lo, hi] = VITAL_RANGES[c];
      const cls = v > hi ? "up" : v < lo ? "down" : "flat", sym = { up: "&#9650;", down: "&#9660;", flat: "&bull;" }[cls];
      return `<div class="vital"><b>${esc(FM[c].label)}</b> ${Math.round(v)} ${unit} <span class="${cls}">${sym}</span></div>`;
    };
    $("view").innerHTML = `<div class="grid2">
      <div class="card"><h3>Triage Assessment</h3>
        <div style="text-align:center;font-size:.98rem">Highest risk: <b>${esc(meta.outcome_display[meta.outcomes[top]])}</b></div>
        ${gauge(probs[top], lift[top])}
        <div class="vital"><b>Age</b> ${Math.round(RAW.age)} years</div>
        <div class="vital"><b>Triage acuity</b> ESI ${Math.round(RAW.triage_acuity)}</div>
        ${vrow("triage_heartrate", "bpm")}${vrow("triage_resprate", "/min")}${vrow("triage_o2sat", "%")}${vrow("triage_sbp", "mmHg")}${vrow("triage_dbp", "mmHg")}
      </div>
      <div><div class="card"><h3>Top Predicted Risks</h3>${topRisks(ranked, probs, lift)}
        <p class="cap">Ranked by risk relative to each condition's baseline prevalence (log-lift), as in Figure 6. Labels show calibrated probability.</p></div>
        <details><summary>+ More conditions (model expansion, coming soon)</summary>
          ${meta.reserved.map(s => `<span class="chip">${esc(s.replace("outcome_", "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()))}</span>`).join("")}
          <p class="cap">Reserved for future model expansion.</p></details></div>
    </div>
    <div class="pid"><div><div class="l">Patient</div><div class="v">Composite case</div></div><div><div class="l">Patient ID</div><div class="v">-</div></div><div><div class="l">Visit ID</div><div class="v">-</div></div></div>
    <p class="cap">Identifiers are fictitious placeholders and do not correspond to any real individual.</p>`;
  } else {
    const t = +tab, name = meta.outcome_display[meta.outcomes[t]], p = probs[t], a = r.attr[t];
    const rows = a.map((v, i) => i).sort((i, j) => Math.abs(a[j]) - Math.abs(a[i])).slice(0, 5)
      .map(i => ({ label: FM[meta.feat_cols[i]].label, value: humanValue(r.raw[i], meta.feat_cols[i]), a: a[i] }));
    $("view").innerHTML = `<div class="grid2 w">
      <div class="card">${gauge(p, lift[t])}<div class="metric"><div class="l">Predicted probability</div><div class="v">${(p * 100).toFixed(1)}%</div><div class="d">calibrated</div></div></div>
      <div class="card expl"><b>What pushed this prediction?</b>${attrChart(rows)}
        <p>Predicted probability for <b>${esc(name)}</b> is <b>${(p * 100).toFixed(1)}%</b> (${p >= thr ? "ABOVE" : "below"} the ${Math.round(thr * 100)}% decision threshold).</p>
        <p style="margin:0">Top contributors for this patient:</p>
        <ul>${rows.map(x => `<li><b>${esc(x.label)}</b> = ${esc(x.value)}: ${x.a > 0 ? "raised" : "lowered"} risk (attribution ${x.a >= 0 ? "+" : ""}${x.a.toFixed(2)}).</li>`).join("")}</ul>
        <p class="cap">Red bars raised the risk, blue bars lowered it (gradient × input on the standardised inputs).</p></div>
    </div>`;
  }
}

function buildForm() {
  const meta = M.meta;
  $("core").innerHTML = meta.core.map(widget).join("");
  const groups = {};
  meta.feat_cols.filter(c => !meta.core.includes(c)).forEach(c => (groups[FM[c].group] ||= []).push(c));
  $("adv").innerHTML = Object.entries(groups).map(([g, cs]) => `<h4>${esc(g)}</h4>${cs.map(widget).join("")}`).join("");
  $("outs").innerHTML = meta.outcomes.map((o, i) => `<label><input type="checkbox" id="o_${i}" checked> ${esc(meta.outcome_display[o])}</label>`).join("");
  $("thr").value = meta.default_threshold;
}

(async () => {
  try {
    M = await loadModel("./");
  } catch (e) {
    $("status").innerHTML = `<span class="warn">Could not load the model files (${esc(e.message)}).</span>`;
    return;
  }
  FM = M.meta.feature_meta;
  buildForm();
  $("status").remove();
  let timer;
  document.querySelector("aside").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(render, 60); });
  $("reset").onclick = () => { buildForm(); tab = "summary"; render(); };
  render();
})();

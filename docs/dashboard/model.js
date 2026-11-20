// MultiDiseasePred DSelect-K forward pass and gradient x input, in plain JavaScript.
// Mirrors dashboard/app/model_utils.py and app/explainability.py of the Streamlit app.
"use strict";

async function loadModel(base) {
  const [meta, buf] = await Promise.all([
    fetch(base + "model.json").then(r => r.json()),
    fetch(base + "model.bin").then(r => r.arrayBuffer()),
  ]);
  const w = new Float32Array(buf);
  const get = name => {
    const t = meta.tensors[name];
    const n = t.shape.reduce((a, b) => a * b, 1);
    return { data: w.subarray(t.offset, t.offset + n), shape: t.shape };
  };
  const A = meta.arch, E = A.experts, H = A.hidden, T = A.tasks, D = meta.feat_cols.length;
  const experts = [];
  for (let e = 0; e < E; e++) {
    const p = `shared_layer.experts.${e}.net.`;
    experts.push({ W1: get(p + "0.weight").data, b1: get(p + "0.bias").data,
                   W2: get(p + "2.weight").data, b2: get(p + "2.bias").data });
  }
  const softmax = (v, temp) => {
    const m = Math.max(...v.map(x => x / temp));
    const ex = v.map(x => Math.exp(x / temp - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map(x => x / s);
  };
  const S = get("shared_layer.S").data, Am = get("shared_layer.A").data;
  // Task-to-expert distribution (independent of the input).
  const pTask = [];
  for (let t = 0; t < T; t++) {
    const s = softmax(Array.from(S.subarray(t * E, (t + 1) * E)), A.temperature);
    const a = softmax(Array.from(Am.subarray(t * E, (t + 1) * E)), A.temperature);
    const p = s.map((x, i) => 0.5 * x + 0.5 * a[i]);
    const z = p.reduce((x, y) => x + y, 0) + 1e-8;
    pTask.push(p.map(x => x / z));
  }
  return {
    meta, D, T, E, H, experts, pTask,
    R: get("shared_layer.shared_router.weight").data,
    rb: get("shared_layer.shared_router.bias").data,
    towers: Array.from({ length: T }, (_, t) => ({
      w: get(`towers.${t}.fc.weight`).data, b: get(`towers.${t}.fc.bias`).data[0] })),
  };
}

function forward(M, x, withGrad) {
  const { D, T, E, H, experts, pTask, R, rb, towers } = M;
  const alpha = M.meta.arch.alpha, temp = M.meta.arch.temperature;
  const h1s = [], h2s = [];
  for (const ex of experts) {
    const h1 = new Float64Array(H), h2 = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      let s = ex.b1[i];
      const row = i * D;
      for (let j = 0; j < D; j++) s += ex.W1[row + j] * x[j];
      h1[i] = s > 0 ? s : 0;
    }
    for (let i = 0; i < H; i++) {
      let s = ex.b2[i];
      const row = i * H;
      for (let j = 0; j < H; j++) s += ex.W2[row + j] * h1[j];
      h2[i] = s > 0 ? s : 0;
    }
    h1s.push(h1); h2s.push(h2);
  }
  const rl = [];
  for (let e = 0; e < E; e++) {
    let s = rb[e];
    for (let j = 0; j < D; j++) s += R[e * D + j] * x[j];
    rl.push(s);
  }
  const m = Math.max(...rl.map(v => v / temp));
  const ex = rl.map(v => Math.exp(v / temp - m));
  const zs = ex.reduce((a, b) => a + b, 0);
  const pShared = ex.map(v => v / zs);

  const logits = new Float64Array(T), grads = [];
  for (let t = 0; t < T; t++) {
    const q = pTask[t].map((p, e) => (1 - alpha) * p + alpha * pShared[e]);
    const Z = q.reduce((a, b) => a + b, 0) + 1e-8;
    const pmix = q.map(v => v / Z);
    const tw = towers[t].w;
    const g = new Float64Array(E); // g_e = tower_t . h2_e
    for (let e = 0; e < E; e++) {
      let s = 0; const h2 = h2s[e];
      for (let i = 0; i < H; i++) s += tw[i] * h2[i];
      g[e] = s;
    }
    let lg = towers[t].b;
    for (let e = 0; e < E; e++) lg += pmix[e] * g[e];
    logits[t] = lg;
    if (!withGrad) continue;

    // d logit / dx = sum_e pmix_e * d g_e/dx  +  alpha/(Z*temp) * sum_e s_e (g_e - gbar) R_e
    const grad = new Float64Array(D);
    const u = new Float64Array(H), v = new Float64Array(H);
    for (let e = 0; e < E; e++) {
      const exr = experts[e], h1 = h1s[e], h2 = h2s[e];
      for (let i = 0; i < H; i++) u[i] = h2[i] > 0 ? tw[i] : 0;
      v.fill(0);
      for (let i = 0; i < H; i++) {
        if (u[i] === 0) continue;
        const row = i * H, ui = u[i];
        for (let j = 0; j < H; j++) v[j] += exr.W2[row + j] * ui;
      }
      for (let j = 0; j < H; j++) {
        if (h1[j] <= 0) continue;
        const c = v[j] * pmix[e], row = j * D;
        for (let k = 0; k < D; k++) grad[k] += exr.W1[row + k] * c;
      }
    }
    let gbar = 0;
    for (let e = 0; e < E; e++) gbar += pShared[e] * g[e];
    const k0 = alpha / (Z * temp);
    for (let e = 0; e < E; e++) {
      const c = k0 * pShared[e] * (g[e] - gbar);
      for (let k = 0; k < D; k++) grad[k] += c * R[e * D + k];
    }
    grads.push(grad);
  }
  return { logits, grads };
}

function isotonic(cal, v) {
  const xs = cal.x, ys = cal.y, n = xs.length;
  if (v <= xs[0]) return ys[0];
  if (v >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= v) lo = mid; else hi = mid; }
  if (xs[hi] === xs[lo]) return ys[lo];
  return ys[lo] + (ys[hi] - ys[lo]) * (v - xs[lo]) / (xs[hi] - xs[lo]);
}

// inputs: {feature: value}. Returns probabilities and per-task attributions.
function predict(M, inputs) {
  const meta = M.meta;
  const raw = meta.feat_cols.map((c, i) => (c in inputs && inputs[c] !== null && inputs[c] !== "" && !Number.isNaN(+inputs[c])) ? +inputs[c] : meta.median[i]);
  const x = raw.map((v, i) => (v - meta.scaler_mean[i]) / meta.scaler_scale[i]);
  const { logits, grads } = forward(M, x, true);
  const probs = Array.from(logits, l => 1 / (1 + Math.exp(-l)));
  const cal = Array.from(logits, (l, t) => isotonic(meta.calibrators[t], l));
  const attr = grads.map(g => Array.from(g, (gi, i) => gi * x[i]));
  return { raw, x, logits: Array.from(logits), probs, cal, attr };
}

if (typeof module !== "undefined") module.exports = { loadModel, forward, predict, isotonic };

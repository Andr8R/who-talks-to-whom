/* "Who talks to whom?" static demo.
   Draws precomputed model outputs (window.DEMO_DATA) and k-NN graphs (pure geometry).
   No ML runs in the browser. State lives in the URL hash, so every view can be linked from a slide. */
(function () {
  "use strict";
  const D = window.DEMO_DATA;
  const NS = "http://www.w3.org/2000/svg";
  const KS = [2, 4, 8, 16, "all"];
  const MODELS = { knn_attn: "k-NN attention (k = 8)", transformer: "Transformer", part: "Transformer + pair bias" };
  const SCORE_NAMES = { deepsets: "Deep Sets", edgeconv: "EdgeConv GNN", knn_attn: "k-NN attention", transformer: "Transformer", part: "Transformer + pair bias" };
  const SCORE_COL = { deepsets: "#e87ba4", edgeconv: "#1baf7a", knn_attn: "#eda100", transformer: "#eb6834", part: "#2a78d6" };
  const HOP = ["#0b2f5e", "#1f5fad", "#5b8fd1", "#9fbfe8", "#cfe0f5"];
  const H_ = D.heads;

  const S = { tab: "graph", jet: 0, k: 8, L: 1, node: 0, model: "transformer", layer: 3, head: "mean", q: 0, layer3: 1, zoom: 0 };

  // ------------------------------------------------------------------ state <-> URL
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    for (const key of Object.keys(S)) {
      if (!p.has(key)) continue;
      const v = p.get(key);
      S[key] = (key === "tab" || key === "model" || (key === "k" && v === "all") || (key === "head" && v === "mean")) ? v : Number(v);
    }
    S.jet = Math.max(0, Math.min(D.jets.length - 1, S.jet | 0));
    if (!MODELS[S.model]) S.model = "transformer";
  }
  function writeHash() {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(S)) p.set(k, v);
    history.replaceState(null, "", "#" + p.toString());
  }

  // ------------------------------------------------------------------ geometry
  const jet = () => D.jets[S.jet];
  function fit(j, focus) {
    if (S.zoom && focus !== null && focus !== undefined) {        // zoom on the selected particle (Delta R window 0.07)
      const r = 0.07, cx = j.eta[focus], cy = j.phi[focus];
      return { x: (e) => 300 + (e - cx) / r * 270, y: (p) => 300 - (p - cy) / r * 270 };
    }
    const xs = j.eta, ys = j.phi;
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const r = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2 + 0.06;
    return { x: (e) => 300 + (e - cx) / r * 270, y: (p) => 300 - (p - cy) / r * 270 };
  }
  const radius = (j, i) => 4 + 13 * Math.sqrt(j.pt[i] / j.pt[0]);

  const knnCache = new Map();
  function knn(j, k) {          // neighbours of each particle (excluding itself), nearest first
    const key = j.id + ":" + k;
    if (knnCache.has(key)) return knnCache.get(key);
    const n = j.n, out = [];
    for (let i = 0; i < n; i++) {
      const d = [];
      for (let m = 0; m < n; m++) if (m !== i) d.push([(j.eta[i] - j.eta[m]) ** 2 + (j.phi[i] - j.phi[m]) ** 2, m]);
      d.sort((a, b) => a[0] - b[0]);
      out.push(d.slice(0, k === "all" ? n - 1 : Math.min(k, n - 1)).map((x) => x[1]));
    }
    knnCache.set(key, out);
    return out;
  }
  function hops(nb, start, L) {  // level[m] = layers needed before m's information reaches `start` (-1: not yet)
    const level = new Array(nb.length).fill(-1);
    level[start] = 0;
    let frontier = [start];
    for (let l = 1; l <= L; l++) {
      const next = [];
      for (const u of frontier) for (const v of nb[u]) if (level[v] < 0) { level[v] = l; next.push(v); }
      frontier = next;
    }
    return level;
  }
  // ------------------------------------------------------------------ attention data
  const attCache = new Map();
  function attn(j, model) {
    const key = j.id + ":" + model;
    if (!attCache.has(key)) {
      const bin = atob(j.attn[model]);
      const a = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
      attCache.set(key, a);
    }
    return attCache.get(key);
  }
  function attRow(j, model, layer, head, q) {   // weights of query q over all keys (layer 1-based, head 0-based or "mean")
    const a = attn(j, model), n = j.n, row = new Float64Array(n);
    const heads = head === "mean" ? [...Array(H_).keys()] : [head];
    for (const h of heads) {
      const off = (((layer - 1) * H_ + h) * n + q) * n;
      for (let m = 0; m < n; m++) row[m] += a[off + m] / 255 / heads.length;
    }
    return row;
  }

  // ------------------------------------------------------------------ drawing
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function drawJet(svg, j, { edges = [], fills = null, halo = null, focus = null, onPick = null, label = null }) {
    svg.innerHTML = "";
    const f = fit(j, focus);
    el("clipPath", { id: svg.id + "-clip" }, el("defs", {}, svg)).appendChild(el("rect", { x: 0, y: 0, width: 600, height: 600 }));
    el("rect", { x: 0, y: 0, width: 600, height: 600, fill: "#fff" }, svg);
    const root = el("g", { "clip-path": `url(#${svg.id}-clip)` }, svg);
    const ge = el("g", {}, root);
    for (const e of edges) {
      el("line", { x1: f.x(j.eta[e.a]), y1: f.y(j.phi[e.a]), x2: f.x(j.eta[e.b]), y2: f.y(j.phi[e.b]),
        stroke: e.color || "#9aa0a8", "stroke-width": e.w || 1.2, "stroke-opacity": e.op ?? 0.6, "stroke-linecap": "round" }, ge);
    }
    for (let i = j.n - 1; i >= 0; i--) {
      const r = radius(j, i);
      if (halo && halo[i] > 0) el("circle", { cx: f.x(j.eta[i]), cy: f.y(j.phi[i]), r: r + 3 + 10 * halo[i], fill: halo.color || "#eb6834", "fill-opacity": 0.25 }, root);
      const c = el("circle", { cx: f.x(j.eta[i]), cy: f.y(j.phi[i]), r, fill: fills ? fills[i] : "#5a5f68",
        stroke: i === focus ? "#b42318" : "#fff", "stroke-width": i === focus ? 3.5 : 1.2 }, root);
      if (onPick) {
        // generous invisible hit area for touch screens
        const hit = el("circle", { cx: f.x(j.eta[i]), cy: f.y(j.phi[i]), r: Math.max(r, 14), fill: "transparent", style: "cursor:pointer" }, root);
        hit.addEventListener("click", () => onPick(i));
        c.addEventListener("click", () => onPick(i));
      }
    }
    el("text", { x: 14, y: 586, "font-size": 15, fill: "#767c86" }, svg).textContent = "Δη →";
    el("text", { x: 14, y: 26, "font-size": 15, fill: "#767c86" }, svg).textContent = label || "Δφ ↑   (marker area ∝ pT)";
    if (S.zoom) el("text", { x: 586, y: 586, "font-size": 15, fill: "#b42318", "text-anchor": "end" }, svg).textContent = "zoom: ΔR window ±0.07";
  }

  function seg(id, values, cur, labelFn, onSet) {
    const box = document.getElementById(id);
    box.innerHTML = "";
    for (const v of values) {
      const b = document.createElement("button");
      b.textContent = labelFn ? labelFn(v) : v;
      if (String(v) === String(cur)) b.className = "on";
      b.addEventListener("click", () => onSet(v));
      box.appendChild(b);
    }
  }

  // ------------------------------------------------------------------ tab 1: graph & message passing
  function renderGraph() {
    const j = jet();
    if (S.node >= j.n) S.node = 0;
    seg("seg-k", KS, S.k, (v) => (v === "all" ? "all" : v), (v) => { S.k = v; render(); });
    document.getElementById("L").value = S.L;
    document.getElementById("L-val").textContent = S.L;
    const nb = knn(j, S.k);
    const level = hops(nb, S.node, S.L);
    const seen = new Set(), edges = [];
    nb.forEach((list, a) => list.forEach((b) => {
      const key = a < b ? a + "-" + b : b + "-" + a;
      if (seen.has(key)) return;
      seen.add(key);
      const inside = level[a] >= 0 && level[b] >= 0;
      edges.push({ a, b, w: inside ? 2.2 : 1, op: S.k === "all" ? (inside ? 0.35 : 0.12) : (inside ? 0.85 : 0.45), color: inside ? "#1f5fad" : "#9aa0a8" });
    }));
    const fills = level.map((l) => (l < 0 ? "#d6d2c9" : HOP[Math.min(l, HOP.length - 1)]));
    drawJet(document.getElementById("svg-graph"), j, { edges, fills, focus: S.node, onPick: (i) => { S.node = i; render(); } });

    const reach = level.filter((l) => l >= 0).length;
    const ptTot = j.pt.reduce((a, b) => a + b, 0);
    const ptReach = j.pt.reduce((a, p, i) => a + (level[i] >= 0 ? p : 0), 0);
    const nEdges = nb.reduce((a, x) => a + x.length, 0);
    const eventually = hops(nb, S.node, j.n).filter((l) => l >= 0).length;
    const ec = S.k === "all" ? null : j.edgeconv_k[String(S.k)];
    document.getElementById("graph-readout").innerHTML =
      `<div class="row"><span>Directed edges (N·k)</span><b>${nEdges}</b></div>` +
      `<div class="row"><span>Reachable after L = ${S.L}</span><b>${reach} of ${j.n} particles · ${(100 * ptReach / ptTot).toFixed(0)} % of pT</b></div>` +
      `<div class="row"><span>Reachable with unlimited layers</span><b>${eventually} of ${j.n} particles</b></div>` +
      (ec === null ? `<div class="row"><span>EdgeConv P(W) for this jet</span><b>—</b></div>`
        : `<div class="row"><span>EdgeConv P(W) for this jet</span><b>${ec.toFixed(2)}</b></div>` +
          (S.k !== 8 ? `<div class="ood">The model was trained with k = 8; this value uses a different graph without retraining.</div>` : ""));
  }

  // ------------------------------------------------------------------ tab 2: attention
  function renderAttention() {
    const j = jet();
    if (S.q >= j.n) S.q = 0;
    seg("seg-model", Object.keys(MODELS), S.model, (v) => MODELS[v], (v) => { S.model = v; render(); });
    seg("seg-layer", [1, 2, 3], S.layer, null, (v) => { S.layer = v; render(); });
    seg("seg-head", ["mean", 0, 1, 2, 3], S.head, (v) => (v === "mean" ? "mean" : v + 1), (v) => { S.head = v; render(); });
    const row = attRow(j, S.model, S.layer, S.head, S.q);
    const n = j.n, uni = 1 / n, edges = [];
    for (let m = 0; m < n; m++) {
      if (m === S.q || row[m] <= uni) continue;
      const r = Math.min(row[m] * n, 5) / 5;
      edges.push({ a: S.q, b: m, w: 0.8 + 6 * r, op: 0.25 + 0.7 * r, color: "#eb6834" });
    }
    edges.sort((x, y) => x.w - y.w);
    const halo = Array.from(row, (w) => Math.min(w * n, 5) / 5);
    halo.color = "#eb6834";
    drawJet(document.getElementById("svg-attn"), j, { edges, halo, focus: S.q, onPick: (i) => { S.q = i; render(); } });

    let H = 0, mx = 0;
    for (const w of row) { if (w > 0) H -= w * Math.log(w); mx = Math.max(mx, w); }
    document.getElementById("attn-readout").innerHTML =
      `<div class="row"><span>Uniform weight 1/N</span><b>${uni.toFixed(3)}</b></div>` +
      `<div class="row"><span>Largest weight</span><b>${mx.toFixed(3)} (${(mx * n).toFixed(1)}× uniform)</b></div>` +
      `<div class="row"><span>Weight on itself</span><b>${row[S.q].toFixed(3)}</b></div>` +
      `<div class="row"><span>Effective no. of keys e<sup>H</sup></span><b>${Math.exp(H).toFixed(1)} of ${n}</b></div>`;
    drawHeat(j);
  }
  function drawHeat(j) {
    const cv = document.getElementById("heat"), n = j.n;
    const cell = Math.floor(300 / n), size = cell * n;
    cv.width = size; cv.height = size;
    const ctx = cv.getContext("2d");
    const nb = S.model === "knn_attn" ? knn(j, 8) : null;
    for (let q = 0; q < n; q++) {
      const row = attRow(j, S.model, S.layer, S.head, q);
      for (let m = 0; m < n; m++) {
        const allowed = !nb || m === q || nb[q].includes(m);
        if (!allowed) { ctx.fillStyle = "#e6e3dc"; }
        else {
          const t = Math.min(row[m] * n, 5) / 5;            // 0 .. 5x uniform
          const c = (a, b) => Math.round(a + (b - a) * t);
          ctx.fillStyle = `rgb(${c(247, 11)},${c(250, 47)},${c(255, 94)})`;
        }
        ctx.fillRect(m * cell, q * cell, cell, cell);
      }
    }
    ctx.strokeStyle = "#b42318"; ctx.lineWidth = 2;
    ctx.strokeRect(1, S.q * cell + 1, size - 2, cell - 2);
    cv.onclick = (ev) => {
      const r = cv.getBoundingClientRect();
      S.q = Math.min(n - 1, Math.floor((ev.clientY - r.top) / r.height * n));
      render();
    };
  }

  // ------------------------------------------------------------------ tab 3: local vs global
  function renderCompare() {
    const j = jet(), n = j.n;
    if (S.q >= n) S.q = 0;
    seg("seg-layer3", [1, 2, 3], S.layer3, null, (v) => { S.layer3 = v; render(); });
    const pick = (i) => { S.q = i; render(); };
    const nb = knn(j, 8);
    const mk = (weights, color) => {
      const edges = [];
      weights.forEach((w, m) => {
        if (m === S.q || w <= 0) return;
        const r = Math.min(w * n, 5) / 5;
        edges.push({ a: S.q, b: m, w: 0.8 + 6 * r, op: 0.25 + 0.7 * r, color });
      });
      return edges.sort((x, y) => x.w - y.w);
    };
    const fixed = new Array(n).fill(0);
    nb[S.q].forEach((m) => (fixed[m] = 1 / 8));
    drawJet(document.getElementById("svg-c1"), j, { edges: mk(fixed, "#1baf7a"), focus: S.q, onPick: pick, label: "weights: 1/k each" });
    const w2 = attRow(j, "knn_attn", S.layer3, "mean", S.q);
    drawJet(document.getElementById("svg-c2"), j, { edges: mk(w2, "#eda100"), focus: S.q, onPick: pick, label: "weights: softmax over 8 + self" });
    const w3 = attRow(j, "transformer", S.layer3, "mean", S.q);
    drawJet(document.getElementById("svg-c3"), j, { edges: mk(w3.map((w) => (w > 1 / n ? w : 0)), "#eb6834"), focus: S.q, onPick: pick, label: "weights: softmax over all (> 1/N shown)" });

    document.getElementById("bars").innerHTML = Object.keys(SCORE_NAMES).map((k) => {
      const p = j.score[k];
      return `<div class="bar"><span>${SCORE_NAMES[k]}</span><span class="track"><span class="fill" style="display:block;width:${(100 * p).toFixed(0)}%;background:${SCORE_COL[k]}"></span><span class="mid"></span></span><span>${p.toFixed(2)}</span></div>`;
    }).join("") + `<p class="small muted">Truth: <b>${j.label ? "W jet" : "QCD jet"}</b>. Dashed line = 0.5.</p>`;
    const mx = Math.max(...Object.values(D.summary).map((v) => v.rej50));
    document.getElementById("summary").innerHTML = Object.keys(SCORE_NAMES).map((k) => {
      const v = D.summary[k];
      return `<div class="bar"><span>${SCORE_NAMES[k]}</span><span class="track"><span class="fill" style="display:block;width:${(100 * v.rej50 / mx).toFixed(0)}%;background:${SCORE_COL[k]}"></span></span><span>${v.rej50.toFixed(1)}</span></div>`;
    }).join("");
  }

  // ------------------------------------------------------------------ shell
  function describe(j, idx) {
    const p = j.score.part, right = j.label ? p > 0.5 : p < 0.5;
    const conf = Math.abs(p - 0.5) < 0.15 ? "borderline" : right ? "correctly classified" : "misclassified";
    return `${j.label ? "W" : "QCD"} jet ${idx + 1} · ${j.n} particles · ${conf}`;
  }
  function render() {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === S.tab));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + S.tab));
    const j = jet();
    document.getElementById("jet").value = S.jet;
    document.querySelectorAll(".zoom").forEach((b) => { b.classList.toggle("on", !!S.zoom); b.textContent = S.zoom ? "Show whole jet" : "Zoom on selected particle"; });
    document.getElementById("jetinfo").innerHTML =
      `<span class="tag ${j.label ? "w" : "q"}">${j.label ? "W → qq̄" : "QCD"}</span> jet mass ${j.mass} GeV · ${j.n} particles`;
    if (S.tab === "graph") renderGraph();
    else if (S.tab === "attention") renderAttention();
    else renderCompare();
    writeHash();
  }

  const sel = document.getElementById("jet");
  D.jets.forEach((j, i) => { const o = document.createElement("option"); o.value = i; o.textContent = describe(j, i); sel.appendChild(o); });
  sel.addEventListener("change", () => { S.jet = Number(sel.value); S.node = 0; S.q = 0; render(); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => { S.tab = b.dataset.tab; render(); }));
  document.querySelectorAll(".zoom").forEach((b) => b.addEventListener("click", () => { S.zoom = S.zoom ? 0 : 1; render(); }));
  document.getElementById("L").addEventListener("input", (e) => { S.L = Number(e.target.value); render(); });
  window.addEventListener("hashchange", () => { readHash(); render(); });
  readHash();
  render();
})();

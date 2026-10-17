// Dashboard and join page. Reads only this coordinator's public JSON API; no third-party requests.
"use strict";

const base = location.pathname.replace(/\/(index\.html|how|join)?$/, "");

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value; else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

function svg(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
  return node;
}

const fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, notation: "compact" });
const num = (v) => (v === null || v === undefined ? "-" : fmt.format(v));

function table(columns, rows, numeric) {
  const head = el("tr", {}, ...columns.map((c, i) => el("th", numeric.includes(i) ? { class: "num" } : {}, c)));
  const body = rows.map((r) => el("tr", {}, ...r.map((v, i) =>
    v instanceof Node ? el("td", {}, v) : el("td", numeric.includes(i) ? { class: "num" } : {}, v))));
  return el("table", {}, el("thead", {}, head), el("tbody", {}, ...body));
}

function curveChart(curve) {
  const names = [...new Set(curve.flatMap((p) => Object.keys(p.metrics || {})))].sort((a, b) =>
    (a.includes(".") - b.includes(".")) || a.localeCompare(b)).slice(0, 7);
  if (!curve.length || !names.length) return el("p", { class: "empty" }, "No held-out scores yet.");
  const values = curve.flatMap((p) => names.map((n) => p.metrics[n]).filter((v) => typeof v === "number"));
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
  const W = 720, H = 260, L = 44, R = 12, T = 10, B = 28;
  const versions = curve.map((p) => p.version);
  const x = (v) => L + ((v - versions[0]) / Math.max(1, versions[versions.length - 1] - versions[0])) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const chart = svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Held-out score per round" });
  for (let i = 0; i <= 4; i++) {
    const v = lo + (i * (hi - lo)) / 4;
    chart.append(svg("line", { class: "axis", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    const label = svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" });
    label.textContent = v.toFixed(2);
    chart.append(label);
  }
  const step = Math.max(1, Math.ceil(versions.length / 10));
  versions.forEach((v, i) => {
    if (i % step) return;
    const label = svg("text", { x: x(v), y: H - 8, "text-anchor": "middle" });
    label.textContent = "v" + v;
    chart.append(label);
  });
  const legend = el("div", { class: "legend" });
  names.forEach((name, i) => {
    const color = `var(--s${i + 1})`;
    const points = curve.filter((p) => typeof p.metrics[name] === "number").map((p) => `${x(p.version)},${y(p.metrics[name])}`);
    chart.append(svg("polyline", { points: points.join(" "), fill: "none", stroke: color, "stroke-width": 2 }));
    for (const point of points) {
      const [cx, cy] = point.split(",");
      chart.append(svg("circle", { cx, cy, r: 2.5, fill: color }));
    }
    const key = el("span", {}, name);
    key.style.setProperty("--c", color);
    legend.append(key);
  });
  return el("div", {}, chart, legend);
}

function render(d) {
  const g = d.global;
  document.getElementById("sub").textContent = g
    ? `Global v${g.version} published ${new Date(g.created * 1000).toLocaleString()} - updated ${new Date(d.generated * 1000).toLocaleTimeString()}`
    : "Waiting for the first global weights.";
  document.getElementById("t-version").textContent = g ? `v${g.version}` : "-";
  document.getElementById("t-online").textContent = `${d.totals.sites_online} / ${d.totals.sites}`;
  document.getElementById("t-day").textContent = num(d.totals.tokens_per_day);
  document.getElementById("t-total").textContent = num(d.totals.tokens_received);
  document.getElementById("t-deltas").textContent = `${d.totals.deltas_accepted} / ${d.totals.deltas_received}`;
  document.getElementById("t-gpu").textContent = num(d.totals.gpu_hours_per_day);
  document.getElementById("curve").replaceChildren(curveChart(d.curve || []));
  const sites = d.sites || [];
  document.getElementById("sites").replaceChildren(sites.length ? table(
    ["#", "Site", "Status", "Tokens/s", "Tokens received", "Deltas accepted", "Pending review"],
    sites.map((s, i) => [i + 1, el("span", s.anonymous ? { class: "anon" } : {}, s.name),
      el("span", { class: "status " + ({ offline: "off", stopped: "off", stale: "warn", paused: "warn" }[s.status] || "on") },
        s.status),
      num(s.tokens_per_s), num(s.tokens_received), s.deltas_accepted, s.deltas_pending]), [0, 3, 4, 5, 6])
    : el("p", { class: "empty" }, "No sites yet."));
  const wanted = d.wanted || {};
  const items = [...(wanted.notes || []).map((n) => el("li", {}, n)),
    ...(wanted.weakest || []).map((w) => el("li", {}, `${w.category}: ${w.eal.toFixed(2)}`))];
  document.getElementById("wanted").replaceChildren(items.length ? el("ul", { class: "plain" }, ...items)
    : el("p", { class: "empty" }, "Nothing published yet."));
  const reasons = Object.entries(d.rejections || {});
  document.getElementById("rejections").replaceChildren(reasons.length
    ? table(["Reason", "Deltas"], reasons.sort((a, b) => b[1] - a[1]), [1]) : el("p", { class: "empty" }, "None."));
  const bench = d.benchmarks;
  document.getElementById("bench").replaceChildren(bench && bench.columns && bench.rows
    ? el("div", {}, table(bench.columns, bench.rows, bench.columns.map((_, i) => i).slice(1)),
      el("p", { class: "sub" }, bench.note || ""))
    : el("p", { class: "empty" }, "Not published yet."));
  document.getElementById("foot").textContent =
    `Participation terms ${d.terms.version}. Data from this coordinator's public API; no trackers, no third-party requests.`;
}

async function refresh() {
  try {
    const response = await fetch(base + "/api/public/dashboard", { cache: "no-store" });
    if (response.ok) render(await response.json());
  } catch (error) {
    document.getElementById("sub").textContent = "Live data unavailable; retrying.";
  }
}

function joinPage() {
  const params = new URLSearchParams(location.hash.slice(1));
  const code = params.get("invite");
  const box = document.getElementById("invite");
  if (!code) { box.textContent = "This page needs an invite link (it ends in #invite=...)."; return; }
  // The code stays in this browser: URL fragments are never sent to the server.
  const command = `drafter-join redeem '${location.origin}${base}/join#invite=${code}' --accept-terms --name "<your display name>"`;
  box.replaceChildren(el("pre", {}, command), el("button", { type: "button", id: "copy" }, "Copy"));
  document.getElementById("copy").addEventListener("click", () => navigator.clipboard.writeText(command));
}

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  if (page === "dashboard") { refresh(); setInterval(refresh, 30000); }
  if (page === "join") joinPage();
});

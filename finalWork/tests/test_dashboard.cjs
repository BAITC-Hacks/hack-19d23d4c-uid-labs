#!/usr/bin/env node
"use strict";

// Optional development check; the dashboard itself does not require Node.js.
// Run from any directory:
//   node tests/test_dashboard.cjs
//   node tests/test_dashboard.cjs path/to/generated/dashboard.html
// The default demo must contain 2,248 synthetic participants. A custom path is
// checked against its own report metadata. This checks JavaScript and DOM logic,
// not browser layout, screenshots, or actual Canvas rendering. No browser or
// network connection is opened. Only Node.js built-in modules are used.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const defaultPath = path.resolve(__dirname, "../results/demo/dashboard.html");
const dashboardPath = process.argv[2] ? path.resolve(process.argv[2]) : defaultPath;
const html = fs.readFileSync(dashboardPath, "utf8");
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)];
const embedded = scripts.find(match => /\bid\s*=\s*["']report-data["']/i.test(match[1]));
assert.ok(embedded, "The generated HTML must include the report-data script");
assert.ok(!embedded[2].includes("__REPORT_JSON__"), "The report placeholder must be filled");
const report = JSON.parse(embedded[2]);
const executable = scripts.filter(match => !/\btype\s*=\s*["']application\/json["']/i.test(match[1]));
assert.equal(executable.length, 1, "Expected one inline application script");
assert.ok(!scripts.some(match => /\bsrc\s*=/i.test(match[1])), "No external script may be required");
const source = executable[0][2];
const compiled = new vm.Script(source, { filename: dashboardPath + ":application-script" });

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName.toLowerCase();
    this.childNodes = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.style = {};
    this.className = "";
    this.value = "";
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this._text = "";
  }

  set textContent(value) {
    this._text = String(value);
    this.childNodes = [];
  }

  get textContent() {
    return this._text + this.childNodes.map(node => node.textContent).join("");
  }

  // The application deliberately writes user-controlled fields as text.
  set innerHTML(_value) {
    throw new Error("Unexpected innerHTML write; review escaping before changing this check");
  }

  append(...nodes) {
    this.childNodes.push(...nodes.map(node => {
      if (node instanceof Element) return node;
      const text = new Element("#text");
      text.textContent = node;
      return text;
    }));
  }

  replaceChildren(...nodes) {
    this._text = "";
    this.childNodes = [];
    this.append(...nodes);
  }

  get lastChild() { return this.childNodes.at(-1) || null; }

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }

  dispatch(type, extra = {}) {
    const event = { type, target: this, preventDefault() {}, ...extra };
    for (const listener of this.listeners.get(type) || []) listener(event);
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  getBoundingClientRect() { return { width: 1000, height: 565, left: 0, top: 0 }; }
  scrollIntoView() {}
  setPointerCapture() {}
  releasePointerCapture() {}
  hasPointerCapture() { return false; }

  cloneNode(deep = false) {
    const clone = new Element(this.tagName);
    for (const key of ["_text", "className", "value", "hidden", "checked", "disabled"]) {
      clone[key] = this[key];
    }
    clone.attributes = new Map(this.attributes);
    if (deep) clone.childNodes = this.childNodes.map(node => node.cloneNode(true));
    return clone;
  }
}

function walk(element) {
  return [element, ...element.childNodes.flatMap(walk)];
}

function byClass(element, className) {
  return walk(element).find(node => node.className.split(/\s+/).includes(className));
}

function boot(data) {
  const ids = new Map();
  // Parse only the static element shells used by the script, not report text.
  const staticMarkup = html.replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi, "");
  for (const match of staticMarkup.matchAll(/<([a-z][\w:-]*)\b([^>]*)>/gi)) {
    const id = match[2].match(/\bid\s*=\s*["']([^"']+)["']/i)?.[1];
    if (!id) continue;
    const element = new Element(match[1]);
    element.hidden = /(?:^|\s)hidden(?:\s|=|$)/i.test(match[2]);
    element.disabled = /(?:^|\s)disabled(?:\s|=|$)/i.test(match[2]);
    element.checked = /(?:^|\s)checked(?:\s|=|$)/i.test(match[2]);
    element.className = match[2].match(/\bclass\s*=\s*["']([^"']*)["']/i)?.[1] || "";
    ids.set(id, element);
  }
  const dataElement = new Element("script");
  dataElement.textContent = JSON.stringify(data);
  ids.set("report-data", dataElement);

  const canvasCalls = new Map();
  const context2d = {};
  const numericCanvasMethods = ["setTransform", "clearRect", "moveTo", "lineTo", "arc", "fillRect"];
  for (const method of [...numericCanvasMethods, "beginPath", "closePath", "stroke", "fill", "setLineDash", "fillText"]) {
    context2d[method] = (...args) => {
      canvasCalls.set(method, (canvasCalls.get(method) || 0) + 1);
      if (numericCanvasMethods.includes(method)) {
        assert.ok(args.every(value => typeof value !== "number" || Number.isFinite(value)), method + " received nonfinite coordinates");
      }
    };
  }
  context2d.measureText = text => ({ width: String(text).length * 6 });
  ids.get("graph-canvas").getContext = name => name === "2d" ? context2d : null;

  const animationFrames = [];
  const consoleErrors = [];
  const sandbox = {
    document: {
      documentElement: new Element("html"),
      getElementById(id) {
        assert.ok(ids.has(id), "Script requested a missing DOM id: " + id);
        return ids.get(id);
      },
      createElement: tagName => new Element(tagName)
    },
    window: {
      devicePixelRatio: 1,
      matchMedia: () => ({ matches: false, addEventListener() {} }),
      addEventListener() {}
    },
    getComputedStyle: () => ({ getPropertyValue: () => "#123456" }),
    requestAnimationFrame: callback => animationFrames.push(callback),
    ResizeObserver: class { observe() {} },
    console: { log() {}, warn() {}, error: (...args) => consoleErrors.push(args) }
    // No fetch, XMLHttpRequest, require, process, or filesystem in the UI context.
  };
  compiled.runInNewContext(sandbox, { timeout: 10000 });

  function flush() {
    let frames = 0;
    while (animationFrames.length) {
      assert.ok(++frames <= 100, "Animation queue did not settle");
      animationFrames.shift()(0);
    }
    assert.deepEqual(consoleErrors, [], "Application emitted console errors");
    assert.equal(ids.get("load-error").hidden, true, "Application displayed a report parse error");
  }
  flush();

  return {
    ids, canvasCalls,
    get: id => ids.get(id),
    flush,
    event(id, type, extra) { ids.get(id).dispatch(type, extra); flush(); },
    search(gid) { ids.get("gid-search").value = gid; this.event("search-form", "submit"); },
    change(id, value) { ids.get(id).value = value; this.event(id, "change"); },
    counts() {
      return ids.get("graph-count").textContent.replace(/\s/g, "").match(/\d+/g).map(Number);
    },
    selectedGid() { return byClass(ids.get("details"), "detail-gid")?.textContent ?? null; }
  };
}

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("ok " + passed + " - " + name);
}

const demo = boot(report);
check("generated report parses; application starts with zero errors", () => {
  assert.ok(Array.isArray(report.nodes));
  assert.equal(demo.get("load-error").textContent, "");
  assert.ok((demo.canvasCalls.get("clearRect") || 0) > 0, "Canvas draw did not run");
});

check("all report nodes are included in the overview", () => {
  assert.equal(Number(demo.get("kpi-nodes").textContent.replace(/\D/g, "")), report.meta.n_nodes);
  assert.deepEqual(demo.counts().slice(0, 2), [report.nodes.length, report.nodes.length]);
  assert.ok((demo.canvasCalls.get("arc") || 0) >= report.nodes.length, "Some fitted nodes were omitted from Canvas drawing");
  if (dashboardPath === defaultPath) {
    assert.equal(report.nodes.length, 2248, "The default fixture must contain 2,248 nodes");
    assert.equal(report.meta.n_nodes, 2248);
  }
});

check("synthetic banner follows report metadata; default demo is synthetic", () => {
  assert.equal(demo.get("synthetic-notice").hidden, !report.meta.synthetic);
  if (dashboardPath === defaultPath) assert.equal(report.meta.synthetic, true);
});

const largeA = "9223372036854775806";
const largeB = "9223372036854775807";
const unsafeText = "</script><img src=x onerror=alert(1)>";
const fixture = {
  meta: { synthetic: true, n_nodes: 4, n_edges: 3, n_transactions: 3, total_kzt: 6000 },
  nodes: [
    { gid: largeA, depth: 0, is_seed: true, role: "collector", cluster_id: 1, metrics: {}, evidence: [unsafeText] },
    { gid: largeB, depth: 1, role: "transit", cluster_id: 1, metrics: { boundary: true } },
    { gid: "003", depth: 2, role: "unknown", cluster_id: 2, metrics: {} },
    { gid: "004", depth: 3, role: "unknown", cluster_id: 2, metrics: {} }
  ],
  edges: [
    { src: largeA, dst: largeB, sum_kzt: 1000, n_tx: 1 },
    { src: largeB, dst: "003", sum_kzt: 2000, n_tx: 1 },
    { src: "003", dst: "004", sum_kzt: 3000, n_tx: 1 }
  ],
  top_nodes: [{ rank: 1, gid: largeA, role: "collector", priority_score: 1, why: "Fixture evidence" }],
  clusters: [{ cluster_id: 1, n_nodes: 2, n_seed: 1, sum_kzt_internal: 1000 }],
  inquiries: [{ gid: largeA, reason: "Fixture reason", request: "Fixture request" }],
  warnings: ["Fixture warning"],
  boundary_witnesses: [{ gid: largeA, observed_same: true, property_changes: true, scenario_a: "A", scenario_b: "B", scope: "Fixture scope" }]
};
const ui = boot(fixture);

check("exact lookup distinguishes adjacent int64 identifiers", () => {
  ui.search(largeA);
  assert.equal(ui.selectedGid(), largeA);
  assert.ok(ui.get("details").textContent.includes(unsafeText), "Evidence should be retained as literal text");
  ui.search(largeB);
  assert.equal(ui.selectedGid(), largeB);
  ui.search("9223372036854775805");
  assert.equal(ui.get("query-status").hidden, false, "An absent adjacent gid must not resolve");
  assert.equal(ui.selectedGid(), largeB, "A failed lookup must retain the previous selection");
});

check("ego mode includes only the selected node and its direct neighbors", () => {
  ui.search(largeA);
  assert.equal(ui.get("ego-filter").checked, true);
  assert.equal(ui.get("ego-filter").disabled, false);
  assert.deepEqual(ui.counts(), [2, 4, 1]);
  ui.search(largeB);
  assert.deepEqual(ui.counts(), [3, 4, 2], "Incoming and outgoing neighbors must both be included");
  ui.get("ego-filter").checked = false;
  ui.event("ego-filter", "change");
  assert.deepEqual(ui.counts(), [4, 4, 3]);
});

check("reset clears selection, search, ego mode, and filters", () => {
  ui.event("reset-graph", "click");
  assert.equal(ui.selectedGid(), null);
  assert.equal(ui.get("gid-search").value, "");
  assert.equal(ui.get("role-filter").value, "");
  assert.equal(ui.get("cluster-filter").value, "");
  assert.equal(ui.get("ego-filter").checked, false);
  assert.equal(ui.get("ego-filter").disabled, true);
  assert.equal(ui.get("query-status").hidden, true);
  assert.deepEqual(ui.counts(), [4, 4, 3]);
});

check("role and cluster filters intersect and handle an empty result", () => {
  ui.change("role-filter", "transit");
  assert.deepEqual(ui.counts(), [1, 4, 0]);
  ui.change("cluster-filter", "2");
  assert.deepEqual(ui.counts(), [0, 4, 0]);
  assert.equal(ui.get("canvas-empty").hidden, false);
  ui.change("role-filter", "");
  assert.deepEqual(ui.counts(), [2, 4, 1]);
  assert.equal(ui.get("canvas-empty").hidden, true);
});

check("exact search clears conflicting filters and preserves leading zeros", () => {
  ui.search(largeA);
  assert.equal(ui.get("cluster-filter").value, "");
  assert.equal(ui.selectedGid(), largeA);
  ui.search("003");
  assert.equal(ui.selectedGid(), "003");
  ui.search("3");
  assert.equal(ui.get("query-status").hidden, false);
  assert.equal(ui.selectedGid(), "003");
});

check("empty and optional report fields do not crash the application", () => {
  boot({});
  boot({ nodes: [], edges: [] });
  const plain = boot({ meta: { synthetic: false }, nodes: [], edges: [] });
  assert.equal(plain.get("synthetic-notice").hidden, true);
});

console.log("\nPASS: " + passed + " checks; " + report.nodes.length + " report nodes; " + dashboardPath);
console.log("Scope: source execution and DOM state in a mock; browser layout and pixels are not checked.");

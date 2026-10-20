/* Taxonomy Editor */

const ACCENTS = ["accent-1", "accent-2", "accent-3", "accent-4", "accent-5"];

const state = { file: null, data: null, dirty: false, activeView: "overview" };

const $ = (s) => document.querySelector(s);
const fileSelect = $("#fileSelect");
const saveBtn = $("#saveBtn");
const statusEl = $("#status");
const tabsEl = $("#tabs");
const contentEl = $("#content");

// --- API ---

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadFileList() {
  const files = await api("/api/taxonomies");
  fileSelect.innerHTML = '<option value="">Select file...</option>';
  files.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    fileSelect.appendChild(opt);
  });
}

async function loadTaxonomy(filename) {
  state.data = await api(`/api/taxonomy/${filename}`);
  state.file = filename;
  setDirty(false);
  state.activeView = "overview";
  render();
}

async function saveTaxonomy() {
  if (!state.file || !state.data) return;
  syncFromDOM();
  await api(`/api/taxonomy/${state.file}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.data),
  });
  setDirty(false);
  statusEl.textContent = "Saved";
  statusEl.className = "saved";
}

// --- Dirty state ---

function setDirty(dirty) {
  state.dirty = dirty;
  saveBtn.disabled = !dirty;
  if (dirty) {
    statusEl.textContent = "Unsaved changes";
    statusEl.className = "dirty";
  } else {
    statusEl.textContent = "";
    statusEl.className = "";
  }
}

function markDirty() {
  if (!state.dirty) setDirty(true);
}

// --- Sync DOM -> state.data ---

function syncFromDOM() {
  if (!state.data) return;
  // Only sync the currently visible taxonomy detail view
  if (state.activeView === "overview" || state.activeView === "graph") return;
  const ai = parseInt(state.activeView.replace("tax-", ""));
  const axisEl = contentEl.querySelector(`.view[data-ai="${ai}"]`);
  if (!axisEl) return;
  const axis = state.data[ai];
  if (!axis) return;
  axis.name = axisEl.querySelector(".axis-name")?.value ?? axis.name;
  axis.description = axisEl.querySelector(".axis-desc")?.value ?? axis.description;
  axisEl.querySelectorAll(".item-card").forEach((itemEl, ii) => {
    const item = axis.items[ii];
    if (!item) return;
    item.name = itemEl.querySelector(".item-name")?.value ?? item.name;
    item.description = itemEl.querySelector(".item-desc")?.value ?? item.description;
  });
}

// --- Navigation ---

function activate(viewId) {
  syncFromDOM();
  state.activeView = viewId;
  tabsEl.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.target === viewId));
  contentEl.querySelectorAll(".view").forEach((v) => v.classList.toggle("visible", v.id === viewId));
}

// --- Render ---

function render() {
  if (!state.data) {
    tabsEl.innerHTML = "";
    contentEl.innerHTML =
      '<p style="text-align:center;color:var(--text-dim);padding:4rem">Select a taxonomy file to begin editing.</p>';
    return;
  }

  // Build tabs
  let tabsHTML = `<button class="tab${state.activeView === "overview" ? " active" : ""}" data-target="overview">Overview</button>`;
  tabsHTML += `<button class="tab${state.activeView === "graph" ? " active" : ""}" data-target="graph">Graph</button>`;
  state.data.forEach((t, i) => {
    const short = t.name.match(/\(([^)]+)\)/)?.[1] || t.name;
    const id = `tax-${i}`;
    tabsHTML += `<button class="tab${state.activeView === id ? " active" : ""}" data-target="${id}">${esc(short)}</button>`;
  });
  tabsEl.innerHTML = tabsHTML;

  // Build content: overview + one view per taxonomy
  let html = `<div class="view${state.activeView === "overview" ? " visible" : ""}" id="overview">${renderOverview()}</div>`;
  html += `<div class="view${state.activeView === "graph" ? " visible" : ""}" id="graph">${renderGraphView()}</div>`;
  state.data.forEach((axis, ai) => {
    const id = `tax-${ai}`;
    html += `<div class="view${state.activeView === id ? " visible" : ""} ${ACCENTS[ai % 5]}" id="${id}" data-ai="${ai}">${renderTaxonomy(axis, ai)}</div>`;
  });
  contentEl.innerHTML = html;
}

function renderOverview() {
  let html = '<div class="overview-grid">';
  state.data.forEach((t, i) => {
    html += `<div class="overview-card ${ACCENTS[i % 5]}" data-goto="tax-${i}">
      <h3>${esc(t.name)}</h3>
      <p>${esc(t.description)}</p>
      <div class="sub-items">
        ${t.items.map((it) => `<span class="sub-item">${esc(it.name)}</span>`).join("")}
      </div>
    </div>`;
  });
  html += "</div>";
  html += '<button class="btn-add-axis" data-add-axis>+ Add Taxonomy Axis</button>';
  return html;
}

function renderTaxonomy(axis, ai) {
  let html = `<div class="taxonomy-header">
    <input class="axis-name" value="${esc(axis.name)}" placeholder="Axis name">
    <textarea class="axis-desc" placeholder="Axis description">${esc(axis.description)}</textarea>
  </div>
  <div class="items-grid">
    ${axis.items.map((item, ii) => renderItem(item, ai, ii)).join("")}
  </div>
  <div class="taxonomy-actions">
    <button class="btn btn-sm" data-add-item="${ai}">+ Add Item</button>
    <button class="btn btn-sm btn-danger" data-delete-axis="${ai}">Delete This Axis</button>
  </div>`;
  return html;
}

function renderItem(item, ai, ii) {
  return `<div class="item-card ${ACCENTS[ai % 5]}" data-ai="${ai}" data-ii="${ii}">
    <div class="item-card-header">
      <input class="item-name" value="${esc(item.name)}" placeholder="Item name">
      <button class="btn btn-danger" data-delete-item="${ai},${ii}">&times;</button>
    </div>
    <textarea class="desc item-desc" placeholder="Description">${esc(item.description)}</textarea>
    <div class="phrases">
      ${renderPhrases(item.phrases, ai, ii)}
      <input class="tag-input" data-add-tag="${ai},${ii},phrase" placeholder="+ phrase (Enter)">
    </div>
    <div class="hard-negative">
      <div class="hard-negative-label">Hard Negatives</div>
      <div class="hard-neg-tags">
        ${renderHardNegs(item.hard_negative, ai, ii)}
        <input class="tag-input-neg" data-add-tag="${ai},${ii},neg" placeholder="+ hard negative (Enter)">
      </div>
    </div>
  </div>`;
}

function renderPhrases(arr, ai, ii) {
  if (!arr) return "";
  return arr
    .map(
      (t, ti) =>
        `<span class="phrase"><span class="phrase-text" contenteditable data-edit-tag="${ai},${ii},phrase,${ti}">${esc(t)}</span><button class="tag-remove" data-remove-tag="${ai},${ii},phrase,${ti}">&times;</button></span>`
    )
    .join("");
}

function renderHardNegs(arr, ai, ii) {
  if (!arr) return "";
  return arr
    .map(
      (t, ti) =>
        `<span class="hard-neg-tag"><span class="hard-neg-text" contenteditable data-edit-tag="${ai},${ii},neg,${ti}">${esc(t)}</span><button class="tag-remove" data-remove-tag="${ai},${ii},neg,${ti}">&times;</button></span>`
    )
    .join("");
}

function esc(s) {
  if (s == null) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// --- Graph View ---

const GR = { W: 170, H: 76, HGAP: 20, VGAP: 56, PAD: 40 };
const GCOLORS = ['var(--accent1)', 'var(--accent2)', 'var(--accent3)', 'var(--accent4)', 'var(--accent5)'];
let graphNodeMap = {};

function buildGTree(item, ai) {
  const node = {
    id: item.chapter || item.name,
    name: item.name,
    chapter: item.chapter || '',
    description: item.description || '',
    ai,
    paperCount: item.papers ? item.papers.length : 0,
    papers: item.papers || [],
    phrases: item.phrases || [],
    hardNeg: item.hard_negative || '',
    children: (item.items || []).map((ch, i) => buildGTree(ch, ai < 0 ? i : ai))
  };
  graphNodeMap[node.id] = node;
  return node;
}

function layoutGTree(node, depth, x0) {
  node.depth = depth;
  if (!node.children.length) {
    node.x = x0;
    node.y = depth * (GR.H + GR.VGAP);
    node.sw = GR.W;
    return GR.W;
  }
  let cx = x0;
  node.children.forEach((ch, i) => {
    const w = layoutGTree(ch, depth + 1, cx);
    cx += w + (i < node.children.length - 1 ? GR.HGAP : 0);
  });
  node.sw = Math.max(GR.W, cx - x0);
  const f = node.children[0], l = node.children[node.children.length - 1];
  node.x = (f.x + l.x + GR.W) / 2 - GR.W / 2;
  node.y = depth * (GR.H + GR.VGAP);
  return node.sw;
}

function gEdges(node) {
  let s = '';
  const c = node.ai >= 0 ? GCOLORS[node.ai % 5] : 'var(--text-dim)';
  node.children.forEach(ch => {
    const x1 = node.x + GR.W / 2, y1 = node.y + GR.H;
    const x2 = ch.x + GR.W / 2, y2 = ch.y;
    const my = (y1 + y2) / 2;
    s += `<path d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}" style="stroke:${c};opacity:.4"/>`;
    s += gEdges(ch);
  });
  return s;
}

function gNodes(node) {
  const c = node.ai >= 0 ? GCOLORS[node.ai % 5] : 'var(--text)';
  const isRoot = node.ai < 0;
  const cls = isRoot ? 'graph-node graph-root' : 'graph-node';
  const badge = node.paperCount > 0
    ? `<span class="gn-badge" style="background:${c}">${node.paperCount}</span>` : '';
  let h = `<div class="${cls}" style="left:${node.x}px;top:${node.y}px;--gc:${c}" data-gnode="${esc(node.id)}">
    <span class="gn-ch">${esc(node.chapter)}</span>
    <span class="gn-name">${esc(node.name.replace(/_/g, ' '))}</span>
    <span class="gn-meta">${badge}</span>
  </div>`;
  node.children.forEach(ch => { h += gNodes(ch); });
  return h;
}

function renderGraphView() {
  if (!state.data || !state.data.length)
    return '<p style="text-align:center;color:var(--text-dim);padding:4rem">No taxonomy loaded.</p>';

  graphNodeMap = {};
  const trees = state.data.map(axis => buildGTree(axis, -1));
  let off = GR.PAD;
  trees.forEach(t => { layoutGTree(t, 0, off); off += t.sw + GR.PAD * 2; });

  let mxX = 0, mxY = 0;
  function walkBounds(n) { mxX = Math.max(mxX, n.x + GR.W); mxY = Math.max(mxY, n.y + GR.H); n.children.forEach(walkBounds); }
  trees.forEach(walkBounds);

  const W = mxX + GR.PAD, H = mxY + GR.PAD;
  let edges = '', nodes = '';
  trees.forEach(t => { edges += gEdges(t); nodes += gNodes(t); });

  return `<div class="graph-scroll">
    <div class="graph-canvas" style="width:${W}px;height:${H}px">
      <svg class="graph-svg" width="${W}" height="${H}">${edges}</svg>
      ${nodes}
    </div>
  </div>
  <div class="graph-detail" id="graphDetail"></div>`;
}

function showGraphDetail(id) {
  const n = graphNodeMap[id];
  if (!n) return;
  const el = document.getElementById('graphDetail');
  if (!el) return;
  const c = n.ai >= 0 ? GCOLORS[n.ai % 5] : 'var(--text)';
  const phrases = n.phrases.length
    ? `<div class="gd-section"><div class="gd-label">Phrases</div><div class="gd-tags">${n.phrases.map(p => `<span class="gd-tag">${esc(p)}</span>`).join('')}</div></div>` : '';
  const hn = n.hardNeg
    ? `<div class="gd-section gd-neg"><div class="gd-label">Hard Negative</div><p>${esc(typeof n.hardNeg === 'string' ? n.hardNeg : Array.isArray(n.hardNeg) ? n.hardNeg.join('; ') : '')}</p></div>` : '';
  const papers = n.papers.length
    ? `<div class="gd-section"><div class="gd-label">Papers (${n.papers.length})</div><div class="gd-papers">${n.papers.map(p => {
        if (p.arxiv) return `<a class="gd-paper" href="https://arxiv.org/abs/${p.arxiv}" target="_blank" rel="noopener">${esc(p.name)} <span class="gd-arxiv">${p.arxiv}</span></a>`;
        return `<span class="gd-paper">${esc(p.name)}</span>`;
      }).join('')}</div></div>` : '';

  el.innerHTML = `<div class="gd-top" style="--gc:${c}">
    <div><span class="gd-ch">${esc(n.chapter)}</span> <span class="gd-name">${esc(n.name.replace(/_/g, ' '))}</span></div>
    <button class="graph-detail-close">&times;</button>
  </div>
  <p class="gd-desc">${esc(n.description)}</p>
  ${phrases}${hn}${papers}`;
  el.classList.add('open');
  contentEl.querySelectorAll('.graph-node.selected').forEach(el => el.classList.remove('selected'));
  const nodeEl = contentEl.querySelector(`[data-gnode="${id}"]`);
  if (nodeEl) nodeEl.classList.add('selected');
}

// --- Helpers ---

function getTagArray(ai, ii, type) {
  const item = state.data[ai]?.items[ii];
  if (!item) return null;
  return type === "phrase" ? item.phrases : item.hard_negative;
}

function parseTagKey(key) {
  const p = key.split(",");
  return [parseInt(p[0]), parseInt(p[1]), p[2], parseInt(p[3])];
}

function reindex() {
  state.data.forEach((axis, i) => (axis.id = i + 1));
}

// --- Events ---

// File select
fileSelect.addEventListener("change", (e) => {
  if (!e.target.value) return;
  if (state.dirty && !confirm("You have unsaved changes. Discard?")) {
    fileSelect.value = state.file || "";
    return;
  }
  loadTaxonomy(e.target.value);
});

// Save
saveBtn.addEventListener("click", () => saveTaxonomy());

// Ctrl+S
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "s") {
    e.preventDefault();
    if (state.dirty) saveTaxonomy();
  }
});

// Tab clicks
tabsEl.addEventListener("click", (e) => {
  if (e.target.classList.contains("tab")) activate(e.target.dataset.target);
});

// Content clicks (delegated)
contentEl.addEventListener("click", (e) => {
  const t = e.target;

  // Graph node click -> show detail
  const gnode = t.closest('[data-gnode]');
  if (gnode) {
    showGraphDetail(gnode.dataset.gnode);
    return;
  }
  // Graph detail close
  if (t.classList.contains('graph-detail-close')) {
    const detail = document.getElementById('graphDetail');
    if (detail) {
      detail.classList.remove('open');
      contentEl.querySelectorAll('.graph-node.selected').forEach(n => n.classList.remove('selected'));
    }
    return;
  }

  // Overview card -> navigate to taxonomy
  const card = t.closest("[data-goto]");
  if (card) {
    activate(card.dataset.goto);
    return;
  }

  // Add axis
  if (t.dataset.addAxis !== undefined) {
    syncFromDOM();
    const nextId = Math.max(0, ...state.data.map((a) => a.id || 0)) + 1;
    state.data.push({ id: nextId, name: "New Taxonomy Axis", description: "", items: [] });
    reindex();
    state.activeView = `tax-${state.data.length - 1}`;
    render();
    markDirty();
    return;
  }

  // Delete axis
  if (t.dataset.deleteAxis !== undefined) {
    const ai = parseInt(t.dataset.deleteAxis);
    if (confirm(`Delete axis "${state.data[ai]?.name}"?`)) {
      syncFromDOM();
      state.data.splice(ai, 1);
      reindex();
      state.activeView = "overview";
      render();
      markDirty();
    }
    return;
  }

  // Add item
  if (t.dataset.addItem !== undefined) {
    syncFromDOM();
    const ai = parseInt(t.dataset.addItem);
    state.data[ai].items.push({ name: "New Item", description: "", phrases: [], hard_negative: [] });
    render();
    markDirty();
    return;
  }

  // Delete item
  if (t.dataset.deleteItem !== undefined) {
    syncFromDOM();
    const [ai, ii] = t.dataset.deleteItem.split(",").map(Number);
    state.data[ai].items.splice(ii, 1);
    render();
    markDirty();
    return;
  }

  // Remove tag
  if (t.dataset.removeTag !== undefined) {
    syncFromDOM();
    const [ai, ii, type, ti] = parseTagKey(t.dataset.removeTag);
    const arr = getTagArray(ai, ii, type);
    if (arr) {
      arr.splice(ti, 1);
      render();
      markDirty();
    }
    return;
  }
});

// Tag input - add on Enter
contentEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.dataset.addTag !== undefined) {
    e.preventDefault();
    const val = e.target.value.trim();
    if (!val) return;
    syncFromDOM();
    const [ai, ii, type] = e.target.dataset.addTag.split(",");
    const arr = getTagArray(parseInt(ai), parseInt(ii), type);
    if (arr) {
      arr.push(val);
      render();
      markDirty();
      const sel = type === "phrase" ? `[data-add-tag="${ai},${ii},phrase"]` : `[data-add-tag="${ai},${ii},neg"]`;
      const newInput = contentEl.querySelector(sel);
      if (newInput) newInput.focus();
    }
  }
});

// Tag edit (contenteditable blur)
contentEl.addEventListener("focusout", (e) => {
  if (e.target.dataset.editTag !== undefined) {
    const [ai, ii, type, ti] = parseTagKey(e.target.dataset.editTag);
    const arr = getTagArray(ai, ii, type);
    if (arr) {
      const newVal = e.target.textContent.trim();
      if (newVal && newVal !== arr[ti]) {
        arr[ti] = newVal;
        markDirty();
      } else if (!newVal) {
        arr.splice(ti, 1);
        render();
        markDirty();
      }
    }
  }
});

// Track changes on inputs/textareas
contentEl.addEventListener("input", (e) => {
  if (e.target.matches("input, textarea")) markDirty();
});

// Warn on page leave
window.addEventListener("beforeunload", (e) => {
  if (state.dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});

// --- Init ---
loadFileList();
render();

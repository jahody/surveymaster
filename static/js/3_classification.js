/* Classification viewer */

const topicSelect = document.getElementById('topicSelect');
const fileSelect = document.getElementById('fileSelect');
const saveBtn = document.getElementById('saveBtn');
const clsStatus = document.getElementById('clsStatus');
const clsWrap = document.getElementById('clsWrap');
const clsHead = document.getElementById('clsHead');
const clsBody = document.getElementById('clsBody');

function genColor(i) {
  const hue = (i * 137.508) % 360;
  return 'hsl(' + Math.round(hue) + ',58%,56%)';
}

let allData = null;         // {papers, models, axes}
let paperTitles = {};       // arxiv → title
let gtMap = {};             // arxiv → GT chapter (for currently selected taxonomy)
let gtStems = [];           // taxonomy stems that have a GT file
let chapterColorMap = {};   // top-level chapter string → hex color
let allChapters = [];       // [{chapter, name}, ...] for manual assignment dropdown
let manualMap = {};         // arxiv → manually assigned chapter
let isDirty = false;        // unsaved manual changes

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function api(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function setStatus(msg, isError) {
  clsWrap.style.display = 'none';
  clsStatus.style.display = '';
  clsStatus.style.color = isError ? 'var(--negative)' : '';
  clsStatus.textContent = msg;
}

function getAxis() {
  if (fileSelect && fileSelect.value) {
    return fileSelect.value.replace('.json', '');
  }
  return allData && allData.axes.length ? allData.axes[0] : '';
}

function axisHasGT(axis) {
  return gtStems.includes(axis);
}

async function buildChapterColorMap(topic, axis) {
  chapterColorMap = {};
  allChapters = [];
  try {
    const tax = await api(`/api/paper-folder/${encodeURIComponent(topic)}/taxonomy/${encodeURIComponent(axis + '.json')}`);
    const items = Array.isArray(tax) ? tax : (tax.items || []);
    let counter = 0;
    function walk(item, isRoot) {
      const ch = String(item.chapter);
      const color = isRoot ? '#e2e4ed' : genColor(counter++);
      chapterColorMap[ch] = color;
      if (item.name) chapterColorMap[String(item.name)] = color;
      allChapters.push({ chapter: ch, name: item.name || '' });
      (item.items || []).forEach(child => walk(child, false));
    }
    items.forEach(item => walk(item, true));
  } catch (e) { /* leave empty */ }
}

async function loadManualForAxis(topic, axis) {
  try {
    manualMap = await api(`/api/paper-folder/${encodeURIComponent(topic)}/manual-gt/${encodeURIComponent(axis)}`);
  } catch (e) {
    manualMap = {};
  }
}

function chapterColor(ch) {
  const s = String(ch);
  if (chapterColorMap[s]) return chapterColorMap[s];
  const parts = s.split('.');
  for (let i = parts.length - 1; i >= 1; i--) {
    const parent = parts.slice(0, i).join('.');
    if (chapterColorMap[parent]) return chapterColorMap[parent];
  }
  return genColor(0);
}

// --- Topic → Classification runs ---

async function loadRunsAndTax() {
  const topic = topicSelect.value;
  allData = null;
  gtMap = {};
  gtStems = [];
  fileSelect.innerHTML = '';
  clsWrap.style.display = 'none';
  clsStatus.style.display = '';
  clsStatus.textContent = 'Select a topic to load classifications.';
  if (!topic) return;

  try {
    const [files, stems, runs] = await Promise.all([
      api(`/api/paper-folder/${encodeURIComponent(topic)}/taxonomies`),
      api(`/api/paper-folder/${encodeURIComponent(topic)}/taxonomies-with-gt`).catch(() => []),
      api(`/api/paper-folder/${encodeURIComponent(topic)}/output-folders`),
    ]);

    gtStems = stems || [];

    fileSelect.innerHTML = files.map(f => {
      const stem = f.replace('.json', '');
      const hasGT = gtStems.includes(stem);
      return `<option value="${esc(f)}">${esc(f)}${hasGT ? ' \u2713' : ''}</option>`;
    }).join('');

    if (!runs.length) {
      setStatus('No classification runs found for this topic.');
      return;
    }
    await loadData(topic, runs[0]);
  } catch (e) {
    setStatus('Error loading: ' + e.message, true);
  }
}

// --- Run → Classifications ---

async function loadData(topic, run) {
  if (!topic || !run) return;

  setStatus('Loading\u2026');

  try {
    const [data, papers] = await Promise.all([
      api(`/api/output-folder/${encodeURIComponent(run)}/topic/${encodeURIComponent(topic)}/classifications`),
      api(`/api/paper-folder/${encodeURIComponent(topic)}/papers`).catch(() => []),
    ]);

    allData = data;

    paperTitles = {};
    for (const p of papers) {
      const arxiv = (p.externalIds || {}).ArXiv || '';
      if (arxiv) paperTitles[arxiv] = p.title || '';
    }

    await Promise.all([loadGTForAxis(topic, getAxis()), buildChapterColorMap(topic, getAxis()), loadManualForAxis(topic, getAxis())]);
    isDirty = false;
    if (saveBtn) saveBtn.disabled = true;
    render();
  } catch (e) {
    setStatus('Error loading classifications: ' + e.message, true);
  }
}

// --- Load GT for selected taxonomy axis ---

async function loadGTForAxis(topic, axis) {
  if (!axisHasGT(axis)) {
    gtMap = {};
    return;
  }
  try {
    gtMap = await api(`/api/paper-folder/${encodeURIComponent(topic)}/taxonomy-gt/${encodeURIComponent(axis)}`);
  } catch (e) {
    gtMap = {};
  }
}

// --- Render table ---

function render() {
  if (!allData) return;
  const axis = getAxis();
  const { papers, models } = allData;
  const arxivIds = Object.keys(papers).sort();

  if (!arxivIds.length) {
    setStatus('No classification data found.');
    return;
  }

  clsStatus.style.display = 'none';
  clsWrap.style.display = '';

  const hasGT = Object.keys(gtMap).length > 0;

  // Header
  let headHtml = '<tr>';
  headHtml += '<th>Paper</th>';
  models.forEach(m => { headHtml += `<th title="${esc(m)}">${esc(m)}</th>`; });
  headHtml += '<th class="th-consensus">Consensus</th>';
  headHtml += '<th class="th-manual">Manual</th>';
  if (hasGT) {
    headHtml += '<th class="th-gt">GT</th>';
    headHtml += '<th class="th-match"></th>';
  }
  headHtml += '</tr>';
  clsHead.innerHTML = headHtml;

  // Body
  let bodyHtml = '';
  let consensusCorrect = 0, consensusTotal = 0;
  const modelCorrect = {};
  const modelTotal = {};
  for (const m of models) {
    modelCorrect[m] = 0;
    modelTotal[m] = 0;
  }

  for (const arxiv of arxivIds) {
    const p = papers[arxiv];
    const title = paperTitles[arxiv] || '';

    bodyHtml += '<tr>';

    // Paper cell
    bodyHtml += `<td>
      <span class="paper-arxiv"><a href="https://arxiv.org/abs/${esc(arxiv)}" target="_blank" rel="noopener">${esc(arxiv)}</a></span>
      ${title ? `<span class="paper-title" title="${esc(title)}">${esc(title)}</span>` : ''}
    </td>`;

    // Model cells
    for (const m of models) {
      const preds = (p.models[m] || {})[axis];
      if (!preds || !preds.length) {
        bodyHtml += '<td><span class="cls-na">\u2014</span></td>';
      } else {
        if (hasGT && gtMap[arxiv]) {
          modelTotal[m]++;
          if (preds[0].chapter === gtMap[arxiv]) modelCorrect[m]++;
        }
        bodyHtml += '<td><div class="cls-preds">';
        preds.slice(0, 3).forEach((pred, i) => {
          const rank = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : 'rank-3';
          const conf = pred.confidence != null ? `<span class="cls-conf">${pred.confidence}%</span>` : '';
          bodyHtml += `<span class="cls-badge ${rank}" style="--gc:${chapterColor(pred.chapter)}" title="${esc(pred.chapter)}">${esc(pred.chapter)}${conf}</span>`;
        });
        bodyHtml += '</div></td>';
      }
    }

    // Consensus cell
    const cons = (p.consensus || {})[axis];
    if (!cons || !cons.chapter) {
      bodyHtml += '<td><span class="cls-na">\u2014</span></td>';
    } else {
      bodyHtml += `<td><div class="cls-preds">
        <span class="cls-badge consensus" style="--gc:${chapterColor(cons.chapter)}" title="${esc(cons.chapter)}">${esc(cons.chapter)}</span>
      </div></td>`;
    }

    // Manual assignment cell
    const currentManual = manualMap[arxiv] || '';
    let opts = `<option value="">— none —</option>`;
    for (const ch of allChapters) {
      const sel = ch.chapter === currentManual ? ' selected' : '';
      opts += `<option value="${esc(ch.chapter)}"${sel}>${esc(ch.chapter)}${ch.name ? ' · ' + esc(ch.name) : ''}</option>`;
    }
    bodyHtml += `<td class="manual-cell${currentManual ? ' has-value' : ''}"><select class="manual-select" data-arxiv="${esc(arxiv)}">${opts}</select></td>`;

    // GT + match cells
    if (hasGT) {
      const gtChapter = gtMap[arxiv];
      if (gtChapter) {
        bodyHtml += `<td><span class="cls-badge gt">${esc(gtChapter)}</span></td>`;
        if (cons && cons.chapter) {
          consensusTotal++;
          const match = cons.chapter === gtChapter;
          if (match) consensusCorrect++;
          bodyHtml += `<td><span class="cls-match ${match ? 'match-yes' : 'match-no'}">${match ? '\u2713' : '\u2717'}</span></td>`;
        } else {
          bodyHtml += '<td><span class="cls-na">\u2014</span></td>';
        }
      } else {
        bodyHtml += '<td><span class="cls-na">\u2014</span></td>';
        bodyHtml += '<td></td>';
      }
    }

    bodyHtml += '</tr>';
  }
  clsBody.innerHTML = bodyHtml;

  // Footer: accuracy
  const clsFoot = document.getElementById('clsFoot');
  if (hasGT && (consensusTotal > 0 || models.some(m => modelTotal[m] > 0))) {
    let footHtml = `<tr>
      <td class="foot-label">Accuracy</td>`;
      
    for (const m of models) {
      if (modelTotal[m] > 0) {
        const pct = Math.round((modelCorrect[m] / modelTotal[m]) * 100);
        footHtml += `<td class="foot-acc">${modelCorrect[m]}/${modelTotal[m]} &nbsp;<strong>${pct}%</strong></td>`;
      } else {
        footHtml += `<td></td>`;
      }
    }
    
    if (consensusTotal > 0) {
      const pct = Math.round((consensusCorrect / consensusTotal) * 100);
      footHtml += `<td class="foot-acc">${consensusCorrect}/${consensusTotal} &nbsp;<strong>${pct}%</strong></td>`;
    } else {
      footHtml += `<td></td>`;
    }
    
    footHtml += `<td></td><td class="foot-gt"></td><td></td></tr>`;
    clsFoot.innerHTML = footHtml;
  } else {
    clsFoot.innerHTML = '';
  }
}

// --- Events ---

document.addEventListener('topicChanged', loadRunsAndTax);

fileSelect.addEventListener('change', async function () {
  if (!allData) return;
  const topic = topicSelect.value;
  await Promise.all([loadGTForAxis(topic, getAxis()), buildChapterColorMap(topic, getAxis()), loadManualForAxis(topic, getAxis())]);
  isDirty = false;
  if (saveBtn) saveBtn.disabled = true;
  render();
});

clsBody.addEventListener('change', function (e) {
  const sel = e.target.closest('.manual-select');
  if (!sel) return;
  const arxiv = sel.dataset.arxiv;
  const val = sel.value;
  if (val) {
    manualMap[arxiv] = val;
  } else {
    delete manualMap[arxiv];
  }
  const cell = sel.closest('td');
  if (cell) {
    cell.classList.toggle('has-value', !!val);
  }
  isDirty = true;
  if (saveBtn) saveBtn.disabled = false;
});

if (saveBtn) {
  saveBtn.addEventListener('click', async function () {
    const topic = topicSelect.value;
    const axis = getAxis();
    if (!topic || !axis) return;
    saveBtn.disabled = true;
    try {
      await fetch(`/api/paper-folder/${encodeURIComponent(topic)}/manual-gt/${encodeURIComponent(axis)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(manualMap),
      });
      isDirty = false;
    } catch (e) {
      saveBtn.disabled = false;
    }
  });
}

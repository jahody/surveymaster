/* Evaluation metrics viewer */

const evalStatus = document.getElementById('evalStatus');
const evalWrap = document.getElementById('evalWrap');

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function shortModel(name) {
  return name.split('/').pop();
}

function kappaClass(k) {
  if (k >= 0.8) return 'kappa-excellent';
  if (k >= 0.6) return 'kappa-good';
  if (k >= 0.4) return 'kappa-moderate';
  if (k >= 0.2) return 'kappa-fair';
  return 'kappa-poor';
}

function pct(v) {
  return (v * 100).toFixed(1) + '%';
}

function setStatus(msg, isError) {
  evalWrap.style.display = 'none';
  evalStatus.style.display = '';
  evalStatus.style.color = isError ? 'var(--negative)' : '';
  evalStatus.textContent = msg;
}

async function loadEvaluation(topic) {
  if (!topic) { setStatus('Select a topic to load evaluation metrics.'); return; }
  setStatus('Loading evaluation data\u2026');

  let data;
  try {
    const r = await fetch(`/api/paper-folder/${encodeURIComponent(topic)}/evaluation`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      setStatus(err.error || `Error ${r.status}`, true);
      return;
    }
    data = await r.json();
  } catch (e) {
    setStatus('Failed to load: ' + e.message, true);
    return;
  }

  evalStatus.style.display = 'none';
  evalWrap.style.display = '';

  const v = data.verdicts || {};

  renderSummary(data, v);
  renderFleiss(data.fleiss_kappa);
  renderVerdict('m1Verdict', v.m1);
  renderPairwise(data.pairwise_agreement);
  renderVerdict('m2Verdict', v.m2);
  renderConsensus(data.consensus_strength);
  renderVerdict('m3Verdict', v.m3);
  renderConfidenceAgreement(data.confidence_agreement);
  renderVerdict('m4Verdict', v.m4);
  renderDeviation(data.model_deviation);
  renderVerdict('m5Verdict', v.m5);
  renderDifficulty(data.taxonomy_difficulty);
  renderVerdict('m6Verdict', v.m6);
  renderCoverage(data.coverage_analysis);
  renderVerdict('m7Verdict', v.m7);
  renderIndependence(data.taxonomy_independence);
  renderVerdict('m8Verdict', v.m8);
  renderPlots(topic);
}

/* --- Verdict banner --- */

function renderVerdict(elementId, verdict) {
  const el = document.getElementById(elementId);
  if (!el || !verdict) { if (el) el.innerHTML = ''; return; }
  const cls = verdict.passed ? 'verdict-pass' : 'verdict-fail';
  const label = verdict.passed ? 'PASS' : 'NEEDS ATTENTION';
  let html = `<div class="${cls}"><strong>${label}</strong> &mdash; ${esc(verdict.detail)}`;
  if (verdict.action) {
    html += `<div class="verdict-action">${esc(verdict.action)}</div>`;
  }
  html += `</div>`;
  el.innerHTML = html;
}

/* --- Summary cards --- */

function renderSummary(data, verdicts) {
  const m = data.metadata;
  const fk = data.fleiss_kappa;

  // Health score from verdicts
  const vList = Object.values(verdicts);
  const passed = vList.filter(v => v.passed).length;
  const total = vList.length;
  const healthCls = total === 0 ? '' : passed === total ? 'kappa-excellent' : passed >= total - 2 ? 'kappa-moderate' : 'kappa-poor';

  const cards = [
    { label: 'Models', value: m.n_models },
    { label: 'Papers', value: m.n_papers },
    { label: 'Taxonomies', value: m.taxonomies.length },
    { label: "Fleiss' \u03ba", value: fk.overall.toFixed(3), cls: kappaClass(fk.overall) },
  ];
  if (total > 0) {
    cards.push({ label: 'Health', value: `${passed}/${total}`, cls: healthCls });
  }

  document.getElementById('summaryCards').innerHTML = cards.map(c =>
    `<div class="card ${c.cls || ''}"><div class="card-value">${esc(String(c.value))}</div><div class="card-label">${esc(c.label)}</div></div>`
  ).join('');
}

/* --- M1: Fleiss' kappa --- */

function renderFleiss(fk) {
  let html = `<div class="metric-row">`;
  html += `<div class="metric-badge ${kappaClass(fk.overall)}">Overall: <strong>${fk.overall.toFixed(3)}</strong></div>`;
  html += `</div>`;
  html += `<div class="bar-chart">`;
  for (const [tax, val] of Object.entries(fk.per_taxonomy).sort((a, b) => b[1] - a[1])) {
    const w = Math.max(0, Math.min(100, ((val + 1) / 2) * 100));
    html += `<div class="bar-row">
      <span class="bar-label">${esc(tax.replace('by_', ''))}</span>
      <div class="bar-track"><div class="bar-fill ${kappaClass(val)}" style="width:${w}%"></div></div>
      <span class="bar-value">${val.toFixed(3)}</span>
    </div>`;
  }
  html += `</div>`;
  document.getElementById('fleissContent').innerHTML = html;
}

/* --- M2: Pairwise agreement matrix --- */

function renderPairwise(pa) {
  const names = pa.model_names.map(shortModel);
  const matrix = pa.matrix;
  let html = `<table class="matrix-table"><thead><tr><th></th>`;
  names.forEach(n => html += `<th>${esc(n)}</th>`);
  html += `</tr></thead><tbody>`;
  for (let i = 0; i < names.length; i++) {
    html += `<tr><th>${esc(names[i])}</th>`;
    for (let j = 0; j < names.length; j++) {
      const v = matrix[i][j];
      const bg = i === j ? '#e8e8e8' : `hsl(${120 * v}, 60%, ${90 - 30 * v}%)`;
      html += `<td style="background:${bg}">${v.toFixed(2)}</td>`;
    }
    html += `</tr>`;
  }
  html += `</tbody></table>`;

  // Per-pair details
  html += `<details class="eval-details"><summary>Pair details</summary><table class="detail-table"><thead><tr><th>Pair</th><th>Agreement</th><th>Cohen's \u03ba</th><th>n</th></tr></thead><tbody>`;
  for (const [pair, info] of Object.entries(pa.pairs)) {
    html += `<tr><td>${esc(pair)}</td><td>${pct(info.agreement_rate)}</td><td class="${kappaClass(info.cohens_kappa)}">${info.cohens_kappa.toFixed(3)}</td><td>${info.n_compared}</td></tr>`;
  }
  html += `</tbody></table></details>`;

  document.getElementById('pairwiseContent').innerHTML = html;
}

/* --- M3: Consensus strength --- */

function renderConsensus(cs) {
  const o = cs.overall;
  let html = `<div class="metric-row">`;
  html += `<span>Mean margin: <strong>${o.mean}</strong></span>`;
  html += `<span>Median: <strong>${o.median}</strong></span>`;
  html += `<span>Weak (&lt;20): <strong>${o.weak_count}/${o.n}</strong></span>`;
  html += `<span>Strong (&ge;20): <strong>${o.strong_count}/${o.n}</strong></span>`;
  html += `</div>`;

  // Per-taxonomy
  html += `<div class="bar-chart">`;
  for (const [tax, stats] of Object.entries(cs.per_taxonomy).sort((a, b) => b[1].mean - a[1].mean)) {
    const maxMargin = 200;
    const w = Math.min(100, (stats.mean / maxMargin) * 100);
    html += `<div class="bar-row">
      <span class="bar-label">${esc(tax.replace('by_', ''))}</span>
      <div class="bar-track"><div class="bar-fill consensus-bar" style="width:${w}%"></div></div>
      <span class="bar-value">${stats.mean.toFixed(0)} (\u00b1${stats.std.toFixed(0)})</span>
    </div>`;
  }
  html += `</div>`;
  document.getElementById('consensusContent').innerHTML = html;
}

/* --- M4: Confidence-Agreement Correlation --- */

function renderConfidenceAgreement(ca) {
  const el = document.getElementById('confidenceContent');
  if (!el || !ca) { if (el) el.innerHTML = ''; return; }
  const r = ca.pearson_r;
  const cls = Math.abs(r) >= 0.5 ? 'kappa-good' : Math.abs(r) >= 0.3 ? 'kappa-moderate' : 'kappa-poor';
  let interp = 'weak';
  if (Math.abs(r) >= 0.7) interp = 'strong';
  else if (Math.abs(r) >= 0.5) interp = 'moderate';
  else if (Math.abs(r) >= 0.3) interp = 'fair';

  let html = `<div class="metric-row">`;
  html += `<div class="metric-badge ${cls}">Pearson r: <strong>${r.toFixed(3)}</strong> (${interp})</div>`;
  html += `<span>${ca.n_points} data points</span>`;
  html += `</div>`;
  el.innerHTML = html;
}

/* --- M5: Model deviation --- */

function renderDeviation(md) {
  const models = Object.entries(md.models).sort((a, b) => a[1].deviation_rate - b[1].deviation_rate);
  let html = `<div class="bar-chart">`;
  for (const [model, stats] of models) {
    const isOutlier = model === md.outlier_model;
    const w = Math.min(100, stats.deviation_rate * 100 * 2);
    html += `<div class="bar-row">
      <span class="bar-label">${esc(shortModel(model))}${isOutlier ? ' <span class="outlier-badge">outlier</span>' : ''}</span>
      <div class="bar-track"><div class="bar-fill ${isOutlier ? 'deviation-outlier' : 'deviation-bar'}" style="width:${w}%"></div></div>
      <span class="bar-value">${pct(stats.deviation_rate)} (${stats.deviations}/${stats.total})</span>
    </div>`;
  }
  html += `</div>`;
  document.getElementById('deviationContent').innerHTML = html;
}

/* --- M6: Taxonomy difficulty --- */

function renderDifficulty(td) {
  const taxs = Object.entries(td.taxonomies).sort((a, b) => b[1].unanimous_rate - a[1].unanimous_rate);
  let html = `<table class="detail-table"><thead><tr><th>Taxonomy</th><th>Unanimous</th><th>Majority</th><th>Avg Confidence</th><th>Entropy</th></tr></thead><tbody>`;
  for (const [tax, stats] of taxs) {
    const label = tax === td.easiest ? ' <span class="easy-badge">easiest</span>' : tax === td.hardest ? ' <span class="hard-badge">hardest</span>' : '';
    html += `<tr>
      <td>${esc(tax.replace('by_', ''))}${label}</td>
      <td>${pct(stats.unanimous_rate)}</td>
      <td>${pct(stats.majority_rate)}</td>
      <td>${stats.mean_top1_confidence.toFixed(0)}</td>
      <td>${stats.entropy.toFixed(2)}</td>
    </tr>`;
  }
  html += `</tbody></table>`;
  document.getElementById('difficultyContent').innerHTML = html;
}

/* --- M7: Coverage Analysis --- */

function renderCoverage(coverage) {
  const el = document.getElementById('coverageContent');
  if (!el || !coverage || Object.keys(coverage).length === 0) {
    if (el) el.innerHTML = '<p class="metric-desc">No coverage data available.</p>';
    return;
  }

  let html = '';
  for (const [tax, info] of Object.entries(coverage).sort()) {
    const status = info.dead_count === 0 && info.fat_count === 0 ? 'balanced' : 'imbalanced';
    const statusCls = status === 'balanced' ? 'kappa-good' : 'kappa-poor';
    html += `<div class="coverage-tax">`;
    html += `<div class="metric-row">`;
    html += `<strong>${esc(tax.replace('by_', ''))}</strong>`;
    html += `<span class="${statusCls}">${status}</span>`;
    html += `<span>Leaves: ${info.n_leaves}</span>`;
    html += `<span>Dead: ${info.dead_count}</span>`;
    html += `<span>Fat: ${info.fat_count}</span>`;
    html += `<span>Gini: ${info.gini.toFixed(3)}</span>`;
    html += `<span>Entropy: ${info.normalized_entropy.toFixed(3)}</span>`;
    html += `</div>`;

    // Per-leaf table
    const leaves = Object.entries(info.per_leaf || {}).sort((a, b) => b[1].count - a[1].count);
    if (leaves.length > 0) {
      html += `<details class="eval-details"><summary>Leaf details (${leaves.length} leaves)</summary>`;
      html += `<table class="detail-table"><thead><tr><th>Chapter</th><th>Name</th><th>Papers</th><th>Status</th></tr></thead><tbody>`;
      for (const [ch, leaf] of leaves) {
        const statusLabel = leaf.dead ? '<span class="coverage-dead">dead</span>'
                          : leaf.fat ? '<span class="coverage-fat">fat</span>'
                          : 'ok';
        html += `<tr>
          <td>${esc(ch)}</td>
          <td>${esc(leaf.name)}</td>
          <td>${leaf.count}</td>
          <td>${statusLabel}</td>
        </tr>`;
      }
      html += `</tbody></table></details>`;
    }
    html += `</div>`;
  }
  el.innerHTML = html;
}

/* --- M8: Taxonomy Independence --- */

function renderIndependence(ti) {
  const el = document.getElementById('independenceContent');
  if (!el || !ti || !ti.pairs || Object.keys(ti.pairs).length === 0) {
    if (el) el.innerHTML = '<p class="metric-desc">Not enough taxonomies to compute independence.</p>';
    return;
  }

  const names = (ti.taxonomy_names || []).map(t => t.replace('by_', ''));
  const mat = ti.matrix || [];
  let html = '';

  // Matrix
  if (names.length >= 2 && mat.length >= 2) {
    html += `<table class="matrix-table"><thead><tr><th></th>`;
    names.forEach(n => html += `<th>${esc(n)}</th>`);
    html += `</tr></thead><tbody>`;
    for (let i = 0; i < names.length; i++) {
      html += `<tr><th>${esc(names[i])}</th>`;
      for (let j = 0; j < names.length; j++) {
        const v = mat[i][j];
        const bg = i === j ? '#e8e8e8' : `hsl(${120 * (1 - v)}, 60%, ${90 - 30 * (1 - v)}%)`;
        html += `<td style="background:${bg}">${v.toFixed(2)}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;
  }

  // Pair list
  const pairs = Object.entries(ti.pairs).sort((a, b) => b[1].cramers_v - a[1].cramers_v);
  html += `<details class="eval-details"><summary>Pair details</summary>`;
  html += `<table class="detail-table"><thead><tr><th>Pair</th><th>Cramer's V</th><th>Status</th></tr></thead><tbody>`;
  for (const [pair, info] of pairs) {
    const flag = info.redundant ? '<span class="outlier-badge">redundant</span>' : 'independent';
    html += `<tr><td>${esc(pair)}</td><td>${info.cramers_v.toFixed(4)}</td><td>${flag}</td></tr>`;
  }
  html += `</tbody></table></details>`;

  el.innerHTML = html;
}

/* --- Plots --- */

function renderPlots(topic) {
  const plotNames = [
    'agreement_heatmap.png',
    'consensus_strength.png',
    'confidence_agreement.png',
    'taxonomy_difficulty.png',
    'model_deviation.png',
    'taxonomy_independence.png',
  ];
  const base = `/api/paper-folder/${encodeURIComponent(topic)}/evaluation/plots`;
  document.getElementById('plotsGrid').innerHTML = plotNames.map(name =>
    `<img src="${base}/${name}" alt="${esc(name)}" onerror="this.style.display='none'">`
  ).join('');
}

/* --- Event wiring --- */

document.addEventListener('topicChanged', function (e) {
  loadEvaluation(e.detail);
});

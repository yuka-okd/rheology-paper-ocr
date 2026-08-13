const state = { runs: [], detail: null, selectedRunId: null, filter: 'all', selectedRow: null, poll: null };
const $ = (selector) => document.querySelector(selector);
const apiKeyStorageKey = 'rheology-evidence.openrouter-api-key';

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || 'Request failed');
  return response.headers.get('content-type')?.includes('application/json') ? response.json() : response;
}

function escape(value) { const div = document.createElement('div'); div.textContent = value ?? ''; return div.innerHTML; }
function effectiveDecision(row) { return row.reviewer_decision || row.decision || 'needs_review'; }
function badge(decision) { return `<span class="badge ${escape(decision)}">${escape(decision.replace('_', ' '))}</span>`; }
function updateFileCount(files) { const count = files.length; $('#file-count').textContent = count ? `${count} file${count === 1 ? '' : 's'} selected` : 'No files selected'; }
function progressLabel(status) { return ({ started: 'Processing', queued: 'Queued', completed: 'Completed', completed_no_findings: 'No findings', failed: 'Failed', blocked_insufficient_credits: 'Blocked' }[status] || status.replaceAll('_', ' ')); }

async function loadRuns() {
  state.runs = await api('/api/runs');
  if (state.selectedRunId && !state.runs.some(run => run.id === state.selectedRunId)) state.selectedRunId = null;
  renderRuns();
  if (!state.selectedRunId && state.runs[0]) await selectRun(state.runs[0].id);
  if (!state.runs.length) showEmpty();
}

function renderRuns() {
  $('#run-list').innerHTML = state.runs.map(run => `<div class="run-entry ${run.id === state.selectedRunId ? 'active' : ''}"><button class="run-item" data-run="${run.id}"><strong>${escape(run.name)}</strong><span>${run.completed_papers}/${run.paper_count} papers · ${escape(run.status)}</span></button><button class="delete-run" data-delete-run="${run.id}" ${run.status === 'running' ? 'disabled' : ''} title="${run.status === 'running' ? 'Cannot delete an active run' : 'Delete run'}">Delete</button></div>`).join('');
  document.querySelectorAll('[data-run]').forEach(button => button.addEventListener('click', () => selectRun(button.dataset.run)));
  document.querySelectorAll('[data-delete-run]').forEach(button => button.addEventListener('click', () => deleteRun(button.dataset.deleteRun)));
}

async function selectRun(id) {
  state.selectedRunId = id;
  state.detail = await api(`/api/runs/${id}`);
  state.selectedRow = null;
  renderRuns(); renderDetail();
  clearInterval(state.poll);
  if (state.detail.run.status === 'running') state.poll = setInterval(() => selectRun(id), 2000);
}

function showEmpty() { $('#empty-state').classList.remove('hidden'); $('#run-view').classList.add('hidden'); $('#actions').classList.add('hidden'); $('#run-title').textContent = 'Extraction runs'; $('#run-subtitle').textContent = 'Choose a run or start a new extraction'; }

function renderDetail() {
  const { run, results, review_queue, progress } = state.detail;
  $('#empty-state').classList.add('hidden'); $('#run-view').classList.remove('hidden'); $('#actions').classList.remove('hidden');
  $('#run-title').textContent = run.name; $('#run-subtitle').textContent = `${run.completed_papers}/${run.paper_count} papers completed`;
  $('#csv-export').href = state.detail.export_csv_url; $('#print-export').href = state.detail.print_url;
  const status = $('#status-banner'); status.className = 'status-banner';
  status.textContent = run.error || ({ ready: 'Ready to extract.', running: 'Extraction is running locally. This view refreshes automatically.', completed: 'Extraction completed. Review rows before export.', blocked: 'Extraction paused because provider credit is exhausted.', failed: 'Extraction failed. Inspect the run details.' }[run.status] || run.status);
  if (['blocked', 'failed'].includes(run.status)) status.classList.add('error'); else if (run.status === 'running' || review_queue.length) status.classList.add('warning');
  renderProgress(run, progress);
  const decisions = results.map(effectiveDecision);
  $('#metrics').innerHTML = [
    ['Extracted rows', results.length], ['Accepted', decisions.filter(v => v === 'accepted').length], ['Needs review', decisions.filter(v => v === 'needs_review').length], ['Review queue', review_queue.length]
  ].map(([label,value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join('');
  renderTable();
}

function renderProgress(run, progress) {
  const panel = $('#run-progress');
  if (run.status !== 'running') { panel.classList.add('hidden'); return; }
  const current = progress.papers.find(paper => paper.status === 'started');
  const percent = progress.total ? ((progress.completed / progress.total) * 100) : 0;
  panel.classList.remove('hidden');
  $('#progress-title').textContent = current ? `Processing ${current.name}` : 'Preparing extraction queue';
  $('#progress-summary').textContent = `${progress.completed}/${progress.total} complete · ${progress.queued} queued`;
  $('#progress-fill').style.width = `${percent}%`;
  $('#progress-list').innerHTML = progress.papers.map(paper => `<li class="progress-paper ${escape(paper.status)}"><span class="paper-state" aria-hidden="true"></span><span class="paper-name">${escape(paper.name)}</span><span class="paper-status">${escape(progressLabel(paper.status))}</span></li>`).join('');
}

function renderTable() {
  const rows = state.detail.results.filter(row => state.filter === 'all' || effectiveDecision(row) === state.filter);
  $('#table-count').textContent = `${rows.length} rows`;
  $('#result-table').innerHTML = rows.map((row, index) => `<tr data-index="${index}">
    <td><div class="cell-title">${escape(row.paper_id)}</div><div class="cell-sub">${escape(row.figure_id || 'Unlabelled figure')}</div></td>
    <td><div class="cell-title">${escape(row.curve_legend_text || row.curve_id)}</div><div class="cell-sub">${escape(row.x_axis_label || 'axis unclear')} → ${escape(row.y_axis_label || 'axis unclear')}</div></td>
    <td><div class="cell-title">${escape(row.sample_display_name || 'Unresolved')}</div><div class="cell-sub">${escape(row.sample_composition || '')}</div></td>
    <td>${escape(row.fibre_outcome)}</td><td><div class="cell-sub">${escape(row.fibre_evidence_text || 'No direct evidence')}</div></td><td>${badge(effectiveDecision(row))}</td>
  </tr>`).join('') || '<tr><td colspan="6" class="cell-sub">No rows in this view.</td></tr>';
  document.querySelectorAll('#result-table tr[data-index]').forEach(element => element.addEventListener('click', () => openReview(rows[Number(element.dataset.index)])));
}

function openReview(row) {
  state.selectedRow = row; $('#review-panel').classList.remove('hidden'); $('#review-title').textContent = row.curve_legend_text || row.curve_id;
  $('#review-chart').innerHTML = row.chart_url ? `<img src="${row.chart_url}" alt="Source chart">` : '<span>No chart crop available</span>';
  $('#review-details').innerHTML = [['Paper',row.paper_id],['Figure',row.figure_id],['Sample',row.sample_display_name],['Composition',row.sample_composition],['Axes',`${row.x_axis_label || '?'} / ${row.y_axis_label || '?'}`],['Outcome',row.fibre_outcome],['Model decision',row.decision],['Reviewer decision',row.reviewer_decision || 'Not reviewed']].map(([label,value]) => `<dt>${escape(label)}</dt><dd>${escape(value || '—')}</dd>`).join('');
  $('#review-evidence').textContent = row.fibre_evidence_text || 'No direct fibre evidence was extracted.';
  $('#review-warnings').innerHTML = (row.warnings || []).map(warning => `<li>${escape(warning)}</li>`).join('') || '<li>No extraction warnings.</li>';
  $('#review-note').value = row.reviewer_note || '';
}

async function saveDecision(decision) {
  if (!state.selectedRow) return;
  await api(`/api/runs/${state.selectedRunId}/decisions`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({curve_id:state.selectedRow.curve_id, figure_id:state.selectedRow.figure_id, decision, note:$('#review-note').value}) });
  await selectRun(state.selectedRunId); openReview(state.detail.results.find(row => row.curve_id === state.selectedRow.curve_id && row.figure_id === state.selectedRow.figure_id));
}

async function deleteRun(runId) {
  const run = state.runs.find(item => item.id === runId);
  if (!run || !window.confirm(`Delete "${run.name}" and all of its local files? This cannot be undone.`)) return;
  try {
    await api(`/api/runs/${runId}`, { method:'DELETE' });
    if (state.selectedRunId === runId) { state.selectedRunId = null; state.detail = null; clearInterval(state.poll); }
    await loadRuns();
  } catch (err) { window.alert(err.message); }
}

async function createRun(event) {
  event.preventDefault();
  if (event.submitter?.value === 'cancel') { $('#upload-dialog').close(); return; }
  const files = $('#pdf-files').files; const error = $('#upload-error'); error.classList.add('hidden');
  if (!files.length) { error.textContent = 'Choose at least one PDF or ZIP archive.'; error.classList.remove('hidden'); return; }
  const apiKey = $('#openrouter-key').value.trim();
  if (!apiKey) { error.textContent = 'Enter an OpenRouter API key to start extraction.'; error.classList.remove('hidden'); $('#openrouter-key').focus(); return; }
  if ($('#remember-key').checked && apiKey) localStorage.setItem(apiKeyStorageKey, apiKey); else localStorage.removeItem(apiKeyStorageKey);
  const form = new FormData(); form.append('name', $('#run-name').value); [...files].forEach(file => form.append('files', file));
  $('#start-upload').disabled = true; $('#start-upload').textContent = 'Uploading…';
  try { const created = await api('/api/runs', {method:'POST', body:form}); await api(`/api/runs/${created.run.id}/start`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({api_key:apiKey || null})}); $('#upload-dialog').close(); await loadRuns(); await selectRun(created.run.id); }
  catch (err) { error.textContent = err.message; error.classList.remove('hidden'); }
  finally { $('#start-upload').disabled = false; $('#start-upload').textContent = 'Start extraction'; }
}

$('#new-run').addEventListener('click', () => $('#upload-dialog').showModal()); document.querySelector('[data-open-upload]').addEventListener('click', () => $('#upload-dialog').showModal());
$('#upload-form').addEventListener('submit', createRun); $('#pdf-files').addEventListener('change', event => updateFileCount(event.target.files));
const dropzone = $('.dropzone');
['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', event => { if (!event.dataTransfer?.files.length) return; $('#pdf-files').files = event.dataTransfer.files; updateFileCount(event.dataTransfer.files); });
const savedApiKey = localStorage.getItem(apiKeyStorageKey); if (savedApiKey) { $('#openrouter-key').value = savedApiKey; $('#remember-key').checked = true; }
$('#close-review').addEventListener('click', () => $('#review-panel').classList.add('hidden')); document.querySelectorAll('[data-decision]').forEach(button => button.addEventListener('click', () => saveDecision(button.dataset.decision)));
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => { document.querySelectorAll('.tab').forEach(item => item.classList.remove('active')); tab.classList.add('active'); state.filter = tab.dataset.filter; renderTable(); }));
loadRuns().catch(error => { console.error(error); $('#run-subtitle').textContent = error.message; });

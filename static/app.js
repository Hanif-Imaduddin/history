'use strict';

// ── Tab switching ──────────────────────────────────────────────────────────────
function initTabs(tabsEl) {
  tabsEl.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.classList.contains('disabled')) return;
      const target = btn.dataset.target;
      tabsEl.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const panel = btn.closest('.card').querySelector('.card-body');
      panel.querySelectorAll('[id^="tab-"]').forEach(t => t.classList.add('d-none'));
      panel.querySelector('#' + target).classList.remove('d-none');
    });
  });
}

initTabs(document.getElementById('left-tabs'));
initTabs(document.getElementById('right-tabs'));

// ── DOM refs ───────────────────────────────────────────────────────────────────
const statusBadge        = document.getElementById('status-badge');
const iterCounter        = document.getElementById('iter-counter');
const iterCurrent        = document.getElementById('iter-current');
const iterMax            = document.getElementById('iter-max');

const inputForm          = document.getElementById('input-form');
const feedbackPanel      = document.getElementById('feedback-panel');
const runningIndicator   = document.getElementById('running-indicator');
const runningAgent       = document.getElementById('running-current-agent');

const inpSector          = document.getElementById('inp-sector');
const inpAudience        = document.getElementById('inp-audience');
const inpPrompt          = document.getElementById('inp-prompt');
const inpIterations      = document.getElementById('inp-iterations');
const btnStart           = document.getElementById('btn-start');

const feedbackAssessment = document.getElementById('feedback-assessment');
const feedbackText       = document.getElementById('feedback-text');
const btnSubmitFeedback  = document.getElementById('btn-submit-feedback');

const tabReviewBtn       = document.getElementById('tab-review-btn');
const reviewVerdict      = document.getElementById('review-verdict');
const reviewFeedback     = document.getElementById('review-feedback');
const btnAccept          = document.getElementById('btn-accept');
const btnRevise          = document.getElementById('btn-revise');

const activityLog        = document.getElementById('activity-log');
const reportPlaceholder  = document.getElementById('report-placeholder');
const reportContent      = document.getElementById('report-content');
const tabReportBtn       = document.getElementById('tab-report-btn');
const tabSessionsBtn     = document.getElementById('tab-sessions-btn');
const sessionsList       = document.getElementById('sessions-list');
const btnRefreshSessions = document.getElementById('btn-refresh-sessions');
const btnNewSession      = document.getElementById('btn-new-session');

// ── State ──────────────────────────────────────────────────────────────────────
let lastOrchestratorFeedback = '';
let lastSynthesis            = '';
let lastEventId              = 0;
const receivedEventIds       = new Set();
let _session_running         = false;

// ── Pipeline visualization ─────────────────────────────────────────────────────
function setPipelineActive(agentKey) {
  const node = document.getElementById(`pipe-${agentKey}`);
  if (!node) return;
  document.querySelectorAll('.pipe-node.active').forEach(n => {
    n.classList.remove('active');
    n.classList.add('done');
  });
  node.classList.remove('done');
  node.classList.add('active');
}

function markPipelineDone(agentKey) {
  const node = document.getElementById(`pipe-${agentKey}`);
  if (!node) return;
  node.classList.remove('active');
  node.classList.add('done');
}

function clearPipeline() {
  document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active', 'done'));
}

// ── Utility ────────────────────────────────────────────────────────────────────
function setStatus(label, variant) {
  statusBadge.textContent = label;
  statusBadge.className = `badge bg-${variant}`;
}

function showLeftPanel(panel) {
  inputForm.classList.add('d-none');
  feedbackPanel.classList.add('d-none');
  runningIndicator.classList.add('d-none');
  panel.classList.remove('d-none');
}

function scrollLog() {
  activityLog.scrollTop = activityLog.scrollHeight;
}

function timestamp() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

const AGENT_CLASS = {
  market_scout:        'entry-market',
  strategic_architect: 'entry-strategic',
  financial_analyst:   'entry-financial',
  ethics_agent:        'entry-ethics',
  lead_orchestrator:   'entry-orchestrator',
  final_summary:       'entry-final',
};

const MSG_TYPE_LABEL = {
  ai:     'Respons Agent',
  tool:   'Hasil Tool',
  human:  'User',
  system: 'System',
};

function addLogEntry(options) {
  const { label, agentKey, statusText, statusClass, messages, badge } = options;

  const placeholder = activityLog.querySelector('p');
  if (placeholder) placeholder.remove();

  const el = document.createElement('div');
  el.className = `activity-entry ${AGENT_CLASS[agentKey] || ''}`;

  let html = `<div class="entry-agent-name">${escapeHtml(label)}`;
  if (badge) {
    html += ` <span class="badge bg-${badge.variant}" style="font-size:0.6rem;letter-spacing:0">${badge.text}</span>`;
  }
  html += `<span class="float-end fw-normal" style="font-size:0.68rem;color:var(--text-muted)">${timestamp()}</span></div>`;

  if (statusText) {
    html += `<div class="${statusClass} mb-1" style="font-size:0.78rem">${statusText}</div>`;
  }

  const hasMessages = messages && messages.length > 0;
  if (hasMessages) {
    const count = messages.length;
    html += `<button class="entry-toggle-btn" data-count="${count}">&#9660; Lihat output (${count} pesan)</button>`;
    html += `<div class="entry-message entry-message-collapsed">`;
    messages.forEach(m => {
      const msgObj  = (typeof m === 'object' && m !== null) ? m : { type: 'ai', content: String(m) };
      const typeKey = (msgObj.type || 'ai').toLowerCase();
      const typeLabel = msgObj.tool_name
        ? `Tool: ${escapeHtml(msgObj.tool_name)}`
        : (MSG_TYPE_LABEL[typeKey] || typeKey);
      const isToolMsg = typeKey === 'tool';
      html += `<div class="entry-msg-block${isToolMsg ? ' entry-msg-tool' : ''}">`;
      html += `<span class="entry-msg-type-label">${typeLabel}</span>`;
      html += `<pre class="entry-msg-content">${escapeHtml(msgObj.content || '')}</pre>`;
      html += `</div>`;
    });
    html += `</div>`;
  }

  el.innerHTML = html;

  if (hasMessages) {
    el.querySelector('.entry-toggle-btn').addEventListener('click', function () {
      const msgEl    = this.nextElementSibling;
      const collapsed = msgEl.classList.toggle('entry-message-collapsed');
      const count    = this.dataset.count;
      this.innerHTML = collapsed
        ? `&#9660; Lihat output (${count} pesan)`
        : `&#9650; Sembunyikan output`;
    });
  }

  activityLog.appendChild(el);
  scrollLog();
}

function escapeHtml(str) {
  return String(str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showToast(msg) {
  document.getElementById('toast-body').textContent = msg;
  const toast = bootstrap.Toast.getOrCreateInstance(document.getElementById('toast'), { delay: 5000 });
  toast.show();
}

function activateReportTab() { tabReportBtn.click(); }

function renderFinalReport(markdown) {
  reportPlaceholder.classList.add('d-none');
  reportContent.classList.remove('d-none');
  reportContent.innerHTML = marked.parse(markdown);
  activateReportTab();
}

function unlockReviewTab(verdict) {
  reviewVerdict.textContent = verdict || '';
  tabReviewBtn.classList.remove('disabled');
}

// ── SSE event handlers ─────────────────────────────────────────────────────────
function handleEvent(event) {
  switch (event.type) {

    case 'connected':
    case 'heartbeat':
      break;

    case 'session_started':
      setStatus('Running', 'primary');
      activityLog.innerHTML = '';
      receivedEventIds.clear();
      lastEventId = 0;
      clearPipeline();
      showLeftPanel(runningIndicator);
      iterCounter.classList.remove('d-none');
      break;

    case 'agent_started':
      setPipelineActive(event.agent);
      if (event.label) runningAgent.textContent = `${event.label} is working…`;
      break;

    case 'agent_complete':
      markPipelineDone(event.agent);
      runningAgent.textContent = '';
      addLogEntry({
        label:    event.label,
        agentKey: event.agent,
        messages: event.messages || [],
      });
      break;

    case 'orchestrator_evaluation': {
      let badgeVariant, statusClass, statusLabel, statusText;
      if (event.status === 'approved') {
        badgeVariant = 'success'; statusClass = 'entry-status-approved'; statusLabel = 'approved';
        statusText = `Decision: APPROVED  (iteration ${event.iteration})`;
      } else if (event.status === 'pending') {
        badgeVariant = 'secondary'; statusClass = 'entry-status-pending'; statusLabel = 'routing';
        statusText = `Routing to agents… (iteration ${event.iteration})`;
      } else {
        badgeVariant = 'danger'; statusClass = 'entry-status-rejected'; statusLabel = 'rejected';
        statusText = `Decision: REJECTED  (iteration ${event.iteration})`;
      }
      addLogEntry({
        label:      event.label || 'Lead Orchestrator',
        agentKey:   'lead_orchestrator',
        statusText,
        statusClass,
        messages:   event.messages && event.messages.length > 0 ? event.messages : [],
        badge:      { text: statusLabel, variant: badgeVariant },
      });
      iterCurrent.textContent        = event.iteration;
      lastOrchestratorFeedback       = event.feedback || '';
      lastSynthesis                  = event.synthesis || '';
      break;
    }

    case 'feedback_required':
      setStatus('Awaiting Feedback', 'warning');
      feedbackAssessment.textContent = event.orchestrator_feedback || '(No specific feedback provided)';
      feedbackText.value = '';
      showLeftPanel(feedbackPanel);
      document.getElementById('tab-input-btn').click();
      break;

    case 'final_result':
      setPipelineActive('final_summary');
      setStatus('Complete', 'success');
      renderFinalReport(event.content);
      unlockReviewTab(lastOrchestratorFeedback);
      showLeftPanel(inputForm);
      iterCounter.classList.add('d-none');
      setTimeout(() => markPipelineDone('final_summary'), 800);
      break;

    case 'done':
      if (statusBadge.textContent === 'Running') setStatus('Complete', 'success');
      showLeftPanel(inputForm);
      enableInputs(true);
      break;

    case 'error':
      setStatus('Error', 'danger');
      showToast(event.message || 'Unknown error');
      showLeftPanel(inputForm);
      enableInputs(true);
      iterCounter.classList.add('d-none');
      clearPipeline();
      break;
  }
}

// ── Enable / disable inputs ────────────────────────────────────────────────────
function enableInputs(enabled) {
  _session_running = !enabled;
  btnNewSession.disabled = !enabled;
  [inpSector, inpAudience, inpPrompt, inpIterations, btnStart].forEach(el => {
    el.disabled = !enabled;
  });
}

// ── SSE connection ─────────────────────────────────────────────────────────────
let evtSource = null;

function connectSSE() {
  if (evtSource) evtSource.close();
  const url  = lastEventId > 0 ? `/api/events?last_event_id=${lastEventId}` : '/api/events';
  evtSource  = new EventSource(url);
  evtSource.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data);
      const eid   = e.lastEventId ? parseInt(e.lastEventId, 10) : null;
      if (eid) {
        if (receivedEventIds.has(eid)) return;
        receivedEventIds.add(eid);
        lastEventId = eid;
      }
      handleEvent(event);
    } catch (_) {}
  };
  evtSource.onerror = () => { setTimeout(connectSSE, 3000); };
}

connectSSE();

// ── Start analysis ─────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  const sector   = inpSector.value.trim();
  const audience = inpAudience.value.trim();
  const prompt   = inpPrompt.value.trim();

  if (!sector || !audience || !prompt) {
    showToast('Please fill in all fields before starting.');
    return;
  }

  enableInputs(false);
  btnNewSession.disabled = true;
  setStatus('Starting…', 'secondary');
  iterMax.textContent     = inpIterations.value;
  iterCurrent.textContent = '0';

  try {
    const res = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sector_and_domain: sector,
        audience,
        initial_prompt:    prompt,
        max_iterations:    parseInt(inpIterations.value, 10),
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(err.detail || 'Failed to start session.');
      enableInputs(true);
      setStatus('Idle', 'secondary');
    } else {
      setStatus('Running', 'primary');
    }
  } catch (e) {
    showToast('Network error: ' + e.message);
    enableInputs(true);
    setStatus('Idle', 'secondary');
  }
});

// ── Submit mid-run feedback ────────────────────────────────────────────────────
btnSubmitFeedback.addEventListener('click', async () => {
  const fb = feedbackText.value.trim();
  if (!fb) { showToast('Please enter your feedback before continuing.'); return; }

  btnSubmitFeedback.disabled = true;
  setStatus('Resuming…', 'primary');
  showLeftPanel(runningIndicator);

  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feedback: fb }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(err.detail || 'Failed to submit feedback.');
      showLeftPanel(feedbackPanel);
      btnSubmitFeedback.disabled = false;
    }
  } catch (e) {
    showToast('Network error: ' + e.message);
    showLeftPanel(feedbackPanel);
    btnSubmitFeedback.disabled = false;
  }
});

// ── Final Review buttons ───────────────────────────────────────────────────────
btnAccept.addEventListener('click', () => {
  setStatus('Accepted', 'success');
  showToast('Report accepted. You may copy the report from the Final Report tab.');
});

btnRevise.addEventListener('click', async () => {
  const fb = reviewFeedback.value.trim();
  if (!fb) { showToast('Enter feedback to guide the revision.'); return; }
  const newPrompt     = inpPrompt.value.trim() + '\n\n[Revision request]: ' + fb;
  inpPrompt.value     = newPrompt;
  reviewFeedback.value = '';
  tabReviewBtn.classList.add('disabled');
  reportContent.innerHTML = '';
  reportPlaceholder.classList.remove('d-none');
  reportContent.classList.add('d-none');
  document.getElementById('tab-activity-btn').click();
  document.getElementById('tab-input-btn').click();
  enableInputs(true);
  btnStart.click();
});

// ── New Session ────────────────────────────────────────────────────────────────
function resetToNewSession() {
  activityLog.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;text-align:center;margin-top:2rem">No activity yet. Start an analysis to see agent logs.</p>';
  reportPlaceholder.classList.remove('d-none');
  reportContent.classList.add('d-none');
  reportContent.innerHTML  = '';
  tabReviewBtn.classList.add('disabled');
  reviewVerdict.textContent = '';
  reviewFeedback.value      = '';
  setStatus('Idle', 'secondary');
  iterCounter.classList.add('d-none');
  iterCurrent.textContent = '0';
  clearPipeline();
  showLeftPanel(inputForm);
  enableInputs(true);
  document.getElementById('tab-activity-btn').click();
  document.getElementById('tab-input-btn').click();
}

btnNewSession.addEventListener('click', () => { if (!_session_running) resetToNewSession(); });

// ── Sessions list ──────────────────────────────────────────────────────────────
function statusVariant(status) {
  if (status === 'approved') return 'success';
  if (status === 'rejected') return 'danger';
  return 'secondary';
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
           d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch (_) { return ''; }
}

async function loadSessions() {
  sessionsList.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;text-align:center;margin-top:1.5rem">Loading…</p>';
  try {
    const res = await fetch('/api/sessions');
    if (!res.ok) throw new Error();
    renderSessionList(await res.json());
  } catch (_) {
    sessionsList.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;text-align:center">Could not load sessions.</p>';
  }
}

function renderSessionList(sessions) {
  if (!sessions || !sessions.length) {
    sessionsList.innerHTML = '<p style="color:var(--text-muted);font-size:0.8rem;text-align:center;margin-top:1.5rem">No sessions found. Start your first analysis!</p>';
    return;
  }
  sessionsList.innerHTML = '';
  sessions.forEach(s => {
    const item          = document.createElement('div');
    item.className      = 'session-item';
    item.dataset.stateId = s.state_id;

    const promptPreview = escapeHtml((s.prompt || '').slice(0, 90)) + ((s.prompt || '').length > 90 ? '…' : '');
    const dateStr       = formatDate(s.created_at);
    const reportIcon    = s.has_report ? '<span class="session-has-report" title="Has final report">&#9679;</span> ' : '';

    item.innerHTML = `
      <div class="d-flex justify-content-between align-items-start mb-1">
        <span class="session-sector">${escapeHtml(s.sector || 'No sector')}</span>
        <span class="badge bg-${statusVariant(s.approval_status)}">${s.approval_status}</span>
      </div>
      <div class="session-prompt">${reportIcon}${promptPreview || '<em style="color:var(--text-muted)">No description</em>'}</div>
      <div class="session-meta">${dateStr ? dateStr + ' &middot; ' : ''}Iter ${s.iteration}/${s.max_iterations}</div>
    `;
    item.addEventListener('click', () => loadSessionDetail(s.state_id));
    sessionsList.appendChild(item);
  });
}

async function loadSessionDetail(stateId) {
  sessionsList.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
  const clicked = sessionsList.querySelector(`[data-state-id="${stateId}"]`);
  if (clicked) clicked.classList.add('active');

  try {
    const res    = await fetch(`/api/sessions/${stateId}`);
    if (!res.ok) throw new Error('Session not found');
    const detail = await res.json();

    if (detail.final_result) {
      reportPlaceholder.classList.add('d-none');
      reportContent.classList.remove('d-none');
      reportContent.innerHTML = marked.parse(detail.final_result);
      tabReportBtn.click();
    } else {
      showToast('This session has no final report yet.');
    }

    inpSector.value   = detail.sector || '';
    inpAudience.value = detail.audience || '';
    inpPrompt.value   = detail.prompt || '';
  } catch (e) {
    showToast('Could not load session: ' + e.message);
  }
}

tabSessionsBtn.addEventListener('click', loadSessions);
btnRefreshSessions.addEventListener('click', loadSessions);

// ── Restore state on page load ─────────────────────────────────────────────────
(async () => {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    if (data.is_running && !data.is_interrupted) {
      setStatus('Running', 'primary');
      showLeftPanel(runningIndicator);
      enableInputs(false);
      iterCounter.classList.remove('d-none');
      if (data.state) {
        iterCurrent.textContent = data.state.iteration || 0;
        iterMax.textContent     = data.state.max_iterations || 3;
      }
    } else if (data.is_interrupted) {
      setStatus('Awaiting Feedback', 'warning');
      enableInputs(false);
      iterCounter.classList.remove('d-none');
      if (data.state) {
        iterCurrent.textContent = data.state.iteration || 0;
        iterMax.textContent     = data.state.max_iterations || 3;
      }
      const intr             = data.interrupt_info || {};
      feedbackAssessment.textContent = intr.orchestrator_feedback || '(No specific feedback provided)';
      feedbackText.value     = '';
      showLeftPanel(feedbackPanel);
      document.getElementById('tab-input-btn').click();
    } else if (data.state && data.state.final_result) {
      renderFinalReport(data.state.final_result);
      setStatus('Complete', 'success');
      unlockReviewTab('');
    }
  } catch (_) {}
})();
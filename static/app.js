'use strict';

// ── Tab switching (shared helper) ──────────────────────────────────────────
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

// ── DOM refs ───────────────────────────────────────────────────────────────
const statusBadge       = document.getElementById('status-badge');
const iterCounter       = document.getElementById('iter-counter');
const iterCurrent       = document.getElementById('iter-current');
const iterMax           = document.getElementById('iter-max');

const inputForm         = document.getElementById('input-form');
const feedbackPanel     = document.getElementById('feedback-panel');
const runningIndicator  = document.getElementById('running-indicator');
const runningAgent      = document.getElementById('running-current-agent');

const inpSector         = document.getElementById('inp-sector');
const inpAudience       = document.getElementById('inp-audience');
const inpPrompt         = document.getElementById('inp-prompt');
const inpIterations     = document.getElementById('inp-iterations');
const btnStart          = document.getElementById('btn-start');

const feedbackAssessment = document.getElementById('feedback-assessment');
const feedbackText      = document.getElementById('feedback-text');
const btnSubmitFeedback = document.getElementById('btn-submit-feedback');

const tabReviewBtn      = document.getElementById('tab-review-btn');
const reviewVerdict     = document.getElementById('review-verdict');
const reviewFeedback    = document.getElementById('review-feedback');
const btnAccept         = document.getElementById('btn-accept');
const btnRevise         = document.getElementById('btn-revise');

const activityLog       = document.getElementById('activity-log');

const reportPlaceholder = document.getElementById('report-placeholder');
const reportContent     = document.getElementById('report-content');
const tabReportBtn      = document.getElementById('tab-report-btn');

// ── State ──────────────────────────────────────────────────────────────────
let lastOrchestratorFeedback = '';
let lastSynthesis = '';

// ── Utility ────────────────────────────────────────────────────────────────
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

function addLogEntry(options) {
  const { label, agentKey, statusText, statusClass, messages, badge } = options;

  // Remove empty-state placeholder
  const placeholder = activityLog.querySelector('p');
  if (placeholder) placeholder.remove();

  const el = document.createElement('div');
  el.className = `activity-entry ${AGENT_CLASS[agentKey] || ''}`;

  let html = `<div class="entry-agent-name">${label}`;
  if (badge) html += ` <span class="badge bg-${badge.variant} ms-1" style="font-size:0.65rem;letter-spacing:0">${badge.text}</span>`;
  html += `<span class="float-end text-secondary fw-normal" style="font-size:0.72rem">${timestamp()}</span></div>`;

  if (statusText) {
    html += `<div class="${statusClass} mb-1" style="font-size:0.8rem;font-weight:500">${statusText}</div>`;
  }

  if (messages && messages.length > 0) {
    const preview = messages[0].slice(0, 500);
    html += `<div class="entry-message">${escapeHtml(preview)}</div>`;
  }

  el.innerHTML = html;
  activityLog.appendChild(el);
  scrollLog();
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showToast(msg) {
  document.getElementById('toast-body').textContent = msg;
  const toast = bootstrap.Toast.getOrCreateInstance(document.getElementById('toast'), { delay: 5000 });
  toast.show();
}

// ── Switch right panel to report tab ──────────────────────────────────────
function activateReportTab() {
  tabReportBtn.click();
}

// ── Show the final report ──────────────────────────────────────────────────
function renderFinalReport(markdown) {
  reportPlaceholder.classList.add('d-none');
  reportContent.classList.remove('d-none');
  reportContent.innerHTML = marked.parse(markdown);
  activateReportTab();
}

// ── Enable Final Review tab ────────────────────────────────────────────────
function unlockReviewTab(verdict) {
  reviewVerdict.textContent = verdict || '';
  tabReviewBtn.classList.remove('disabled');
}

// ── SSE event handlers ─────────────────────────────────────────────────────
function handleEvent(event) {
  switch (event.type) {

    case 'connected':
    case 'heartbeat':
      break;

    case 'session_started':
      setStatus('Running', 'primary');
      activityLog.innerHTML = '';
      showLeftPanel(runningIndicator);
      iterCounter.classList.remove('d-none');
      break;

    case 'agent_complete':
      runningAgent.textContent = '';
      addLogEntry({
        label:    event.label,
        agentKey: event.agent,
        messages: event.messages || [],
      });
      runningAgent.textContent = 'Waiting for next agent…';
      break;

    case 'orchestrator_evaluation': {
      const isApproved = event.status === 'approved';
      addLogEntry({
        label:       event.label || 'Lead Orchestrator',
        agentKey:    'lead_orchestrator',
        statusText:  `Decision: ${event.status.toUpperCase()}  (iteration ${event.iteration})`,
        statusClass: isApproved ? 'entry-status-approved' : 'entry-status-rejected',
        messages:    event.feedback ? [event.feedback] : [],
        badge:       { text: event.status, variant: isApproved ? 'success' : 'danger' },
      });
      iterCurrent.textContent = event.iteration;
      lastOrchestratorFeedback = event.feedback || '';
      lastSynthesis = event.synthesis || '';
      break;
    }

    case 'feedback_required':
      setStatus('Awaiting Feedback', 'warning');
      feedbackAssessment.textContent = event.orchestrator_feedback || '(No specific feedback provided)';
      feedbackText.value = '';
      showLeftPanel(feedbackPanel);
      // Auto-switch to input tab if not already there
      document.getElementById('tab-input-btn').click();
      break;

    case 'final_result':
      setStatus('Complete', 'success');
      renderFinalReport(event.content);
      unlockReviewTab(lastOrchestratorFeedback);
      showLeftPanel(inputForm);
      iterCounter.classList.add('d-none');
      break;

    case 'done':
      if (statusBadge.textContent === 'Running') {
        setStatus('Complete', 'success');
      }
      showLeftPanel(inputForm);
      enableInputs(true);
      break;

    case 'error':
      setStatus('Error', 'danger');
      showToast(event.message || 'Unknown error');
      showLeftPanel(inputForm);
      enableInputs(true);
      iterCounter.classList.add('d-none');
      break;
  }
}

// ── Enable / disable input form ────────────────────────────────────────────
function enableInputs(enabled) {
  [inpSector, inpAudience, inpPrompt, inpIterations, btnStart].forEach(el => {
    el.disabled = !enabled;
  });
}

// ── SSE connection ─────────────────────────────────────────────────────────
let evtSource = null;

function connectSSE() {
  if (evtSource) { evtSource.close(); }
  evtSource = new EventSource('/api/events');
  evtSource.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); } catch (_) {}
  };
  evtSource.onerror = () => {
    setTimeout(connectSSE, 3000);
  };
}

connectSSE();

// ── Start analysis ─────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  const sector = inpSector.value.trim();
  const audience = inpAudience.value.trim();
  const prompt = inpPrompt.value.trim();

  if (!sector || !audience || !prompt) {
    showToast('Please fill in all fields before starting.');
    return;
  }

  enableInputs(false);
  setStatus('Starting…', 'secondary');
  iterMax.textContent = inpIterations.value;
  iterCurrent.textContent = '0';

  try {
    const res = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sector_and_domain: sector,
        audience,
        initial_prompt: prompt,
        max_iterations: parseInt(inpIterations.value, 10),
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(err.detail || 'Failed to start session.');
      enableInputs(true);
      setStatus('Idle', 'secondary');
    }
  } catch (e) {
    showToast('Network error: ' + e.message);
    enableInputs(true);
    setStatus('Idle', 'secondary');
  }
});

// ── Submit mid-run feedback ────────────────────────────────────────────────
btnSubmitFeedback.addEventListener('click', async () => {
  const fb = feedbackText.value.trim();
  if (!fb) {
    showToast('Please enter your feedback before continuing.');
    return;
  }

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

// ── Final Review buttons ───────────────────────────────────────────────────
btnAccept.addEventListener('click', () => {
  setStatus('Accepted', 'success');
  showToast('Report accepted. You may download or copy the report from the Final Report tab.');
});

btnRevise.addEventListener('click', async () => {
  const fb = reviewFeedback.value.trim();
  if (!fb) {
    showToast('Enter feedback to guide the revision.');
    return;
  }

  // Restart with the same constraints but new feedback as initial_prompt addendum
  const newPrompt = inpPrompt.value.trim() + '\n\n[Revision request]: ' + fb;
  inpPrompt.value = newPrompt;
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

// ── Restore state on page load ─────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    if (data.is_running && !data.is_interrupted) {
      setStatus('Running', 'primary');
      showLeftPanel(runningIndicator);
      enableInputs(false);
      iterCounter.classList.remove('d-none');
      if (data.state) {
        iterCurrent.textContent = data.state.iteration || 0;
        iterMax.textContent = data.state.max_iterations || 3;
      }
    } else if (data.is_interrupted) {
      setStatus('Awaiting Feedback', 'warning');
    } else if (data.state && data.state.final_result) {
      renderFinalReport(data.state.final_result);
      setStatus('Complete', 'success');
      unlockReviewTab('');
    }
  } catch (_) {}
})();

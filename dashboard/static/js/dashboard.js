/* ═══════════════════════════════════════════════════════════════
   SpamSentry Dashboard — JavaScript
═══════════════════════════════════════════════════════════════ */

"use strict";

// ── State ──────────────────────────────────────────────────────
const App = {
  trained:    false,
  training:   false,
  results:    {},
  bestModel:  null,
  pollTimer:  null,
  classChart: null,
  algoChart:  null,
};

// ── Chart.js defaults ──────────────────────────────────────────
Chart.defaults.color          = "#8896b0";
Chart.defaults.font.family    = "'Space Mono', monospace";
Chart.defaults.font.size      = 11;
Chart.defaults.borderColor    = "#1e2535";

// ── Toast ──────────────────────────────────────────────────────
function toast(msg, type = "info", duration = 3200) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className   = `toast show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = "toast"; }, duration);
}

// ── Navigation ─────────────────────────────────────────────────
function initNav() {
  const titles = {
    overview: "Overview",
    classify: "Classify Email",
    models:   "Model Performance",
    eda:      "Data Analysis",
    train:    "Training Console",
  };

  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const sec = link.dataset.section;
      document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
      document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
      link.classList.add("active");
      document.getElementById(`section-${sec}`).classList.add("active");
      document.getElementById("pageTitle").textContent = titles[sec] || sec;

      // Lazy-load figures when switching to those sections
      if (sec === "eda")    loadEdaFigures();
      if (sec === "models") loadModelFigures();
    });
  });
}

// ── Status polling ─────────────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    App.trained  = data.trained;
    App.training = data.training;

    updateGlobalStatus(data);
    updateConsoleLog(data.log || []);

    if (data.training) {
      updatePipelineStages(data.log || []);
      startPolling();
    } else {
      stopPolling();
      if (data.trained) {
        loadAll();
        document.getElementById("consoleStatus").className = "console-status done";
        document.getElementById("consoleStatus").textContent = "COMPLETE";
      }
      if (data.error) {
        toast(`Error: ${data.error}`, "error", 6000);
      }
    }
  } catch {
    // Server might not be running; ignore silently
  }
}

function startPolling() {
  if (!App.pollTimer) {
    App.pollTimer = setInterval(pollStatus, 1200);
  }
}
function stopPolling() {
  if (App.pollTimer) { clearInterval(App.pollTimer); App.pollTimer = null; }
}

function updateGlobalStatus(data) {
  const dot  = document.getElementById("globalStatus");
  const text = document.getElementById("globalStatusText");
  if (data.training) {
    dot.className  = "status-dot training";
    text.textContent = "Training…";
    document.querySelectorAll(".btn-train, #trainBtn2").forEach(b => {
      b.disabled = true; b.textContent = "Training…";
    });
  } else if (data.trained) {
    dot.className  = "status-dot ready";
    text.textContent = "Ready";
    document.querySelectorAll(".btn-train, #trainBtn2").forEach(b => {
      b.disabled = false; b.textContent = "▶ Retrain";
    });
  } else if (data.error) {
    dot.className  = "status-dot error";
    text.textContent = "Error";
    document.querySelectorAll(".btn-train, #trainBtn2").forEach(b => {
      b.disabled = false; b.textContent = "▶ Train Pipeline";
    });
  } else {
    dot.className  = "status-dot";
    text.textContent = "Not trained";
    document.querySelectorAll(".btn-train, #trainBtn2").forEach(b => {
      b.disabled = false; b.textContent = "▶ Train Pipeline";
    });
  }
}

// ── Console log ────────────────────────────────────────────────
let _lastLogLen = 0;
function updateConsoleLog(lines) {
  const el = document.getElementById("consoleLog");
  if (lines.length === _lastLogLen) return;
  _lastLogLen = lines.length;

  el.innerHTML = lines.map(line => {
    let cls = "";
    if (line.includes("✓") || line.includes("complete")) cls = "success";
    else if (line.includes("ERROR")) cls = "error";
    else if (line.includes("Training") || line.includes("Evaluating") || line.includes("Loading")) cls = "info";
    return `<div class="log-line ${cls}">${escHtml(line)}</div>`;
  }).join("") || '<div class="log-line dim">No output yet.</div>';

  el.scrollTop = el.scrollHeight;
}

function updatePipelineStages(lines) {
  const log = lines.join(" ").toLowerCase();
  const stages = [
    ["stage-load",       "Loading dataset"],
    ["stage-eda",        "Running EDA"],
    ["stage-preprocess", "Preprocessing text"],
    ["stage-features",   "Extracting TF-IDF"],
    ["stage-train",      "Training all"],
    ["stage-eval",       "Evaluating"],
  ];
  let foundActive = false;
  [...stages].reverse().forEach(([id, keyword]) => {
    const el = document.getElementById(id);
    if (!el) return;
    const done = log.includes(keyword.toLowerCase());
    if (done && !foundActive) {
      el.className = "stage active";
      el.querySelector(".stage-icon").textContent = "◉";
      foundActive = true;
    } else if (log.includes(keyword.toLowerCase()) && foundActive) {
      el.className = "stage done";
      el.querySelector(".stage-icon").textContent = "●";
    }
  });
}

// ── Train pipeline ─────────────────────────────────────────────
async function startTraining() {
  if (App.training) return;
  document.querySelectorAll(".btn-train, #trainBtn2").forEach(b => {
    b.disabled = true; b.textContent = "Training…";
  });
  document.getElementById("consoleStatus").className = "console-status running";
  document.getElementById("consoleStatus").textContent = "RUNNING";
  _lastLogLen = 0;

  // Reset stage icons
  document.querySelectorAll(".stage").forEach(s => {
    s.className = "stage";
    s.querySelector(".stage-icon").textContent = "○";
  });

  try {
    const res  = await fetch("/api/train", { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      toast("Pipeline started. This may take a minute…", "warning", 5000);
      App.training = true;
      startPolling();
    } else {
      toast(data.error || "Failed to start training.", "error");
      document.querySelectorAll(".btn-train, #trainBtn2").forEach(b => {
        b.disabled = false; b.textContent = "▶ Train Pipeline";
      });
    }
  } catch {
    toast("Could not connect to server.", "error");
  }
}

// ── Load all data after training ───────────────────────────────
async function loadAll() {
  await loadDatasetStats();
  await loadModelResults();
  loadEdaFigures();
  loadModelFigures();
}

// ── Dataset stats ──────────────────────────────────────────────
async function loadDatasetStats() {
  try {
    const res  = await fetch("/api/dataset/stats");
    if (!res.ok) return;
    const d    = await res.json();
    setValue("sv-total", d.total_samples?.toLocaleString());
    setValue("sv-spam",  d.spam_count?.toLocaleString());
    setValue("sv-ham",   d.ham_count?.toLocaleString());

    drawClassChart(d.ham_count, d.spam_count);
  } catch {}
}

// ── Model results ──────────────────────────────────────────────
async function loadModelResults() {
  try {
    const res  = await fetch("/api/models");
    if (!res.ok) return;
    const d    = await res.json();
    App.results   = d.results || {};
    App.bestModel = d.best_model;

    // Topbar chip
    if (App.bestModel) {
      document.getElementById("bestModelName").textContent =
        App.bestModel.replace(/_/g, " ");
    }

    // Best F1
    const bestF1 = App.results[App.bestModel]?.f1;
    if (bestF1 !== undefined) setValue("sv-f1", bestF1.toFixed(4));

    renderMetricsTable(App.results, App.bestModel);
    drawAlgoChart(App.results);
    populateCmSelect(Object.keys(App.results));
  } catch {}
}

// ── Metrics table ──────────────────────────────────────────────
function renderMetricsTable(results, bestModel) {
  const tbody = document.getElementById("metricsBody");
  if (!results || Object.keys(results).length === 0) return;

  const sorted = Object.entries(results).sort((a, b) => b[1].f1 - a[1].f1);
  tbody.innerHTML = sorted.map(([name, m], i) => {
    const rank = i + 1;
    const rankBadge = rank <= 3
      ? `<span class="rank-badge rank-${rank}">${rank === 1 ? "🥇" : rank === 2 ? "🥈" : "🥉"} #${rank}</span>`
      : `<span class="rank-badge rank-other">#${rank}</span>`;
    const isBest = name === bestModel ? "best-row" : "";
    const bar = (v) => `
      <div class="bar-cell">
        <div class="metric-bar" style="width:${Math.round(v * 80)}px;"></div>
        ${v.toFixed(4)}
      </div>`;
    return `
      <tr class="${isBest}">
        <td>${fmt(name)}</td>
        <td>${bar(m.accuracy)}</td>
        <td>${bar(m.precision)}</td>
        <td>${bar(m.recall)}</td>
        <td>${bar(m.f1)}</td>
        <td>${isNaN(m.auc) ? "N/A" : bar(m.auc)}</td>
        <td>${rankBadge}</td>
      </tr>`;
  }).join("");
}

// ── EDA figures ────────────────────────────────────────────────
const EDA_FIGS = [
  "class_distribution",
  "text_length_distribution",
  "word_count_distribution",
  "top_words",
];
async function loadEdaFigures() {
  for (const name of EDA_FIGS) {
    const wrap = document.getElementById(`eda-${name}`);
    if (!wrap || wrap.querySelector("img")) continue;
    try {
      const res  = await fetch(`/api/eda/${name}`);
      if (!res.ok) continue;
      const d    = await res.json();
      wrap.innerHTML = `<img src="${d.image}" alt="${name}" />`;
    } catch {}
  }
}

// ── Model figures (ROC, CM, metrics comparison) ────────────────
async function loadModelFigures() {
  // ROC curves
  const rocWrap = document.getElementById("rocFigure");
  if (rocWrap && !rocWrap.querySelector("img")) {
    try {
      const res = await fetch("/api/eda/roc_curves");
      if (res.ok) {
        const d = await res.json();
        rocWrap.innerHTML = `<img src="${d.image}" alt="ROC curves" />`;
      }
    } catch {}
  }

  // Metrics comparison
  const mcWrap = document.getElementById("metricsFigure");
  if (mcWrap && !mcWrap.querySelector("img")) {
    try {
      const res = await fetch("/api/eda/metrics_comparison");
      if (res.ok) {
        const d = await res.json();
        mcWrap.innerHTML = `<img src="${d.image}" alt="metrics comparison" />`;
      }
    } catch {}
  }
}

function populateCmSelect(names) {
  const sel = document.getElementById("cmModelSelect");
  sel.innerHTML = '<option value="">— Select model —</option>' +
    names.map(n => `<option value="${n}">${fmt(n)}</option>`).join("");
}

document.getElementById("cmModelSelect")?.addEventListener("change", async function () {
  const name = this.value;
  if (!name) return;
  const wrap = document.getElementById("cmFigure");
  wrap.innerHTML = `<div class="figure-placeholder">Loading…</div>`;
  try {
    const res = await fetch(`/api/eda/cm_${name}`);
    if (!res.ok) { wrap.innerHTML = `<div class="figure-placeholder">Figure not available</div>`; return; }
    const d = await res.json();
    wrap.innerHTML = `<img src="${d.image}" alt="Confusion matrix ${name}" />`;
  } catch {
    wrap.innerHTML = `<div class="figure-placeholder">Failed to load figure</div>`;
  }
});

// ── Charts ─────────────────────────────────────────────────────
function drawClassChart(ham, spam) {
  const ctx = document.getElementById("classChart")?.getContext("2d");
  if (!ctx) return;
  if (App.classChart) App.classChart.destroy();
  App.classChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Ham (Legitimate)", "Spam"],
      datasets: [{
        data: [ham, spam],
        backgroundColor: ["#00d4aa30", "#ff4d4d30"],
        borderColor    : ["#00d4aa",   "#ff4d4d"],
        borderWidth    : 2,
        hoverOffset    : 6,
      }],
    },
    options: {
      cutout: "68%",
      plugins: {
        legend: { position: "bottom", labels: { padding: 16, boxWidth: 12 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw.toLocaleString()}` } },
      },
    },
  });
}

function drawAlgoChart(results) {
  const ctx = document.getElementById("algoChart")?.getContext("2d");
  if (!ctx || !results) return;
  if (App.algoChart) App.algoChart.destroy();

  const sorted = Object.entries(results).sort((a, b) => b[1].f1 - a[1].f1);
  const labels  = sorted.map(([n]) => fmt(n));
  const f1s     = sorted.map(([, m]) => m.f1);
  const accs    = sorted.map(([, m]) => m.accuracy);

  App.algoChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "F1-Score", data: f1s, backgroundColor: "#f5a62370", borderColor: "#f5a623", borderWidth: 1.5, borderRadius: 4 },
        { label: "Accuracy", data: accs, backgroundColor: "#00d4aa30", borderColor: "#00d4aa", borderWidth: 1.5, borderRadius: 4 },
      ],
    },
    options: {
      scales: {
        y: { min: 0.5, max: 1.0, grid: { color: "#1e2535" } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { position: "bottom", labels: { padding: 14, boxWidth: 10 } } },
    },
  });
}

// ── Email classification ────────────────────────────────────────
document.getElementById("classifyBtn")?.addEventListener("click", classifyEmail);
document.getElementById("clearBtn")?.addEventListener("click", () => {
  document.getElementById("emailInput").value = "";
  resetVerdict();
});
document.getElementById("sampleSpamBtn")?.addEventListener("click", () => {
  document.getElementById("emailInput").value =
    "CONGRATULATIONS! You have been SELECTED as our LUCKY WINNER! " +
    "Click HERE immediately to claim your FREE prize of $5,000! " +
    "Limited time offer. Act NOW before it expires!!!";
});
document.getElementById("sampleHamBtn")?.addEventListener("click", () => {
  document.getElementById("emailInput").value =
    "Hi team, just a reminder that the project status meeting is scheduled " +
    "for Thursday at 2pm in conference room B. Please bring your progress " +
    "reports. Let me know if you have any questions.";
});

async function classifyEmail() {
  const text = document.getElementById("emailInput").value.trim();
  if (!text) { toast("Please enter email text to classify.", "warning"); return; }
  if (!App.trained) { toast("Train the pipeline first.", "warning"); return; }

  const btn = document.getElementById("classifyBtn");
  btn.disabled = true;
  btn.textContent = "Analysing…";

  try {
    const res  = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const d = await res.json();
    if (!res.ok) { toast(d.error || "Prediction failed.", "error"); return; }

    showVerdict(d);
    renderPredictionsTable(d.all_predictions);
  } catch {
    toast("Could not reach server.", "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse Email";
  }
}

function showVerdict(d) {
  document.getElementById("verdictPlaceholder").classList.add("hidden");
  const result = document.getElementById("verdictResult");
  result.classList.remove("hidden");

  const badge = document.getElementById("verdictBadge");
  badge.textContent = d.verdict.toUpperCase();
  badge.className   = `verdict-badge ${d.verdict}`;

  const conf = d.confidence != null
    ? `Confidence: ${(d.confidence * 100).toFixed(1)}%`
    : "Confidence: N/A";
  document.getElementById("verdictConf").textContent = conf;
  document.getElementById("verdictModel").textContent =
    `via ${fmt(d.best_model)} (best model)`;
}

function resetVerdict() {
  document.getElementById("verdictPlaceholder").classList.remove("hidden");
  document.getElementById("verdictResult").classList.add("hidden");
  document.getElementById("predictionsBody").innerHTML =
    '<tr><td colspan="3" class="empty-row">Run analysis to see predictions</td></tr>';
}

function renderPredictionsTable(preds) {
  const tbody = document.getElementById("predictionsBody");
  if (!preds) return;
  tbody.innerHTML = Object.entries(preds).map(([name, p]) => {
    const cls  = p.is_spam ? "tag-spam" : "tag-ham";
    const conf = p.probability != null ? `${(p.probability * 100).toFixed(1)}%` : "—";
    return `
      <tr>
        <td>${fmt(name)}</td>
        <td class="${cls}">${p.label.toUpperCase()}</td>
        <td>${conf}</td>
      </tr>`;
  }).join("");
}

// ── Utility ────────────────────────────────────────────────────
function fmt(name) {
  return (name || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
function setValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? "—";
}
function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Train button wiring ────────────────────────────────────────
document.getElementById("trainBtn")?.addEventListener("click",  startTraining);
document.getElementById("trainBtn2")?.addEventListener("click", startTraining);

// ── Init ───────────────────────────────────────────────────────
initNav();
pollStatus();           // immediate check on load
startPolling();         // keep polling (stops itself when idle)

// ── Scan Emails section ────────────────────────────────────────────────────
(function () {
  const titles = {
    overview: "Overview", classify: "Classify Email",
    models: "Model Performance", eda: "Data Analysis",
    train: "Training Console", scan: "Scan Emails"
  };

  // Patch nav so "scan" section works
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const sec = link.dataset.section;
      if (sec === "scan") {
        document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
        document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
        link.classList.add("active");
        document.getElementById("section-scan").classList.add("active");
        document.getElementById("pageTitle").textContent = "Scan Emails";
      }
    });
  });

  let _scanPollTimer = null;

  document.getElementById("scanBtn")?.addEventListener("click", async () => {
    const path = (document.getElementById("mboxPath")?.value || "").trim();
    if (!path) { toast("Please enter a file or folder path.", "warning"); return; }

    const btn = document.getElementById("scanBtn");
    btn.disabled = true;
    btn.textContent = "Scanning…";

    // Show progress area
    document.getElementById("scanProgress").style.display = "block";
    document.getElementById("scanResultsCard").style.display = "none";
    document.getElementById("scanError").style.display = "none";
    document.getElementById("scanCsvNote").style.display = "none";
    document.getElementById("scanStatus").className = "console-status running";
    document.getElementById("scanStatus").textContent = "SCANNING";
    ["scanProcessed","scanSpam","scanHam"].forEach(id => {
      document.getElementById(id).textContent = "0";
    });

    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const d = await res.json();
      if (!res.ok) {
        toast(d.error || "Scan failed to start.", "error", 6000);
        btn.disabled = false; btn.textContent = "▶ Scan Emails";
        return;
      }
      toast("Scan started. Polling for progress…", "warning", 4000);
      _scanPollTimer = setInterval(_pollScan, 1500);
    } catch {
      toast("Could not reach server.", "error");
      btn.disabled = false; btn.textContent = "▶ Scan Emails";
    }
  });

  async function _pollScan() {
    try {
      const res = await fetch("/api/scan/status");
      const d   = await res.json();

      document.getElementById("scanProcessed").textContent = d.processed?.toLocaleString() || "0";
      document.getElementById("scanSpam").textContent      = d.spam_count?.toLocaleString() || "0";
      document.getElementById("scanHam").textContent       = d.ham_count?.toLocaleString()  || "0";

      if (d.error) {
        clearInterval(_scanPollTimer);
        document.getElementById("scanStatus").className = "console-status";
        document.getElementById("scanStatus").textContent = "ERROR";
        const errEl = document.getElementById("scanError");
        errEl.textContent = "Error: " + d.error;
        errEl.style.display = "block";
        const btn = document.getElementById("scanBtn");
        btn.disabled = false; btn.textContent = "▶ Scan Emails";
        return;
      }

      if (d.done && !d.scanning) {
        clearInterval(_scanPollTimer);
        document.getElementById("scanStatus").className = "console-status done";
        document.getElementById("scanStatus").textContent = "COMPLETE";

        const btn = document.getElementById("scanBtn");
        btn.disabled = false; btn.textContent = "▶ Scan Again";

        if (d.csv_path) {
          const note = document.getElementById("scanCsvNote");
          note.textContent = "Full CSV report saved to: " + d.csv_path;
          note.style.display = "block";
        }

        toast(`Scan complete. ${d.spam_count} spam found in ${d.processed} emails.`,
              d.spam_count > 0 ? "error" : "success", 6000);

        _renderScanResults(d.results || []);
      }
    } catch {
      // server hiccup — keep polling
    }
  }

  function _renderScanResults(results) {
    const card = document.getElementById("scanResultsCard");
    const tbody = document.getElementById("scanResultsBody");

    if (!results || results.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No emails classified.</td></tr>';
      card.style.display = "block";
      return;
    }

    tbody.innerHTML = results.map(r => {
      const isSpam = r.verdict === "spam";
      const tag    = isSpam
        ? '<span class="tag-spam">SPAM</span>'
        : '<span class="tag-ham">HAM</span>';
      const conf   = r.confidence != null
        ? (r.confidence * 100).toFixed(1) + "%" : "—";
      const subj   = (r.subject || "(no subject)").substring(0, 60);
      const from   = (r.from   || "").substring(0, 35);
      const row_cls = isSpam ? 'style="background:#ff000008;"' : "";
      return `<tr ${row_cls}>
        <td>${r.index}</td>
        <td>${r.date || ""}</td>
        <td title="${r.from || ""}">${from}</td>
        <td title="${r.subject || ""}">${subj}</td>
        <td>${tag}</td>
        <td>${conf}</td>
      </tr>`;
    }).join("");

    card.style.display = "block";
  }
})();

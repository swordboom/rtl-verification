const statusBox = document.getElementById("status-box");
const predictionBox = document.getElementById("prediction-box");
const uploadBox = document.getElementById("upload-box");
const uploadResults = document.getElementById("upload-results");
const severityCanvas = document.getElementById("severity-chart");
const moduleFilter = document.getElementById("module-filter");
const apiBase = "";

let severityChart = null;
let analyticsSource = "auto";
let knownModules = new Set();

function setStatus(text) {
  statusBox.textContent = text;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function updateModuleFilterOptions(modules) {
  modules.forEach((m) => knownModules.add(m));
  const current = moduleFilter.value;
  const sorted = Array.from(knownModules).sort();
  moduleFilter.innerHTML = '<option value="">All Modules</option>';
  sorted.forEach((moduleName) => {
    const opt = document.createElement("option");
    opt.value = moduleName;
    opt.textContent = moduleName;
    moduleFilter.appendChild(opt);
  });
  moduleFilter.value = sorted.includes(current) ? current : "";
}

async function trainModel() {
  setStatus("Training model and generating synthetic dataset...");
  const res = await fetch(`${apiBase}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row_count: 12000 }),
  });
  const data = await res.json();
  setStatus(pretty(data));
  analyticsSource = "dataset";
  await loadAnalytics(analyticsSource);
}

async function predictFromLog() {
  const log = document.getElementById("log-input").value;
  predictionBox.textContent = "Scoring log...";
  const res = await fetch(`${apiBase}/predict-from-log`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log }),
  });
  const data = await res.json();
  predictionBox.textContent = pretty(data);
  analyticsSource = data.analytics_source || "uploaded";
  await loadAnalytics(analyticsSource);
}

async function uploadLogs() {
  const fileInput = document.getElementById("log-file");
  if (!fileInput.files.length) {
    uploadBox.textContent = "Select a .log or .txt file first.";
    return;
  }
  uploadBox.textContent = "Uploading and scoring logs...";
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const res = await fetch(`${apiBase}/upload-logs`, {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  uploadBox.textContent = pretty({
    total_logs: data.total_logs,
    categorized_error_summary: data.categorized_error_summary,
    unique_failures_top: (data.unique_failures || []).slice(0, 5),
  });
  renderCollapsibleUploadedJson(data.uploaded_records || []);
  analyticsSource = data.analytics_source || "uploaded";
  await loadAnalytics(analyticsSource);
}

function renderCollapsibleUploadedJson(records) {
  uploadResults.innerHTML = "";
  if (!records.length) {
    uploadResults.textContent = "No parsed records available.";
    return;
  }
  const maxRender = 200;
  const visible = records.slice(0, maxRender);
  if (records.length > maxRender) {
    const note = document.createElement("pre");
    note.textContent = `Showing ${maxRender} of ${records.length} records to keep UI responsive.`;
    uploadResults.appendChild(note);
  }
  visible.forEach((record) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const prediction = record.prediction || {};
    const parsed = record.parsed || {};
    summary.textContent = `#${record.index} | ${parsed.module_name || "Unknown"} | ${prediction.predicted_priority_label || "-"} | Score ${prediction.priority_score || "-"}`;
    const content = document.createElement("pre");
    content.textContent = pretty(record);
    details.appendChild(summary);
    details.appendChild(content);
    uploadResults.appendChild(details);
  });
}

function drawSeverityChart(distribution) {
  const labels = Object.keys(distribution);
  const values = Object.values(distribution);

  if (severityChart) {
    severityChart.destroy();
  }

  severityChart = new Chart(severityCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Failures",
          data: values,
          backgroundColor: ["#b00020", "#1f7a4f", "#f4a300"],
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

function drawPriorityHistogram(priorityDistribution) {
  const x = Object.keys(priorityDistribution);
  const y = Object.values(priorityDistribution);
  Plotly.newPlot(
    "priority-plot",
    [{ type: "bar", x, y, marker: { color: "#2a9d8f" } }],
    { margin: { t: 20 }, xaxis: { title: "Priority Label" }, yaxis: { title: "Count" } },
    { responsive: true }
  );
}

function drawCoveragePlot(coverageByModule) {
  const x = coverageByModule.map((item) => item.module_name);
  const y = coverageByModule.map((item) => item.avg_coverage_drop);
  Plotly.newPlot(
    "coverage-plot",
    [{ type: "scatter", mode: "lines+markers", x, y, marker: { color: "#264653" } }],
    { margin: { t: 20 }, xaxis: { title: "Module" }, yaxis: { title: "Avg Coverage Drop" } },
    { responsive: true }
  );
}

function drawHeatmap(heatmap) {
  Plotly.newPlot(
    "heatmap-plot",
    [
      {
        type: "heatmap",
        x: heatmap.severities,
        y: heatmap.modules,
        z: heatmap.values,
        colorscale: "Viridis",
      },
    ],
    { margin: { t: 20 }, xaxis: { title: "Severity" }, yaxis: { title: "Module" } },
    { responsive: true }
  );
}

async function loadAnalytics(source = "auto") {
  setStatus("Loading analytics...");
  const moduleName = moduleFilter.value || "";
  const res = await fetch(
    `${apiBase}/analytics?source=${encodeURIComponent(source)}&module_name=${encodeURIComponent(moduleName)}`
  );
  const data = await res.json();
  setStatus(pretty({ source: data.source, rows: data.rows, columns: data.columns }));
  if (Array.isArray(data.coverage_by_module)) {
    updateModuleFilterOptions(data.coverage_by_module.map((x) => x.module_name));
  }
  drawSeverityChart(data.severity_distribution);
  drawPriorityHistogram(data.priority_distribution);
  drawCoveragePlot(data.coverage_by_module);
  drawHeatmap(data.module_severity_heatmap);
}

async function init() {
  const healthRes = await fetch(`${apiBase}/health`);
  const health = await healthRes.json();
  setStatus(pretty(health));
  if (health.model_ready && health.dataset_exists) {
    await loadAnalytics(analyticsSource);
  }
}

document.getElementById("train-btn").addEventListener("click", trainModel);
document.getElementById("refresh-btn").addEventListener("click", async () => {
  await loadAnalytics(analyticsSource);
});
document.getElementById("predict-btn").addEventListener("click", predictFromLog);
document.getElementById("upload-btn").addEventListener("click", uploadLogs);
moduleFilter.addEventListener("change", async () => {
  await loadAnalytics(analyticsSource);
});

init();

const state = {
  graphPath: "results/knowledge_graph_full_gemma.json",
  dashboard: null,
  selectedEntityId: null,
  lastRetrievedIds: [],
  pipelineJobId: null,
  pipelineTimer: null,
};

const qs = (selector) => document.querySelector(selector);

async function apiGet(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  const response = await fetch(url);
  return readApiResponse(response);
}

async function apiPost(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readApiResponse(response);
}

async function readApiResponse(response) {
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function setText(selector, text) {
  qs(selector).textContent = text;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function formatScore(value) {
  return Number(value || 0).toFixed(4);
}

async function loadDashboard() {
  setText("#graphPathLabel", state.graphPath);
  qs("#graphPathInput").value = state.graphPath;
  const data = await apiGet("/api/dashboard", { graph: state.graphPath });
  state.dashboard = data;
  setText("#entityCount", formatNumber(data.entity_count));
  setText("#relationCount", formatNumber(data.relation_count));
  setText("#consistencyScore", Number(data.validation.consistency_score).toFixed(3));
  const badge = qs("#validationBadge");
  badge.textContent = `${data.validation.status} · ${data.validation.violation_count}`;
  badge.classList.toggle("invalid", data.validation.status !== "valid");
  renderTypeDistribution(data.type_distribution);
  renderRankList("#topDegree", data.top_by_degree, "degree");
  renderRankList("#topPagerank", data.top_by_pagerank, "pagerank");
  fillSelect("#retrievalMode", data.retrieval_modes, "hybrid_rrf");
  fillSelect("#entityTypeFilter", ["", ...data.entity_types], "");
  fillSelect("#predicateFilter", ["", ...data.predicates], "");
  await searchEntities();
}

function renderTypeDistribution(distribution) {
  const container = qs("#typeDistribution");
  container.innerHTML = "";
  const entries = Object.entries(distribution);
  const maxValue = Math.max(...entries.map(([, value]) => value), 1);
  entries.forEach(([label, value], index) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(4, (value / maxValue) * 100)}%`;
    fill.style.background = ["#2f5fbd", "#087f7a", "#a86c00", "#6c4aa8"][index % 4];
    row.innerHTML = `<span>${escapeHtml(label)}</span><div class="bar-track"></div><strong>${formatNumber(value)}</strong>`;
    row.querySelector(".bar-track").appendChild(fill);
    container.appendChild(row);
  });
}

function renderRankList(selector, items, metric) {
  const list = qs(selector);
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `${escapeHtml(item.label)} <span class="type-pill">${escapeHtml(item.entity_type)}</span><br><span class="muted">${metric}: ${formatScore(item[metric])}</span>`;
    li.addEventListener("click", () => selectEntity(item.id));
    list.appendChild(li);
  });
}

function fillSelect(selector, values, selected) {
  const select = qs(selector);
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value || "All";
    option.selected = value === selected;
    select.appendChild(option);
  });
}

async function searchEntities() {
  const data = await apiGet("/api/entities", {
    graph: state.graphPath,
    query: qs("#entitySearchInput").value,
    entity_type: qs("#entityTypeFilter").value,
    limit: 30,
  });
  const list = qs("#entitySearchResults");
  list.innerHTML = "";
  data.items.forEach((entity) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "entity-item";
    row.innerHTML = `<strong>${escapeHtml(entity.label)}</strong><span class="type-pill">${escapeHtml(entity.entity_type)}</span><br><span class="muted">degree ${entity.degree} · PageRank ${formatScore(entity.pagerank)}</span>`;
    row.addEventListener("click", () => selectEntity(entity.id));
    list.appendChild(row);
  });
}

async function selectEntity(entityId) {
  state.selectedEntityId = entityId;
  await loadSubgraph(entityId);
}

async function loadSubgraph(entityId) {
  const data = await apiGet("/api/subgraph", {
    graph: state.graphPath,
    entity_id: entityId,
    depth: qs("#depthInput").value,
    limit_nodes: qs("#limitNodesInput").value,
    entity_type: qs("#entityTypeFilter").value,
    predicate: qs("#predicateFilter").value,
  });
  setText("#graphStatus", `${data.nodes.length} nodes · ${data.relations.length} relations`);
  renderGraph(data);
}

async function submitAsk(event) {
  event.preventDefault();
  qs("#askStatus").textContent = "Running";
  qs("#answerBox").textContent = "";
  qs("#graphContext").textContent = "";
  qs("#retrievedEntities").innerHTML = "";
  qs("#showRetrievedSubgraph").disabled = true;
  try {
    const data = await apiPost("/api/ask", {
      graph_path: state.graphPath,
      query: qs("#questionInput").value,
      retrieval_mode: qs("#retrievalMode").value,
      top_k: Number(qs("#topK").value),
      llm: qs("#llmMode").value,
    });
    qs("#answerBox").textContent = data.answer;
    qs("#graphContext").textContent = data.graph_context;
    state.lastRetrievedIds = data.retrieved_entities.map((item) => item.entity_id);
    renderRetrievedEntities(data.retrieved_entities);
    qs("#showRetrievedSubgraph").disabled = state.lastRetrievedIds.length === 0;
    qs("#askStatus").textContent = "Done";
  } catch (error) {
    qs("#askStatus").textContent = "Failed";
    qs("#answerBox").textContent = error.message;
  }
}

function renderRetrievedEntities(items) {
  const list = qs("#retrievedEntities");
  list.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "entity-item";
    row.innerHTML = `<strong>${item.rank}. ${escapeHtml(item.label)}</strong><span class="type-pill">${escapeHtml(item.entity_type)}</span><br><span class="muted">score ${formatScore(item.score)}</span>`;
    row.addEventListener("click", () => selectEntity(item.entity_id));
    list.appendChild(row);
  });
}

async function showRetrievedSubgraph() {
  const data = await apiPost("/api/retrieved-subgraph", {
    graph_path: state.graphPath,
    entity_ids: state.lastRetrievedIds,
    depth: Number(qs("#depthInput").value),
    limit_nodes: Number(qs("#limitNodesInput").value),
  });
  renderGraph(data);
  setText("#graphStatus", `${data.nodes.length} nodes · ${data.relations.length} relations`);
}

function renderGraph(data) {
  const svg = qs("#graphSvg");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", "0 0 1000 540");
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa5b5"></path></marker>`;
  svg.appendChild(defs);
  if (!data.nodes.length) {
    const text = svgText("No graph slice", 500, 270, "middle");
    text.setAttribute("class", "muted");
    svg.appendChild(text);
    return;
  }
  const positions = layoutNodes(data.nodes);
  const nodeById = new Map(data.nodes.map((node) => [node.id, node]));
  data.relations.forEach((relation) => {
    const source = positions.get(relation.subject_id);
    const target = positions.get(relation.object_id);
    if (!source || !target) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("class", "graph-link");
    line.setAttribute("marker-end", "url(#arrow)");
    svg.appendChild(line);
    const label = svgText(truncate(relation.predicate, 22), (source.x + target.x) / 2, (source.y + target.y) / 2 - 5, "middle");
    label.setAttribute("class", "graph-link-label");
    svg.appendChild(label);
  });
  data.nodes.forEach((node) => {
    const position = positions.get(node.id);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `graph-node${node.selected ? " selected" : ""}`);
    group.setAttribute("tabindex", "0");
    group.addEventListener("click", () => showEntityPanel(node));
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", position.x);
    circle.setAttribute("cy", position.y);
    circle.setAttribute("r", String(14 + Math.min(12, node.degree)));
    circle.setAttribute("fill", colorForType(node.entity_type));
    group.appendChild(circle);
    const label = svgText(truncate(node.label, 28), position.x, position.y + 35, "middle");
    group.appendChild(label);
    svg.appendChild(group);
    nodeById.set(node.id, node);
  });
  const selected = data.nodes.find((node) => node.selected) || data.nodes[0];
  showEntityPanel(selected);
}

function layoutNodes(nodes) {
  const positions = new Map();
  const selected = nodes.filter((node) => node.selected);
  const others = nodes.filter((node) => !node.selected);
  if (selected.length === 1) {
    positions.set(selected[0].id, { x: 500, y: 270 });
  } else {
    selected.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(selected.length, 1);
      positions.set(node.id, { x: 500 + Math.cos(angle) * 90, y: 270 + Math.sin(angle) * 70 });
    });
  }
  others.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(others.length, 1);
    const radiusX = 320 + (index % 3) * 38;
    const radiusY = 180 + (index % 4) * 18;
    positions.set(node.id, {
      x: 500 + Math.cos(angle) * radiusX,
      y: 270 + Math.sin(angle) * radiusY,
    });
  });
  return positions;
}

function showEntityPanel(node) {
  qs("#entityPanelContent").innerHTML = `
    <strong>${escapeHtml(node.label)}</strong>
    <span class="type-pill">${escapeHtml(node.entity_type)}</span>
    <span>Degree: ${node.degree} · In: ${node.in_degree} · Out: ${node.out_degree}</span>
    <span>PageRank: ${formatScore(node.pagerank)}</span>
    <span>${escapeHtml(node.description || "No description")}</span>
    <span class="muted">${escapeHtml((node.aliases || []).join(", "))}</span>
  `;
}

async function submitPipeline(event) {
  event.preventDefault();
  const files = Array.from(qs("#documentInput").files || []);
  if (!files.length) {
    qs("#pipelineStatus").textContent = "Choose Markdown files";
    return;
  }
  qs("#pipelineStatus").textContent = "Uploading";
  const documents = await Promise.all(
    files.map(async (file) => ({
      filename: file.name,
      content: await file.text(),
    })),
  );
  const job = await apiPost("/api/pipeline/jobs", {
    documents,
    out_path: qs("#pipelineOutPath").value,
  });
  state.pipelineJobId = job.id;
  renderPipelineJob(job);
  pollPipelineJob();
}

async function pollPipelineJob() {
  if (!state.pipelineJobId) return;
  window.clearTimeout(state.pipelineTimer);
  const job = await apiGet(`/api/pipeline/jobs/${state.pipelineJobId}`);
  renderPipelineJob(job);
  if (job.status === "running" || job.status === "queued") {
    state.pipelineTimer = window.setTimeout(pollPipelineJob, 1500);
  } else if (job.status === "completed") {
    state.graphPath = job.graph_path;
    await loadDashboard();
  }
}

function renderPipelineJob(job) {
  qs("#pipelineStatus").textContent = job.status;
  const list = qs("#stageList");
  list.innerHTML = "";
  job.stages.forEach((stage) => {
    const row = document.createElement("div");
    row.className = "stage-item";
    row.innerHTML = `<strong>${escapeHtml(stage.label)}</strong><span class="stage-status ${stage.status}">${escapeHtml(stage.status)}</span><span class="muted">${escapeHtml(stage.detail || "")}</span>`;
    list.appendChild(row);
  });
}

async function exportNeo4j() {
  qs("#neo4jStatus").textContent = "Running";
  try {
    const result = await apiPost("/api/neo4j/export", {
      graph_path: state.graphPath,
      clear: qs("#clearNeo4j").checked,
    });
    qs("#neo4jStatus").textContent = `${formatNumber(result.entity_count)} entities imported`;
  } catch (error) {
    qs("#neo4jStatus").textContent = error.message;
  }
}

function svgText(textContent, x, y, anchor) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.textContent = textContent;
  text.setAttribute("x", x);
  text.setAttribute("y", y);
  text.setAttribute("text-anchor", anchor);
  return text;
}

function colorForType(type) {
  const colors = {
    Drug: "#2f5fbd",
    Condition: "#b33a3a",
    Procedure: "#087f7a",
    Symptom: "#a86c00",
    AnatomicalStructure: "#6c4aa8",
    DosageRegimen: "#2f7d4b",
    LabTest: "#596579",
    Organization: "#4f6f52",
    Other: "#7a8495",
  };
  return colors[type] || colors.Other;
}

function truncate(value, maxLength) {
  const text = String(value || "");
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function wireEvents() {
  qs("#graphForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    state.graphPath = qs("#graphPathInput").value;
    await loadDashboard();
  });
  qs("#askForm").addEventListener("submit", submitAsk);
  qs("#entitySearchInput").addEventListener("input", debounce(searchEntities, 250));
  qs("#entityTypeFilter").addEventListener("change", async () => {
    await searchEntities();
    if (state.selectedEntityId) await loadSubgraph(state.selectedEntityId);
  });
  qs("#predicateFilter").addEventListener("change", async () => {
    if (state.selectedEntityId) await loadSubgraph(state.selectedEntityId);
  });
  qs("#depthInput").addEventListener("change", async () => {
    if (state.selectedEntityId) await loadSubgraph(state.selectedEntityId);
  });
  qs("#limitNodesInput").addEventListener("change", async () => {
    if (state.selectedEntityId) await loadSubgraph(state.selectedEntityId);
  });
  qs("#showRetrievedSubgraph").addEventListener("click", showRetrievedSubgraph);
  qs("#pipelineForm").addEventListener("submit", submitPipeline);
  qs("#exportNeo4jButton").addEventListener("click", exportNeo4j);
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

wireEvents();
loadDashboard().catch((error) => {
  qs("#validationBadge").textContent = "Error";
  qs("#answerBox").textContent = error.message;
});

const state = {
  graphPath: "results/knowledge_graph_full_gemma.json",
  dashboard: null,
  selectedEntityId: null,
  lastRetrievedIds: [],
  lastAskPayload: null,
  currentGraph: { nodes: [], relations: [] },
  pipelineJobId: null,
  pipelineTimer: null,
  entityOffset: 0,
  entityHasMore: false,
  graphViewBox: { x: 0, y: 0, width: 1000, height: 560 },
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
  return new Intl.NumberFormat("ru-RU").format(value || 0);
}

function formatScore(value) {
  return Number(value || 0).toFixed(4);
}

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(3)} s`;
}

async function loadDashboard() {
  setText("#graphPathLabel", state.graphPath);
  qs("#graphPathInput").value = state.graphPath;
  const data = await apiGet("/api/dashboard", { graph: state.graphPath });
  state.dashboard = data;

  setText("#entityCount", formatNumber(data.entity_count));
  setText("#relationCount", formatNumber(data.relation_count));
  setText("#consistencyScore", Number(data.validation.consistency_score).toFixed(3));
  setText("#componentCount", formatNumber(data.graph_metrics.connected_components));
  setText("#sourceChunkCount", formatNumber(data.graph_metrics.source_chunk_count));
  setText("#isolatedCount", formatNumber(data.graph_metrics.isolated_entities));

  const badge = qs("#validationBadge");
  badge.textContent = `${data.validation.status} · ${data.validation.violation_count}`;
  badge.classList.toggle("invalid", data.validation.status !== "valid");

  renderBarList("#typeDistribution", data.type_distribution);
  renderBarList(
    "#topPredicates",
    Object.fromEntries(data.top_predicates.map((item) => [item.predicate, item.count])),
  );
  renderRankList("#topDegree", data.top_by_degree, "degree");
  renderRankList("#topPagerank", data.top_by_pagerank, "pagerank");
  fillSelect("#retrievalMode", data.retrieval_modes, "lexical_structural");
  fillSelect("#entityTypeFilter", ["", ...data.entity_types], "");
  fillSelect("#predicateFilter", ["", ...data.predicates], "");
  await searchEntities(true);
  await loadQuality();
  await testNeo4j(false);
}

function renderBarList(selector, distribution) {
  const container = qs(selector);
  container.innerHTML = "";
  const entries = Object.entries(distribution || {});
  if (!entries.length) {
    container.innerHTML = `<span class="muted">Нет данных</span>`;
    return;
  }
  const maxValue = Math.max(...entries.map(([, value]) => Number(value)), 1);
  entries.forEach(([label, value], index) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(4, (Number(value) / maxValue) * 100)}%`;
    fill.style.background = ["#2f5fbd", "#087f7a", "#a86c00", "#6c4aa8"][index % 4];
    row.innerHTML = `
      <span title="${escapeHtml(label)}">${escapeHtml(label)}</span>
      <div class="bar-track"></div>
      <strong>${formatNumber(value)}</strong>
    `;
    row.querySelector(".bar-track").appendChild(fill);
    container.appendChild(row);
  });
}

function renderRankList(selector, items, metric) {
  const list = qs(selector);
  list.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <button type="button" class="link-row">
        <strong>${escapeHtml(item.label)}</strong>
        <span class="type-pill">${escapeHtml(item.entity_type)}</span>
        <span class="muted">${metric}: ${formatScore(item[metric])}</span>
      </button>
    `;
    li.querySelector("button").addEventListener("click", () => selectEntity(item.id));
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

async function searchEntities(reset = false) {
  if (reset) {
    state.entityOffset = 0;
    qs("#entitySearchResults").innerHTML = "";
  }
  const data = await apiGet("/api/entities", {
    graph: state.graphPath,
    query: qs("#entitySearchInput").value,
    entity_type: qs("#entityTypeFilter").value,
    limit: 30,
    offset: state.entityOffset,
  });
  const list = qs("#entitySearchResults");
  data.items.forEach((entity) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "entity-item";
    row.innerHTML = `
      <strong>${escapeHtml(entity.label)}</strong>
      <span class="type-pill">${escapeHtml(entity.entity_type)}</span>
      <span class="muted">degree ${entity.degree} · PageRank ${formatScore(entity.pagerank)}</span>
    `;
    row.addEventListener("click", () => selectEntity(entity.id));
    list.appendChild(row);
  });
  state.entityOffset += data.items.length;
  state.entityHasMore = data.has_more;
  qs("#loadMoreEntities").disabled = !data.has_more;
}

async function selectEntity(entityId) {
  state.selectedEntityId = entityId;
  activateView("explorer");
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
  qs("#askResults").classList.remove("hidden");
  qs("#askStatus").textContent = "Выполняется";
  qs("#answerBox").textContent = "";
  qs("#graphContext").textContent = "";
  qs("#retrievedEntities").innerHTML = "";
  qs("#contextRelations").innerHTML = "";
  qs("#askDiagnostics").innerHTML = "";
  qs("#sourceChunks").innerHTML = "";
  qs("#compareResults").innerHTML = "";
  qs("#showRetrievedSubgraph").disabled = true;
  try {
    const data = await apiPost("/api/ask", {
      graph_path: state.graphPath,
      query: qs("#questionInput").value,
      retrieval_mode: qs("#retrievalMode").value,
      top_k: Number(qs("#topK").value),
      llm: qs("#llmMode").value,
    });
    state.lastAskPayload = data;
    qs("#answerBox").textContent = data.answer;
    qs("#graphContext").textContent = data.graph_context;
    state.lastRetrievedIds = data.retrieved_entities.map((item) => item.entity_id);
    renderRetrievedEntities(data.retrieved_entities);
    renderRelations("#contextRelations", data.context_relations);
    renderDiagnostics(data.diagnostics);
    renderSourceChunks(data.source_chunks);
    qs("#showRetrievedSubgraph").disabled = state.lastRetrievedIds.length === 0;
    qs("#askStatus").textContent = "Готово";
    activateSubtab("answerTab");
  } catch (error) {
    qs("#askResults").classList.remove("hidden");
    qs("#askStatus").textContent = "Ошибка";
    qs("#answerBox").textContent = error.message;
  }
}

function renderRetrievedEntities(items) {
  const list = qs("#retrievedEntities");
  list.innerHTML = "";
  (items || []).forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "entity-item retrieved-item";
    row.innerHTML = `
      <strong>${item.rank}. ${escapeHtml(item.label)}</strong>
      <span class="type-pill">${escapeHtml(item.entity_type)}</span>
      <span class="muted">score ${formatScore(item.score)}</span>
      ${componentScoresHtml(item.component_scores)}
    `;
    row.addEventListener("click", () => selectEntity(item.entity_id));
    list.appendChild(row);
  });
}

function componentScoresHtml(scores) {
  const entries = Object.entries(scores || {});
  if (!entries.length) return "";
  return `
    <span class="score-row">
      ${entries
        .map(([key, value]) => `<span>${escapeHtml(key)} ${formatScore(value)}</span>`)
        .join("")}
    </span>
  `;
}

function renderRelations(selector, relations) {
  const container = qs(selector);
  container.innerHTML = "";
  if (!relations || !relations.length) {
    container.innerHTML = `<span class="muted">Нет связей</span>`;
    return;
  }
  relations.forEach((relation) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "relation-item";
    row.innerHTML = `
      <strong>${escapeHtml(relation.subject_label)}</strong>
      <span>${escapeHtml(relation.predicate)}</span>
      <strong>${escapeHtml(relation.object_label)}</strong>
      <span class="muted">confidence ${formatScore(relation.confidence)}</span>
    `;
    row.addEventListener("click", () => selectEntity(relation.subject_id));
    container.appendChild(row);
  });
}

function renderDiagnostics(diagnostics) {
  const values = {
    graph: diagnostics.graph_path,
    mode: diagnostics.retrieval_mode,
    top_k: diagnostics.top_k,
    llm: diagnostics.llm,
    retrieved: diagnostics.retrieved_count,
    context_chars: diagnostics.graph_context_chars,
    retrieval: formatSeconds(diagnostics.retrieval_s),
    answer: formatSeconds(diagnostics.answer_s),
    total: formatSeconds(diagnostics.total_s),
  };
  renderKeyValues("#askDiagnostics", values);
}

function renderSourceChunks(chunks) {
  const container = qs("#sourceChunks");
  container.innerHTML = "";
  if (!chunks || !chunks.length) {
    container.innerHTML = `<span class="muted">Нет source chunks</span>`;
    return;
  }
  chunks.forEach((chunk) => {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = chunk;
    container.appendChild(span);
  });
}

async function compareRetrievalModes() {
  const query = qs("#questionInput").value.trim();
  if (!query) {
    qs("#askStatus").textContent = "Введите вопрос";
    return;
  }
  qs("#askResults").classList.remove("hidden");
  qs("#askStatus").textContent = "Сравнение";
  try {
    const data = await apiPost("/api/retrieval/compare", {
      graph_path: state.graphPath,
      query,
      top_k: Number(qs("#topK").value),
    });
    renderCompareResults(data);
    activateSubtab("diagnosticsTab");
    qs("#askStatus").textContent = "Готово";
  } catch (error) {
    qs("#askStatus").textContent = "Ошибка";
    qs("#compareResults").textContent = error.message;
  }
}

function renderCompareResults(data) {
  const container = qs("#compareResults");
  container.innerHTML = "";
  Object.entries(data.modes || {}).forEach(([mode, payload]) => {
    const block = document.createElement("div");
    block.className = "compare-block";
    const topItems = (payload.items || [])
      .slice(0, 5)
      .map(
        (item) => `
          <li>
            <button type="button" class="link-row" data-entity-id="${escapeHtml(item.entity_id)}">
              <strong>${escapeHtml(item.label)}</strong>
              <span class="muted">${formatScore(item.score)}</span>
            </button>
          </li>
        `,
      )
      .join("");
    block.innerHTML = `
      <h4>${escapeHtml(mode)}</h4>
      <span class="muted">${formatSeconds(payload.duration_s)}</span>
      <ol class="rank-list">${topItems}</ol>
    `;
    block.querySelectorAll("button[data-entity-id]").forEach((button) => {
      button.addEventListener("click", () => selectEntity(button.dataset.entityId));
    });
    container.appendChild(block);
  });
}

async function showRetrievedSubgraph() {
  const data = await apiPost("/api/retrieved-subgraph", {
    graph_path: state.graphPath,
    entity_ids: state.lastRetrievedIds,
    depth: Number(qs("#depthInput").value),
    limit_nodes: Number(qs("#limitNodesInput").value),
  });
  activateView("explorer");
  renderGraph(data);
  setText("#graphStatus", `${data.nodes.length} nodes · ${data.relations.length} relations`);
}

function renderGraph(data) {
  state.currentGraph = data;
  state.graphViewBox = { x: 0, y: 0, width: 1000, height: 560 };
  const svg = qs("#graphSvg");
  svg.innerHTML = "";
  setGraphViewBox();
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML =
    '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa5b5"></path></marker>';
  svg.appendChild(defs);
  if (!data.nodes.length) {
    const text = svgText("No graph slice", 500, 280, "middle");
    text.setAttribute("class", "graph-empty");
    svg.appendChild(text);
    showEntityPanel(null);
    return;
  }

  const positions = layoutNodes(data.nodes);
  const showEdgeLabels = qs("#edgeLabelsToggle").checked;
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
    if (showEdgeLabels) {
      const label = svgText(
        truncate(relation.predicate, 24),
        (source.x + target.x) / 2,
        (source.y + target.y) / 2 - 6,
        "middle",
      );
      label.setAttribute("class", "graph-link-label");
      svg.appendChild(label);
    }
  });

  data.nodes.forEach((node) => {
    const position = positions.get(node.id);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `graph-node${node.selected ? " selected" : ""}`);
    group.setAttribute("tabindex", "0");
    group.addEventListener("click", () => showEntityPanel(node));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") showEntityPanel(node);
    });
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", position.x);
    circle.setAttribute("cy", position.y);
    circle.setAttribute("r", String(15 + Math.min(12, node.degree)));
    circle.setAttribute("fill", colorForType(node.entity_type));
    group.appendChild(circle);
    const label = svgText(truncate(node.label, 30), position.x, position.y + 38, "middle");
    group.appendChild(label);
    svg.appendChild(group);
  });
  const selected = data.nodes.find((node) => node.selected) || data.nodes[0];
  showEntityPanel(selected);
}

function layoutNodes(nodes) {
  const positions = new Map();
  const selected = nodes.filter((node) => node.selected);
  const others = nodes.filter((node) => !node.selected);
  const centerX = 500;
  const centerY = 280;
  if (selected.length === 1) {
    positions.set(selected[0].id, { x: centerX, y: centerY });
  } else {
    selected.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(selected.length, 1);
      positions.set(node.id, {
        x: centerX + Math.cos(angle) * 95,
        y: centerY + Math.sin(angle) * 75,
      });
    });
  }
  others.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(others.length, 1);
    const radiusX = 315 + (index % 4) * 34;
    const radiusY = 175 + (index % 5) * 15;
    positions.set(node.id, {
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    });
  });
  return positions;
}

function showEntityPanel(node) {
  const container = qs("#entityPanelContent");
  if (!node) {
    container.innerHTML = `<span class="muted">Выберите узел</span>`;
    return;
  }
  const relations = (state.currentGraph.relations || []).filter(
    (relation) => relation.subject_id === node.id || relation.object_id === node.id,
  );
  const relationHtml = relations.length
    ? relations
        .slice(0, 12)
        .map(
          (relation) => `
            <button type="button" class="relation-item compact-relation" data-entity-id="${escapeHtml(
              relation.subject_id === node.id ? relation.object_id : relation.subject_id,
            )}">
              <strong>${escapeHtml(relation.subject_label)}</strong>
              <span>${escapeHtml(relation.predicate)}</span>
              <strong>${escapeHtml(relation.object_label)}</strong>
            </button>
          `,
        )
        .join("")
    : '<span class="muted">Нет связей в текущем срезе</span>';
  container.innerHTML = `
    <strong>${escapeHtml(node.label)}</strong>
    <span class="type-pill solo">${escapeHtml(node.entity_type)}</span>
    <span>Degree: ${node.degree} · In: ${node.in_degree} · Out: ${node.out_degree}</span>
    <span>PageRank: ${formatScore(node.pagerank)}</span>
    <span>${escapeHtml(node.description || "No description")}</span>
    <div class="chip-list">${chipsHtml(node.aliases || [], "aliases")}</div>
    <div class="chip-list">${chipsHtml(node.source_chunks || [], "source")}</div>
    <h4>Relations</h4>
    <div class="relation-list">${relationHtml}</div>
  `;
  container.querySelectorAll("button[data-entity-id]").forEach((button) => {
    button.addEventListener("click", () => selectEntity(button.dataset.entityId));
  });
}

function chipsHtml(values, emptyLabel) {
  if (!values.length) return `<span class="muted">No ${emptyLabel}</span>`;
  return values
    .slice(0, 12)
    .map((value) => `<span class="chip">${escapeHtml(value)}</span>`)
    .join("");
}

async function submitPipeline(event) {
  event.preventDefault();
  const files = Array.from(qs("#documentInput").files || []);
  if (!files.length) {
    qs("#pipelineStatus").textContent = "Выберите Markdown файлы";
    return;
  }
  qs("#pipelineStatus").textContent = "Загрузка";
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
    row.innerHTML = `
      <strong>${escapeHtml(stage.label)}</strong>
      <span class="stage-status ${escapeHtml(stage.status)}">${escapeHtml(stage.status)}</span>
      <span class="muted">${escapeHtml(stage.detail || "")}</span>
      <span class="muted">${stage.duration_s ? formatSeconds(stage.duration_s) : ""}</span>
    `;
    list.appendChild(row);
  });
  if (job.summary) {
    renderKeyValues("#pipelineSummary", {
      documents: job.summary.documents_processed,
      chunks: job.summary.chunks_processed,
      entities_raw: job.summary.entity_count_raw,
      relations_raw: job.summary.relation_count_raw,
      entities_final: job.summary.entity_count_final,
      relations_final: job.summary.relation_count_final,
      dedup_merged: job.summary.dedup_merged_count,
      consistency_final: Number(job.summary.consistency_final || 0).toFixed(3),
      graph_path: job.summary.graph_path,
    });
  }
  if (job.error) {
    qs("#pipelineSummary").innerHTML = `<pre class="text-output">${escapeHtml(job.error)}</pre>`;
  }
}

async function exportNeo4j() {
  qs("#neo4jStatus").textContent = "Импорт";
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

async function testNeo4j(showRunning = true) {
  if (showRunning) qs("#neo4jStatus").textContent = "Проверка";
  try {
    const result = await apiGet("/api/neo4j/status");
    const target = result.target || {};
    const status = result.connected ? "connected" : "not connected";
    qs("#neo4jStatus").textContent =
      `${status}: ${target.uri || ""} / ${target.database || ""}`;
  } catch (error) {
    qs("#neo4jStatus").textContent = error.message;
  }
}

async function loadQuality() {
  const container = qs("#qualitySummary");
  container.innerHTML = `<span class="muted">Загрузка</span>`;
  try {
    const data = await apiGet("/api/evaluation/summary");
    renderQuality(data);
  } catch (error) {
    container.textContent = error.message;
  }
}

function renderQuality(data) {
  const container = qs("#qualitySummary");
  container.innerHTML = "";
  if (!data.has_any_metrics) {
    container.innerHTML = `<span class="muted">Метрики еще не рассчитаны</span>`;
    return;
  }
  if (data.retrieval_metrics) {
    container.appendChild(metricTable("Retrieval", data.retrieval_metrics));
  }
  if (data.generation_metrics) {
    container.appendChild(metricTable("Generation", data.generation_metrics));
  }
  if (data.latency_metrics?.aggregated) {
    container.appendChild(metricTable("Latency", data.latency_metrics.aggregated));
  }
}

function metricTable(title, metrics) {
  const block = document.createElement("div");
  block.className = "quality-block";
  const rows = Object.entries(metrics)
    .map(([name, values]) => {
      if (values && typeof values === "object" && !Array.isArray(values)) {
        return `
          <tr>
            <th>${escapeHtml(name)}</th>
            <td>${Object.entries(values)
              .map(([key, value]) => `${escapeHtml(key)}: ${escapeHtml(formatMetricValue(value))}`)
              .join("<br>")}</td>
          </tr>
        `;
      }
      return `<tr><th>${escapeHtml(name)}</th><td>${escapeHtml(formatMetricValue(values))}</td></tr>`;
    })
    .join("");
  block.innerHTML = `
    <h3>${escapeHtml(title)}</h3>
    <table class="metric-table"><tbody>${rows}</tbody></table>
  `;
  return block;
}

function renderKeyValues(selector, values) {
  const container = qs(selector);
  container.innerHTML = "";
  Object.entries(values || {}).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "kv-item";
    row.innerHTML = `<span>${escapeHtml(key)}</span><strong>${escapeHtml(formatMetricValue(value))}</strong>`;
    container.appendChild(row);
  });
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? formatNumber(value) : Number(value).toFixed(4);
  }
  return String(value ?? "");
}

function activateSubtab(tabId) {
  document.querySelectorAll(".subtab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
}

function activateView(viewId) {
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === viewId);
  });
  document.querySelectorAll(".main-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });
  if (window.location.hash !== `#${viewId}`) {
    window.history.replaceState(null, "", `#${viewId}`);
  }
}

function activateInitialView() {
  const viewId = window.location.hash.replace("#", "");
  const panel = viewId ? document.getElementById(viewId) : null;
  activateView(panel?.classList.contains("view-panel") ? viewId : "ask");
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
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setGraphViewBox() {
  const { x, y, width, height } = state.graphViewBox;
  qs("#graphSvg").setAttribute("viewBox", `${x} ${y} ${width} ${height}`);
}

function zoomGraph(factor) {
  const view = state.graphViewBox;
  const centerX = view.x + view.width / 2;
  const centerY = view.y + view.height / 2;
  const width = Math.max(260, Math.min(1600, view.width * factor));
  const height = Math.max(150, Math.min(900, view.height * factor));
  state.graphViewBox = {
    x: centerX - width / 2,
    y: centerY - height / 2,
    width,
    height,
  };
  setGraphViewBox();
}

function resetGraphView() {
  state.graphViewBox = { x: 0, y: 0, width: 1000, height: 560 };
  setGraphViewBox();
}

function setupGraphPan() {
  const svg = qs("#graphSvg");
  let dragStart = null;
  svg.addEventListener("pointerdown", (event) => {
    if (event.target !== svg) return;
    dragStart = {
      clientX: event.clientX,
      clientY: event.clientY,
      view: { ...state.graphViewBox },
    };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!dragStart) return;
    const rect = svg.getBoundingClientRect();
    const dx = ((event.clientX - dragStart.clientX) / rect.width) * dragStart.view.width;
    const dy = ((event.clientY - dragStart.clientY) / rect.height) * dragStart.view.height;
    state.graphViewBox = {
      ...dragStart.view,
      x: dragStart.view.x - dx,
      y: dragStart.view.y - dy,
    };
    setGraphViewBox();
  });
  svg.addEventListener("pointerup", () => {
    dragStart = null;
  });
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomGraph(event.deltaY < 0 ? 0.88 : 1.12);
  });
}

function wireEvents() {
  document.querySelectorAll(".main-tab").forEach((button) => {
    button.addEventListener("click", () => activateView(button.dataset.view));
  });
  qs("#graphForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    state.graphPath = qs("#graphPathInput").value;
    await loadDashboard();
  });
  qs("#askForm").addEventListener("submit", submitAsk);
  qs("#compareModesButton").addEventListener("click", compareRetrievalModes);
  qs("#entitySearchInput").addEventListener("input", debounce(() => searchEntities(true), 250));
  qs("#entityTypeFilter").addEventListener("change", async () => {
    await searchEntities(true);
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
  qs("#edgeLabelsToggle").addEventListener("change", () => renderGraph(state.currentGraph));
  qs("#loadMoreEntities").addEventListener("click", () => searchEntities(false));
  qs("#showRetrievedSubgraph").addEventListener("click", showRetrievedSubgraph);
  qs("#pipelineForm").addEventListener("submit", submitPipeline);
  qs("#testNeo4jButton").addEventListener("click", () => testNeo4j(true));
  qs("#exportNeo4jButton").addEventListener("click", exportNeo4j);
  qs("#reloadQualityButton").addEventListener("click", loadQuality);
  qs("#graphZoomIn").addEventListener("click", () => zoomGraph(0.85));
  qs("#graphZoomOut").addEventListener("click", () => zoomGraph(1.15));
  qs("#graphResetView").addEventListener("click", resetGraphView);
  document.querySelectorAll(".subtab").forEach((button) => {
    button.addEventListener("click", () => activateSubtab(button.dataset.tab));
  });
  setupGraphPan();
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

wireEvents();
activateInitialView();
loadDashboard().catch((error) => {
  qs("#validationBadge").textContent = "Ошибка";
  qs("#answerBox").textContent = error.message;
});

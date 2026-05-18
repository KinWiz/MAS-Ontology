const state = {
  graphPath: "results/knowledge_graph_full_gemma.json",
  dashboard: null,
  selectedEntityId: null,
  focusEntityId: null,
  lastRetrievedIds: [],
  lastAskPayload: null,
  currentGraph: { nodes: [], relations: [] },
  pipelineJobId: null,
  pipelineTimer: null,
  entityOffset: 0,
  entityHasMore: false,
  graphViewBox: { x: 0, y: 0, width: 1000, height: 560 },
  defaultGraphViewBox: { x: 0, y: 0, width: 1000, height: 560 },
};

const GRAPH_LAYOUT = {
  minWidth: 1000,
  minHeight: 560,
  margin: 120,
  firstRingRadius: 185,
  ringGap: 145,
  layerGap: 110,
  minArcSpacing: 175,
  relaxationPasses: 160,
  maxLabelChars: 24,
};

const ENTITY_TYPE_ORDER = [
  "Condition",
  "Drug",
  "Procedure",
  "LabTest",
  "DosageRegimen",
  "Symptom",
  "AnatomicalStructure",
  "Organization",
  "Other",
];

const PREDICATE_STYLES = {
  treats: { color: "#2f5fbd", dash: "" },
  diagnosed_by: { color: "#087f7a", dash: "" },
  has_protocol: { color: "#a86c00", dash: "" },
  causes: { color: "#b33a3a", dash: "" },
  improves: { color: "#2f7d4b", dash: "" },
  related_to: { color: "#9aa5b5", dash: "5 5" },
  inferred_related_to: { color: "#6c4aa8", dash: "3 5" },
  default: { color: "#7a8495", dash: "" },
};

const SELECT_OPTION_LABELS = {
  "": "Все",
  all: "Все",
  lexical_only: "Только текстовый поиск",
  lexical_structural: "Текст + связи графа",
  structural_only: "Только структура графа",
  semantic_only: "Только смысловой поиск",
  hybrid_rrf: "Гибридный поиск (RRF)",
  demo: "Демо embeddings",
  auto: "Авто",
  ollama: "Ollama локально",
  openai: "OpenAI / ChatGPT API",
  deepseek: "DeepSeek API",
  gemma: "Gemma локально",
  deterministic: "Детерминированный тест",
  Condition: "Состояние / диагноз",
  Drug: "Препарат",
  Procedure: "Процедура",
  LabTest: "Анализ",
  DosageRegimen: "Схема дозирования",
  Symptom: "Симптом",
  AnatomicalStructure: "Анатомия",
  Organization: "Организация",
  Other: "Другое",
  treats: "лечит",
  treated_by: "лечится с помощью",
  diagnosed_by: "диагностируется через",
  has_protocol: "имеет протокол",
  causes: "вызывает",
  caused_by: "вызвано",
  contraindicated_for: "противопоказано при",
  dosage_is: "дозировка",
  enhances: "усиливает",
  improves: "улучшает",
  includes: "включает",
  part_of: "часть",
  prevents: "предотвращает",
  reduces: "снижает",
  related_to: "связано с",
  requires: "требует",
  inferred_related_to: "предположительно связано",
  type_cluster: "Группировать по типам",
  neighborhood: "Показать окружение",
  important: "Основные связи",
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

function setElementHint(element, title) {
  if (!element) return;
  element.setAttribute("title", title);
  element.setAttribute("aria-label", title);
}

function setButtonCopy(selector, text, title = text) {
  const button = qs(selector);
  if (!button) return;
  button.textContent = text;
  setElementHint(button, title);
}

function setLabelForControl(selector, text, title = text) {
  const control = qs(selector);
  const label = control?.closest("label");
  if (!label) return;
  const textNode = Array.from(label.childNodes).find(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
  );
  if (textNode) {
    textNode.textContent = ` ${text} `;
  }
  label.setAttribute("title", title);
}

function setSelectOptionLabels(selector, labels) {
  const select = qs(selector);
  if (!select) return;
  select.querySelectorAll("option").forEach((option) => {
    option.textContent = labels[option.value] || SELECT_OPTION_LABELS[option.value] || option.textContent;
  });
}

function localizeDoctorControls() {
  const graphSummary = qs(".graph-settings summary");
  if (graphSummary) {
    graphSummary.textContent = "Граф знаний";
    graphSummary.setAttribute("title", "Выбрать JSON-файл графа знаний");
  }
  const graphPathInput = qs("#graphPathInput");
  if (graphPathInput) {
    graphPathInput.setAttribute("aria-label", "Путь к файлу графа знаний");
    graphPathInput.setAttribute("title", "JSON-файл графа знаний, который нужно открыть");
  }
  const advancedSummary = qs(".advanced-options summary");
  if (advancedSummary) {
    advancedSummary.textContent = "Настройки поиска и сравнение методов";
    advancedSummary.setAttribute(
      "title",
      "Выбрать алгоритм поиска, модель ответа и режим смыслового поиска",
    );
  }
  setLabelForControl(
    "#retrievalMode",
    "Алгоритм поиска",
    "Как искать сущности и связи в графе знаний",
  );
  setLabelForControl("#topK", "Сколько результатов", "Максимальное число найденных сущностей");
  setLabelForControl("#llmMode", "Модель ответа", "Какая LLM сформирует ответ врачу");
  setLabelForControl(
    "#embeddingMode",
    "Смысловой поиск",
    "Провайдер embeddings для semantic/hybrid поиска",
  );
  setButtonCopy(
    "#graphForm button[type='submit']",
    "Открыть граф",
    "Открыть выбранный JSON-файл графа знаний",
  );
  setButtonCopy(
    ".main-tab[data-view='ask']",
    "Вопрос врачу",
    "Задать клинический вопрос по графу знаний",
  );
  setButtonCopy(
    ".main-tab[data-view='dashboard']",
    "Сводка графа",
    "Посмотреть количество сущностей, связей и состояние графа",
  );
  setButtonCopy(
    ".main-tab[data-view='explorer']",
    "Связи графа",
    "Исследовать сущности и клинические связи графа",
  );
  setButtonCopy(
    ".main-tab[data-view='quality']",
    "Качество",
    "Посмотреть метрики качества поиска и ответов",
  );
  setButtonCopy(
    ".main-tab[data-view='pipeline']",
    "Сборка графа",
    "Построить новый граф знаний из документов",
  );
  setButtonCopy(
    "#askForm .ask-actions button[type='submit']",
    "Получить ответ по графу",
    "Найти релевантные сущности и сформировать ответ на вопрос",
  );
  setButtonCopy(
    "#compareModesButton",
    "Сравнить методы поиска",
    "Показать, какие сущности находят разные режимы retrieval",
  );
  setButtonCopy(
    ".subtab[data-tab='answerTab']",
    "Ответ",
    "Показать сформированный ответ",
  );
  setButtonCopy(
    ".subtab[data-tab='entitiesTab']",
    "Найденные сущности",
    "Показать сущности и факты, найденные в графе",
  );
  setButtonCopy(
    ".subtab[data-tab='contextTab']",
    "Контекст графа",
    "Показать фрагмент графа, использованный для ответа",
  );
  setButtonCopy(
    ".subtab[data-tab='diagnosticsTab']",
    "Диагностика поиска",
    "Показать режимы, время выполнения и служебные показатели",
  );
  setButtonCopy(
    ".subtab[data-tab='sourcesTab']",
    "Источники",
    "Показать исходные фрагменты документов",
  );
  setButtonCopy(
    "#showRetrievedSubgraph",
    "Показать найденные связи",
    "Открыть на графе сущности, найденные для ответа",
  );
  setButtonCopy(
    "#loadMoreEntities",
    "Показать еще сущности",
    "Загрузить следующую страницу найденных сущностей",
  );
  setButtonCopy("#graphZoomIn", "+", "Приблизить граф");
  setButtonCopy("#graphZoomOut", "-", "Отдалить граф");
  qs("#graphZoomIn")?.classList.add("icon-button");
  qs("#graphZoomOut")?.classList.add("icon-button");
  setButtonCopy(
    "#graphResetView",
    "Весь граф",
    "Вернуть граф к полному обзору",
  );
  setButtonCopy(
    "#reloadQualityButton",
    "Обновить метрики",
    "Перечитать сохраненные метрики качества из папки results",
  );
  const entitySearchInput = qs("#entitySearchInput");
  if (entitySearchInput) {
    entitySearchInput.placeholder = "Найти протокол, симптом, препарат...";
    setElementHint(entitySearchInput, "Поиск сущности в графе знаний");
  }
  const edgeLabelsToggle = qs("#edgeLabelsToggle");
  if (edgeLabelsToggle) {
    setLabelForControl(
      "#edgeLabelsToggle",
      "Подписи связей",
      "Показать названия связей на ребрах графа",
    );
  }
  setLabelForControl("#entityTypeFilter", "Тип сущности", "Отфильтровать узлы графа по типу");
  setLabelForControl("#predicateFilter", "Тип связи", "Отфильтровать ребра графа по типу связи");
  setLabelForControl("#depthInput", "Глубина связей", "Сколько шагов от выбранной сущности показать");
  setLabelForControl("#limitNodesInput", "Лимит узлов", "Сколько узлов максимум показывать на графе");
  setLabelForControl("#layoutMode", "Раскладка", "Как расположить узлы на графе");
  setSelectOptionLabels("#layoutMode", SELECT_OPTION_LABELS);
  setLabelForControl("#relationView", "Связи", "Показать основные или все связи");
  setSelectOptionLabels("#relationView", SELECT_OPTION_LABELS);
  setLabelForControl(
    "#focusModeToggle",
    "Фокус на выбранной сущности",
    "Приглушать дальние узлы и оставлять рядом ближайшие связи",
  );
  const graphSvg = qs("#graphSvg");
  if (graphSvg) {
    graphSvg.setAttribute("aria-label", "Интерактивный граф знаний");
  }
  const qualityTitle = qs(".quality-surface h2");
  if (qualityTitle) {
    qualityTitle.textContent = "Качество поиска";
  }
  const pipelineTitle = qs(".pipeline-surface h2");
  if (pipelineTitle) {
    pipelineTitle.textContent = "Сборка графа";
  }
  setLabelForControl("#pipelineLlmMode", "LLM для извлечения", "Модель для извлечения сущностей");
  const pipelineOutPath = qs("#pipelineOutPath");
  if (pipelineOutPath) {
    setElementHint(pipelineOutPath, "Куда сохранить собранный JSON-граф");
  }
  setButtonCopy(
    "#pipelineForm button[type='submit']",
    "Построить граф из документов",
    "Запустить извлечение сущностей и связей из выбранных Markdown-документов",
  );
  setLabelForControl(
    "#clearNeo4j",
    "Очистить Neo4j перед загрузкой",
    "Удалить старые узлы и связи перед импортом нового графа",
  );
  setButtonCopy(
    "#testNeo4jButton",
    "Проверить Neo4j",
    "Проверить, доступна ли база Neo4j",
  );
  setButtonCopy(
    "#exportNeo4jButton",
    "Загрузить граф в Neo4j",
    "Импортировать текущий граф знаний в Neo4j",
  );
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
  fillSelect("#llmMode", data.answer_llms || ["gemma", "openai", "deepseek", "deterministic"], "gemma");
  fillSelect("#embeddingMode", data.embedding_providers || ["demo", "auto", "ollama", "openai"], "demo");
  fillSelect("#pipelineLlmMode", data.pipeline_llms || ["gemma", "openai", "deepseek"], "gemma");
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
    const button = li.querySelector("button");
    setElementHint(button, `Открыть связи сущности: ${item.label}`);
    button.addEventListener("click", () => selectEntity(item.id));
    list.appendChild(li);
  });
}

function fillSelect(selector, values, selected) {
  const select = qs(selector);
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = SELECT_OPTION_LABELS[value] || value || "Все";
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
    setElementHint(row, `Открыть связи сущности: ${entity.label}`);
    row.addEventListener("click", () => selectEntity(entity.id));
    list.appendChild(row);
  });
  state.entityOffset += data.items.length;
  state.entityHasMore = data.has_more;
  qs("#loadMoreEntities").disabled = !data.has_more;
}

async function selectEntity(entityId) {
  state.selectedEntityId = entityId;
  state.focusEntityId = entityId;
  activateView("explorer");
  await loadSubgraph(entityId);
  qs(".graph-zone").scrollIntoView({ block: "start", inline: "nearest" });
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
      embedding_provider: qs("#embeddingMode").value,
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
    setElementHint(row, `Открыть связи найденной сущности: ${item.label}`);
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
    setElementHint(row, `Открыть субъект связи: ${relation.subject_label}`);
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
    embeddings: diagnostics.embedding_provider || "demo",
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
      embedding_provider: qs("#embeddingMode").value,
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
      const label = button.querySelector("strong")?.textContent || "сущность";
      setElementHint(button, `Открыть связи сущности: ${label}`);
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
  state.defaultGraphViewBox = { ...state.graphViewBox };
  const svg = qs("#graphSvg");
  svg.innerHTML = "";
  setGraphViewBox();
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = graphMarkerDefs();
  svg.appendChild(defs);
  if (!data.nodes.length) {
    const text = svgText("No graph slice", 500, 280, "middle");
    text.setAttribute("class", "graph-empty");
    svg.appendChild(text);
    renderGraphLegend([], []);
    state.focusEntityId = null;
    showEntityPanel(null);
    return;
  }

  const layout = layoutGraph(data);
  const positions = layout.positions;
  state.graphViewBox = { ...layout.viewBox };
  state.defaultGraphViewBox = { ...layout.viewBox };
  setGraphViewBox();
  const showEdgeLabels = qs("#edgeLabelsToggle").checked;
  const visibleRelations = graphRelationsForDisplay(data.nodes, data.relations);
  renderGraphLegend(data.nodes, visibleRelations);
  setText(
    "#graphStatus",
    `${data.nodes.length} nodes В· ${visibleRelations.length}/${data.relations.length} relations`,
  );
  visibleRelations.forEach((relation) => {
    const source = positions.get(relation.subject_id);
    const target = positions.get(relation.object_id);
    if (!source || !target) return;
    const start = edgeEndpoint(source, target, source.radius + 5);
    const end = edgeEndpoint(target, source, target.radius + 8);
    const style = edgeStyleForPredicate(relation.predicate);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", start.x);
    line.setAttribute("y1", start.y);
    line.setAttribute("x2", end.x);
    line.setAttribute("y2", end.y);
    line.setAttribute("class", "graph-link");
    line.setAttribute("data-subject-id", relation.subject_id);
    line.setAttribute("data-object-id", relation.object_id);
    line.setAttribute("stroke", style.color);
    line.setAttribute("marker-end", `url(#arrow-${style.key})`);
    if (style.dash) line.setAttribute("stroke-dasharray", style.dash);
    svg.appendChild(line);
    if (showEdgeLabels) {
      const label = svgText(
        truncate(relation.predicate, 24),
        (source.x + target.x) / 2,
        (source.y + target.y) / 2 - 6,
        "middle",
      );
      label.setAttribute("class", "graph-link-label");
      label.setAttribute("fill", style.color);
      svg.appendChild(label);
    }
  });

  data.nodes.forEach((node) => {
    const position = positions.get(node.id);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `graph-node${node.selected ? " selected" : ""}`);
    group.setAttribute("data-entity-id", node.id);
    group.setAttribute("data-label-priority", String(labelPriority(node)));
    group.setAttribute("tabindex", "0");
    group.addEventListener("click", () => {
      setGraphFocus(node.id);
      showEntityPanel(node);
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        setGraphFocus(node.id);
        showEntityPanel(node);
      }
    });
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", position.x);
    circle.setAttribute("cy", position.y);
    circle.setAttribute("r", String(position.radius));
    circle.setAttribute("fill", colorForType(node.entity_type));
    group.appendChild(circle);
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${node.label} (${node.entity_type})`;
    group.appendChild(title);
    const labelY = position.y + position.radius + 23;
    const labelBox = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    labelBox.setAttribute("x", position.x - position.labelWidth / 2);
    labelBox.setAttribute("y", labelY - 15);
    labelBox.setAttribute("width", position.labelWidth);
    labelBox.setAttribute("height", "22");
    labelBox.setAttribute("rx", "6");
    labelBox.setAttribute("class", "graph-node-label-bg");
    group.appendChild(labelBox);
    const label = svgText(
      truncate(node.label, GRAPH_LAYOUT.maxLabelChars),
      position.x,
      labelY,
      "middle",
    );
    label.setAttribute("class", "graph-node-label");
    group.appendChild(label);
    svg.appendChild(group);
  });
  const selected = data.nodes.find((node) => node.selected) || data.nodes[0];
  setGraphFocus(state.focusEntityId || selected.id);
  showEntityPanel(selected);
  updateGraphLabelVisibility();
}

function graphMarkerDefs() {
  return Object.entries(PREDICATE_STYLES)
    .map(
      ([key, style]) => `
        <marker id="arrow-${key}" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="${style.color}"></path>
        </marker>
      `,
    )
    .join("");
}

function graphRelationsForDisplay(nodes, relations) {
  if (qs("#relationView").value === "all") return relations;
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const centerIds = new Set(nodes.filter((node) => node.selected).map((node) => node.id));
  const focusId = state.focusEntityId && nodeById.has(state.focusEntityId)
    ? state.focusEntityId
    : [...centerIds][0];
  const limit = Math.min(relations.length, Math.max(32, Math.ceil(nodes.length * 1.25)));
  const forced = [];
  const optional = [];
  relations.forEach((relation) => {
    const touchesFocus =
      relation.subject_id === focusId ||
      relation.object_id === focusId ||
      centerIds.has(relation.subject_id) ||
      centerIds.has(relation.object_id);
    const item = {
      relation,
      score: relationImportanceScore(relation, nodeById, focusId, centerIds),
    };
    if (touchesFocus) {
      forced.push(item);
    } else {
      optional.push(item);
    }
  });
  forced.sort(compareRelationScore);
  optional.sort(compareRelationScore);
  return [...forced, ...optional]
    .slice(0, limit)
    .map((item) => item.relation);
}

function relationImportanceScore(relation, nodeById, focusId, centerIds) {
  const source = nodeById.get(relation.subject_id);
  const target = nodeById.get(relation.object_id);
  const predicateWeight = {
    treats: 18,
    diagnosed_by: 15,
    has_protocol: 14,
    improves: 12,
    causes: 10,
    related_to: 3,
  }[relation.predicate] || 6;
  const focusWeight =
    relation.subject_id === focusId || relation.object_id === focusId ? 30 : 0;
  const centerWeight =
    centerIds.has(relation.subject_id) || centerIds.has(relation.object_id) ? 20 : 0;
  const degreeScore = Math.sqrt(Number(source?.degree || 0) + Number(target?.degree || 0));
  const pagerankScore =
    (Number(source?.pagerank || 0) + Number(target?.pagerank || 0)) * 150;
  return (
    focusWeight +
    centerWeight +
    predicateWeight +
    degreeScore +
    pagerankScore +
    Number(relation.confidence || 0)
  );
}

function compareRelationScore(left, right) {
  return (
    right.score - left.score ||
    String(left.relation.predicate).localeCompare(String(right.relation.predicate)) ||
    String(left.relation.subject_label).localeCompare(String(right.relation.subject_label)) ||
    String(left.relation.object_label).localeCompare(String(right.relation.object_label))
  );
}

function renderGraphLegend(nodes, relations) {
  const legend = qs("#graphLegend");
  if (!nodes.length) {
    legend.innerHTML = "";
    return;
  }
  const types = [...new Set(nodes.map((node) => node.entity_type))]
    .sort(compareEntityTypes)
    .slice(0, 8);
  const predicates = [...new Set(relations.map((relation) => relation.predicate))]
    .sort((a, b) => {
      const styleDelta = edgeStyleOrder(a) - edgeStyleOrder(b);
      return styleDelta || a.localeCompare(b);
    })
    .slice(0, 5);
  legend.innerHTML = `
    <div class="legend-section">
      ${types
        .map(
          (type) => `
            <span class="legend-item">
              <span class="legend-dot" style="background:${colorForType(type)}"></span>
              ${escapeHtml(type)}
            </span>
          `,
        )
        .join("")}
    </div>
    <div class="legend-section edge-legend">
      ${predicates
        .map((predicate) => {
          const style = edgeStyleForPredicate(predicate);
          return `
            <span class="legend-item">
              <span class="legend-edge" style="background:${style.color}"></span>
              ${escapeHtml(predicate)}
            </span>
          `;
        })
        .join("")}
    </div>
  `;
}

function labelPriority(node) {
  if (node.selected || node.id === state.focusEntityId) return 3;
  const degree = Number(node.degree || 0);
  if (degree >= 12) return 2;
  if (degree >= 4) return 1;
  return 0;
}

function setGraphFocus(entityId) {
  const graphNodeIds = new Set((state.currentGraph.nodes || []).map((node) => node.id));
  state.focusEntityId = graphNodeIds.has(entityId) ? entityId : null;
  applyGraphFocus();
  updateGraphLabelVisibility();
}

function applyGraphFocus() {
  const focusId = state.focusEntityId;
  const focusEnabled = qs("#focusModeToggle").checked && Boolean(focusId);
  const neighborIds = new Set([focusId]);
  if (focusEnabled) {
    (state.currentGraph.relations || []).forEach((relation) => {
      if (relation.subject_id === focusId) neighborIds.add(relation.object_id);
      if (relation.object_id === focusId) neighborIds.add(relation.subject_id);
    });
  }
  document.querySelectorAll("#graphSvg .graph-node").forEach((group) => {
    const entityId = group.dataset.entityId;
    group.classList.toggle("focused", focusEnabled && entityId === focusId);
    group.classList.toggle("neighbor", focusEnabled && neighborIds.has(entityId));
    group.classList.toggle("dimmed", focusEnabled && !neighborIds.has(entityId));
  });
  document.querySelectorAll("#graphSvg .graph-link").forEach((line) => {
    const connected =
      line.dataset.subjectId === focusId || line.dataset.objectId === focusId;
    line.classList.toggle("focused", focusEnabled && connected);
    line.classList.toggle("dimmed", focusEnabled && !connected);
  });
}

function updateGraphLabelVisibility() {
  const svg = qs("#graphSvg");
  if (!svg) return;
  const zoomRatio = state.graphViewBox.width / Math.max(state.defaultGraphViewBox.width, 1);
  const minimumPriority = zoomRatio > 1.95 ? 3 : zoomRatio > 1.45 ? 2 : zoomRatio > 1.12 ? 1 : 0;
  svg.querySelectorAll(".graph-node").forEach((group) => {
    const priority = Number(group.dataset.labelPriority || 0);
    const focused = group.dataset.entityId === state.focusEntityId;
    group.classList.toggle("labels-hidden", !focused && priority < minimumPriority);
  });
}

function layoutGraph(data) {
  const positions = layoutNodes(data.nodes, data.relations);
  return {
    positions,
    viewBox: graphViewBoxForPositions(positions),
  };
}

function layoutNodes(nodes, relations = []) {
  const adjacency = buildAdjacency(nodes, relations);
  const selectedIds = nodes.filter((node) => node.selected).map((node) => node.id);
  const focusIds = selectedIds.length
    ? selectedIds
    : [sortedNodesForLayout(nodes)[0]?.id].filter(Boolean);
  const distances = graphDistances(focusIds, adjacency);
  const grouped = groupNodesByDepth(nodes, distances, focusIds);
  if (qs("#layoutMode").value === "type_cluster") {
    return layoutNodesByType(nodes, relations, grouped);
  }
  return layoutNodesByNeighborhood(relations, grouped);
}

function layoutNodesByNeighborhood(relations, grouped) {
  const positions = new Map();
  const selected = grouped.get(0) || [];

  if (selected.length === 1) {
    positions.set(selected[0].id, nodePosition(selected[0], 0, 0, 0, true));
  } else if (selected.length > 1) {
    placeRingGroup({
      nodes: selected,
      positions,
      startRadius: 90,
      depth: 0,
      angleOffset: -Math.PI / 2,
      locked: false,
    });
  }

  let radiusCursor = GRAPH_LAYOUT.firstRingRadius;
  const depths = [...grouped.keys()].filter((depth) => depth > 0).sort((a, b) => a - b);
  depths.forEach((depth) => {
    const group = grouped.get(depth) || [];
    const usedRadius = placeRingGroup({
      nodes: group,
      positions,
      startRadius: radiusCursor,
      depth,
      angleOffset: -Math.PI / 2 + depth * 0.43,
      locked: false,
    });
    radiusCursor = usedRadius + GRAPH_LAYOUT.layerGap;
  });

  relaxLayout({
    positions,
    relations,
  });
  return positions;
}

function layoutNodesByType(nodes, relations, grouped) {
  const positions = new Map();
  const selected = grouped.get(0) || [];
  if (selected.length === 1) {
    positions.set(selected[0].id, nodePosition(selected[0], 0, 0, 0, true));
  } else if (selected.length > 1) {
    placeRingGroup({
      nodes: selected,
      positions,
      startRadius: 90,
      depth: 0,
      angleOffset: -Math.PI / 2,
      locked: false,
    });
  }

  const groupedByType = new Map();
  sortedNodesForLayout(nodes)
    .filter((node) => !positions.has(node.id))
    .forEach((node) => {
      if (!groupedByType.has(node.entity_type)) groupedByType.set(node.entity_type, []);
      groupedByType.get(node.entity_type).push(node);
    });
  const typeEntries = [...groupedByType.entries()].sort(([left], [right]) =>
    compareEntityTypes(left, right),
  );
  typeEntries.forEach(([type, typeNodes], typeIndex) => {
    placeTypeCluster({
      type,
      nodes: typeNodes,
      positions,
      typeIndex,
      typeCount: typeEntries.length,
    });
  });

  relaxLayout({
    positions,
    relations,
  });
  return positions;
}

function placeTypeCluster({ type, nodes, positions, typeIndex, typeCount }) {
  const centerAngle = -Math.PI / 2 + (Math.PI * 2 * typeIndex) / Math.max(typeCount, 1);
  const sectorWidth = Math.min(Math.PI * 0.72, (Math.PI * 2) / Math.max(typeCount, 1) * 0.72);
  let index = 0;
  let ring = 0;
  while (index < nodes.length) {
    const radius = GRAPH_LAYOUT.firstRingRadius + ring * GRAPH_LAYOUT.ringGap;
    const capacity = Math.max(
      3,
      Math.floor((Math.max(sectorWidth, 0.35) * radius) / GRAPH_LAYOUT.minArcSpacing),
    );
    const take = Math.min(capacity, nodes.length - index);
    for (let localIndex = 0; localIndex < take; localIndex += 1) {
      const node = nodes[index + localIndex];
      const fraction = take === 1 ? 0.5 : localIndex / (take - 1);
      const jitter = (deterministicAngle(`${type}:${node.id}`) - Math.PI) * 0.04;
      const angle = centerAngle - sectorWidth / 2 + sectorWidth * fraction + jitter;
      positions.set(
        node.id,
        nodePosition(
          node,
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          ring + 1,
          false,
          radius,
          angle,
        ),
      );
    }
    index += take;
    ring += 1;
  }
}

function buildAdjacency(nodes, relations) {
  const adjacency = new Map(nodes.map((node) => [node.id, new Set()]));
  relations.forEach((relation) => {
    if (!adjacency.has(relation.subject_id) || !adjacency.has(relation.object_id)) {
      return;
    }
    adjacency.get(relation.subject_id).add(relation.object_id);
    adjacency.get(relation.object_id).add(relation.subject_id);
  });
  return adjacency;
}

function graphDistances(startIds, adjacency) {
  const distances = new Map();
  const queue = [];
  startIds.forEach((id) => {
    if (!adjacency.has(id) || distances.has(id)) return;
    distances.set(id, 0);
    queue.push(id);
  });
  while (queue.length) {
    const current = queue.shift();
    const nextDepth = distances.get(current) + 1;
    (adjacency.get(current) || []).forEach((neighbor) => {
      if (distances.has(neighbor)) return;
      distances.set(neighbor, nextDepth);
      queue.push(neighbor);
    });
  }
  return distances;
}

function groupNodesByDepth(nodes, distances, focusIds) {
  const grouped = new Map();
  const focusSet = new Set(focusIds);
  const maxKnownDepth = Math.max(1, ...[...distances.values()]);
  sortedNodesForLayout(nodes).forEach((node) => {
    const depth = focusSet.has(node.id)
      ? 0
      : Math.min(distances.get(node.id) ?? maxKnownDepth + 1, maxKnownDepth + 1);
    if (!grouped.has(depth)) grouped.set(depth, []);
    grouped.get(depth).push(node);
  });
  return grouped;
}

function sortedNodesForLayout(nodes) {
  return [...nodes].sort((a, b) => {
    const selectedDelta = Number(b.selected) - Number(a.selected);
    if (selectedDelta) return selectedDelta;
    const degreeDelta = Number(b.degree || 0) - Number(a.degree || 0);
    if (degreeDelta) return degreeDelta;
    const typeDelta = String(a.entity_type || "").localeCompare(String(b.entity_type || ""));
    if (typeDelta) return typeDelta;
    const labelDelta = String(a.label || "").localeCompare(String(b.label || ""));
    if (labelDelta) return labelDelta;
    return String(a.id).localeCompare(String(b.id));
  });
}

function placeRingGroup({ nodes, positions, startRadius, depth, angleOffset, locked }) {
  let index = 0;
  let ring = 0;
  let usedRadius = startRadius;
  while (index < nodes.length) {
    const radius = startRadius + ring * GRAPH_LAYOUT.ringGap;
    const capacity = Math.max(6, Math.floor((Math.PI * 2 * radius) / GRAPH_LAYOUT.minArcSpacing));
    const take = Math.min(capacity, nodes.length - index);
    const ringOffset = angleOffset + (ring % 2 ? Math.PI / Math.max(take, 1) : 0);
    for (let localIndex = 0; localIndex < take; localIndex += 1) {
      const node = nodes[index + localIndex];
      const angle = ringOffset + (Math.PI * 2 * localIndex) / Math.max(take, 1);
      positions.set(
        node.id,
        nodePosition(
          node,
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          depth,
          locked,
          radius,
          angle,
        ),
      );
    }
    usedRadius = radius;
    index += take;
    ring += 1;
  }
  return usedRadius;
}

function nodePosition(
  node,
  x,
  y,
  depth,
  locked,
  anchorRadius = Math.hypot(x, y),
  anchorAngle = Math.atan2(y, x),
) {
  const radius = nodeRadius(node);
  const labelWidth = labelBoxWidth(node);
  return {
    x,
    y,
    radius,
    labelWidth,
    collisionRadius: Math.max(radius + 40, labelWidth / 2 + 24),
    depth,
    locked,
    anchorRadius,
    anchorAngle,
  };
}

function nodeRadius(node) {
  return 15 + Math.min(13, Math.sqrt(Math.max(0, Number(node.degree || 0))) * 3.6);
}

function labelBoxWidth(node) {
  const label = String(node.label || "");
  const visibleChars = Math.min(GRAPH_LAYOUT.maxLabelChars, label.length);
  return Math.max(72, Math.min(174, visibleChars * 7 + 22));
}

function relaxLayout({ positions, relations }) {
  const positioned = [...positions.entries()].map(([id, position]) => ({
    id,
    position,
  }));
  const relationPairs = relations
    .map((relation) => [
      positions.get(relation.subject_id),
      positions.get(relation.object_id),
    ])
    .filter(([source, target]) => source && target);

  for (let pass = 0; pass < GRAPH_LAYOUT.relaxationPasses; pass += 1) {
    separateAllNodes(positioned);
    relationPairs.forEach(([source, target]) => pullLinkedNodes(source, target));
    positioned.forEach(({ position }) => pullToAnchor(position));
  }
  for (let pass = 0; pass < 40; pass += 1) {
    separateAllNodes(positioned);
  }
}

function separateAllNodes(positioned) {
  for (let i = 0; i < positioned.length; i += 1) {
    const current = positioned[i].position;
    for (let j = i + 1; j < positioned.length; j += 1) {
      const other = positioned[j].position;
      separateNodes(current, other, positioned[i].id, positioned[j].id);
    }
  }
}

function separateNodes(a, b, firstId, secondId) {
  let dx = b.x - a.x;
  let dy = b.y - a.y;
  let distance = Math.hypot(dx, dy);
  if (distance < 0.001) {
    const angle = deterministicAngle(`${firstId}:${secondId}`);
    dx = Math.cos(angle) * 0.01;
    dy = Math.sin(angle) * 0.01;
    distance = 0.01;
  }
  const minimum = a.collisionRadius + b.collisionRadius + 28;
  if (distance >= minimum) return;
  const push = ((minimum - distance) / distance) * 0.52;
  const aWeight = a.locked ? 0 : b.locked ? 1 : 0.5;
  const bWeight = b.locked ? 0 : a.locked ? 1 : 0.5;
  a.x -= dx * push * aWeight;
  a.y -= dy * push * aWeight;
  b.x += dx * push * bWeight;
  b.y += dy * push * bWeight;
}

function pullLinkedNodes(source, target) {
  if (source.locked && target.locked) return;
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const desired = source.collisionRadius + target.collisionRadius + 72;
  const force = ((distance - desired) / distance) * 0.012;
  if (!source.locked) {
    source.x += dx * force;
    source.y += dy * force;
  }
  if (!target.locked) {
    target.x -= dx * force;
    target.y -= dy * force;
  }
}

function pullToAnchor(position) {
  if (position.locked) return;
  const targetX = Math.cos(position.anchorAngle) * position.anchorRadius;
  const targetY = Math.sin(position.anchorAngle) * position.anchorRadius;
  position.x += (targetX - position.x) * 0.018;
  position.y += (targetY - position.y) * 0.018;
}

function deterministicAngle(value) {
  let hash = 0;
  String(value)
    .split("")
    .forEach((char) => {
      hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    });
  return (hash / 2 ** 32) * Math.PI * 2;
}

function edgeEndpoint(from, to, offset) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  return {
    x: from.x + (dx / distance) * offset,
    y: from.y + (dy / distance) * offset,
  };
}

function graphViewBoxForPositions(positions) {
  const values = [...positions.values()];
  if (!values.length) return { x: 0, y: 0, width: 1000, height: 560 };
  const bounds = values.reduce(
    (acc, position) => {
      const horizontal = Math.max(position.collisionRadius, position.labelWidth / 2) + 18;
      const vertical = position.radius + 48;
      return {
        minX: Math.min(acc.minX, position.x - horizontal),
        maxX: Math.max(acc.maxX, position.x + horizontal),
        minY: Math.min(acc.minY, position.y - vertical),
        maxY: Math.max(acc.maxY, position.y + vertical),
      };
    },
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );
  const width = Math.max(GRAPH_LAYOUT.minWidth, bounds.maxX - bounds.minX + GRAPH_LAYOUT.margin * 2);
  const height = Math.max(
    GRAPH_LAYOUT.minHeight,
    bounds.maxY - bounds.minY + GRAPH_LAYOUT.margin * 2,
  );
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  return {
    x: centerX - width / 2,
    y: centerY - height / 2,
    width,
    height,
  };
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
    <span>${escapeHtml(node.description || "Описание не указано")}</span>
    <div class="chip-list">${chipsHtml(node.aliases || [], "aliases")}</div>
    <div class="chip-list">${chipsHtml(node.source_chunks || [], "source")}</div>
    <h4>Связи</h4>
    <div class="relation-list">${relationHtml}</div>
  `;
  container.querySelectorAll("button[data-entity-id]").forEach((button) => {
    const label = button.textContent.trim().replace(/\s+/g, " ");
    setElementHint(button, `Открыть соседнюю сущность по связи: ${label}`);
    button.addEventListener("click", () => selectEntity(button.dataset.entityId));
  });
}

function chipsHtml(values, emptyLabel) {
  const emptyLabels = {
    aliases: "синонимов",
    source: "источников",
  };
  if (!values.length) {
    return `<span class="muted">Нет ${emptyLabels[emptyLabel] || emptyLabel}</span>`;
  }
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
    llm: qs("#pipelineLlmMode").value,
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
      llm: job.summary.llm || job.llm,
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
  if (data.baseline_metrics) {
    container.appendChild(metricTable("Baselines", data.baseline_metrics));
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

function compareEntityTypes(left, right) {
  const leftIndex = ENTITY_TYPE_ORDER.indexOf(left);
  const rightIndex = ENTITY_TYPE_ORDER.indexOf(right);
  return (
    (leftIndex === -1 ? ENTITY_TYPE_ORDER.length : leftIndex) -
      (rightIndex === -1 ? ENTITY_TYPE_ORDER.length : rightIndex) ||
    left.localeCompare(right)
  );
}

function edgeStyleForPredicate(predicate) {
  const key = Object.prototype.hasOwnProperty.call(PREDICATE_STYLES, predicate)
    ? predicate
    : "default";
  return { key, ...PREDICATE_STYLES[key] };
}

function edgeStyleOrder(predicate) {
  return Object.keys(PREDICATE_STYLES).indexOf(edgeStyleForPredicate(predicate).key);
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
  updateGraphLabelVisibility();
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
  state.graphViewBox = { ...state.defaultGraphViewBox };
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
  qs("#layoutMode").addEventListener("change", () => renderGraph(state.currentGraph));
  qs("#relationView").addEventListener("change", () => renderGraph(state.currentGraph));
  qs("#focusModeToggle").addEventListener("change", applyGraphFocus);
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

localizeDoctorControls();
wireEvents();
activateInitialView();
loadDashboard().catch((error) => {
  qs("#validationBadge").textContent = "Ошибка";
  qs("#answerBox").textContent = error.message;
});

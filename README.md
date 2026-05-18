# HiRAG-Ontology

HiRAG-Ontology is a Python prototype for a
multi-agent RAG system that constructs and queries a typed medical knowledge
graph from Markdown documents.

This repository is a Python research prototype scaffold with a local Gemma 4
runtime through Ollama plus optional ChatGPT/OpenAI and DeepSeek API backends.
Tests use deterministic test doubles and do not require network access, cloud
services, or API keys.

## MVP Architecture

The planned MVP follows a small, testable architecture:

1. Document ingestion reads Markdown files from a local directory.
2. Chunking creates deterministic overlapping word-level chunks.
3. Extraction uses an LLM abstraction to produce entities and relations.
4. Typing maps entities to ontology classes.
5. Deduplication merges near-duplicate entities.
6. Validation checks graph consistency against ontology constraints.
7. Reasoning adds simple inferred relations.
8. Retrieval uses Russian-aware lexical matching, optional structural PageRank,
   semantic retrieval, and RRF fusion modes.
9. The CLI runs graph building and question answering with Gemma, OpenAI, or
   DeepSeek backends.
10. Tests use internal test doubles for deterministic coverage.

For the MVP, graph storage uses NetworkX plus JSON persistence by default. Neo4j
is available as an optional export/query backend; it is not required for tests or
basic question answering.

## Current Layout

```text
src/hirag_ontology/
  cli.py
  config.py
  llm.py
  ontology.py
  pipeline/
  retrieval/
  storage/
  evaluation/
  app/
tests/
docs/
data/
```

## Development

Install and run commands with `uv`:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
```

or run the same checks through Make:

```bash
make verify
```

Run the pipeline with local Gemma 4:

```bash
uv run hirag-ontology run-demo --input data/sample_docs --out results/demo_graph.json --llm gemma
```

or:

```bash
uv run python -m hirag_ontology.cli run-demo --input data/sample_docs --out results/demo_graph.json --llm gemma
```

This reads `data/sample_docs`, builds a graph, writes `results/demo_graph.json`,
writes `results/run_summary.json`, and runs one sample retrieval query.

Ask a question against a saved graph with:

```bash
uv run python -m hirag_ontology.cli ask \
  --graph results/knowledge_graph_repaired.json \
  --query "How is Ph+ ALL treated?" \
  --llm gemma \
  --retrieval-mode lexical_structural \
  --show-context
```

## Prebuilt Full Graph

The repository includes a prebuilt full graph converted from
`ekaesha/hirag-ontology`:

```text
results/knowledge_graph_full_gemma.json
```

It also includes a validation-repaired graph used by the Web UI by default:

```text
results/knowledge_graph_repaired.json
results/knowledge_graph_repaired.repair_report.json
```

The repaired graph preserves the same entities, reverses ontology-compatible
backward relations, relaxes unresolved domain/range conflicts to `related_to`,
deduplicates exact duplicate relations, and reaches zero ontology violations.

Use it to test question answering without rebuilding the graph:

```bash
uv run python -m hirag_ontology.cli ask \
  --graph results/knowledge_graph_repaired.json \
  --query "Как лечить Острый лимфобластный лейкоз (ОЛЛ)?" \
  --llm gemma \
  --retrieval-mode lexical_structural \
  --show-context
```

For Russian clinical questions, `lexical_structural` is the default and usually
works better than `hybrid_rrf` while the semantic embedding provider is still an
MVP/demo component. Retrieval normalizes common variants such as `ОЛЛ`, `BCR ABL`
/ `BCR-ABL` / `BCR::ABL`, `ТКИ`, and `ингибиторы тирозинкиназы`.

Print graph statistics with:

```bash
uv run python -m hirag_ontology.cli graph-stats \
  --graph results/knowledge_graph_repaired.json
```

Repair a graph again with:

```bash
uv run python -m hirag_ontology.cli repair-graph \
  --graph results/knowledge_graph_full_gemma.json \
  --out results/knowledge_graph_repaired.json
```

## Optional Neo4j Export

Neo4j is optional. The JSON graph remains the portable source artifact. Install
the Neo4j extra only when you want to export/query the graph in Neo4j:

```bash
uv sync --extra neo4j
```

Configure Neo4j through `.env` or environment variables:

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
```

Export the prebuilt graph:

```bash
uv run python -m hirag_ontology.cli export-neo4j \
  --graph results/knowledge_graph_repaired.json \
  --clear
```

The export command never logs the Neo4j password. Unit tests use a fake driver;
live Neo4j tests should be marked with `pytest.mark.neo4j`.

## Web UI

Start the local Web UI:

```bash
uv --cache-dir .uv-cache run python -m hirag_ontology.cli web
```

Then open:

```text
http://127.0.0.1:8765
```

The UI includes:

- dashboard metrics: entity count, relation count, type distribution, PageRank,
  degree, and ontology validation status;
- question answering with retrieval mode, `top_k`, answer text, retrieved
  entities, graph context, and retrieved subgraph rendering;
- graph explorer with entity search, type/predicate filters, depth 1-3, node
  limits, clickable nodes, and an entity panel;
- pipeline upload for Markdown files, A1-A6 stage status, saved graph output,
  and optional import of the result to Neo4j.

By default, the UI opens:

```text
results/knowledge_graph_repaired.json
```

To choose another graph at startup:

```bash
uv --cache-dir .uv-cache run python -m hirag_ontology.cli web \
  --graph results/demo_graph.json \
  --host 127.0.0.1 \
  --port 8765
```

Ask mode can use local Gemma, ChatGPT/OpenAI API, DeepSeek API, or deterministic
graph-only answers. The default retrieval mode is `lexical_structural`, which
prioritizes Russian medical term matches and uses PageRank only as a secondary
signal. Pipeline runs can use Gemma, OpenAI, or DeepSeek and may take a long
time on large document sets. Neo4j import requires `NEO4J_PASSWORD` in `.env`
and the optional Neo4j package.

Semantic and hybrid retrieval can optionally use real multilingual embeddings.
The default `demo` provider remains deterministic for tests. Set
`EMBEDDING_PROVIDER=ollama` with a local embedding model such as
`mxbai-embed-large`, or `EMBEDDING_PROVIDER=openai` with an OpenAI-compatible
`/embeddings` endpoint. Runtime logs and errors never print embedding API keys.

## Local Gemma 4 Runtime

Gemma calls are local and never used by tests. Install Ollama, pull Gemma 4,
and start the local Ollama service:

```bash
ollama pull gemma4
ollama serve
```

Create a local `.env` from `.env.example`:

```bash
cp .env.example .env
```

Optional settings:

```text
GEMMA_BASE_URL=http://localhost:11434
GEMMA_MODEL=gemma4:latest
GEMMA_MAX_RETRIES=2
GEMMA_MIN_REQUEST_INTERVAL_SECONDS=0.5
GEMMA_TEMPERATURE=0.0
GEMMA_REQUEST_TIMEOUT_SECONDS=600
```

## Optional Remote LLM APIs

OpenAI and DeepSeek are opt-in. Tests never require these keys, and runtime
logs only show whether a key is configured, never the key value.

Configure OpenAI / ChatGPT API in `.env`:

```text
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_RETRIES=2
OPENAI_MIN_REQUEST_INTERVAL_SECONDS=0.5
OPENAI_TEMPERATURE=0.0
OPENAI_REQUEST_TIMEOUT_SECONDS=120
```

Configure DeepSeek API in `.env`:

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_RETRIES=2
DEEPSEEK_MIN_REQUEST_INTERVAL_SECONDS=0.5
DEEPSEEK_TEMPERATURE=0.0
DEEPSEEK_REQUEST_TIMEOUT_SECONDS=120
```

Use a backend from the CLI:

```bash
uv run python -m hirag_ontology.cli ask \
  --graph results/knowledge_graph_repaired.json \
  --query "Какой протокол используется при ОПЛ?" \
  --llm openai

uv run python -m hirag_ontology.cli ask \
  --graph results/knowledge_graph_repaired.json \
  --query "Какой протокол используется при ОПЛ?" \
  --llm deepseek
```

The Web UI exposes the same choice in the answer settings and in the pipeline
form. Remote API calls use rate limiting, retry handling, JSON mode for
extraction/typing, and graph-grounded answer prompts.

Run the pipeline with a selected LLM backend:

```bash
uv run python -m hirag_ontology.cli run-demo \
  --input data/sample_docs \
  --out results/demo_graph.json \
  --llm gemma
```

The runtime loads `.env` and environment variables, applies retry handling and a
minimum interval between requests, and logs cache hits/misses by cache filename.

Real answer generation is also local:

```bash
uv run python -m hirag_ontology.cli ask \
  --graph results/demo_graph.json \
  --query "How is Ph+ ALL treated?" \
  --llm gemma
```

## Evaluation

The repository includes an evaluation suite in the same spirit as
`ekaesha/hirag-ontology`: retrieval metrics, generation metrics, latency
measurement, and deduplication ablation. The default 50-question benchmark
annotations live in:

```text
evaluation/ground_truth.json
```

Run the full deterministic evaluation suite against the bundled graph:

```bash
uv run python -m hirag_ontology.cli evaluate \
  --kg results/knowledge_graph_repaired.json \
  --gt evaluation/ground_truth.json \
  --out-dir results
```

This writes reproducible JSON artifacts:

```text
results/retrieval_metrics.json
results/retrieval_metrics_per_question.json
results/baseline_metrics.json
results/baseline_metrics_per_question.json
results/generation_metrics.json
results/generation_metrics_per_question.json
results/latency_results.json
results/dedup_ablation.json
results/full_evaluation_report.json
results/evaluation_report.md
```

The default generation evaluation is graph-only and deterministic, so it does
not call Ollama. LLM-as-judge helpers are available in
`src/hirag_ontology/evaluation/llm_judge.py`, but tests do not require a live
model.

You can run individual evaluation modules too:

```bash
uv run python -m hirag_ontology.evaluation.retrieval_eval \
  --kg results/knowledge_graph_repaired.json \
  --gt evaluation/ground_truth.json \
  --top-k 10

uv run python -m hirag_ontology.evaluation.generation_eval \
  --kg results/knowledge_graph_repaired.json \
  --gt evaluation/ground_truth.json \
  --mode lexical_structural

uv run python -m hirag_ontology.evaluation.latency_eval \
  --kg results/knowledge_graph_repaired.json \
  --gt evaluation/ground_truth.json \
  --n-queries 20

uv run python -m hirag_ontology.evaluation.dedup_ablation \
  --kg results/knowledge_graph_repaired.json
```

The included ground truth is a 50-question benchmark whose
`relevant_entity_labels` are checked against the bundled graph labels and
aliases. Treat it as reproducible MVP evaluation; add external clinical review
before using the numbers as clinical-quality evidence.

## Safety

This project is a research prototype and is not intended for clinical
decision-making. Tests must use test doubles and must not require Ollama,
network access, or API keys.

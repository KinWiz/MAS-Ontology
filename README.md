# HiRAG-Ontology

HiRAG-Ontology is a Python prototype for a
multi-agent RAG system that constructs and queries a typed medical knowledge
graph from Markdown documents.

This repository is a Python research prototype scaffold with a local Gemma 4
runtime through Ollama. Tests use deterministic test doubles and do not require
network access, cloud services, or API keys.

## MVP Architecture

The planned MVP follows a small, testable architecture:

1. Document ingestion reads Markdown files from a local directory.
2. Chunking creates deterministic overlapping word-level chunks.
3. Extraction uses an LLM abstraction to produce entities and relations.
4. Typing maps entities to ontology classes.
5. Deduplication merges near-duplicate entities.
6. Validation checks graph consistency against ontology constraints.
7. Reasoning adds simple inferred relations.
8. Retrieval combines semantic, lexical, and structural rankings through RRF.
9. The CLI runs graph building and question answering with local Gemma 4.
10. Tests use internal test doubles for deterministic coverage.

For the MVP, graph storage should prefer NetworkX plus JSON persistence. Neo4j,
real LLM providers, web serving, and full-corpus evaluation can be added later as
optional runtime features.

## Current Layout

```text
src/hirag_ontology/
  cli.py
  config.py
  llm.py
  ontology.py
  pipeline/
  retrieval/
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
  --graph results/knowledge_graph_full_gemma.json \
  --query "How is Ph+ ALL treated?" \
  --llm gemma \
  --show-context
```

## Prebuilt Full Graph

The repository includes a prebuilt full graph converted from
`ekaesha/hirag-ontology`:

```text
results/knowledge_graph_full_gemma.json
```

Use it to test question answering without rebuilding the graph:

```bash
uv run python -m hirag_ontology.cli ask \
  --graph results/knowledge_graph_full_gemma.json \
  --query "How is Ph+ ALL treated?" \
  --llm gemma \
  --retrieval-mode lexical_only \
  --show-context
```

## Local Gemma 4 Runtime

LLM calls are local and never used by tests. Install Ollama, pull Gemma 4, and
start the local Ollama service:

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

Run the pipeline with local Gemma 4:

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

## Safety

This project is a research prototype and is not intended for clinical
decision-making. Tests must use test doubles and must not require Ollama,
network access, or API keys.

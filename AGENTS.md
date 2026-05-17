\# AGENTS.md



\## Project goal



Rebuild a research prototype of HiRAG-Ontology:

a multi-agent RAG system that builds a typed medical knowledge graph from Markdown documents

and retrieves entities using hybrid RRF over semantic, lexical, and structural signals.



\## Repository layout



\- src/hirag\_ontology/pipeline/: graph construction agents

\- src/hirag\_ontology/retrieval/: BM25, embeddings, PageRank, RRF retrieval

\- src/hirag\_ontology/evaluation/: baselines and LLM-as-judge scripts

\- tests/: deterministic unit and integration tests

\- data/sample\_docs/: tiny sample documents for tests

\- data/documents/minzdrav\_dataset/: real corpus, ignored by default in tests



\## Engineering rules



\- Use Python 3.11+.

\- Use src/ package layout.

\- Use Pydantic models or dataclasses for structured data.

\- Do not commit API keys or secrets.

\- Do not use real LLM calls in unit tests.

\- Use FakeLLMClient fixtures for extraction and typing tests.

\- Keep all LLM prompts in code as explicit templates.

\- Keep pipeline stages independently testable.

\- Prefer small, reviewable changes.

\- Never invent evaluation numbers; write scripts that compute them.



\## Commands



Run these before considering a task complete:



```bash

uv run pytest -q

uv run ruff check .

uv run python -m hirag\_ontology.cli run-demo --input data/sample\_docs --out results/demo\_graph.json


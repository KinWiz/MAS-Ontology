# HiRAG-Ontology Rebuild Specification

## 1. Project summary

This repository rebuilds a research prototype of **HiRAG-Ontology**: a multi-agent Retrieval-Augmented Generation (RAG) system that constructs, improves, validates, and queries a typed medical knowledge graph from unstructured clinical guideline documents.

The system should process Markdown documents, extract medical entities and semantic relations, type entities according to a domain ontology, merge duplicates, validate logical constraints, infer simple missing relations, compute structural graph signals, and answer questions using a hybrid entity retriever.

The MVP must be a **testable Python research prototype**, not a production clinical decision-support system.

## 2. Goals

### 2.1 MVP goals

The MVP must:

1. Read Markdown documents from a local directory.
2. Split documents into deterministic overlapping word-level chunks.
3. Extract entities and relations from chunks through an LLM abstraction.
4. Support deterministic test doubles for tests.
5. Store a knowledge graph in memory using NetworkX and persist it as JSON.
6. Load a small JSON ontology defining entity classes, relation types, and domain/range constraints.
7. Type entities using an LLM abstraction or deterministic test double.
8. Deduplicate entities using hybrid lexical-semantic similarity.
9. Validate the graph against ontology constraints.
10. Auto-repair simple validation errors.
11. Infer simple missing relations.
12. Compute PageRank over the graph.
13. Retrieve relevant entities with semantic, lexical, structural, and hybrid RRF modes.
14. Build graph context for a downstream answer-generation call.
15. Provide a CLI demo that runs without API keys.
16. Include pytest coverage for all core modules.

### 2.2 Non-goals for the MVP

The MVP does **not** need to:

1. Implement a production Neo4j backend.
2. Implement full OWL 2 reasoning through HermiT/Pellet.
3. Reproduce all thesis metrics exactly.
4. Use real clinical documents in tests.
5. Perform OCR.
6. Provide medical advice.
7. Deploy a public web service.
8. Require real cloud LLM calls for unit tests.

Neo4j, OWL reasoner integration, web demo, LangChain integration, and full-corpus evaluation may be added later after the MVP is stable.

## 3. Source project being rebuilt

The original thesis describes a system for automatic construction and iterative improvement of ontologies from unstructured documents to improve RAG. The system introduces a structured intermediate layer consisting of an ontology, knowledge graph, and specialized agents. The key technical contribution is a hybrid entity retriever combining lexical BM25, semantic embeddings, and PageRank structural centrality through Reciprocal Rank Fusion (RRF).

The implementation in this repository should follow the thesis architecture, but simplify storage to NetworkX + JSON for the MVP.

## 4. Core terminology

- **Document**: one Markdown file containing clinical guideline text.
- **Chunk**: a contiguous word-level segment produced by sliding-window segmentation.
- **Entity**: a typed medical concept, such as a drug, condition, procedure, lab test, or dosage regimen.
- **Relation**: a directed typed edge between two entities.
- **Triplet**: subject-predicate-object representation of one relation.
- **Ontology**: JSON specification of entity classes, relation types, and constraints.
- **Knowledge graph**: directed graph of entities and typed semantic relations.
- **Canonical entity**: selected representative of a cluster of duplicate entities.
- **Hybrid retrieval**: retrieval that fuses semantic, lexical, and structural ranked lists.
- **RRF**: Reciprocal Rank Fusion, used to combine ranked lists.

## 5. Formal model

Let the input corpus be:

```text
D = {d1, d2, ..., dN}
```

Each document is split into chunks by a segmentation function:

```text
S(di) = {ci1, ci2, ..., ciK}
```

The knowledge graph is a directed labeled graph:

```text
G = (V, E)
```

where:

- `V` is the set of entity nodes.
- `E` is the set of directed typed relations.
- Each relation is represented as `(subject_id, predicate, object_id)`.

The ontology is:

```text
O = (C, R, A)
```

where:

- `C` is the set of entity classes.
- `R` is the set of relation types.
- `A` is the set of validation constraints.

The graph quality functional is:

```text
Q(G) = λ1 * Coverage(G) + λ2 * Cons(G, O) + λ3 * Precision(G) - λ4 * Redundancy(G)
```

Default weights:

```text
λ1 = 0.3
λ2 = 0.3
λ3 = 0.2
λ4 = 0.2
```

For the MVP, `Precision(G)` may be estimated from confidence scores or set to a configurable default when no annotations exist.

## 6. Expected repository layout

Use a `src/` package layout:

```text
hirag-ontology-rebuild/
  AGENTS.md
  README.md
  pyproject.toml
  ontology.json
  .env.example
  docs/
    spec.md
    thesis_summary.md
  data/
    sample_docs/
      all_guideline.md
      apl_guideline.md
    documents/
      minzdrav_dataset/
  results/
    .gitkeep
  src/
    hirag_ontology/
      __init__.py
      config.py
      ontology.py
      llm.py
      embedding.py
      cli.py
      pipeline/
        __init__.py
        chunking.py
        knowledge_graph.py
        extractor.py
        typing_agent.py
        deduplication.py
        validator.py
        reasoning.py
        quality.py
        runner.py
      retrieval/
        __init__.py
        rrf.py
        retriever.py
      evaluation/
        __init__.py
        llm_judge.py
        run_eval.py
      app/
        __init__.py
        web_demo.py
  tests/
    conftest.py
    test_chunking.py
    test_knowledge_graph.py
    test_ontology.py
    test_extractor.py
    test_typing_agent.py
    test_deduplication.py
    test_validator.py
    test_reasoning.py
    test_quality.py
    test_retriever.py
    test_pipeline_demo.py
```

## 7. Technology stack

Use Python 3.11+.

Core dependencies:

```text
networkx
pydantic
rank-bm25
numpy
pytest
ruff
mypy
```

Optional dependencies:

```text
Ollama local runtime with Gemma 4
sentence-transformers
pymorphy3
fastapi
uvicorn
langchain
neo4j
```

The MVP tests must not require optional dependencies unless the specific test is skipped when the dependency is absent.

## 8. Configuration

Configuration must be loaded from environment variables and/or CLI arguments.

`.env.example` should include:

```text
GEMMA_BASE_URL=http://localhost:11434
GEMMA_MODEL=gemma4:latest
GEMMA_MAX_RETRIES=2
GEMMA_MIN_REQUEST_INTERVAL_SECONDS=0.5
GEMMA_TEMPERATURE=0.0
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
HIRAG_CACHE_DIR=.cache/hirag_ontology
HIRAG_RESULTS_DIR=results
```

Unit tests must not require `.env`, Ollama, network access, or API keys.

## 9. Data ingestion and chunking

### 9.1 Input format

The MVP reads `.md` files from a directory. Markdown heading hierarchy, table text, and section text should be preserved as plain text. Files should be processed in deterministic sorted order.

Default input directory:

```text
data/documents/minzdrav_dataset/
```

Test/demo input directory:

```text
data/sample_docs/
```

### 9.2 Chunking

Implement word-level sliding-window chunking:

```python
chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]
```

Requirements:

- Split by whitespace into words.
- Use target chunk size of 800 words.
- Use overlap of 100 words.
- Preserve deterministic chunk order.
- Do not emit empty chunks.
- Validate that `overlap < chunk_size`.

Also implement a richer document loader returning chunk metadata:

```python
@dataclass
class TextChunk:
    document_id: str
    chunk_id: str
    text: str
    source_path: str
    start_word: int
    end_word: int
```

`document_id` should be stable across runs, e.g. based on file stem or normalized relative path.

## 10. Ontology specification

Create `ontology.json` in the repository root.

### 10.1 Entity classes

The ontology must include exactly these MVP classes:

```text
Drug
Condition
Procedure
Symptom
AnatomicalStructure
DosageRegimen
LabTest
Organization
Other
```

Suggested class definitions:

```json
{
  "Drug": "Medication, active substance, chemotherapy drug, targeted therapy, or pharmacological agent.",
  "Condition": "Disease, diagnosis, clinical condition, cancer type, subtype, complication, or syndrome.",
  "Procedure": "Medical intervention, treatment phase, diagnostic procedure, surgery, monitoring action, or care protocol.",
  "Symptom": "Clinical sign, symptom, toxicity, side effect, or patient-observed manifestation.",
  "AnatomicalStructure": "Body part, organ, tissue, cell type, or anatomical location.",
  "DosageRegimen": "Dose, frequency, duration, route, schedule, cycle, or treatment regimen.",
  "LabTest": "Laboratory test, biomarker test, imaging test, cytogenetic analysis, PCR, FISH, or diagnostic measurement.",
  "Organization": "Medical institution, authority, guideline publisher, clinical group, or regulator.",
  "Other": "Fallback class for entities that do not fit another class."
}
```

### 10.2 Relation types

The ontology must include exactly these MVP relation types:

```text
treats
causes
contraindicated_for
part_of
diagnosed_by
dosage_is
related_to
```

### 10.3 Domain/range constraints

Use the following constraints for the MVP:

| Relation | Domain | Range | Meaning |
|---|---|---|---|
| `treats` | `Drug` | `Condition` | A drug treats a condition. |
| `causes` | `Drug` | `Symptom` | A drug may cause a symptom or adverse event. |
| `contraindicated_for` | `Drug` | `Condition` | A drug is contraindicated for a condition. |
| `part_of` | `Procedure` | `Procedure` | A procedure or phase is part of another procedure/protocol. |
| `diagnosed_by` | `Condition` | `LabTest` | A condition is diagnosed by a test or diagnostic procedure. |
| `dosage_is` | `Drug` | `DosageRegimen` | A drug has a dosage/regimen. |
| `related_to` | `Other` | `Other` | Fallback relation for weak or general associations. |

`Other` as domain/range means the validator should not enforce a strict type check for that side.

### 10.4 Ontology JSON structure

`ontology.json` should use this structure:

```json
{
  "classes": {
    "Drug": {"description": "..."},
    "Condition": {"description": "..."},
    "Procedure": {"description": "..."},
    "Symptom": {"description": "..."},
    "AnatomicalStructure": {"description": "..."},
    "DosageRegimen": {"description": "..."},
    "LabTest": {"description": "..."},
    "Organization": {"description": "..."},
    "Other": {"description": "..."}
  },
  "relations": {
    "treats": {"domain": "Drug", "range": "Condition", "description": "..."},
    "causes": {"domain": "Drug", "range": "Symptom", "description": "..."},
    "contraindicated_for": {"domain": "Drug", "range": "Condition", "description": "..."},
    "part_of": {"domain": "Procedure", "range": "Procedure", "description": "..."},
    "diagnosed_by": {"domain": "Condition", "range": "LabTest", "description": "..."},
    "dosage_is": {"domain": "Drug", "range": "DosageRegimen", "description": "..."},
    "related_to": {"domain": "Other", "range": "Other", "description": "..."}
  },
  "axioms": [
    "valid_entity_type",
    "valid_predicate",
    "domain_constraint",
    "range_constraint",
    "no_self_loops"
  ]
}
```

## 11. Knowledge graph data model

Implement `src/hirag_ontology/pipeline/knowledge_graph.py`.

### 11.1 Entity

```python
@dataclass
class Entity:
    label: str
    entity_type: str = "Other"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Entity IDs must be deterministic:

```text
entity_id = md5(normalized_label).hexdigest()
```

Normalization:

- lowercase;
- trim whitespace;
- collapse repeated whitespace;
- normalize punctuation spacing;
- do not remove medically meaningful symbols such as `+`, `-`, `/`, or digits.

### 11.2 Relation

```python
@dataclass
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source_chunk: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 11.3 KnowledgeGraph

```python
class KnowledgeGraph:
    entities: dict[str, Entity]
    relations: list[Relation]
    graph: nx.DiGraph
    pagerank: dict[str, float]
```

Required methods:

```python
add_entity(entity: Entity) -> str
add_relation(subject_label: str, predicate: str, object_label: str, confidence: float = 1.0, source_chunk: str | None = None) -> None
add_relation_by_ids(subject_id: str, predicate: str, object_id: str, confidence: float = 1.0, source_chunk: str | None = None) -> None
compute_pagerank(alpha: float = 0.85, max_iter: int = 200) -> dict[str, float]
save(path: str | Path) -> None
load(path: str | Path) -> KnowledgeGraph
neighbors(entity_id: str, depth: int = 1) -> list[str]
get_entity(entity_id: str) -> Entity
```

Rules:

- `add_entity` should merge aliases and source chunks if an entity ID already exists.
- `add_relation` should create missing subject/object entities if necessary.
- Self-loops should be rejected unless explicitly allowed by a parameter.
- Graph edges should store `predicate` and `confidence`.
- JSON roundtrips must preserve all entities, relations, aliases, source chunks, and PageRank scores.

## 12. LLM abstraction

Implement `src/hirag_ontology/llm.py`.

### 12.1 LLM interface

```python
class LLMClient(Protocol):
    def complete_json(self, prompt: str, *, schema_name: str | None = None) -> dict[str, Any]:
        ...

    def complete_text(self, prompt: str) -> str:
        ...
```

### 12.2 Test LLM client

The test LLM client must be deterministic and usable in tests.

It may support fixtures such as:

```python
FakeLLMClient(json_responses={
    "extract": {...},
    "type:imatinib": {"class": "Drug", "confidence": 0.95}
})
```

The test client must not make network calls.

### 12.3 Local Gemma 4 client

Implement an optional runtime wrapper using local Ollama with Gemma 4:

```python
class GemmaOllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"): ...
```

Requirements:

- Support Ollama JSON mode for structured extraction and typing.
- Use temperature 0 by default.
- Add retry handling for transient failures.
- Do not require or log API keys.
- Not used by unit tests.

## 13. Embedding abstraction

Implement `src/hirag_ontology/embedding.py`.

```python
class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str], *, normalize: bool = True) -> list[list[float]]:
        ...
```

Implement:

1. `FakeEmbeddingProvider` for deterministic tests.
2. Optional `SentenceTransformerEmbeddingProvider` for real runtime.

Tests must use `FakeEmbeddingProvider`.

## 14. Multi-agent pipeline

The complete agent sequence is:

```text
A1 Extraction -> A2 Typing -> A3 Deduplication -> A4 Validation -> A5 Reasoning -> A6 Update/PageRank/Persistence
```

Each agent should be independently testable.

## 15. A1 ExtractionAgent

Implement `src/hirag_ontology/pipeline/extractor.py`.

### 15.1 Responsibility

Extract entities and relations from a text chunk.

### 15.2 Input

```python
TextChunk | str
```

### 15.3 Output schema

The LLM must return JSON with this structure:

```json
{
  "entities": [
    {
      "label": "imatinib",
      "type": "Drug",
      "description": "BCR-ABL tyrosine kinase inhibitor"
    }
  ],
  "relations": [
    {
      "subject": "imatinib",
      "predicate": "treats",
      "object": "Ph+ acute lymphoblastic leukemia",
      "confidence": 0.9
    }
  ]
}
```

### 15.4 Prompt template

The prompt must include:

- role: medical knowledge extraction system;
- allowed entity types;
- allowed relation types;
- instruction to return only valid JSON;
- the chunk text.

### 15.5 Caching

Cache extraction output by MD5 hash of chunk text:

```text
.cache/hirag_ontology/extraction/{md5}.json
```

Cache behavior must be testable.

### 15.6 Error handling

- Malformed JSON should raise a clear exception or return a structured error object.
- Unknown entity types should be allowed initially but repaired later by the validator.
- Unknown predicates should be captured but flagged by validation.

## 16. A2 TypingAgent

Implement `src/hirag_ontology/pipeline/typing_agent.py`.

### 16.1 Responsibility

Assign exactly one ontology class to each entity.

### 16.2 Input

```python
Entity(label, description, aliases)
```

### 16.3 Output schema

```json
{
  "class": "Drug",
  "confidence": 0.95,
  "rationale": "The label refers to a pharmacological substance."
}
```

### 16.4 Requirements

- Use ontology class names and definitions in the prompt.
- Cache results by normalized entity label.
- If returned class is unknown, assign `Other`.
- Expose `type_graph(kg: KnowledgeGraph) -> dict[str, Any]` returning counts and cache stats.
- Tests must use `FakeLLMClient`.

## 17. A3 DeduplicationAgent

Implement `src/hirag_ontology/pipeline/deduplication.py`.

### 17.1 Responsibility

Merge near-duplicate entities into canonical nodes.

### 17.2 Blocking

To avoid O(n²) comparisons on large graphs, use token-level blocking:

- Normalize labels.
- Remove a small set of stopwords.
- Candidate duplicates must share at least one non-stopword token.

### 17.3 Similarity

Use hybrid similarity:

```text
sim(a, b) = alpha * sim_sem(a, b) + (1 - alpha) * sim_lex(a, b)
```

Defaults:

```text
alpha = 0.6
threshold = 0.85
```

Where:

- `sim_lex` is token sort ratio using `difflib.SequenceMatcher`.
- `sim_sem` is cosine similarity of embeddings if both entities have embeddings.
- If embeddings are unavailable, use lexical similarity as semantic fallback.

### 17.4 Merge algorithm

Use Union-Find clustering.

For each cluster:

- choose canonical representative by highest graph degree;
- merge aliases;
- merge source chunks;
- redirect incoming and outgoing relations;
- discard self-loops introduced by merging;
- preserve relation confidence and source metadata.

### 17.5 Required tests

Tests must cover:

- punctuation variants;
- word-order variants;
- aliases preserved;
- no merge below threshold;
- relation redirection;
- self-loops removed after merge;
- deterministic canonical selection.

## 18. A4 ValidationAgent

Implement `src/hirag_ontology/pipeline/validator.py`.

### 18.1 Responsibility

Check whether graph entities and relations satisfy ontology constraints.

### 18.2 Validation checks

Implement five checks:

1. Valid entity type.
2. Valid predicate.
3. Domain constraint.
4. Range constraint.
5. No self-loops.

### 18.3 Output

```python
{
  "consistency_score": 0.9733,
  "violations": [
    {
      "type": "domain_violation",
      "relation_index": 3,
      "predicate": "treats",
      "expected": "Drug",
      "actual": "LabTest",
      "subject_id": "...",
      "object_id": "..."
    }
  ],
  "counts": {
    "valid_entity_type": 0,
    "valid_predicate": 1,
    "domain_constraint": 2,
    "range_constraint": 0,
    "no_self_loops": 0
  }
}
```

Consistency score:

```text
Cons(G, O) = 1 - violations / total_checks
```

For the MVP:

```text
total_checks = len(ontology.axioms) * max(len(kg.relations), 1)
```

### 18.4 Auto-repair

Implement:

```python
auto_repair(kg: KnowledgeGraph, validation_result: dict | None = None) -> dict[str, Any]
```

Auto-repair should:

- convert unknown entity types to `Other`;
- remove self-loop relations;
- optionally drop relations with unknown predicates if `drop_invalid_predicates=True`.

It should not silently rewrite domain/range violations unless explicitly configured.

## 19. A5 ReasoningAgent

Implement `src/hirag_ontology/pipeline/reasoning.py`.

### 19.1 Responsibility

Infer simple missing relations from graph structure.

### 19.2 MVP inference rule

If two `Drug` entities both have `treats` relations to the same `Condition`, suggest a `related_to` relation between the two drugs.

Example:

```text
Drug A --treats--> Condition C
Drug B --treats--> Condition C
=> Drug A --related_to--> Drug B
```

### 19.3 Requirements

- Suggested relations should have `confidence = 0.7`.
- Do not add duplicate relations.
- Do not add self-loops.
- Return suggestion stats.

## 20. A6 UpdateAgent / graph finalization

The MVP may implement A6 as functions in `runner.py` rather than a separate class.

Responsibilities:

1. Compute PageRank.
2. Save final graph JSON.
3. Save run metadata.
4. Report graph statistics.

Required output files:

```text
results/knowledge_graph_final.json
results/run_summary.json
```

`run_summary.json` should include:

```json
{
  "documents_processed": 2,
  "chunks_processed": 10,
  "entity_count_raw": 100,
  "relation_count_raw": 80,
  "entity_count_final": 90,
  "relation_count_final": 85,
  "dedup_merged_count": 10,
  "consistency_before": 0.92,
  "consistency_after": 0.97,
  "pagerank_computed": true
}
```

## 21. Quality functional

Implement `src/hirag_ontology/pipeline/quality.py`.

```python
@dataclass
class QualityScores:
    coverage: float
    consistency: float
    precision: float
    redundancy: float
    q: float
```

### 21.1 Coverage

```text
Coverage(G) = number of ontology classes represented by at least one entity / number of ontology classes excluding Other
```

### 21.2 Consistency

Use the validator's consistency score.

### 21.3 Precision

For MVP, implement one of:

- average relation confidence; or
- configurable default if confidence is missing; or
- annotation-based precision when gold labels exist.

### 21.4 Redundancy

For MVP, use:

```text
Redundancy(G) = number_of_aliases / max(number_of_entities + number_of_aliases, 1)
```

This is an approximation of the fraction of non-canonical nodes.

## 22. Retrieval

Implement retrieval modules in `src/hirag_ontology/retrieval/`.

## 22.1 Retrieval modes

```python
class RetrievalMode(str, Enum):
    SEMANTIC_ONLY = "semantic_only"
    LEXICAL_ONLY = "lexical_only"
    STRUCTURAL_ONLY = "structural_only"
    HYBRID_RRF = "hybrid_rrf"
```

## 22.2 Entity document representation

Each entity should be represented for retrieval as:

```text
<label> <entity_type> <description> <aliases>
```

## 22.3 Semantic retrieval

Use query/entity embeddings through `EmbeddingProvider`.

Requirements:

- Use normalized embeddings when possible.
- Rank entities by cosine similarity.
- Lazy-compute missing entity embeddings.
- Tests must use deterministic embeddings.

## 22.4 Lexical retrieval

Use BM25 over entity document representations.

Requirements:

- Include labels, aliases, descriptions, and class names.
- Tokenize by lowercase whitespace for MVP.
- Optional future improvement: Russian lemmatization with `pymorphy3`.

## 22.5 Structural retrieval

Use PageRank scores computed over the graph.

Requirements:

- Compute PageRank if missing.
- Rank by descending PageRank.
- Tie-break deterministically by entity label or ID.

## 22.6 Reciprocal Rank Fusion

Implement `rrf.py`:

```python
def rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    ...
```

Formula:

```text
RRF(entity) = Σ 1 / (k + rank_i(entity))
```

where rank is 1-based.

Default:

```text
k = 60
```

## 22.7 HybridRetriever

```python
class HybridRetriever:
    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_provider: EmbeddingProvider,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        rrf_k: int = 60,
    ): ...

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedEntity]: ...
```

`RetrievedEntity` should include:

```python
@dataclass
class RetrievedEntity:
    entity_id: str
    entity: Entity
    score: float
    rank: int
    retrieval_mode: str
    component_scores: dict[str, float] = field(default_factory=dict)
```

### 22.8 Required retrieval tests

Tests must verify:

- semantic-only ranking;
- lexical-only ranking;
- structural-only ranking;
- RRF fusion correctness;
- deterministic tie-breaking;
- `top_k` behavior;
- behavior on empty graph.

## 23. Graph context and answer generation

Implement a utility that converts retrieved entities and local relations into LLM context.

```python
def build_graph_context(
    kg: KnowledgeGraph,
    retrieved: list[RetrievedEntity],
    include_neighbors: bool = True,
    max_relations: int = 50,
) -> str:
    ...
```

Context format:

```text
Entities:
- imatinib [Drug]: BCR-ABL tyrosine kinase inhibitor. Aliases: ...
- Ph+ acute lymphoblastic leukemia [Condition]: ...

Relations:
- imatinib --treats--> Ph+ acute lymphoblastic leukemia (confidence=0.90)
- Ph+ acute lymphoblastic leukemia --diagnosed_by--> RT-PCR (confidence=0.85)
```

Answer generation should be optional and should use `LLMClient.complete_text`.

The answer prompt must instruct the model:

- answer only from provided graph context;
- say when the context is insufficient;
- avoid inventing unsupported medical claims.

## 24. CLI

Implement `src/hirag_ontology/cli.py`.

Required commands:

### 24.1 `run-demo`

```bash
python -m hirag_ontology.cli run-demo \
  --input data/sample_docs \
  --out results/demo_graph.json \
  --llm gemma
```

Behavior:

1. Load sample Markdown documents.
2. Chunk documents.
3. Extract graph using local Gemma 4 through Ollama.
4. Type entities.
5. Deduplicate entities.
6. Validate and auto-repair.
7. Apply reasoning.
8. Compute PageRank.
9. Save graph JSON.
10. Run one retrieval query.
11. Save run summary.

### 24.2 `build-graph`

```bash
python -m hirag_ontology.cli build-graph \
  --input data/documents/minzdrav_dataset \
  --out results/knowledge_graph_final.json \
  --llm gemma
```

This command requires a local Ollama Gemma 4 runtime and is not used by tests.

### 24.3 `ask`

```bash
python -m hirag_ontology.cli ask \
  --graph results/knowledge_graph_final.json \
  --query "How is Ph+ ALL managed differently?" \
  --llm gemma
```

## 25. Demo data

Add small Markdown files in `data/sample_docs/`.

The sample docs should include enough content to test:

- a `Drug` treating a `Condition`;
- a `Condition` diagnosed by a `LabTest`;
- a drug with a `DosageRegimen`;
- duplicate surface forms that should merge;
- two drugs treating the same condition so the reasoning agent can infer `related_to`.

Example entities:

```text
imatinib
BCR-ABL tyrosine kinase inhibitor
Ph+ acute lymphoblastic leukemia
RT-PCR
FISH
induction therapy
consolidation therapy
maintenance therapy
```

The sample docs are for deterministic tests only and should not be presented as clinical advice.

## 26. Evaluation

Implement minimal evaluation scaffolding in `src/hirag_ontology/evaluation/`.

### 26.1 Baseline configurations

Prepare code that can run these modes, even if the MVP only demonstrates them on small sample data:

1. Naive RAG: flat semantic retrieval over chunks.
2. HiRAG baseline: semantic-only entity retrieval.
3. HiRAG + Dedup: semantic-only entity retrieval after deduplication.
4. HiRAG-Ontology: hybrid RRF retrieval after typing, deduplication, validation, reasoning, and PageRank.

### 26.2 LLM-as-judge dimensions

Evaluation code should support these dimensions:

```text
Comprehensiveness
Empowerment
Diversity
Overall
```

For tests, use a deterministic judge. Do not call real APIs.

### 26.3 Benchmark question schema

```json
{
  "id": "q001",
  "question": "How is Ph+ ALL managed differently from Ph- ALL?",
  "type": "relation_inference",
  "document": "ALL guidelines",
  "expected_entities": ["Ph+ acute lymphoblastic leukemia", "imatinib", "BCR-ABL"],
  "notes": "Sample question for retrieval testing, not clinical advice."
}
```

## 27. Testing strategy

All core logic must be covered with deterministic pytest tests.

### 27.1 Unit tests

Required tests:

```text
test_chunking.py
test_knowledge_graph.py
test_ontology.py
test_extractor.py
test_typing_agent.py
test_deduplication.py
test_validator.py
test_reasoning.py
test_quality.py
test_retriever.py
```

### 27.2 Integration tests

Required integration test:

```text
test_pipeline_demo.py
```

This test should run the demo pipeline with deterministic test clients and assert:

- output graph file exists;
- graph has at least one entity;
- graph has at least one relation;
- validation score is present;
- PageRank is present;
- retrieval returns at least one entity for a known query.

### 27.3 Test constraints

Tests must:

- never require external API keys;
- never make network calls;
- not depend on full Minzdrav dataset;
- use temporary directories for cache/output;
- be deterministic;
- pass with `pytest -q`.

## 28. Quality gates

Before a task is complete, these commands should pass:

```bash
uv run pytest -q
uv run ruff check .
uv run python -m hirag_ontology.cli ask --graph results/knowledge_graph_full_gemma.json --query "How is Ph+ ALL managed differently?" --llm gemma
```

If `uv` is not available, equivalent commands may be used:

```bash
python -m pytest -q
python -m ruff check .
python -m hirag_ontology.cli ask --graph results/knowledge_graph_full_gemma.json --query "How is Ph+ ALL managed differently?" --llm gemma
```

## 29. Security and privacy requirements

- Do not commit API keys.
- Do not log secrets.
- Do not send test data to external APIs.
- Real LLM calls must use local Gemma through Ollama.
- Unit tests must use deterministic test doubles.
- Output files should not include raw secrets or environment variables.

## 30. Medical safety requirements

This project is a research prototype. Generated answers must not be presented as medical advice.

The answer-generation prompt must include a statement equivalent to:

```text
Use only the provided graph context. If the context is insufficient, say that the answer is not supported by the graph context.
```

The README should include a disclaimer that the system is not intended for clinical decision-making without expert validation.

## 31. Implementation milestones

### Milestone 1: Project scaffold

Create the package layout, `pyproject.toml`, README, `.env.example`, empty modules, and initial tests.

Done when:

- package imports successfully;
- `pytest -q` passes;
- `ruff check .` passes.

### Milestone 2: Ontology and knowledge graph

Implement ontology loading, entity/relation models, graph add/save/load, and PageRank.

Done when:

- graph JSON roundtrip works;
- self-loops are rejected;
- PageRank test passes.

### Milestone 3: Chunking and document loading

Implement Markdown loading and chunking with metadata.

Done when:

- chunk size and overlap tests pass;
- deterministic file ordering is tested.

### Milestone 4: LLM abstraction and extraction

Implement `LLMClient`, deterministic test client, local Gemma 4 wrapper, and `ExtractionAgent`.

Done when:

- deterministic extraction populates graph;
- cache behavior is tested;
- malformed JSON behavior is tested.

### Milestone 5: Typing and validation

Implement `TypingAgent` and `ValidationAgent`.

Done when:

- invalid class fallback works;
- all five validation checks are covered;
- auto-repair is tested.

### Milestone 6: Deduplication

Implement token blocking, hybrid similarity, Union-Find clustering, canonical selection, alias merge, and edge redirection.

Done when:

- duplicate merge tests pass;
- false merge prevention is tested;
- relation redirection is tested.

### Milestone 7: Reasoning and quality

Implement simple missing relation inference and quality functional.

Done when:

- shared-condition `related_to` inference works;
- no duplicate suggestions are added;
- quality score is computed.

### Milestone 8: Hybrid retriever

Implement semantic, lexical, structural, and RRF retrieval.

Done when:

- each retrieval mode has deterministic tests;
- RRF formula is tested;
- empty graph behavior is tested.

### Milestone 9: End-to-end demo CLI

Implement `run-demo` with local Gemma and sample docs.

Done when:

- demo command creates graph and summary;
- integration test passes;
- no API keys are required.

### Milestone 10: Local Gemma runtime mode

Add local Gemma 4 execution through Ollama.

Done when:

- tests still use deterministic test doubles;
- public CLI mode uses local Gemma;
- README explains setup.

## 32. Acceptance criteria for the MVP

The MVP is accepted when:

1. `python -m pytest -q` passes.
2. `python -m ruff check .` passes.
3. `python -m hirag_ontology.cli ask --graph results/knowledge_graph_full_gemma.json --query "How is Ph+ ALL managed differently?" --llm gemma` succeeds.
4. The demo graph JSON contains typed entities and relations.
5. The graph has a computed PageRank dictionary.
6. The validator reports a consistency score.
7. The retriever returns top-K entities for a sample question.
8. Tests do not require real API keys or network access.
9. README explains how to run local Gemma mode and the prebuilt graph.
10. Public behavior is documented.

## 33. Suggested Codex instructions

When using Codex to implement this project:

- Work milestone by milestone.
- Ask for a plan before writing code for complex tasks.
- Keep changes small and reviewable.
- Write tests before or alongside implementation.
- Do not implement real API calls in tests.
- Use test doubles and deterministic embeddings for test behavior.
- Do not invent evaluation scores; implement scripts that compute them.
- Prefer NetworkX + JSON for MVP.
- Add Neo4j only after the MVP works.

## 34. Known limitations to address later

Future versions may add:

1. Neo4j graph backend.
2. OWL 2 reasoner integration.
3. Russian morphological normalization with `pymorphy3`.
4. Domain-adapted Russian medical embeddings.
5. SPARQL-style structured graph retrieval.
6. Web demo.
7. LangChain-compatible retriever wrapper.
8. Full 78-document evaluation.
9. More robust deduplication for abbreviations, INN/brand names, and transliteration variants.
10. Retrieval-oriented quality metric such as coverage@K.

## 35. Minimal deterministic test fixture expectation

The deterministic test fixture should be able to produce a tiny graph similar to:

```text
imatinib [Drug]
Ph+ acute lymphoblastic leukemia [Condition]
RT-PCR [LabTest]
FISH [LabTest]
induction therapy [Procedure]
consolidation therapy [Procedure]
maintenance therapy [Procedure]

imatinib --treats--> Ph+ acute lymphoblastic leukemia
Ph+ acute lymphoblastic leukemia --diagnosed_by--> RT-PCR
Ph+ acute lymphoblastic leukemia --diagnosed_by--> FISH
induction therapy --part_of--> ALL treatment protocol
consolidation therapy --part_of--> ALL treatment protocol
maintenance therapy --part_of--> ALL treatment protocol
```

A sample query such as:

```text
How is Ph+ ALL managed differently?
```

should retrieve at least:

```text
Ph+ acute lymphoblastic leukemia
imatinib
RT-PCR
FISH
```

The generated test answer can be deterministic, but the retrieved graph context must be real output from the pipeline.

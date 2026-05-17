# Thesis Summary: HiRAG-Ontology / Multi-Agent Ontology Construction

This document is an implementation-oriented summary of the thesis **“Development of a Multi-Agent System for Automatic Construction and Improvement of Ontologies”** by Eva Karimova and Alexey Popov, HSE and University of London Double Degree Programme, Moscow, 2026.

It is meant to give a coding agent enough project context to rebuild the system. For exact implementation requirements, use this file together with `docs/spec.md` and `AGENTS.md`.

---

## 1. Executive summary

The thesis proposes **HiRAG-Ontology**, a multi-agent system that improves Retrieval-Augmented Generation (RAG) by constructing and refining a typed medical knowledge graph from unstructured clinical guideline documents.

The system addresses three weaknesses of ordinary chunk-based or graph-based RAG:

1. **No explicit entity model**: flat vector retrieval treats each text chunk independently, so the same medical concept may appear under multiple names without being linked.
2. **No formal consistency layer**: retrieved chunks may contain logically incompatible or stale statements, while the generator has no structured signal that a contradiction exists.
3. **Entity duplication**: large corpora accumulate near-duplicate nodes such as brand names, abbreviations, transliterations, and grammatical variants.

The proposed solution adds a structured intermediate layer between documents and generation:

```text
raw clinical documents
  -> chunks
  -> extracted entities and relations
  -> typed ontology + knowledge graph
  -> deduplication
  -> validation / auto-repair
  -> reasoning / inferred relations
  -> PageRank + persistence
  -> hybrid entity retrieval
  -> RAG answer generation
```

The key retrieval contribution is replacing HiRAG’s semantic-only entity retriever with a **hybrid RRF retriever** that fuses three ranked lists:

- semantic retrieval via BERT embeddings,
- lexical retrieval via BM25,
- structural retrieval via PageRank.

The main reported result is an improvement over Naive RAG on a clinical benchmark: **Overall = 6.68 vs. 5.34**, a **25.1% improvement**. The ablation study identifies the hybrid RRF retriever as the most important component.

---

## 2. Metadata

| Field | Value |
|---|---|
| Thesis title | Development of a Multi-Agent System for Automatic Construction and Improvement of Ontologies |
| Authors | Eva Karimova, Alexey Popov |
| Institution | National Research University Higher School of Economics, Faculty of Computer Science |
| Programme | HSE and University of London Double Degree Programme in Data Science and Business Analytics |
| Year | 2026 |
| Project type | Bachelor’s thesis, software project |
| Domain | Medical / clinical NLP, oncology guidelines |
| Core method | Multi-agent ontology and knowledge graph construction for graph-based RAG |
| Proposed system name | HiRAG-Ontology |
| Baseline family | Naive RAG, GraphRAG, LightRAG, HiRAG |

---

## 3. Motivation and problem statement

RAG improves factuality by retrieving external evidence at inference time. However, naive RAG usually indexes a flat collection of text chunks and retrieves by dense-vector cosine similarity. The thesis argues that this design is insufficient for high-stakes domains such as medicine.

The project introduces a structured representation:

```text
O = ontology
G = knowledge graph
M = set of specialized agents
```

Given a corpus of unstructured documents:

```text
D = {d1, d2, ..., dN}
```

construct a knowledge representation:

```text
G* = arg max_G Q(G)
```

where `Q(G)` is a quality functional combining coverage, consistency, precision, and redundancy. The optimized graph is then used to improve downstream RAG answer quality.

---

## 4. Research question

The central research question is:

> Can a formal ontological layer and a multi-signal hybrid retriever, maintained by a specialized multi-agent system, measurably improve RAG quality relative to a graph-based baseline system such as HiRAG?

The thesis breaks this into five objectives:

1. Define a formal model of the system: documents, chunks, graph, ontology, deduplication, consistency, quality functional.
2. Implement a modular multi-agent pipeline for extracting, typing, deduplicating, validating, reasoning over, and updating entities and relations.
3. Build a hybrid entity retriever using Reciprocal Rank Fusion over BM25, embeddings, and PageRank.
4. Run ablations to quantify component contributions.
5. Evaluate on a domain-specific clinical guideline benchmark.

---

## 5. Scientific contributions

### 5.1 Unified quality functional

The thesis defines:

```text
Q(G) = λ1 * Coverage(G)
     + λ2 * Cons(G, O)
     + λ3 * Precision(G)
     - λ4 * Redundancy(G)
```

Default weights:

```text
λ1 = 0.3
λ2 = 0.3
λ3 = 0.2
λ4 = 0.2
```

where:

- `Coverage(G)` measures how many domain concept classes are represented.
- `Cons(G, O)` measures ontology consistency.
- `Precision(G)` estimates factual correctness of extracted triplets.
- `Redundancy(G)` penalizes duplicate entities.

### 5.2 Hybrid entity retrieval

The system replaces semantic-only retrieval with rank fusion over:

```text
semantic:   cosine(embedding(entity), embedding(query))
lexical:    BM25(entity_label + aliases + description + type, query)
structural: PageRank(entity, graph)
```

The ranked lists are fused with Reciprocal Rank Fusion:

```text
RRF(v) = Σ_r 1 / (k + rank_r(v))
```

with:

```text
k = 60
```

### 5.3 Ontology validation and auto-repair

The validation layer checks ontology constraints such as valid classes, valid predicates, domain constraints, range constraints, and self-loop violations. It can automatically repair detectable issues, especially unknown types and self-loops.

### 5.4 Multi-agent graph construction

Each agent is modeled as a graph transformation:

```text
Ai: G -> G'
```

The full pipeline is a sequential composition:

```text
GT = (A6 ◦ A5 ◦ A4 ◦ A3 ◦ A2 ◦ A1)(G0)
```

---

## 6. Literature gap addressed by the thesis

The thesis positions HiRAG-Ontology at the intersection of:

- graph-based RAG,
- automatic ontology learning,
- entity deduplication,
- multi-agent LLM systems,
- hybrid retrieval.

Existing GraphRAG-style systems improve retrieval by using knowledge graphs, but the thesis argues that GraphRAG, LightRAG, and HiRAG still share two major limitations:

1. they rely mainly on semantic embedding similarity for entity retrieval;
2. they lack a formal ontology layer that enforces types, domain/range constraints, and consistency.

Ontology learning methods can induce structured knowledge from text, but they are rarely integrated into downstream RAG pipelines. Multi-agent systems can coordinate complex workflows, but they often lack a shared quality objective. HiRAG-Ontology combines these directions.

---

## 7. Core architecture

The architecture consists of six specialized agents.

| Agent | Name | Purpose | Input | Output |
|---|---|---|---|---|
| A1 | Extraction Agent | Extract entities and relations from chunks | text chunks | raw triplets |
| A2 | Typing Agent | Assign ontology classes to entities | graph entities | typed entities |
| A3 | Deduplication Agent | Merge near-duplicate entities | typed graph | canonical graph |
| A4 | Validation Agent | Check ontology constraints and repair violations | canonical graph | consistent graph |
| A5 | Reasoning Agent | Infer missing relations | validated graph | enriched graph |
| A6 | Update Agent | Persist graph and compute structural scores | enriched graph | final graph with PageRank |

Recommended rebuild data flow:

```text
1. Load Markdown clinical guideline files.
2. Split documents into overlapping chunks.
3. Use A1 to extract entities and relations as structured JSON.
4. Insert results into KnowledgeGraph.
5. Use A2 to assign ontology classes.
6. Use A3 to merge duplicates and redirect edges.
7. Use A4 to validate and repair graph consistency.
8. Use A5 to add simple inferred relations.
9. Use A6 to compute PageRank and persist graph.
10. Use HybridRetriever to retrieve entities for a query.
11. Build a graph context and generate a grounded RAG answer.
```

---

## 8. Ontology design

The extraction prompt and ontology define nine entity types:

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

The thesis uses seven relation types:

```text
treats
causes
contraindicated_for
part_of
diagnosed_by
dosage_is
related_to
```

A minimal domain/range schema for rebuilding:

| Relation | Domain | Range | Meaning |
|---|---|---|---|
| `treats` | Drug | Condition | a drug treats a condition |
| `causes` | Drug | Symptom | a drug can cause a symptom/adverse event |
| `contraindicated_for` | Drug | Condition | a drug is contraindicated for a condition |
| `part_of` | Procedure | Procedure | one procedure is part of another |
| `diagnosed_by` | Condition | LabTest | a condition is diagnosed by a test |
| `dosage_is` | Drug | DosageRegimen | a drug has a dosage regimen |
| `related_to` | Other | Other | generic fallback relation |

The thesis also includes an OWL/Turtle excerpt with classes such as `Drug`, `ChemotherapeuticDrug`, `Condition`, and `OncologicalCondition`, and properties such as `treats` and `contraindicated_for`.

For an MVP, `ontology.json` is enough. For a production-like version, add OWL serialization and a real OWL 2 reasoner.

---

## 9. Formal model, simplified for implementation

### 9.1 Documents and chunks

Documents are segmented with a sliding window:

```text
chunk_size = 800 words
overlap = 100 words
```

The thesis uses word-level chunking because Russian BPE token counts are less predictable than word counts.

### 9.2 Knowledge graph

The knowledge graph is a directed labelled graph:

```text
G = (V, E, ℓV, ℓE)
```

where:

- `V` = entity nodes,
- `E` = directed relation edges,
- `ℓV` = node labels,
- `ℓE` = edge labels / predicates.

A relation is a triplet:

```text
(subject_entity, predicate, object_entity)
```

### 9.3 Ontology

The ontology is:

```text
O = (C, R, A)
```

where:

- `C` = concept classes,
- `R` = relation types with domain/range constraints,
- `A` = logical axioms.

Typing function:

```text
τ: V -> C
```

A graph is type-consistent when every triplet satisfies the predicate’s domain and range constraints.

### 9.4 Deduplication

Entity similarity combines semantic and lexical similarity:

```text
sim(vi, vj) = α * sim_sem(vi, vj) + (1 - α) * sim_lex(vi, vj)
```

Default parameters:

```text
α = 0.6
θ = 0.85
```

Entities are duplicates when:

```text
sim(vi, vj) >= θ
```

The canonical representative is selected as the entity with the highest graph degree.

---

## 10. Implementation details from the thesis

### 10.1 Repository structure in the original implementation

The thesis describes a modular Python project with these key folders:

```text
pipeline/
  knowledge_graph.py
  extractor.py
  typing_agent.py
  deduplication.py
  validator.py
  quality.py

retrieval/
  retriever.py

evaluation/
  run_eval.py
  llm_judge.py
```

### 10.2 Technology stack

The thesis mentions the following stack:

```text
sentence-transformers
rank-bm25
networkx
local Gemma 4 / Ollama clients
python-dotenv
Neo4j
JSON graph persistence
```

For the rebuild, start with:

```text
NetworkX + JSON
```

and add Neo4j later as an optional backend.

### 10.3 KnowledgeGraph data structure

The `KnowledgeGraph` class should use:

```text
Entity:
  id
  label
  entity_type
  description
  aliases
  source_chunks
  embedding

Relation:
  subject_id
  predicate
  object_id
  confidence
  source_chunk
```

Entity IDs are generated as MD5 hashes of normalized labels.

The graph should support:

```text
add_entity()
add_relation()
save()
load()
compute_pagerank()
```

### 10.4 Extraction Agent

A1 processes each chunk and asks an LLM to return JSON:

```json
{
  "entities": [
    {"label": "...", "type": "EntityType", "description": "..."}
  ],
  "relations": [
    {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}
  ]
}
```

The thesis uses structured prompts and response caching by MD5 hash of the chunk text.

### 10.5 Typing Agent

A2 assigns exactly one ontology class to each entity. It receives:

```text
entity label
entity description
list of ontology classes and definitions
```

It returns:

```json
{"class": "Drug", "confidence": 0.91}
```

Invalid returned classes should fall back to `Other`.

### 10.6 Deduplication Agent

A3 uses:

- token-level blocking to reduce comparisons,
- token sort ratio for lexical similarity,
- embedding cosine similarity for semantic similarity,
- hybrid score with `α = 0.6`,
- threshold `θ = 0.85`,
- Union-Find clustering,
- highest-degree node as canonical representative,
- alias preservation,
- edge redirection,
- self-loop removal.

Observed duplicate patterns:

```text
word-order variants
punctuation variants
Russian morphological variants
brand / abbreviation / INN variants
transliteration variants
```

### 10.7 Validation Agent

A4 checks five violation categories:

```text
valid entity types
valid predicates
domain constraints
range constraints
no self-loops
```

Auto-repair handles at least:

```text
unknown entity type -> Other
self-loop relation -> remove
```

### 10.8 Reasoning Agent

A5 identifies missing relations with simple rules. Example:

```text
If Drug A treats Condition C
and Drug B treats Condition C,
then Drug A related_to Drug B.
```

Suggested inferred relations are added with lower confidence, e.g.:

```text
confidence = 0.7
```

### 10.9 Update Agent

A6 persists the refined graph and computes PageRank:

```text
nx.pagerank(alpha=0.85, max_iter=200)
```

The structural scores are used by the HybridRetriever.

---

## 11. HybridRetriever details

The retriever supports four modes:

```text
SEMANTIC_ONLY
LEXICAL_ONLY
STRUCTURAL_ONLY
HYBRID_RRF
```

### 11.1 Semantic retrieval

Compute query embedding and compare it to entity embeddings.

Entity text can be constructed from:

```text
label + aliases + description + entity_type
```

The thesis describes using BERT-style multilingual embeddings and lazy batch embedding computation.

### 11.2 Lexical retrieval

Build a BM25 index over entity documents:

```text
canonical label + aliases + description + ontology class name
```

BM25 is important because exact domain terms, acronyms, and surface-form matches can beat dense embeddings in clinical text.

### 11.3 Structural retrieval

Use PageRank as a query-independent prior. It favors central entities that are likely to appear in answer chains.

Risk: generic high-degree nodes such as “treatment” or “diagnosis” may dominate. Consider down-weighting generic node types or adding a stoplist.

### 11.4 RRF fusion

For each ranked list, add:

```text
1 / (k + rank)
```

Use:

```text
k = 60
top_k = 10
```

---

## 12. Evaluation setup

### 12.1 Dataset

The primary corpus is clinical guideline documents from the Russian Ministry of Health, focused on oncological conditions.

Reported dataset details include:

```text
78 Russian clinical guideline documents
preprocessed Markdown format
file sizes roughly 108 KB to 566 KB
50-question benchmark
secondary corpus: 12 English NCCN guidelines
```

The thesis text contains some ambiguity about experiment scale: some sections describe the first 10 documents being used for graph construction, while main result tables describe the full 78-document corpus. For a rebuild, make corpus size a configuration parameter and report it explicitly in every experiment output.

### 12.2 Benchmark questions

The benchmark contains 50 questions across three types:

```text
single-entity lookup
relation inference
multi-hop reasoning
```

Question examples include:

```text
What is the treatment protocol for acute lymphoblastic leukemia in children?
How is Ph+ ALL managed differently from Ph- ALL?
What is the role of ATRA in APL treatment?
What monitoring is required during mitotane therapy?
```

### 12.3 Evaluation metrics

The thesis uses LLM-as-judge evaluation. In this rebuild, judge prompts should
run through the local Gemma 4 runtime. Answers are scored on a 0–10 scale across:

```text
Comprehensiveness
Empowerment
Diversity
Overall
```

### 12.4 Baselines

Systems compared:

| System | Description |
|---|---|
| Naive RAG | Flat vector retrieval over chunks, no graph, no ontology |
| HiRAG baseline | Graph-based HiRAG with semantic-only retrieval |
| HiRAG + Dedup | HiRAG with deduplication but no full ontology/hybrid retriever |
| HiRAG-Ontology | Proposed system with typed ontology, deduplication, validation, reasoning, and hybrid RRF retrieval |

### 12.5 Main reported results

Main table, 50-question benchmark, 0–10 scale:

| System | Comprehensiveness | Empowerment | Diversity | Overall |
|---|---:|---:|---:|---:|
| Naive RAG | 5.21 | 5.10 | 5.42 | 5.34 |
| HiRAG baseline | 6.15 | 6.05 | 6.20 | 6.28 |
| HiRAG + Dedup | 5.00 | 4.95 | 5.35 | 5.10 |
| HiRAG-Ontology | 6.55 | 6.45 | 6.90 | 6.68 |

Key claimed improvements:

```text
HiRAG-Ontology vs Naive RAG:
Overall 6.68 vs 5.34 = +25.1%

HiRAG-Ontology vs HiRAG baseline:
Overall 6.68 vs 6.28 = +6.4%
```

### 12.6 Ablation results

Ablation table, 10-question subset, 0–10 scale:

| Configuration | Overall | Δ vs. Full |
|---|---:|---:|
| Full System / HiRAG-Ontology | 7.10 | — |
| w/o Hybrid Retriever | 5.40 | -23.9% |
| w/o Deduplication | 7.20 | +1.4% |
| Baseline, no dedup, semantic-only | 5.20 | -26.8% |

Interpretation: the hybrid retriever is the most important component. Deduplication’s effect is more complex; it can help quality and graph cleanliness, but aggressive merging may also remove useful distinctions.

---

## 13. Qualitative example

The thesis compares responses to this question:

```text
What is the treatment protocol for acute lymphoblastic leukemia in children,
and how is Ph+ ALL managed differently?
```

Naive RAG retrieves relevant but weak context and cannot produce a detailed treatment protocol.

HiRAG-Ontology retrieves structured typed relations and produces a more complete answer covering:

```text
induction therapy
consolidation therapy
maintenance therapy
Ph+ ALL targeted therapy using BCR-ABL tyrosine kinase inhibitors such as imatinib
Ph+ ALL diagnosis via cytogenetic analysis, FISH, RT-PCR/PCR-RV
```

The qualitative point is not that the LLM became medically smarter; it was given a better structured graph context.

---

## 14. Figures and visual artifacts in the thesis

The thesis includes several useful visuals:

| Figure | Meaning |
|---|---|
| Figure 5.1 | Entity type distribution in the knowledge graph. Condition entities dominate, reflecting the disease-centric clinical guideline corpus. |
| Figure 5.2 | Knowledge graph fragment showing top entities by degree. Node size is proportional to degree and color encodes entity type. |
| Figure 5.3 | Iterative multi-agent pipeline convergence. The graph gains inferred relations and converges quickly. |

For a rebuild, recreate these figures from the actual generated graph rather than hard-coding thesis numbers.

---

## 15. Demonstration and deployment tools

The thesis mentions supplementary tooling:

```text
graph_explorer.ipynb
langchain_integration.py
web_demo.py
hirag_ontology_colab.ipynb
iterative_pipeline.py
```

### 15.1 Jupyter graph explorer

Used for:

```text
basic graph statistics
PageRank computation
entity type distribution
top-20 entities by degree
NetworkX spring layout visualization
```

### 15.2 LangChain integration

A wrapper named something like:

```text
HiRAGOntologyRetriever
```

can expose the retriever as a LangChain-compatible component:

```text
retrieve -> format -> prompt -> generate
```

### 15.3 Web demo

A local web interface runs on port 5000 and allows browser-based querying of the knowledge graph.

### 15.4 Iterative pipeline

The iterative pipeline repeats validation, reasoning, and update until quality improvement falls below a threshold:

```text
threshold = 0.0005
max_iterations = 5
```

The thesis reports convergence after one iteration, adding 10 inferred relations.

---

## 16. Error analysis

The thesis identifies three dominant error patterns.

### 16.1 Out-of-graph questions

Some questions require information not present in the processed documents. The answer quality is low because retrieval cannot surface missing evidence.

Mitigation:

```text
process more documents
track document coverage
add answerability checks
return “not supported by context” when evidence is absent
```

### 16.2 Over-retrieval of generic procedure nodes

High-degree generic nodes such as “treatment” or “diagnosis” can appear in many queries due to PageRank.

Mitigation:

```text
stoplist generic nodes
down-weight structural signal for generic labels
penalize nodes with overly broad descriptions
require lexical/semantic support for high-PageRank nodes
```

### 16.3 Russian morphological mismatch

Russian morphology can cause BM25 to miss relevant terms.

Mitigation:

```text
pymorphy3 lemmatization
normalization of Russian medical terms
alias expansion
transliteration handling
```

Additional failure categories:

```text
extraction errors: wrong boundaries or relation types
deduplication errors: false merges and missed aliases
reasoning failures: weak multi-hop inference chains of length >= 3
```

---

## 17. Limitations

Important limitations to preserve when reporting results:

1. **Experimental scale ambiguity**: some sections mention 10 documents, others full 78-document evaluation. Rebuild experiments must log corpus size clearly.
2. **HiRAG baseline reproduction**: the baseline is described as a reimplementation rather than necessarily the official HiRAG codebase.
3. **Fixed ontology schema**: entities outside the nine classes fall into `Other`, limiting precision and coverage.
4. **Language/domain specificity**: the corpus is Russian oncology text, while the embedding model is not necessarily medical-domain-tuned.
5. **Q(G) stability**: the thesis notes that Q(G) can remain stable even when RAG quality improves, because the metric measures static graph properties rather than retrieval-oriented utility.

For a robust rebuild, do not claim thesis-level performance until the evaluation scripts reproduce it on the intended corpus.

---

## 18. Future work suggested by the thesis

Suggested extensions:

```text
full 78-document or larger 100+ question evaluation
comparison against official HiRAG and GraphRAG repositories
domain-adapted Russian medical embeddings
automatic ontology extension via Formal Concept Analysis
real OWL 2 reasoner integration, e.g. HermiT / Pellet
SPARQL-style structured graph retrieval
better coverage@K / answerability metrics
```

---

## 19. Rebuild guidance for Codex or another coding agent

When rebuilding the project, implement it as a testable Python research prototype.

### 19.1 Recommended MVP choices

Use:

```text
Python 3.11+
NetworkX for graph operations
JSON for graph persistence
Pydantic or dataclasses for structured objects
rank-bm25 for lexical retrieval
sentence-transformers for embeddings
FakeLLMClient for tests
FakeEmbeddingProvider for tests
pytest for deterministic tests
```

Avoid at MVP stage:

```text
Neo4j hard dependency
real LLM calls in tests
real embedding calls in unit tests
uncontrolled API costs
hard-coded evaluation scores
medical claims not backed by retrieved context
```

### 19.2 Suggested module order

Build in this order:

```text
1. ontology.py
2. pipeline/chunking.py
3. pipeline/knowledge_graph.py
4. pipeline/validator.py
5. llm.py and FakeLLMClient
6. pipeline/extractor.py
7. pipeline/typing_agent.py
8. pipeline/deduplication.py
9. pipeline/reasoning.py
10. retrieval/rrf.py
11. retrieval/retriever.py
12. pipeline/runner.py
13. cli.py
14. evaluation/llm_judge.py
15. app/web_demo.py
```

### 19.3 Acceptance criteria for MVP

The MVP is complete when:

```text
- sample Markdown documents can be loaded and chunked;
- a graph can be built from deterministic extraction fixtures;
- entities are typed by a deterministic test client;
- duplicate entities can be merged;
- invalid edges are detected and repaired;
- PageRank is computed;
- BM25, semantic, structural, and hybrid RRF retrieval modes work;
- graph context can be generated for an LLM prompt;
- all tests pass without API keys;
- a CLI demo writes results/demo_graph.json.
```

### 19.4 Testing strategy

Unit tests should cover:

```text
chunk overlap and deterministic ordering
Entity / Relation serialization
KnowledgeGraph add/save/load/PageRank
ontology loading and domain/range validation
malformed LLM JSON handling
extraction cache hits
invalid type fallback to Other
deduplication alias preservation and edge redirection
RRF ranking determinism
retrieval mode differences
end-to-end deterministic test pipeline
```

### 19.5 Reporting principle

Do not write static claims like “the system improves by 25.1%” into runtime output unless the script computed that value. Store thesis results as historical reference in docs; compute new results from evaluation scripts.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation |
| GraphRAG | RAG using a knowledge graph or graph-derived summaries |
| HiRAG | Hierarchical graph-based RAG baseline |
| HiRAG-Ontology | Proposed ontology-aware HiRAG extension |
| Ontology | Formal set of classes, relations, and axioms |
| Knowledge graph | Entity-relation graph built from documents |
| Triplet | Subject-predicate-object relation |
| Canonicalization | Choosing one representative for duplicate entities |
| BM25 | Sparse lexical retrieval algorithm |
| PageRank | Graph centrality score used as structural retrieval signal |
| RRF | Reciprocal Rank Fusion, rank aggregation method |
| Cons(G, O) | Ontology consistency score |
| Q(G) | Graph quality functional |
| LLM-as-judge | Evaluation where an LLM scores generated answers |

---

## 21. Minimal project description for README

HiRAG-Ontology is a research prototype that builds a typed medical knowledge graph from clinical guideline documents and uses it to improve RAG. It extracts entities and relations with an LLM, assigns ontology classes, merges duplicate entities, validates graph consistency, infers simple missing relations, computes PageRank, and retrieves graph entities using hybrid Reciprocal Rank Fusion over BM25, embeddings, and structural centrality. The system is designed for Russian oncology guideline documents but can be adapted to other domains by changing the ontology, prompts, and corpus.

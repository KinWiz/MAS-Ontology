# HiRAG-Ontology Evaluation Report

- timestamp: `2026-05-18T08:48:14.469606+00:00`
- graph: `results\knowledge_graph_full_gemma.json`
- benchmark: `evaluation\ground_truth.json`
- top_k: `10`
- total_elapsed_s: `62.3363`

## Retrieval

| mode | Hit@5 | Hit@10 | MRR | MAP@10 | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 50 |
| lexical_only | 0.6000 | 0.6400 | 0.5667 | 0.2658 | 50 |
| lexical_structural | 0.6400 | 0.6400 | 0.5380 | 0.2597 | 50 |
| structural_only | 0.2000 | 0.2600 | 0.1311 | 0.0411 | 50 |
| hybrid_rrf | 0.5600 | 0.6600 | 0.4249 | 0.1698 | 50 |

## Generation

| metric | value |
| --- | ---: |
| faithfulness | 0.8488 |
| answer_relevance | 0.0837 |
| context_precision | 0.1120 |
| context_recall | 0.3517 |

## Latency

| stage | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| retrieval_s | 0.2564 | 0.0114 | 0.2419 | 0.2849 |
| context_format_s | 0.0010 | 0.0002 | 0.0007 | 0.0018 |
| generation_s | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| total_s | 0.2574 | 0.0115 | 0.2429 | 0.2867 |

## Deduplication Ablation

| alpha | threshold | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.85 | 1.0000 | 0.1250 | 0.2222 |

## Notes

- Generation metrics are deterministic unless an explicit LLM client is passed.
- Remote LLM APIs are not called by the default evaluation command.
- Treat these metrics as reproducible MVP metrics, not clinical validation.

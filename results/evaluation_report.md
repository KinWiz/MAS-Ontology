# HiRAG-Ontology Evaluation Report

- timestamp: `2026-05-18T10:42:48.232224+00:00`
- graph: `results\knowledge_graph_repaired.json`
- benchmark: `evaluation\ground_truth.json`
- top_k: `10`
- total_elapsed_s: `101.6124`

## Retrieval

| mode | Hit@5 | Hit@10 | MRR | MAP@10 | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic_only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 50 |
| lexical_only | 0.6000 | 0.6400 | 0.5667 | 0.2658 | 50 |
| lexical_structural | 0.6400 | 0.6400 | 0.5380 | 0.2615 | 50 |
| structural_only | 0.2000 | 0.3200 | 0.1342 | 0.0387 | 50 |
| hybrid_rrf | 0.5400 | 0.6800 | 0.4352 | 0.1806 | 50 |

## Baselines

| system | Hit@5 | Hit@10 | MRR | MAP@10 | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive_rag | 0.6000 | 0.6400 | 0.5667 | 0.2658 | 50 |
| hirag | 0.5400 | 0.6800 | 0.4352 | 0.1806 | 50 |
| hirag_ontology | 0.6400 | 0.6400 | 0.5380 | 0.2615 | 50 |

## Generation

| metric | value |
| --- | ---: |
| faithfulness | 0.8488 |
| answer_relevance | 0.0827 |
| context_precision | 0.1120 |
| context_recall | 0.3517 |

## Latency

| stage | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| retrieval_s | 0.2515 | 0.0084 | 0.2434 | 0.2742 |
| context_format_s | 0.0011 | 0.0002 | 0.0007 | 0.0013 |
| generation_s | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| total_s | 0.2526 | 0.0084 | 0.2443 | 0.2756 |

## Deduplication Ablation

| alpha | threshold | precision | recall | f1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.85 | 1.0000 | 0.2500 | 0.4000 |

## Notes

- Generation metrics are deterministic unless an explicit LLM client is passed.
- Remote LLM APIs are not called by the default evaluation command.
- Treat these metrics as reproducible MVP metrics, not clinical validation.

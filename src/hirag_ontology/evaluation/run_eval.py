"""Compatibility entry point for running the full evaluation suite."""

from __future__ import annotations

from hirag_ontology.evaluation.run_full_eval import main, run_full_evaluation

__all__ = ["main", "run_full_evaluation"]


if __name__ == "__main__":
    main()

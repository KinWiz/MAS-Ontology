"""Ontology loading utilities for HiRAG-Ontology."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class OntologyClass:
    """Entity class definition from the ontology JSON."""

    description: str


@dataclass(frozen=True)
class OntologyRelation:
    """Relation definition with domain and range constraints."""

    domain: str
    range: str
    description: str


@dataclass(frozen=True)
class Ontology:
    """Loaded ontology specification."""

    classes: dict[str, OntologyClass]
    relations: dict[str, OntologyRelation]
    axioms: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Build an ontology object from a JSON-compatible dictionary."""
        classes = {
            name: OntologyClass(description=str(data.get("description", "")))
            for name, data in payload.get("classes", {}).items()
        }
        relations = {
            name: OntologyRelation(
                domain=str(data["domain"]),
                range=str(data["range"]),
                description=str(data.get("description", "")),
            )
            for name, data in payload.get("relations", {}).items()
        }
        axioms = [str(axiom) for axiom in payload.get("axioms", [])]
        ontology = cls(classes=classes, relations=relations, axioms=axioms)
        ontology.validate_schema()
        return ontology

    @property
    def class_names(self) -> set[str]:
        """Return valid entity class names."""
        return set(self.classes)

    @property
    def relation_names(self) -> set[str]:
        """Return valid relation predicate names."""
        return set(self.relations)

    def validate_schema(self) -> None:
        """Validate internal ontology references."""
        missing_classes: set[str] = set()
        for relation in self.relations.values():
            if relation.domain not in self.classes:
                missing_classes.add(relation.domain)
            if relation.range not in self.classes:
                missing_classes.add(relation.range)

        if missing_classes:
            missing = ", ".join(sorted(missing_classes))
            msg = f"Ontology relation constraints reference unknown classes: {missing}"
            raise ValueError(msg)


def default_ontology_path() -> Path:
    """Return the repository-root ontology path."""
    return Path(__file__).resolve().parents[2] / "ontology.json"


def load_ontology(path: str | Path | None = None) -> Ontology:
    """Load ontology JSON from a path or the repository default."""
    ontology_path = Path(path) if path is not None else default_ontology_path()
    payload = json.loads(ontology_path.read_text(encoding="utf-8"))
    return Ontology.from_dict(payload)

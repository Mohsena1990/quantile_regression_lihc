"""Serializable split identities shared by every measurement family."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

@dataclass(frozen=True, slots=True)
class SplitAssignment:
    household_id: str
    partition: str
    fold: int | None = None
    country: str | None = None
    nuts1: str | None = None
    split_version: str | None = None
    seed: int | None = None
    def __post_init__(self) -> None:
        if not self.household_id:
            raise ValueError("household_id cannot be empty")
        if self.partition not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid partition: {self.partition}")
        if self.fold is not None and self.fold < 0:
            raise ValueError("fold must be non-negative")

def validate_assignments(assignments: Iterable[SplitAssignment]) -> list[SplitAssignment]:
    rows = list(assignments)
    ids = [row.household_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Each household must occur exactly once in a split registry")
    return rows

def write_split_registry(assignments: Iterable[SplitAssignment], path: str | Path, *, metadata: Mapping[str, object]) -> Path:
    rows = validate_assignments(assignments)
    if "seed" not in metadata or "strategy" not in metadata:
        raise ValueError("metadata must record both seed and strategy")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"metadata": dict(metadata), "assignments": [asdict(row) for row in rows]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target

def read_split_registry(path: str | Path) -> tuple[dict[str, object], list[SplitAssignment]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload["metadata"]), validate_assignments(SplitAssignment(**row) for row in payload["assignments"])

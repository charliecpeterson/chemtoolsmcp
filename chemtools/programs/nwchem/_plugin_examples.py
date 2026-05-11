"""NWChem ExamplesCorpus sub-protocol implementation.

Loads `examples/index.json` and serves curated NWChem input templates.
Indexed by task type, method, and tags so an agent can ask for a specific
template ("CASSCF optimization with a transition metal") and get a
verified starting point instead of drafting from scratch.

This is essentially RAG-for-input-files — agents drafting unfamiliar
methods can use templates as ground truth and adapt only what changes.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from chemtools.core.types import ExampleEntry, TaskKind


_THIS_DIR = Path(__file__).resolve().parent / "examples"
_INDEX_PATH = _THIS_DIR / "index.json"


def _load_index() -> list[ExampleEntry]:
    """Read the example index. Cached at first call; cheap to re-read."""
    if not _INDEX_PATH.exists():
        return []
    with _INDEX_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("entries") or [])


def _score(entry: ExampleEntry, task: TaskKind | None, tags: list[str] | None, methods: list[str] | None) -> int:
    """Score an example against a filter. Higher is a better match.
    Score components (combined into one int for sortable ordering):
      +10  exact task_type match
      +1   per tag overlap
      +5   per method overlap (case-insensitive)
    Entries that fail a hard task_type filter return 0.
    """
    if task is not None and entry.get("task_type") != task:
        return 0
    score = 10 if task is not None else 0
    if tags:
        entry_tags = set(entry.get("tags") or [])
        for t in tags:
            if t in entry_tags:
                score += 1
    if methods:
        entry_methods_lc = {m.lower() for m in (entry.get("methods") or [])}
        for m in methods:
            if m.lower() in entry_methods_lc:
                score += 5
    return score


class _NwchemExamples:
    """Implements chemtools.core.program.ExamplesCorpus for NWChem."""

    def find_example(
        self,
        task: TaskKind | None = None,
        tags: list[str] | None = None,
        methods: list[str] | None = None,
    ) -> ExampleEntry | None:
        entries = _load_index()
        if not entries:
            return None
        scored = [
            (_score(e, task, tags, methods), e)
            for e in entries
        ]
        scored = [(s, e) for s, e in scored if s > 0]
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]

    def list_examples(
        self,
        task: TaskKind | None = None,
        tags: list[str] | None = None,
    ) -> list[ExampleEntry]:
        entries = _load_index()
        if task is None and not tags:
            return list(entries)
        out: list[ExampleEntry] = []
        for e in entries:
            if task is not None and e.get("task_type") != task:
                continue
            if tags:
                entry_tags = set(e.get("tags") or [])
                if not any(t in entry_tags for t in tags):
                    continue
            out.append(e)
        return out

    def read_example(self, name: str) -> str:
        """Return the raw .nw template text for an example by name."""
        entries = _load_index()
        for e in entries:
            if e.get("name") == name:
                file_rel = e.get("file")
                if not file_rel:
                    raise ValueError(f"Example {name!r} has no 'file' field in the index")
                full = _THIS_DIR / file_rel
                if not full.exists():
                    raise FileNotFoundError(f"Example file missing: {full}")
                return full.read_text(encoding="utf-8")
        raise KeyError(f"No example named {name!r}; use list_examples() to see what's available")


NWCHEM_EXAMPLES = _NwchemExamples()


__all__ = ["NWCHEM_EXAMPLES"]

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DATA_DIR = Path(__file__).resolve().parent / "data" / "nwchem"
DOCS_ROOT = _DATA_DIR / "docs"


@dataclass(frozen=True)
class _DocMatch:
    file_path: Path
    line_number: int
    line_text: str
    score: float


def list_docs() -> list[dict[str, Any]]:
    files = sorted(item for item in DOCS_ROOT.iterdir() if item.is_file())
    return [
        {
            "name": file_path.name,
            "size_bytes": file_path.stat().st_size,
        }
        for file_path in files
    ]


def search_docs(
    query: str,
    *,
    max_results: int = 8,
    context_lines: int = 2,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty")
    tokens = _tokenize(query)
    phrase = query.casefold()
    matches: list[_DocMatch] = []

    for file_path in sorted(DOCS_ROOT.iterdir()):
        if not file_path.is_file():
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        file_name_lc = file_path.name.casefold()
        file_name_bonus = 1.5 if any(token in file_name_lc for token in tokens) else 0.0
        for idx, line in enumerate(lines, start=1):
            line_lc = line.casefold()
            score = file_name_bonus
            if phrase in line_lc:
                score += 12.0
            token_hits = sum(1 for token in tokens if token in line_lc)
            if token_hits == 0:
                continue
            score += float(token_hits * 2)
            if _looks_like_heading(line):
                score += 1.0
            matches.append(
                _DocMatch(
                    file_path=file_path,
                    line_number=idx,
                    line_text=line.strip(),
                    score=score,
                )
            )

    ranked = sorted(matches, key=lambda item: (-item.score, item.file_path.name, item.line_number))
    deduped: list[_DocMatch] = []
    seen: set[tuple[str, int]] = set()
    for match in ranked:
        key = (str(match.file_path), match.line_number)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
        if len(deduped) >= max_results:
            break

    results = [
        _format_search_match(match, context_lines=context_lines)
        for match in deduped
    ]
    return {
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def lookup_block_syntax(
    block_name: str,
    *,
    max_results: int = 6,
) -> dict[str, Any]:
    block = block_name.strip()
    if not block:
        raise ValueError("block_name must be non-empty")
    query = f"{block} end {block}"
    result = search_docs(query, max_results=max_results, context_lines=3)
    result["block_name"] = block
    return result


def find_examples(
    topic: str,
    *,
    max_results: int = 6,
) -> dict[str, Any]:
    topic = topic.strip()
    if not topic:
        raise ValueError("topic must be non-empty")
    tokens = _tokenize(topic)
    candidates: list[_DocMatch] = []
    # These stems match the actual bundled doc filenames (case-insensitive)
    _EXAMPLE_DOC_STEMS = ("19-examples", "23-tutorials", "11_quantummechanicalmethods", "11_quantummethods-2")
    for file_path in sorted(DOCS_ROOT.iterdir()):
        if not file_path.is_file():
            continue
        name_lc = file_path.name.casefold()
        if not any(label in name_lc for label in _EXAMPLE_DOC_STEMS):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            line_lc = line.casefold()
            token_hits = sum(1 for token in tokens if token in line_lc)
            if token_hits == 0:
                continue
            score = float(token_hits * 3)
            if _looks_like_heading(line):
                score += 1.5
            candidates.append(
                _DocMatch(
                    file_path=file_path,
                    line_number=idx,
                    line_text=line.strip(),
                    score=score,
                )
            )
    ranked = sorted(candidates, key=lambda item: (-item.score, item.file_path.name, item.line_number))
    results = [_format_search_match(match, context_lines=4) for match in ranked[:max_results]]
    return {
        "topic": topic,
        "result_count": len(results),
        "results": results,
    }


def read_doc_excerpt(
    doc_name: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    query: str | None = None,
    context_lines: int = 8,
) -> dict[str, Any]:
    # Accept either a bare filename or a full path (for internal use)
    path = Path(doc_name)
    if not path.is_absolute():
        path = DOCS_ROOT / doc_name
    path = path.resolve()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if query:
        query_lc = query.casefold()
        for idx, line in enumerate(lines, start=1):
            if query_lc in line.casefold():
                start = max(1, idx - context_lines)
                end = min(len(lines), idx + context_lines)
                return _excerpt_payload(path, lines, start, end, matched_line=idx)
        raise ValueError(f"query not found in {path.name}")
    start = max(1, start_line or 1)
    end = min(len(lines), end_line or min(len(lines), start + context_lines * 2))
    return _excerpt_payload(path, lines, start, end, matched_line=None)


def get_topic_guide(topic: str) -> dict[str, Any]:
    normalized = _normalize_topic(topic)
    if normalized == "scf_open_shell":
        return {
            "topic": normalized,
            "summary": "Use the SCF documentation for open-shell forms such as ROHF/UHF with NOPEN or multiplicity keywords; do not guess syntax from memory.",
            "recommended_tools": [
                "lookup_nwchem_block_syntax(scf)",
                "search_nwchem_docs('NOPEN rohf uhf singlet sextet scf')",
            ],
            "results": search_docs(
                "NOPEN rohf uhf singlet sextet scf",
                max_results=6,
                context_lines=3,
            )["results"],
        }
    if normalized == "mcscf":
        return {
            "topic": normalized,
            "summary": "For MCSCF, verify ACTIVE, ACTELEC, MULTIPLICITY, STATE, VECTORS, HESSIAN, and LEVEL directly from docs before drafting.",
            "recommended_tools": [
                "lookup_nwchem_block_syntax(mcscf)",
                "search_nwchem_docs('MCSCF STATE ACTIVE ACTELEC MULTIPLICITY VECTORS LEVEL HESSIAN')",
            ],
            "results": search_docs(
                "MCSCF STATE ACTIVE ACTELEC MULTIPLICITY VECTORS LEVEL HESSIAN",
                max_results=8,
                context_lines=3,
            )["results"],
        }
    if normalized == "fragment_guess":
        return {
            "topic": normalized,
            "summary": "Fragment guess workflows should be based on documented examples that use vectors input fragment and separate fragment wavefunctions.",
            "recommended_tools": [
                "find_nwchem_examples(fragment guess)",
                "search_nwchem_docs('vectors input fragment')",
            ],
            "results": (
                find_examples("fragment guess", max_results=4)["results"]
                + search_docs("vectors input fragment", max_results=4, context_lines=3)["results"]
            )[:8],
        }
    if normalized == "tce":
        return {
            "topic": normalized,
            "summary": "TCE inputs should be drafted from the documented TCE block and task forms such as TASK TCE ENERGY/OPTIMIZE/FREQUENCIES.",
            "recommended_tools": [
                "lookup_nwchem_block_syntax(tce)",
                "search_nwchem_docs('TASK TCE ENERGY OPTIMIZE FREQUENCIES ACTIVE_OA ACTIVE_OB')",
            ],
            "results": search_docs(
                "TASK TCE ENERGY OPTIMIZE FREQUENCIES ACTIVE_OA ACTIVE_OB",
                max_results=8,
                context_lines=3,
            )["results"],
        }
    raise ValueError("unsupported topic; use one of: scf_open_shell, mcscf, fragment_guess, tce")


def _excerpt_payload(path: Path, lines: list[str], start: int, end: int, matched_line: int | None) -> dict[str, Any]:
    excerpt = [
        {
            "line_number": idx,
            "text": lines[idx - 1],
            "matched": idx == matched_line,
        }
        for idx in range(start, end + 1)
    ]
    return {
        "file_name": path.name,
        "start_line": start,
        "end_line": end,
        "matched_line": matched_line,
        "excerpt": excerpt,
    }


def _normalize_topic(topic: str) -> str:
    value = topic.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "scf": "scf_open_shell",
        "scf_open": "scf_open_shell",
        "open_shell_scf": "scf_open_shell",
        "mcscf_block": "mcscf",
        "fragment": "fragment_guess",
        "fragmentguess": "fragment_guess",
        "tce_block": "tce",
    }
    return aliases.get(value, value)


def _format_search_match(match: _DocMatch, *, context_lines: int) -> dict[str, Any]:
    excerpt = read_doc_excerpt(
        str(match.file_path),
        start_line=max(1, match.line_number - context_lines),
        end_line=match.line_number + context_lines,
    )
    return {
        "file_name": match.file_path.name,
        "line_number": match.line_number,
        "line_text": match.line_text,
        "score": match.score,
        "excerpt": excerpt["excerpt"],
    }


def _tokenize(text: str) -> list[str]:
    tokens = [token for token in re.split(r"[^a-zA-Z0-9_+-]+", text.casefold()) if token]
    return [token for token in tokens if len(token) >= 2]


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) < 120 and stripped == stripped.title():
        return True
    return stripped.endswith(":") or stripped.startswith(("Example", "Examples", "SCF", "MCSCF", "DFT", "TCE"))

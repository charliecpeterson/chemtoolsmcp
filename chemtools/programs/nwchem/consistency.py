"""Compare an explicit NWChem input with evidence in one output file.

Checks stay conservative when a multi-state deck or sparse output prevents
an unambiguous comparison.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

from chemtools.programs.nwchem.geometry_consistency import (
    compare_single_geometry,
)
from chemtools.programs.nwchem.parse.geometry import (
    OutputGeometryScanner,
)
from chemtools.programs.nwchem.output_task_state import (
    OutputTaskStateScanner,
)
from chemtools.programs.nwchem.task_consistency import (
    compare_single_task_electronic_state_checks,
    compare_task_states,
    normalize_operation,
)


_ECHO_START_RE = re.compile(
    r"^=+\s*echo of input deck\s*=+\s*$",
    re.IGNORECASE,
)
_ECHO_END_RE = re.compile(r"^={20,}\s*$")
_CHARGE_RE = re.compile(
    r"^\s*Charge\s*:\s*([+-]?\d+)\s*$",
    re.IGNORECASE,
)
_MULTIPLICITY_RE = re.compile(
    r"^\s*Spin multiplicity\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_ATOM_COUNT_RE = re.compile(
    r"^\s*No\.\s+of atoms\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_ELECTRON_COUNT_RE = re.compile(
    r"^\s*No\.\s+of electrons\s*:\s*(\d+)\s*$",
    re.IGNORECASE,
)
_START_RE = re.compile(
    r"^\s*(start|restart)(?:\s+(\S+))?",
    re.IGNORECASE,
)
_VECTORS_SUBKEYWORDS = {
    "lock",
    "output",
    "project",
    "reorder",
    "rotate",
    "swap",
}
_DIRECT_TASK_METHODS = {
    "CCSD",
    "CCSD(T)",
    "DFT",
    "DPLOT",
    "MCSCF",
    "MP2",
    "SCF",
    "TDDFT",
}


class NwchemRunConsistency:
    def compare_input_output(
        self,
        input_path: str,
        output_path: str,
        parsed_input: Mapping[str, Any],
        parsed_output: Mapping[str, Any],
        artifact_paths: tuple[str, ...],
    ) -> Mapping[str, Any]:
        input_source = Path(input_path)
        input_text = input_source.read_text(
            encoding="utf-8",
            errors="replace",
        )
        output_evidence = _read_output_evidence(
            Path(output_path),
            parsed_output.get("tasks") or [],
        )
        checks = [
            _input_deck_check(
                input_text,
                output_evidence["echoed_deck_digest"],
            ),
            _task_method_check(parsed_input, parsed_output),
            _task_operation_check(parsed_input, parsed_output),
            _single_value_check(
                "charge",
                parsed_input.get("charges_seen") or [],
                output_evidence["charges"],
                "No single explicit input charge and output charge were available.",
            ),
            _single_value_check(
                "multiplicity",
                parsed_input.get("multiplicities_seen") or [],
                output_evidence["multiplicities"],
                (
                    "No single explicit input multiplicity and output "
                    "multiplicity were available."
                ),
            ),
            _atom_count_check(
                parsed_input,
                output_evidence["atom_counts"],
            ),
            compare_single_geometry(
                input_source,
                parsed_input,
                output_evidence["first_geometry"],
            ),
        ]
        task_state_check = compare_task_states(
            input_source,
            parsed_input,
            parsed_output,
            output_evidence["task_states"],
        )
        checks.extend(
            compare_single_task_electronic_state_checks(
                input_source,
                parsed_input,
                output_evidence["task_states"],
            )
        )
        if task_state_check is not None:
            checks.append(task_state_check)
        checks.append(
            _restart_artifact_check(
                input_source,
                input_text,
                artifact_paths,
            )
        )
        summary = {
            status: sum(check["status"] == status for check in checks)
            for status in ("match", "mismatch", "not_checked")
        }
        if summary["mismatch"]:
            status = "mismatch"
        elif summary["match"]:
            status = "checked"
        else:
            status = "not_checked"
        return {
            "status": status,
            "input_path": str(input_source.resolve()),
            "summary": summary,
            "checks": checks,
        }


def _input_deck_check(
    input_text: str,
    echoed_deck_digest: str | None,
) -> dict[str, Any]:
    if echoed_deck_digest is None:
        return _not_checked(
            "input_deck",
            "The output contains no complete NWChem input-deck echo.",
        )
    input_digest = _normalized_digest(input_text)
    return {
        "field": "input_deck",
        "status": (
            "match"
            if input_digest == echoed_deck_digest
            else "mismatch"
        ),
        "input": {"normalized_sha256": input_digest},
        "output": {"normalized_sha256": echoed_deck_digest},
        "basis": (
            "SHA-256 after newline normalization and trailing-space removal."
        ),
    }


def _read_output_evidence(
    path: Path,
    parsed_tasks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    echo_hasher = sha256()
    echo_started = False
    echo_finished = False
    echo_has_content = False
    pending_blank_lines = 0
    charges = []
    multiplicities = []
    atom_counts = []
    electron_counts = []
    geometry_scanner = OutputGeometryScanner()
    task_state_scanner = OutputTaskStateScanner(parsed_tasks)

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not echo_started and _ECHO_START_RE.match(line):
                echo_started = True
                continue
            if echo_started and not echo_finished:
                if _ECHO_END_RE.match(line):
                    echo_finished = True
                    if echo_has_content:
                        echo_hasher.update(b"\n")
                    continue
                normalized = line.rstrip()
                if not normalized:
                    if echo_has_content:
                        pending_blank_lines += 1
                    continue
                if echo_has_content:
                    echo_hasher.update(
                        b"\n" * (pending_blank_lines + 1)
                    )
                echo_hasher.update(normalized.encode("utf-8"))
                echo_has_content = True
                pending_blank_lines = 0
                continue

            geometry_scanner.feed(line)
            if match := _CHARGE_RE.match(line):
                charges.append(int(match.group(1)))
            if match := _MULTIPLICITY_RE.match(line):
                multiplicities.append(int(match.group(1)))
            if match := _ATOM_COUNT_RE.match(line):
                atom_counts.append(int(match.group(1)))
            if match := _ELECTRON_COUNT_RE.match(line):
                electron_counts.append(int(match.group(1)))
            task_state_scanner.feed(line)

    return {
        "echoed_deck_digest": (
            echo_hasher.hexdigest()
            if echo_finished and echo_has_content
            else None
        ),
        "charges": charges,
        "multiplicities": multiplicities,
        "atom_counts": atom_counts,
        "electron_counts": electron_counts,
        "first_geometry": geometry_scanner.first_geometry,
        "task_states": task_state_scanner.finish(),
    }


def _normalized_digest(text: str) -> str:
    normalized = "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip("\n")
    return sha256((normalized + "\n").encode("utf-8")).hexdigest()


def _task_method_check(
    parsed_input: Mapping[str, Any],
    parsed_output: Mapping[str, Any],
) -> dict[str, Any]:
    input_tasks = (
        parsed_input.get("task_states")
        or parsed_input.get("tasks")
        or []
    )
    input_methods = [
        str(task.get("method") or task.get("module") or "").upper()
        for task in input_tasks
        if task.get("method") or task.get("module")
    ]
    output_methods = [
        str(task.get("method", "")).upper()
        for task in parsed_output.get("tasks") or []
        if task.get("method")
    ]
    if (
        not input_methods
        or len(input_methods) != len(output_methods)
        or any(method not in _DIRECT_TASK_METHODS for method in input_methods)
    ):
        return _not_checked(
            "task_methods",
            (
                "Input and output did not expose the same number of directly "
                "comparable task methods."
            ),
            input_methods,
            output_methods,
        )
    return _comparison("task_methods", input_methods, output_methods)


def _task_operation_check(
    parsed_input: Mapping[str, Any],
    parsed_output: Mapping[str, Any],
) -> dict[str, Any]:
    input_tasks = (
        parsed_input.get("task_states")
        or parsed_input.get("tasks")
        or []
    )
    input_operations = [
        normalize_operation(task.get("operation"))
        for task in input_tasks
    ]
    output_operations = [
        normalize_operation(task.get("kind"))
        for task in parsed_output.get("tasks") or []
    ]
    if (
        not input_operations
        or None in input_operations
        or len(input_operations) != len(output_operations)
        or any(
            operation in {None, "other", "unknown"}
            for operation in output_operations
        )
    ):
        return _not_checked(
            "task_operations",
            (
                "Input and output did not expose the same number of comparable "
                "task operations."
            ),
            input_operations,
            output_operations,
        )
    return _comparison(
        "task_operations",
        input_operations,
        output_operations,
    )


def _single_value_check(
    field: str,
    input_values: list[Any],
    output_values: list[int],
    reason: str,
) -> dict[str, Any]:
    unique_input = list(dict.fromkeys(input_values))
    unique_output = list(dict.fromkeys(output_values))
    if len(unique_input) != 1 or len(unique_output) != 1:
        return _not_checked(
            field,
            reason,
            unique_input,
            unique_output,
        )
    return _comparison(field, unique_input[0], unique_output[0])


def _atom_count_check(
    parsed_input: Mapping[str, Any],
    output_counts: list[int],
) -> dict[str, Any]:
    output_counts = list(dict.fromkeys(output_counts))
    if (
        parsed_input.get("geometry_block_count") != 1
        or parsed_input.get("atom_count") is None
        or len(output_counts) != 1
    ):
        return _not_checked(
            "atom_count",
            (
                "A single input geometry and a single output atom count were "
                "not both available."
            ),
            parsed_input.get("atom_count"),
            output_counts,
        )
    return _comparison(
        "atom_count",
        parsed_input["atom_count"],
        output_counts[0],
    )


def _restart_artifact_check(
    input_path: Path,
    input_text: str,
    artifact_paths: tuple[str, ...],
) -> dict[str, Any]:
    references = _external_restart_references(input_path, input_text)
    if not references:
        return _not_checked(
            "restart_artifacts",
            "The input declares no external restart artifacts.",
        )
    supplied = {str(Path(path).resolve()) for path in artifact_paths}
    missing = [
        reference
        for reference in references
        if reference["path"] not in supplied
    ]
    return {
        "field": "restart_artifacts",
        "status": "mismatch" if missing else "match",
        "input": {"references": references},
        "output": {
            "supplied_paths": sorted(supplied),
            "missing_paths": [item["path"] for item in missing],
        },
        "basis": "Explicit related-artifact paths supplied to inspect_run.",
    }


def _external_restart_references(
    input_path: Path,
    text: str,
) -> list[dict[str, str]]:
    logical_text = re.sub(r"\\\s*\n\s*", " ", text)
    references: list[dict[str, str]] = []
    produced: set[str] = set()
    for line in logical_text.splitlines():
        uncommented = line.split("#", 1)[0]
        if match := _START_RE.match(uncommented):
            if match.group(1).lower() == "restart" and match.group(2):
                name = match.group(2)
                references.append(
                    _reference(
                        input_path,
                        f"{name}.db",
                        "restart",
                        declared=name,
                    )
                )

        try:
            tokens = shlex.split(uncommented)
        except ValueError:
            continue
        lowered = [token.lower() for token in tokens]
        if len(tokens) < 3 or lowered[:2] != ["vectors", "input"]:
            continue

        input_names = _vectors_input_names(tokens[2:])
        for name in input_names:
            if name not in produced:
                references.append(
                    _reference(input_path, name, "vectors_input")
                )
        if "output" in lowered[2:]:
            output_index = lowered.index("output", 2)
            if output_index + 1 < len(tokens):
                produced.add(tokens[output_index + 1])
    return list({
        (item["directive"], item["path"]): item
        for item in references
    }.values())


def _vectors_input_names(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    first = tokens[0].lower()
    if first == "atomic":
        return []
    if first != "fragment":
        return [] if first in _VECTORS_SUBKEYWORDS else [tokens[0]]

    names = []
    for token in tokens[1:]:
        if token.lower() in _VECTORS_SUBKEYWORDS:
            break
        names.append(token)
    return names


def _reference(
    input_path: Path,
    name: str,
    directive: str,
    *,
    declared: str | None = None,
) -> dict[str, str]:
    path = Path(name)
    if not path.is_absolute():
        path = input_path.parent / path
    return {
        "directive": directive,
        "declared": declared or name,
        "path": str(path.resolve()),
    }


def _comparison(
    field: str,
    input_value: Any,
    output_value: Any,
) -> dict[str, Any]:
    return {
        "field": field,
        "status": "match" if input_value == output_value else "mismatch",
        "input": input_value,
        "output": output_value,
    }


def _not_checked(
    field: str,
    reason: str,
    input_value: Any = None,
    output_value: Any = None,
) -> dict[str, Any]:
    check = {
        "field": field,
        "status": "not_checked",
        "reason": reason,
    }
    if input_value is not None:
        check["input"] = input_value
    if output_value is not None:
        check["output"] = output_value
    return check


NWCHEM_RUN_CONSISTENCY = NwchemRunConsistency()


__all__ = ["NWCHEM_RUN_CONSISTENCY", "NwchemRunConsistency"]

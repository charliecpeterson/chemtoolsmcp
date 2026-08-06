"""Generate and validate every complete f-block catalog CSF list with GRASP.

Run this script inside a GRASP2018 environment. All generated artifacts and
the machine-readable summary stay under the caller-provided scratch directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

from chemtools.reference import (
    load_fblock_catalog,
    validate_grasp_fblock_artifacts,
)


def check_catalog(arguments: argparse.Namespace) -> dict[str, object]:
    scratch = arguments.scratch.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    catalog = load_fblock_catalog()
    checked: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    started = monotonic()

    for element in catalog.elements:
        for state in element.states:
            identity = f"{element.symbol}-{state.slug}"
            if not state.confline or state.core_menu is None:
                skipped.append({"state": identity, "reason": "incomplete_grasp_input"})
                continue
            work = scratch / identity
            work.mkdir(exist_ok=True)
            stdin = "\n".join([
                "*",
                str(state.core_menu),
                state.confline,
                "",
                state.active_set,
                state.jrange,
                "0",
                "n",
                "",
            ])
            try:
                completed = subprocess.run(
                    [arguments.rcsfgenerate],
                    input=stdin,
                    text=True,
                    cwd=work,
                    capture_output=True,
                    timeout=arguments.timeout,
                    check=False,
                )
                (work / "rcsfgenerate.stdout").write_text(
                    completed.stdout,
                    encoding="utf-8",
                )
                (work / "rcsfgenerate.stderr").write_text(
                    completed.stderr,
                    encoding="utf-8",
                )
                if completed.returncode != 0:
                    raise ValueError(
                        f"rcsfgenerate exited {completed.returncode}"
                    )
                artifact = work / "rcsf.out"
                validated = validate_grasp_fblock_artifacts(
                    element.symbol,
                    state.slug,
                    artifact,
                )
                checked.append({
                    "state": identity,
                    "electron_count": validated["csf"]["electron_count"],
                    "block_count": len(validated["csf"]["blocks"]),
                    "csf_count": validated["csf"]["csf_count"],
                })
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                failures.append({"state": identity, "error": str(error)})

    return {
        "schema_version": "chemtools.fblock-grasp-catalog-check/1",
        "catalog_sha256": catalog.metadata.catalog_sha256,
        "elapsed_seconds": monotonic() - started,
        "checked_count": len(checked),
        "skipped_count": len(skipped),
        "failure_count": len(failures),
        "checked": checked,
        "skipped": skipped,
        "failures": failures,
        "success": not failures and len(checked) == 616 and len(skipped) == 17,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--rcsfgenerate",
        default="/apps/grasp/bin/rcsfgenerate",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    arguments = parser.parse_args()
    evidence = check_catalog(arguments)
    evidence_path = arguments.scratch.expanduser().resolve() / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "evidence": str(evidence_path),
        "checked_count": evidence["checked_count"],
        "skipped_count": evidence["skipped_count"],
        "failure_count": evidence["failure_count"],
        "success": evidence["success"],
    }, sort_keys=True))
    return 0 if evidence["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

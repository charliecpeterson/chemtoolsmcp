"""Verify every named artifact in an external reference manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chemtools.reference.external_corpus import (
    DEFAULT_CASE_BYTE_LIMIT,
    load_reference_manifest,
    verify_reference_case,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify pinned artifacts without running chemistry parsers."
    )
    parser.add_argument("manifest")
    parser.add_argument(
        "--corpus",
        default=os.environ.get("CHEMTOOLS_REFERENCE_CORPUS"),
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Case ID to verify; repeat to select more than one.",
    )
    parser.add_argument(
        "--byte-limit",
        type=int,
        default=DEFAULT_CASE_BYTE_LIMIT,
        help="Maximum required bytes per case before filesystem access.",
    )
    arguments = parser.parse_args(argv)

    manifest = load_reference_manifest(arguments.manifest)
    selected = arguments.case_ids or [
        case["id"] for case in manifest["cases"]
    ]
    records = [
        verify_reference_case(
            manifest,
            case_id,
            arguments.corpus,
            byte_limit=arguments.byte_limit,
        )
        for case_id in selected
    ]
    report = {
        "schema": "chemtools.reference-corpus-verification/1",
        "manifest": str(Path(arguments.manifest).resolve()),
        "corpus_root": (
            str(Path(arguments.corpus).expanduser().resolve())
            if arguments.corpus
            else None
        ),
        "case_count": len(records),
        "verified_count": sum(
            record["outcome"] == "verified" for record in records
        ),
        "no_reference_count": sum(
            record["outcome"] == "no_reference" for record in records
        ),
        "records": records,
    }
    print(json.dumps(report, indent=2))
    return 3 if report["no_reference_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

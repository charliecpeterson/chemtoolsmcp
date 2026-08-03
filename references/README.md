# External reference manifest

This directory records reviewed files from a local chemistry corpus without
copying that corpus into Chemtools. Each entry has a relative path, exact byte
size, SHA-256 hash, validation status, and comparison contract.

The current Orbitron file has eight cases: the original QE and Molcas checks,
the resolved Molcas vibration case, and three QE geometry comparisons for Si,
Fe, and FeO. The eighth case pins the `last_attempted` role and source text for
a failed benzene relaxation. [ADR 005](../docs/adr/005-reference-corpus-boundaries.md)
proposes the general schema and keeps storage tier, redistribution permission,
case purpose, and scientific status as separate fields.

Set `CHEMTOOLS_REFERENCE_CORPUS` to the corpus root. The checker resolves only
the listed paths and never scans the tree. A missing file, size change, or hash
change is reported as `no_reference` and is not compared until the case is
reviewed. Size is checked before the checker hashes or parses a source.

Run the current Orbitron slice with:

```bash
export CHEMTOOLS_ORBITRON_CLI=/path/to/orbitron
export CHEMTOOLS_REFERENCE_CORPUS=/path/to/input_examples
.venv/bin/python scripts/check_orbitron_contract.py
```

The JSON report uses `agree`, `disagree`, `tool_refused`, and `no_reference`
outcomes. Exit codes are 0 for full agreement, 1 for a disagreement, 2 when
Orbitron refuses a case, and 3 when a pinned reference is unavailable.

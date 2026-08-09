# External reference manifest

This directory records selected files from a local chemistry corpus without
copying that corpus into Chemtools. Each entry has a relative path, exact byte
size, SHA-256 hash, scientific status, and intended purpose.

`../chemtools/data/reference_cases/nwchem_behavior_cases.json` pins the first
five expert NWChem workflows for
the simplification behavior lock. All five remain `exploratory`; verification
establishes that the local bytes match the audit, not that the scientific
expectations have been approved.

`../chemtools/data/reference_cases/non_nwchem_review_cases.json` pins two
review candidates each for Molcas,
DIRAC, GRASP, QE, and QMCPACK. Every case remains `exploratory`. Its recorded
inspection verdict is an observation used to organize review, not an approved
scientific expectation. The review queue and unresolved questions are in
[`notes/non-nwchem-reference-review-queue.md`](../notes/non-nwchem-reference-review-queue.md).

The current Orbitron file has eight cases: the original QE and Molcas checks,
the resolved Molcas vibration case, and three QE geometry comparisons for Si,
Fe, and FeO. The eighth case pins the `last_attempted` role and source text for
a failed benzene relaxation. [ADR 005](../docs/adr/005-reference-corpus-boundaries.md)
proposes the general schema and keeps storage tier, redistribution permission,
case purpose, and scientific status as separate fields.

`../chemtools/data/reference_cases/orca_experimental_cases.json` pins nineteen
ORCA 6.1.1 serial experiments. They cover basic DFT and post-HF calculations,
SCF failure and recovery, metal and f-block systems, RIJCOSX, QM/MM and
Crystal-QMMM, CASSCF with NEVPT2 or CASPT2, MRCI, TD-DFT, EOM-CCSD, and
ORCA_ESD vibronic spectra with radiative rates. They
remain `exploratory` or `regression_failure` while the ORCA backend and
scientific expectations are reviewed.

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

Verify only the NWChem artifact identities with:

```bash
export CHEMTOOLS_REFERENCE_CORPUS=/path/to/input_examples
.venv/bin/python scripts/check_reference_manifest.py \
  chemtools/data/reference_cases/nwchem_behavior_cases.json
```

This command verifies only listed files. It refuses an oversized case before
filesystem access, then checks containment, exact size, and SHA-256. It does
not parse outputs or change scientific status.

Use the same command with
`chemtools/data/reference_cases/non_nwchem_review_cases.json` to verify the ten
non-NWChem candidates. These two manifests live under packaged data so
`find_reference_case` works from an installed wheel.

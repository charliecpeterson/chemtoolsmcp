# Testing Baseline

This document records the test collection at the end of the Phase 0 behavior
lock. It is a point-in-time inventory, not a coverage percentage.

Baseline date: 2026-07-30
Baseline repository commit: `3ea17e4`

Run the default suite from the repository root:

```bash
.venv/bin/python -m pytest -q
```

Inspect the collected nodes without running them:

```bash
.venv/bin/python -m pytest --collect-only -q
```

## Current collection

The working tree collects 60 tests. Of those, 33 are in the three test modules
present at the baseline commit; 27 are Phase 0 tests added in the current
worktree.

| Module | Collected tests | At `3ea17e4` | Contract protected |
| --- | ---: | --- | --- |
| `tests/test_mcp_golden.py` | 6 | No | Five program and generic `tools/call` requests, response shape, and pinned scientific fields |
| `tests/test_mcp_protocol.py` | 5 | No | Supported MCP version policy and initialization negotiation in both servers |
| `tests/test_nwchem_input_parsing.py` | 12 | Yes | Fragment-vector parsing, lint findings, and local resource warnings |
| `tests/test_orbitron_contract.py` | 8 | No | QE task, geometry, and failed-run provenance agreement; Molcas vibration checks; reference identity; manifest containment; and outcome exit codes |
| `tests/test_orbitron_integration.py` | 36 | No | CLI resolution, fixed subprocess arguments, schema checks, geometry-role validation, and error preservation |
| `tests/test_orbitron_analysis_mcp.py` | 14 | No | Analysis response schemas, canonical mappings, structured uncertainty, and fixed local-path-only tool definitions |
| `tests/test_registration.py` | 4 | Yes | Tool imports, schema and handler integrity, uniqueness, and compatibility imports |
| `tests/test_session_parsers.py` | 17 | Yes | DIRAC, Molcas, NWChem, and GRASP parser and scientific-verdict regressions |
| `tests/test_tool_inventory.py` | 4 | No | Exact tool counts, ownership metadata, rendering stability, and generated documents |
| Total | 60 | 33 at baseline | Default local collection |

`tests/test_mcp_golden.py` defines two test functions. Its first function is
parameterized over five case files, so pytest collects six nodes from the
module.

The default suite has no network dependency and does not run chemistry
executables, schedulers, containers, or the installed Orbitron binary. The
Orbitron subprocess adapter is unit-tested with controlled command results.
The external Orbitron corpus check remains an explicit command:

```bash
export CHEMTOOLS_ORBITRON_CLI=/path/to/orbitron
export CHEMTOOLS_REFERENCE_CORPUS=/path/to/input_examples
.venv/bin/python scripts/check_orbitron_contract.py
```

### Post-baseline companion-runtime additions

The Phase 0 counts above are historical. The Open Babel conversion fixture
corpus adds two self-contained structure tests in
`tests/test_openbabel_fixtures.py` and one explicit runtime check. It remains
outside the default suite because it invokes the configured companion
interpreter:

```bash
export CHEMTOOLS_SCIENCE_PYTHON=/path/to/chemtools-science/bin/python
.venv/bin/python scripts/check_openbabel_fixture_corpus.py
```

## Historical test inventory

Commit `8f88027f` deleted three test modules on 2026-04-19. Its parent,
`1b7a79d`, is the last revision before deletion. Counting the test methods in
that revision gives:

| Historical module | Test methods | Removed-corpus dependency | Current assessment |
| --- | ---: | ---: | --- |
| `tests/test_chemtools.py` | 84 | 72 | Broad NWChem library tests; restore selected behavior with owned fixtures |
| `tests/test_chemtools_mcp.py` | 52 | 44 | MCP mirrors of many library tests; keep only distinct boundary contracts |
| `tests/test_nwchem_docs.py` | 7 | 0 | Useful and runnable against the bundled documentation after import updates |
| Total | 143 | 116 | 27 methods do not refer to the removed corpora |

The old suite expected repository-local copies of `nwchem-test`,
`nwchemaitest`, and Orbitron's test corpus. Those paths are ignored and are
absent from this checkout. The ignored `test_phase1` and `test2` directories
are also absent, so there are no local test artifacts to recover from them.

The 27 methods without removed-corpus references are not all immediate copy
candidates. Some duplicate current inventory or protocol tests. Others launch
a subprocess or send a signal and need a clear integration-test boundary.

## Recovery queue

Restore behavior, not the old file layout.

| Priority | Cases to recover | Form |
| --- | --- | --- |
| 1 | NWChem document listing, search, syntax lookup, excerpts, and topic guides | Update imports to `chemtools.programs.nwchem.docs`; assert exact files and selected matches from the bundled docs |
| 1 | Task-title parsing, multiplicity handling, invalid task syntax, incomplete optimization, oscillatory SCF, and divergent geometry decisions | Small pytest modules with inline input and output fixtures |
| 1 | High-value guided MCP calls for progress review and recovery decisions | Add a small number of golden cases at the real `tools/call` boundary |
| 2 | Runner-profile parsing, launch-plan rendering, and adaptive watch intervals | Unit tests with temporary profiles and controlled process state |
| 2 | Fe wrong-state and fragment-guess cases, SCF nonconvergence, interrupted frequency, imaginary modes, MCSCF retry, population analysis, cube files, and actinide basis/ECP behavior | Curated, hash-pinned fixtures under the reference-corpus policy in ADR 005 |
| 3 | Direct process launch and termination | Marked integration tests with isolated child processes and bounded timeouts |

Do not restore all 52 historical MCP tests as a mirror of library tests. That
would pin the same details twice and make changes expensive. Keep library
tests for scientific logic, then use MCP golden cases for request routing,
public names, response shape, capability gates, and a representative set of
guided workflows.

Do not restore absolute assumptions about `nwchem-test`, `nwchemaitest`, or an
Orbitron source checkout. A fixture must be committed and redistributable, or
selected through a manifest that records identity, permission, purpose, and
scientific status.

## Gaps not supplied by the historical suite

The historical files do not cover several current boundaries:

- JSON-RPC errors for unknown methods, unknown tools, malformed calls, and
  capability-filtered tools.
- Execution-target and artifact-provenance contracts proposed for Phases 2
  and 3.
- QMCPACK production-output support beyond the bounded scalar, population, and
  fixed-layout HDF5 inspection slices. XML input parsing and linting have
  regression coverage; Quantum ESPRESSO has a broader input and output review
  slice.
- A measured line or branch coverage baseline.

These are new tests to design. They should not be described as recovered
historical coverage.

## Updating this baseline

Update the collection table when test modules or parameterized case counts
change. Update the historical section only when another source of old tests is
found. Keep external executables and non-redistributable corpora outside the
default pytest command.

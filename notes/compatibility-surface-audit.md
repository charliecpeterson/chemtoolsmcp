# Compatibility surface audit

The repository has one canonical executable and one catalog-driven MCP
composition path. Compatibility names remain, but they are now separated by
surface and recorded in `docs/tool-inventory.json`.

Audit date: 2026-08-09

## Current inventory

| Surface | Current state |
| --- | --- |
| MCP definitions | 329 canonical and 9 advertised legacy definitions |
| Hidden MCP aliases | 15 callable names omitted from `tools/list` |
| Executable aliases | `chemtools-nwchem` and `chemtools-nwchem-docs` |
| Python compatibility imports | 6 facades or shim modules |

The inventory assigns `deprecated_since: 0.1.0` to each compatibility surface
for the final compatibility release and leaves `remove_after` unset. The
standalone docs executable is marked as a distinct legacy surface because its
six-tool server is not contract-equivalent to the main server.

The 15 hidden aliases now live in a validated metadata registry. Registration
rejects missing canonical targets, name collisions, alias chains, and program
or mode availability broader than the target. The inventory records each
argument adapter, availability boundary, reason, and contract status. Missing
historical schemas and effects remain `null` rather than inferred from current
handlers.

The complete external-corpus suite passed 1,779 tests with the registry in
place. Base and DIRAC-extra checks also passed against an isolated installation
of the rebuilt wheel.

## First-party usage

The Stampede3 MCP example now launches `chemtools`. No maintained example or
configuration launches `chemtools-nwchem`. Project documentation still names
both compatibility executables so users can identify old installations.

Normal MCP startup imports `_nwchem_provider.py`. After the `v0.1.0` tag, the
unused `chemtools.mcp.tools.nwchem` aggregator and `_nwchem_base.py` wildcard
facade were removed. Absence tests now reject reinstalling either module, and
no production MCP module imports the top-level `chemtools` facade.

First-party recovery payloads now recommend canonical MCP names. State and SCF
recovery recommendations call `suggest_nwchem_recovery` with an explicit
`mode`, and case-review recommendations call `analyze_nwchem_case`. An AST
contract scans first-party Python sources for deprecated names in `tool`
fields, while branch tests pin the argument translation.

The complete external-corpus suite passed 1,767 tests after this migration.
Base and DIRAC-extra checks also passed against an isolated installation of
the rebuilt wheel.

## Direct Python caller evidence

The broad top-level `chemtools` facade exports 121 names. No maintained
library module imports it. The only non-test caller is:

- `scripts/check_wheel_install.py`, which imports the broad facades on purpose
  to verify compatibility during the migration window.

`scripts/check_nwchem_atomic_input.py` now imports
`draft_nwchem_atom_input` from its focused NWChem owner. A source-only scan of
the other maintained repositories under `/home/charlie/projects` and
`/home/charlie/mcps`, plus `/home/charlie/projects/scripts`, found no
`chemtools`, `chemtools.api`, `chemtools.api_input`, or
`chemtools.api_strategy` Python imports. The scan excluded virtual
environments, dependency trees, build products, caches, and this repository.
The broader scan was repeated on 2026-08-09 for the deprecated executables,
all 15 hidden MCP names, and all 7 Python shims; it found no maintained caller
outside this repository.

Other maintained scripts already import focused application, program,
integration, execution, or reference modules. Six test imports cover the
broad facade, mostly to pin compatibility identity. Repository and
maintained-workspace evidence does not show a current personal workflow that
requires the top-level `chemtools` facade, `chemtools.api`,
`chemtools.api_input`, or `chemtools.api_strategy` as supported public APIs.

The smallest supported policy would make the `chemtools` command and guided
MCP surface the product API. Focused module paths would remain available for
scientific Python work, but the 121-name top-level facade and its three broad
aggregators would be compatibility-only for one tagged release. A new small
Python facade should wait until a real notebook or script needs one; the
repository does not yet identify that caller.

## Disposition

- Keep both executable aliases through the `v0.1.0` migration window. Decide
  whether the standalone docs server needs an exact replacement before
  assigning it a removal version.
- Treat the top-level Python facade and three `api*` aggregators as
  compatibility-only through the final compatibility release. Supported
  direct Python use goes through focused application, program, execution,
  integration, persistence, and reference modules. Do not add a replacement
  facade until a real caller needs one.
- Keep all hidden MCP aliases callable through the `v0.1.0` migration window.
  Their historical schemas remain unverified, so no removal version is set.
- Recover historical alias schemas and effects, then add exact result, error,
  and effect checks before marking contracts verified or hiding the nine
  advertised legacy definitions.

## Post-release removals

`chemtools.mcp.tools.nwchem` and `_nwchem_base.py` were removed after
`v0.1.0`. The files had no maintained caller, normal catalog composition
already used `_nwchem_provider.py`, and their routing helpers duplicated the
focused generic MCP owner. The generated inventory now reports six Python
compatibility shims. The `v0.1.0` tag remains the migration source for callers
that still need the former wildcard or dynamic handler import behavior. The
post-release package and MCP server version is `0.2.0.dev0`; compatibility
metadata remains anchored to `deprecated_since: 0.1.0`. Focused checks passed
75 tests, followed by all 1,891 tests with the external corpus. Base and
DIRAC-extra isolated installs of wheel SHA-256
`ce3a2406c8e54c050b967ea9d8e7262980cdc145f2a99b5d1c71591911087a0e`
confirmed both removed modules are absent.

The exact `chemtools.execution.legacy_profiles` import facade was also removed
after `v0.1.0`. Its version 1 schema and conversion implementation remain in
`chemtools.execution.profiles`; only the duplicate import path left. A
maintained-workspace scan found no external caller. Focused checks passed 81
tests, followed by all 1,890 tests with the external corpus. Base and
DIRAC-extra isolated installs of wheel SHA-256
`6f54f7d000e5871b9bb9d5d6697dc09e070f31bdb8f0ef65ec9f5c05e59978a1`
confirmed the removed module is absent and all eight bundled profiles still
load through the canonical owner.

## Final-release readiness

The canonical MCP server now reports `chemtools` instead of the old
NWChem-specific identity. The `chemtools-nwchem` compatibility command names
`chemtools` as its replacement and gives a removal warning. The concise
[migration note](../docs/compatibility-migration.md) covers executable, MCP,
and Python callers. The compatibility ledger starts at `0.1.0` without
promising a removal version.

Focused compatibility and protocol checks passed 33 tests. The full suite
passed 1,891 tests against the external corpus both in the development tree
and in a clean detached worktree. Annotated tag `v0.1.0` points to commit
`c9a4298`. Base and DIRAC-extra isolated installs of the exact-tag wheel
SHA-256
`aecf564b9f9677d6de1c153c2c86a245568926780caa4e67eeaacee4b49b3e3e`
negotiated MCP `2025-11-25`, reported server name `chemtools`, listed the
eleven guided tools, and preserved the compatibility imports.

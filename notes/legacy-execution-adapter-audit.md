# Legacy execution adapter audit

Audit date: 2026-08-07

The legacy execution modules are not ready for wholesale removal. The old
profile import path is now isolated, but the renderer, response translators,
unowned status functions, and scheduler wrappers still have first-party
callers.

## Profile ownership

`chemtools.execution.profiles` now owns version 1 profile loading, default
merging, and conversion into typed resources, installations, and Slurm target
settings. Six application adapters and the NWChem, Molcas, DIRAC, GRASP, QE,
and QMCPACK launch providers import that owner directly.

`chemtools.execution.legacy_profiles` is an exact compatibility facade. No
runtime module imports it. The installed-wheel check still imports it on
purpose to protect the old Python path until the final compatibility release.

Removal gate: tag the compatibility release, confirm that external callers
have moved to `chemtools.execution.profiles`, then remove the facade and its
identity test.

Verification: 127 focused profile and execution tests passed, followed by all
1,854 tests with the external corpus. Wheel SHA-256
`0d25226a195fccac80f88bb3dd5ad5a6744679e43606ee98774d4b197b938e23` passed
the isolated installed-copy check. The old and canonical profile imports were
identical, and the installed MCP command negotiated protocol `2025-11-25`,
listed the eleven guided tools, inspected a representative NWChem output, and
prepared an approval-gated launch without writing output files.

## Legacy renderer and launcher

Eight first-party runtime modules still import `execution.legacy_runner`:

- NWChem, QE, and QMCPACK application adapters use the old renderer to keep
  their low-level response shapes.
- The three public scheduler modules still call the old render, launch,
  status, watch, and cancellation functions.
- The NWChem compatibility runner still calls the old render and launch
  functions.
- `core/runner.py` remains a pure compatibility facade with no first-party
  caller.

The guided interface has a typed, approval-gated launch and owned monitoring
path for the five retained NWChem workflows. Its provider now reads version 1
profile values from `execution.profiles` and builds the typed NWChem plan
directly. No guided runtime path imports the old renderer.

The direct provider produced the same prepared plan and executor rendering as
the former path across all seven supported bundled local and Slurm profiles.
Focused launch, model, and boundary checks passed 30 tests, followed by all
1,855 tests with the external corpus. Installed wheel SHA-256
`f6fbc933a82c3e80ed5c47c0c2c6417316799c7dc784415bd064a714a105488f`
passed the isolated MCP exchange.

Removal gates:

1. Replace each retained low-level preview with direct typed preparation after
   that program has an accepted guided launch provider.
2. Reimplement retained scheduler calls over typed targets after their unowned
   status contract is decided.

The QE and QMCPACK comparison confirmed that typed plans cover commands and
artifacts but not the full version 1 preview dictionaries. A shared replacement
would recreate the old renderer as another response projector. Keep these two
calls until their guided results define which preview fields survive.

The retention decision and per-program evidence are recorded in
[`low-level-execution-retention-audit.md`](low-level-execution-retention-audit.md).

## Legacy output archival

`execution/legacy_archive.py` now owns the timestamped, collision-safe rename
policy used before compatibility launches. All six program application
adapters import it directly. `execution.legacy_runner` keeps exact imports of
both archive functions for its old direct Python surface and its remaining
version 1 launch implementation.

Focused execution and import-boundary checks passed 69 tests, followed by all
1,856 tests with the external corpus. The archive integration cases for
Molcas, DIRAC, and GRASP retained their exact response paths and file contents.
Installed wheel SHA-256
`63e6dd6e97a688293c6c37ba8ab8ae417405b5fc96c25e90c7bf5c742338158e`
preserved both old archive imports as exact identities and passed the guided
MCP exchange.

## Resource inspection

`execution/resource_inspection.py` now owns local CPU and memory budgeting plus
Slurm and PBS partition discovery. The generic resource tool and NWChem
preflight import that owner directly. `execution.legacy_runner` keeps exact
imports for the old Python surface.

Focused resource, workflow, MCP, compatibility, and import-boundary checks
passed 69 tests, followed by all 1,859 tests with the external corpus.
Installed wheel SHA-256
`0fc350cf33074453e3dc945b4fe4da32f908c30f7eaa59d92c38b94a8d55a416`
included the focused owner, preserved the old import identities, and passed
the guided MCP exchange.

## Legacy status

`execution.legacy_status` accepts arbitrary PIDs, scheduler job IDs, output
paths, `.jobid` files, PBS jobs, and LSF jobs. Guided `monitor_run` accepts
only a launch ID owned by the current execution service. These are different
contracts.

The old status implementation is still reached through the three scheduler
wrappers and `programs.nwchem.legacy_status`. Program-specific monitoring
uses typed status for owned launches, then retains the legacy path for
unowned identifiers.

Removal gate: decide whether arbitrary unowned process and scheduler
inspection is still part of the personal Python workflow. If it is retained,
it needs a clearly named direct API outside the guided surface. If it is not,
remove the fallback tools and wrappers after the compatibility release.

The current evidence and proposed Slurm-plus-file scope are recorded in
[`unowned-status-scope-audit.md`](unowned-status-scope-audit.md).

## Legacy response projection

`application.legacy_execution` translates typed results into old dictionaries
for six program application adapters. It contains no execution mechanism, but
it cannot leave while those low-level MCP responses remain supported.

Removal gate: retire or version those response contracts after the final
compatibility release. Guided `launch_run` and `monitor_run` do not depend on
this projection.

## Current disposition

Keep the canonical `profiles.py` owner and focused `legacy_archive.py` seam.
Keep the remaining compatibility modules until their callers are removed in
the order above. The guided NWChem path is the reference for direct typed plan
preparation; do not add another profile schema or execution abstraction for
the remaining compatibility callers.

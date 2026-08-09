# Legacy execution adapter audit

Audit date: 2026-08-09

The legacy execution modules are not ready for wholesale removal. The old
profile import, status, and non-NWChem scheduler paths have been removed, but
the renderer still serves the old direct Python runner and `render_job_script`.
The response translator remains for the retained low-level NWChem MCP launch.

## Profile ownership

`chemtools.execution.profiles` now owns version 1 profile loading, default
merging, and conversion into typed resources, installations, and Slurm target
settings. The NWChem compatibility application adapter and the NWChem, Molcas,
DIRAC, GRASP, QE, and QMCPACK launch providers import that owner directly.

`chemtools.execution.legacy_profiles` was an exact compatibility facade. No
runtime module imported it, and a maintained-workspace scan found no external
caller. It was removed after the final compatibility release, `v0.1.0`.

The canonical `chemtools.execution.profiles` owner and version 1 profile schema
remain unchanged. Source and installed-wheel absence checks prevent the old
facade from returning on the `0.2.0.dev0` line.

Before removal, 127 focused profile and execution tests and all 1,854 tests
with the external corpus established that the facade and canonical imports
were identical. After removal, 81 focused checks and all 1,890 tests with the
external corpus passed. Base and DIRAC-extra isolated installs of wheel
SHA-256
`6f54f7d000e5871b9bb9d5d6697dc09e070f31bdb8f0ef65ec9f5c05e59978a1`
confirmed the old module is absent while the canonical owner still loads all
eight bundled profiles. The same wheel is installed in the repository-local
`venv`.

## Legacy renderer and launcher

Two first-party runtime modules still import `execution.legacy_runner`:

- The NWChem compatibility runner still calls the old render and launch
  functions.
- `core/runner.py` remains a pure compatibility facade with no first-party
  caller.

The low-level NWChem MCP adapter now prepares through the same typed provider
as guided launch. Its dry-run and live responses retain the established keys
but expose the executor's exact command or Slurm script. The adapter no longer
imports the legacy renderer or the broad NWChem runner, and environment
overrides now enter the typed plan before rendering.

The guided interface has typed, approval-gated NWChem, OpenMolcas, DIRAC,
GRASP workflow, Quantum ESPRESSO, and QMCPACK launch providers. Each reads
version 1 profile values from `execution.profiles` or selects a schema-2 target
from the server catalog and builds its program-owned plan directly. No guided
runtime path imports the old renderer. Equivalent named local MPI and Slurm
targets produce the same approval-bound plan as their profile migration
adapters.

The direct provider produced the same prepared plan and executor rendering as
the former path across all seven supported bundled local and Slurm profiles.
Focused launch, model, and boundary checks passed 30 tests, followed by all
1,855 tests with the external corpus. Installed wheel SHA-256
`f6fbc933a82c3e80ed5c47c0c2c6417316799c7dc784415bd064a714a105488f`
passed the isolated MCP exchange.

Removal gates:

1. Replace each retained low-level preview with direct typed preparation after
   that program has an accepted guided launch provider. This is complete for
   the low-level NWChem launch.
2. Migrate or retire `render_job_script` and the old direct Python launch
   functions before removing `execution.legacy_runner`.

The NWChem adapter migration passed 95 focused execution, launch, model,
profile, and boundary checks, followed by all 1,918 tests with the external
corpus. Base and DIRAC-extra isolated installs of wheel SHA-256
`fa6ee2215e4305e4abb08bf9a535fde8bb694769ca6dad03e21746291b6eed0a`
passed, and the same wheel is installed in the repository-local `venv`.

The QE, QMCPACK, Molcas, DIRAC, and asynchronous GRASP low-level MCP execution surfaces were
removed after their guided providers passed the parity and external-corpus
gates. The GRASP comparison confirmed that its typed one-file plan replaces
the asynchronous workflow wrappers but does not cover the interactive and
structured-workflow response contracts. Those distinct synchronous calls
remain without importing the legacy renderer.

The GRASP removal passed 103 focused checks and all 1,915 tests with the
external corpus. Base and DIRAC-extra isolated installs of wheel SHA-256
`48c09af051cf95d4228adfae1dd6fb515d82cb3ac7e587e10777d2774366cdaa`
confirmed that the scheduler and monitoring modules are absent while the
guided provider and synchronous interactive contract remain available.

The Molcas and DIRAC removal passed 123 focused architecture and guided checks,
all 27 retained program test modules, and all 1,923 tests with the external
corpus. Base and DIRAC-extra isolated installs of wheel SHA-256
`64ad9e361a9f7926b492abf886a97c23394cf8674edbb2d6f6ae916fbb5ad8b7`
confirmed that their six compatibility modules are absent.

The retention decision and per-program evidence are recorded in
[`low-level-execution-retention-audit.md`](low-level-execution-retention-audit.md).

## Legacy output archival

`execution/legacy_archive.py` now owns the timestamped, collision-safe rename
policy used before compatibility launches. The NWChem application adapter
imports it directly. `execution.legacy_runner` keeps exact
imports of both archive functions for its old direct Python surface and its
remaining version 1 launch implementation.

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

## External status

`execution.external_status` now owns the approved retained contract: read-only
file inspection and attachment to an external Slurm job through an explicit
profile and job ID. `programs.nwchem.external_status` adds NWChem progress
interpretation. Other programs use the generic guided monitor or call the
external-status owner explicitly.

The post-`v0.1.0` cleanup removed both former `legacy_status` modules,
arbitrary PID probing, PBS and LSF status parsing, `.jobid` inference, and
direct Python cancellation wrappers. Local process status and every MCP
cancellation require a launch owned by the current execution service. The
decision evidence is recorded in
[`unowned-status-scope-audit.md`](unowned-status-scope-audit.md).

Focused checks passed 109 tests, followed by all 1,899 tests with the external
corpus. Base and DIRAC-extra isolated installs of wheel SHA-256
`4c5b6fba061968a2016510792523d9466edd290d41a9da87111b13084b8eccf7`
confirmed the removed import paths and retained external-status boundary.

## Legacy response projection

`application.legacy_execution` translates typed results into old dictionaries
for the NWChem compatibility adapter. It contains no execution mechanism, but
it cannot leave while those low-level MCP responses remain supported.

The post-release audit found that this is a real shared boundary rather than a
removable facade. Its 82 lines keep launch IDs, effective argv, Slurm
submission fields, timeout translation, `.jobid` compatibility writes, and
scheduler cancellation results stable for NWChem. Renaming it would only hide
that the response contract is legacy. Before the redundant program adapters
were removed, the six execution contract suites passed 48 tests.

Removal gate: remove the projector when the remaining low-level tools are
retired or their response contracts are replaced. Do not remove or rename it
as an independent cleanup. Guided `launch_run` and `monitor_run` do not depend
on this projection.

## Current disposition

Keep the canonical `profiles.py` owner and focused `legacy_archive.py` seam.
Keep the remaining compatibility modules until their callers are removed in
the order above. Both NWChem MCP launch paths now use the typed provider. The
remaining decision concerns `render_job_script` and direct Python compatibility,
not another program-specific execution adapter.

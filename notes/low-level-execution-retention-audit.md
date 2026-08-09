# Low-level execution retention audit

Audit date: 2026-08-09

The remaining low-level asynchronous execution tools are active compatibility
code for NWChem. GRASP retains separate synchronous per-executable and
structured workflow calls. The guided `launch_run` contract has validated
providers for those programs plus OpenMolcas, DIRAC, Quantum ESPRESSO, and
QMCPACK.

## Decision

Retain the low-level NWChem calls and distinct GRASP interactive and structured
workflow calls behind explicit program or developer toolsets. Do not add them
to the default guided surface. The redundant QE, QMCPACK, Molcas, DIRAC, and
asynchronous GRASP execution surfaces were removed after their guided providers
passed the accepted parity and corpus gates.

A low-level launch call may leave after one of these conditions is met:

1. A guided provider covers the same program behavior and passes its accepted
   reference cases.
2. The owner explicitly decides that Chemtools will analyze that program but
   will no longer execute it.

This keeps existing capability without forcing every exploratory backend into
the eleven-tool guided contract.

## Evidence by program

NWChem has the approved guided launch path, but its low-level launch, watch,
registration, and recovery recommendations still form one compatibility
workflow. Remove those names together after the final release and migration
window, rather than peeling them away one at a time.

The retained low-level launch now prepares through the guided NWChem provider
and typed execution service. Its established response keys remain, while both
dry-run and live calls report the executor's exact command or Slurm script.
The application adapter no longer depends on the legacy renderer. The old
direct Python runner and `render_job_script` remain a separate compatibility
decision.

QE now exposes the shared guided launch path through a schema-2 named target or
version 1 migration profile. Its redundant `render_qe_launch` and
`launch_qe_run` MCP tools and application adapter were removed. The typed
`pw.x` plan and profile migration adapter remain program-owned.

QMCPACK now exposes ordinary and initialization-only execution through the
shared guided launch path. Its redundant `render_qmcpack_launch` and
`launch_qmcpack_run` MCP tools and application adapter were removed. The typed
plan, profile migration adapter, and explicit `initialization_only` option
remain program-owned.

The removal passed 114 focused architecture and launch checks, all 294 QE and
QMCPACK tests, and the complete 1,939-test external-corpus suite. Base and
DIRAC-extra isolated installs of wheel SHA-256
`33bbc2c00d3794c7b1f6cee4e33011ec31d84bdd3e733177e5175f4e1458fd0f`
confirmed the three retired modules are absent while both backends retain
guided launch planning. The same wheel is installed in the repository-local
`venv`.

Molcas now exposes the shared guided launch path through a schema-2 named
target or version 1 migration profile. Named targets conservatively serialize
CASPT2 and show the requested-to-effective rank change. Its redundant launch,
status, watch, and cancellation MCP tools plus execution, monitoring, and
scheduler compatibility modules were removed. The read-only
`prepare_molcas_launch` helper, typed plan, and profile adapter remain.

The Molcas local and Slurm parity matrix covered ordinary and CASPT2 input.
Focused launch, backend, boundary, inventory, and MCP checks passed 116 tests,
followed by all 1,930 tests with the external corpus. Base and DIRAC-extra
isolated installs of wheel SHA-256
`9bd74fb8384ebb9e4f8fa47dfb57bca5f1edec4b334ea19eb68afd9166205d86`
loaded the provider and portable Molcas target entries.

DIRAC now exposes the shared guided launch path through a schema-2 named target
or version 1 migration profile. Its approval snapshot includes the paired
`.inp` and `.mol` identities. Named targets use DIRAC's installation memory
defaults, while profiles retain explicit `--mw` and `--nw` values. Its
redundant launch, status, watch, and cancellation MCP tools plus execution,
monitoring, and scheduler compatibility modules were removed. The read-only
`prepare_dirac_launch` helper retains advanced checkpoint flags while the
typed plan and profile adapter remain program-owned.

The DIRAC local and Slurm parity matrix covered the paired input command,
artifacts, target rendering, and approval token. A dispatch-level check proved
that the `.mol` identity reaches the shared MCP handler. Focused launch,
backend, boundary, inventory, and MCP checks passed 119 tests, followed by all
1,936 tests with the external corpus. Base and DIRAC-extra isolated installs
of wheel SHA-256
`fdd03dd2a41a2ae25ff092eb441c8031ba8d63834b0acce46a9ce134255619ec`
loaded the provider and portable DIRAC target entries. The same wheel is
installed in the repository-local `venv`.

The joint Molcas and DIRAC low-level removal passed 123 focused architecture
and guided checks, all 27 retained program test modules, and the complete
1,923-test external-corpus suite. Base and DIRAC-extra isolated installs of
wheel SHA-256
`64ad9e361a9f7926b492abf886a97c23394cf8674edbb2d6f6ae916fbb5ad8b7`
confirmed that all six retired compatibility modules are absent while both
guided providers and advanced read-only preparers remain available. The same
wheel is installed in the repository-local `venv`.

GRASP now exposes whole workflow-script execution through the shared guided
launch path. Named targets own the container prefix and `bash` command; version
1 profiles remain the migration fallback. Approval binds the workflow script
and rendered command. Interactive stdin-driven executable and structured
workflow calls remain low-level because they do not have one input-file
contract.

The GRASP local and Slurm parity matrix covered the container command,
workflow-script identity, artifacts, target rendering, unsupported input-review
evidence, and approval token. Focused workflow, launch, backend, boundary,
inventory, and MCP checks passed 142 tests, followed by all 1,938 tests with
the external corpus. Base and DIRAC-extra isolated installs of wheel SHA-256
`8f547099d8b8d52ad784e3c594d3727cd4159462b50559fd8070a818e7643f52`
loaded the provider and portable GRASP target entries. The same wheel is
installed in the repository-local `venv`.

The GRASP guided plan replaces the asynchronous workflow-script launch,
status, watch, and cancellation surface. The retained interactive and
structured workflow calls still expose session-log behavior, captured stdin
and output, and per-step results outside the one-file guided contract. Their
application adapter no longer imports the legacy renderer, response projector,
archive policy, or external-status path.

The removal passed 103 focused GRASP, catalog, inventory, boundary, runner,
and launch checks, followed by all 1,915 tests with the external corpus. Base
and DIRAC-extra isolated installs of wheel SHA-256
`48c09af051cf95d4228adfae1dd6fb515d82cb3ac7e587e10777d2774366cdaa`
confirmed that the guided provider and retained interactive tools remain while
the monitoring and scheduler modules are absent. The same wheel is installed
in the repository-local `venv`.

## Consequences

- Keep the retained interactive tools out of the default guided toolset.
- Keep the remaining calls stable until their explicit removal gate is met.
- Move a program to guided execution only after its review cases establish the
  program-specific arguments, artifacts, and failure behavior.
- Do not build a second generic runner layer. Reuse the typed execution service
  and add a small program-owned launch provider when a backend is promoted.
The next engineering target is the old direct Python runner and
`render_job_script`. The low-level NWChem MCP workflow still needs one removal
decision for its public names, but no longer needs another execution adapter.

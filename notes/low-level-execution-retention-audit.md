# Low-level execution retention audit

Audit date: 2026-08-09

The low-level execution tools are not dead compatibility code. They remain the
only execution interface for several program backends. The guided `launch_run`
contract has validated providers for NWChem, Quantum ESPRESSO, and QMCPACK.

## Decision

Retain the low-level NWChem, QE, QMCPACK, Molcas, DIRAC, and GRASP launch calls
behind explicit program or developer toolsets through the compatibility
release. Do not add them to the default guided surface.

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

QE now exposes the shared guided launch path through a schema-2 named target or
version 1 migration profile. `render_qe_launch` and `launch_qe_run` remain
because their low-level response fields are a separate compatibility contract.

QMCPACK now exposes ordinary and initialization-only execution through the
shared guided launch path. `render_qmcpack_launch` and `launch_qmcpack_run`
remain because their version 1 preview fields are a compatibility contract.

The QE and QMCPACK guided plans do not replace their full version 1 preview
contracts. Those previews still expose `launcher_command`, `launcher_kind`,
`restart_prefix`, `shell`, resolved profile paths, and scheduler-template
fields. The QE adapter replaces the displayed command with its typed rendering;
the QMCPACK dry run still displays the legacy command and only uses the typed
rendering during a live launch. Changing either response now would be a public
compatibility change.

Do not add a second generic preview projector to remove these renderer calls.
Define each program's guided launch result first, then retire the extra version
1 fields with the low-level response contract.

Molcas and DIRAC application adapters use their scheduler wrappers to preserve
the current preview contract before a typed launch. GRASP does the same while
its typed launch path owns workflow and interactive execution. After the final
compatibility release, their monitoring fallbacks were narrowed to file-only
inspection and explicit external Slurm attachment.

## Consequences

- Keep these tools out of the default guided toolset.
- Keep their current behavior stable through the final compatibility release.
- Move a program to guided execution only after its review cases establish the
  program-specific arguments, artifacts, and failure behavior.
- Do not build a second generic runner layer. Reuse the typed execution service
  and add a small program-owned launch provider when a backend is promoted.
- Treat the program scheduler render and launch wrappers as active
  compatibility code. The status wrappers now delegate to the focused
  external-status owner.

The next engineering target is narrower: remove old renderer calls only when a
typed program plan already supplies the same preview and launch behavior.

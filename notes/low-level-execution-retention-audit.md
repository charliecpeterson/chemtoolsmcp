# Low-level execution retention audit

Audit date: 2026-08-07

The low-level execution tools are not dead compatibility code. They remain the
only execution interface for several program backends, while the guided
`launch_run` contract currently has a validated provider only for NWChem.

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

QE exposes `render_qe_launch` and `launch_qe_run`. The application adapter and
smoke-test documentation cover a real `pw.x` launch through the typed execution
service. Removing these tools now would remove QE execution rather than remove
a duplicate guided entry point.

QMCPACK exposes `render_qmcpack_launch` and `launch_qmcpack_run`. Its launch
contract also carries the QMCPACK-specific initialization-only option. That
option belongs in a future QMCPACK guided provider if the backend is promoted.

The QE and QMCPACK typed plans do not replace their full version 1 preview
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
the current preview contract before a typed launch. Their monitoring adapters
also use the wrappers for unowned status inspection. GRASP retains the same
unowned monitoring fallback, while its typed launch path owns workflow and
interactive execution.

## Consequences

- Keep these tools out of the default guided toolset.
- Keep their current behavior stable through the final compatibility release.
- Move a program to guided execution only after its review cases establish the
  program-specific arguments, artifacts, and failure behavior.
- Do not build a second generic runner layer. Reuse the typed execution service
  and add a small program-owned launch provider when a backend is promoted.
- Treat the program scheduler wrappers and unowned status functions as active
  compatibility code, not immediate deletion candidates.

The next engineering target is narrower: remove old renderer calls only when a
typed program plan already supplies the same preview and launch behavior.

# Unowned status scope audit

Audit date: 2026-08-07

The version 1 status adapter combines file inspection, arbitrary local process
probing, and three scheduler families. Those capabilities do not need the same
retention decision.

## Active evidence

NWChem, Molcas, DIRAC, and GRASP status and watch tools first try the typed
execution service. When a process or scheduler identifier is not owned by that
service, they fall back to `execution.legacy_status`. Tests explicitly pin this
fallback for all four programs.

Output-only inspection is also used without a process or scheduler identifier.
It reports file presence and, for NWChem, injects the program progress reader.
This remains useful for calculations launched outside Chemtools.

Slurm job attachment fits the retained personal workflow: a caller can supply a
profile and job ID for an existing chemistry run, then combine scheduler state
with output progress. The current implementation uses the configured status
command and does not claim ownership of the job.

No maintained example, smoke test, or target profile establishes a current PBS
or LSF workflow. The bundled PBS profile is a compatibility example. There is
also no maintained workflow that requires attaching to an arbitrary local PID;
owned local launches use retained process handles instead.

Unowned cancellation is not part of the typed application contract. Program
cancellation functions resolve a recorded launch owned by the current service
instance before sending a signal or scheduler command.

## Approved retained contract

After the final compatibility release:

- Keep read-only output, input, and error-file inspection.
- Keep read-only attachment to an existing Slurm job through an explicit
  profile and job ID.
- Keep these calls behind program or developer toolsets, outside the guided
  `monitor_run` contract.
- Retire arbitrary local PID probing.
- Retire PBS and LSF status parsing unless a real personal or coworker workflow
  is identified before the compatibility release.
- Continue requiring execution-service ownership for cancellation.

The owner approved this scope on 2026-08-07. Compatibility behavior remains in
place through the final compatibility release and migration window. Removal
of PID, PBS, and LSF inspection is release-gated, not part of this decision
record.

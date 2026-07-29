# Run-layer hardening notes (NWChem GA hang + MCP defects)

Written 2026-07-25, from a session running two water opt jobs (MP2/cc-pVDZ,
CCSD(T)/6-31G) through the local `local_apptainer` profile. The jobs surfaced
one NWChem/Global Arrays issue and four defects in how this MCP runs and
watches apps. This is a design note, not a task list — pick up whichever
threads are worth it.

## What happened

The MP2 optimization hung. NWChem reached the CPHF linear solve for the MP2
analytic gradient and then froze: 15 of 16 MPI ranks pinned at ~99% CPU, rank 0
idle, output file frozen for >10 minutes on a job that finishes in seconds.
Killed it, reran both jobs at 4 ranks (`resource_overrides={"mpi_ranks": 4}`),
and they converged immediately.

Final results (both physically sane):

- MP2/cc-pVDZ: E = -76.234904885 Ha, 3 steps, r(O-H)=0.9641 A, angle=101.75 deg
- CCSD(T)/6-31G: E = -76.121822 Ha, 4 steps, r(O-H)=0.9767 A, angle=109.04 deg
  (wide angle is the expected 6-31G no-polarization artifact, not a run problem)

## Root cause of the GA hang (best hypothesis, UNVERIFIED)

Not classic CPU oversubscription — 16 ranks on 20 physical cores fits. Notes:

- The ~99% CPU is a red herring. NWChem's Global Arrays runs over ARMCI, and
  the container's Intel MPI (oneAPI `mpiexec.hydra`) busy-polls by default, so
  healthy ranks also spin at 100%. "Hung" = 100% CPU **plus** zero output growth.
- Best guess: an ARMCI/GA progress deadlock that triggers when the distributed
  problem is tiny relative to rank count. The CPHF solve here is ~100 response
  variables over 25 basis functions; when a GA dimension is comparable to or
  smaller than `nproc`, some ranks own zero elements and a collective can wait
  on progress that never arrives. Build- and transport-dependent, which is why
  4 ranks dodges it and 16 hits it.
- This is a hypothesis, not confirmed. Confirming it means a rank sweep in the
  container plus toggling `ARMCI_*` / progress-thread env vars on the water job,
  not asserting from memory.

The MCP fixes below don't fix the GA bug — they make the MCP *notice* when any
job hangs. Diagnosing GA itself is a separate track.

## Structural verdict: do NOT rearchitect

The run layer's bones are sound. `core/runner.py` owns launch/watch/registry;
per-program wrappers add only domain heuristics; DIRAC already reuses the core
cleanly as a thin wrapper (`programs/dirac/scheduler.py`). Launch glue is not
duplicated across nwchem/molcas/dirac/grasp. The shared-core design is right.
What exists is four specific, cheap-to-fix defects on a good base. Fixing those
beats a restructure.

(Minor smell, not urgent: the generic core function is named `run_nwchem()` even
though DIRAC calls it — misleading naming / slightly leaky abstraction.)

## The four defects, in priority order

### 1. `auto_watch=true` is a silent no-op for the local (direct) runner
This is the real reason launch "returned immediately with status: started."

- A `direct` launcher (`local_apptainer` is one) spawns via
  `os.spawnve(P_NOWAIT)` and returns — `core/runner.py:236-248`.
- The `auto_watch` block only fires for `launcher_kind == "scheduler"` —
  `mcp/tools/nwchem_jobs.py:72-94`.
- So the tool docstring ("auto_watch=true (default) blocks until completion") is
  false for exactly the local, interactive path used most.

Fix (cheapest honest option): keep direct launches non-blocking but make the
docstring truthful ("direct launches return immediately; call `watch_nwchem_run`
separately") and make that separate watch trustworthy (see #2). Truly blocking a
long local job ties up the MCP call, so I'd avoid making direct actually block.

### 2. Stall detection already exists but is inert (arm it)
`watch_nwchem_run` (`core/runner.py:700-738`) already tracks a progress signature
including output `size_bytes` and breaks with `stop_reason="stalled_no_progress"`
— but only when `stall_timeout_seconds` is set, and it defaults to `None` and
isn't exposed on the nwchem watch tool. So it never armed; we hit the plain 600s
timeout while it reported "running" the whole time.

Fix: give `stall_timeout_seconds` a sane default (~180s) and expose it. Optional:
augment the signature with a CPU check to distinguish frozen-but-alive from
I/O-bound. This single change would have caught today's hang. Near-trivial, high
value.

### 3. Intervention verdict is phase-blind
`_assess_nwchem_progress_intervention` (`programs/nwchem/runner.py:582-741`) keys
entirely off SCF/optimization *trends*. Our hang was in the CPHF gradient phase —
SCF already converged, no trend to read — so it fell through to the
`continue_monitoring` default (lines 731-740).

Fix: feed the "process alive + output frozen for N minutes" signal from #2 into
the verdict as a low-confidence kill recommendation. Phase-agnostic — covers freq
displacements, post-HF, gradient, everything, not just this GA case.

### 4. Registry `program`-column error recurs on every launch
`core/run_registry.py:106-119` has a self-healing migration (catch
"no such column", ALTER, retry), but `registry_error` showed up on all three
launches this session. If the migration were sticking, launches 2 and 3 would be
clean. So the ALTER isn't persisting across calls — likely a commit/connection
-lifecycle bug, or more than one DB file. Small fix, but it's noise on every run.

## Rank sizing / "make it smarter"
Keep it advisory, not a silent override. Auto-scaling ranks from basis-function
count is the over-engineered version — skip it. Cheapest real improvement: the
`local_apptainer` profile default of 16 ranks is just wrong for the small
interactive jobs run here; drop the default to ~8, or add a preflight warning
when ranks far exceed problem size. Domain/HPC call.

## Suggested order of work
#2 (arm stall detection) and #4 (registry) are near-trivial and high value.
#1 (honest auto_watch contract) and #3 (phase-blind intervention) are slightly
more involved. GA root-cause diagnosis is a separate track — do it in the
container with a rank sweep + ARMCI env vars if/when the NWChem behavior itself
matters, rather than just avoiding the trigger.

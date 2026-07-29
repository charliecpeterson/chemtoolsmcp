# NWChem failure diagnosis: lessons from the SRD 46 sweep

Derived from a debugging session on 2026-07-25/26 against a 225-job PBE0/cc-pVTZ
COSMO geometry sweep on Hoffman2 (UCLA, Slurm). The session began by
misdiagnosing a convergence bug as a filesystem outage and spent hours on the
wrong theory. Most of what follows is about not repeating that.

Written as input for the chemtools MCP, not as a spec. Nothing here is
implemented yet. Sections are ordered by how much damage the gap causes.

Context: sweep lives at `~/projects/reackt/geometry-sweeps/srd46-pbe0-tz-001`,
with a fuller incident write-up in that directory's `INCIDENT-2026-07-23.md`.

---

## 1. The NWChem error string is not diagnostic (highest priority)

**This is the big one.** A failed `task dft optimize` writes to stderr:

```
0:dft optimize failed:Received an Error in Communication
Abort(-1) on node 0 (rank 0 in comm 496): application called MPI_Abort(comm=0x84000003, -1) - process 0
```

That string reads like an MPI or interconnect fault. It is not. It is NWChem's
generic `errquit` message for **any** failed `task dft optimize`. Verified
empirically: three unrelated jobs (39005, 39008, 40022) that merely exhausted
`driver maxiter 250` emit the byte-identical two lines. Same for jobs that died
of entirely different causes.

`grep -r "Error in Communication"` over this repo returns nothing today, so an
agent hitting it will reason from the plain English and conclude "MPI problem."
That is exactly what happened in this session and it cost hours.

**What the MCP should do**

- Add this to a known-non-diagnostic-strings table. When it appears, the tool
  must say so explicitly rather than surfacing it as the cause:
  `"generic errquit; the real cause is in the .out driver/SCF trace, not stderr"`.
- Never classify a failure from stderr alone. Classification belongs to the
  `.out` trace (§3).
- Same treatment for any `MPI_Abort` line that follows an `errquit`: it is a
  consequence, not evidence.

---

## 2. Buffered stdout eats the real error

NWChem's stdout is block-buffered when redirected to a file, and `MPI_Abort`
terminates without flushing. The last few KB, **including the `errquit` detail
block that says what actually went wrong**, never reach disk.

Symptom: the `.out` file stops mid-SCF-iteration with no error block at all.
It looks truncated by an external kill. It is not.

**What the MCP should do**

- Launch with line-buffered stdout so the error survives:
  `stdbuf -oL -eL mpirun ... nwchem x.nw > x.out`. Worth defaulting to on for
  every run; the throughput cost on a multi-minute job is irrelevant next to
  losing the diagnosis.
- When parsing an output that ends mid-record, emit a distinct status like
  `truncated_by_abort` rather than something that implies the job was killed.
  Add the hint that the error block was lost to buffering and that a rerun with
  `stdbuf -oL` will show it.
- Do not report the last timestamp in a truncated `.out` as the time of death.
  It is the last flush, which lags.

---

## 3. Parse the driver convergence table; it is where the answer lives

NWChem prints a per-step table that is the single most useful artifact for
diagnosing an optimization:

```
@ Step       Energy      Delta E   Gmax     Grms     Xrms     Xmax   Walltime
@    0    -205.11732557  0.0D+00  0.00051  0.00025  0.00000  0.00000      4.6
@  250    -205.11732557 -2.3D-11  0.00051  0.00025  0.00000  0.00000    731.2
```

Step 250 is identical to step 0 in every column. `Xmax` and `Xrms` are exactly
zero for all 250 iterations: **the geometry never moved.** The job burned 250
steps at ~2.9 s each accomplishing nothing, then exited via the §1 string.

A `stuck` state is qualitatively different from `slowly converging` and calls
for a different fix, but both look like `maxiter` from the outside.

**What the MCP should do**

Add a `diagnose_optimization` tool (or extend `nwchem_parse`) returning:

| field | meaning |
|---|---|
| `steps_taken` | rows in the `@` table |
| `displacement_zero_streak` | trailing steps with `Xmax == 0` |
| `energy_flat_streak` | trailing steps with abs(Delta E) below ~1e-9 |
| `criteria` | per-criterion `{threshold, achieved, passes}` |
| `verdict` | `converged` / `stuck_zero_step` / `converging_slowly` / `oscillating` / `diverging` |

`stuck_zero_step` when the displacement streak is long and energy is flat.
That verdict plus the per-criterion breakdown is the whole diagnosis.

**Read thresholds from the output, not from a hardcoded table.** NWChem prints
them for the run in question:

```
 maximum gradient threshold         (gmax) =   0.000450
 rms gradient threshold             (grms) =   0.000300
 maximum cartesian step threshold   (xmax) =   0.001800
 rms cartesian step threshold       (xrms) =   0.001200
```

Defaults change between versions and are overridden by `driver loose/tight` and
by explicit keywords. Parsing the printed block is both easier and correct.

**Report criteria individually.** The case that mattered here passed three of
four. "Not converged" is useless; "gmax fails by 13%, everything else passes,
energy converged to 11 digits" tells you what to do.

---

## 4. COSMO has a gradient noise floor that can sit above the default threshold

The root cause in this sweep. Small rigid anions under COSMO carry a residual
gradient of roughly 5-6e-4 from the discretized solvation surface. NWChem's
default `gmax` is 4.5e-4. The criterion is therefore **unreachable** and the
optimizer spins until `maxiter` no matter how good the geometry is.

Observed plateaus (PBE0/cc-pVTZ, COSMO water, `dielec 78.4`):

| slug | species | gmax | grms |
|---|---|---|---|
| lig-10091-hl | | 0.00046 | 0.00026 |
| lig-10100-l | | 0.00050 | 0.00027 |
| lig-10108-l | HNO2 / nitrite | 0.00051 | 0.00025 |
| lig-10107-l | NO⁻ | 0.00062 | 0.00036 |

This repo's project notes already record the same pathology one tier tighter:
`driver tight` (gmax 1.5e-5) was abandoned because jobs sat at gmax ~5e-5 for
hundreds of steps. The lesson generalizes and was not carried forward: *for
COSMO, pick the convergence tier from the observed noise floor, not from taste.*

**Critical distinction the MCP must make.** A fifth slug, `lig-10096-l`, also
plateaus with `Xmax == 0` but at **gmax 0.00488 / grms 0.00259**, an order of
magnitude higher. That is a genuine residual force on a structure that is not at
a minimum. Loosening thresholds past it would certify a bad geometry as
converged. Any auto-recommendation must separate these:

```
noise_floor_plateau   : Xmax==0, gmax within ~2x of threshold, grms near/below its threshold
unconverged_plateau   : Xmax==0, gmax an order of magnitude above threshold
```

Only the first is safe to loosen. The second needs a different starting
geometry, a better solvation surface, or a look at whether the species is
sensible at all. **Never auto-apply a threshold change to the second class.**

When recommending a loosened tier, preserve NWChem's default `grms/gmax` ratio
of 0.667 rather than touching gmax alone. In this sweep `lig-10107-l` failed
`grms` too (0.00036 vs 0.00030), so a gmax-only change would not have released
it. The values used were `gmax 0.0010, grms 0.00067`: about 1.6x headroom over
the worst observed floor and still 4.5x tighter than `driver loose`.

**A restart does not fix a noise-limited plateau.** Both affected jobs were
reseeded from their last geometry and hit `maxiter` again within minutes. Before
any tool recommends "restart from the last step", it must check whether the
geometry was actually moving. Reseeding a zero-displacement plateau just burns
the allocation again.

---

## 5. Scheduler facts that get misread

**Exit codes are `exitcode:signal`, not a signal.** `sacct` reporting `15:0`
means exit status 15 with **no signal**. It does not mean SIGTERM. Signal 0
across a whole failed batch positively rules out the OOM killer, which is a
strong and cheap negative result. Getting this backwards sent this session
chasing a memory theory. `255:0` is the usual `MPI_Abort` code.

**Slurm `COMPLETED` does not mean the science converged**, and `FAILED` does not
mean the science is wrong. Always join scheduler state with output parsing.
Three jobs here were `COMPLETED 0:0` with truncated, unconverged output.

**Memory accounting may be off entirely.** This cluster has
`JobAcctGatherType = (null)`, so `MaxRSS` and `MaxVMSize` are empty for every
job and memory pressure cannot be confirmed after the fact. A tool that reports
"memory usage: unknown" as if that were reassuring is misleading. Detect the
null plugin via `scontrol show config` and say memory is **unverifiable on this
cluster**, then fall back to comparing the NWChem `memory` directive against
`--mem`.

That arithmetic is worth surfacing unprompted:
`memory stack 800 mb heap 100 mb global 300 mb` is 1.2 GB/rank, times 48 ranks
is 57.6 GB against a `--mem=64G` cgroup. 90% consumed before MPI buffers and
page cache. Nothing flagged this.

**Node-local scratch may not exist.** `TmpDisk=0` with `TmpFS=/tmp` on tmpfs
means there is no local disk and `/tmp` is RAM. All per-rank scratch necessarily
goes to the network filesystem, and writing to `/tmp` would eat the memory
cgroup. Query this before advising on `scratch_dir` / `permanent_dir`.

**Check the mount before blaming the filesystem.** `/u/scratch` here is NFSv3
mounted `hard`, which blocks indefinitely on a stalled server rather than
returning errors. Under a hard mount, a storage outage produces **hung jobs at
walltime, not fast failures**. A job that aborts in 7 minutes is therefore not a
storage outage, and that single fact would have killed the wrong theory early.

---

## 6. Correlated failures are usually correlated submissions

41 of 44 jobs failed within a 103-second window and it looked unmistakably like
an infrastructure event. It was not. They had been submitted together and
started within 4 seconds of each other, so similar molecules doing similar work
reach the same point at the same time. Synchronized failure follows from
synchronized launch with no external cause at all.

**Before concluding "cluster event", check the submission pattern.** If a tool
reports correlated failures it should also report start-time spread. Tight start
clustering plus similar runtimes is the null hypothesis, not evidence.

The falsification that settled it: the failure **reproduced with 2 concurrent
jobs on an idle cluster**. Cheap, decisive, and it should have been the first
test rather than the last. Worth encoding as a recommended next action whenever
a load-related cause is suspected: rerun one case alone.

---

## 7. Batch and sync hazards worth encoding

If the MCP generates job arrays or manages remote sync, these bit hard.

**Array task-to-work-item mapping is resolved at task start.** A pattern like
`slug=$(sed -n "${SLURM_ARRAY_TASK_ID}p" run_queue.txt)` re-reads the queue file
whenever a task launches. Regenerating that file while an array is live means a
requeued task computes the **wrong molecule and files it under the wrong name**.
Silent data corruption with no error anywhere. Any generated array must version
its queue file per submission (`run_queue_r1.txt`, `_r2`, ...) and never reuse
the name while an earlier array has tasks outstanding.

**rsync exit 23 and 24 are normal against a live run tree.** Files are being
rewritten and deleted mid-transfer. A sync script under `set -e` treating these
as fatal will abort before its downstream parse step. This silently froze the
local view of the sweep for two days: every sync appeared to run and quietly
died before re-summarizing. Treat 23/24 as success; only other codes are errors.

**Never blanket-sync into a directory tree with running jobs.** Pushing a whole
`nw/` tree upward overwrites live output with stale local copies. Scope pushes
to the specific job directories being submitted (`--files-from`).

**Restaging tools that select by status will eat live jobs.** A status like
`running_or_incomplete` covers both dead and running work. A restage that moves
directories based on status alone, with no liveness check, will pull the working
directory out from under running jobs. Always re-query the scheduler immediately
before any destructive step, and never trust a snapshot taken minutes earlier.

---

## 8. Smaller things

- `direct` in the `dft` block means **no `aoints` files are ever written**.
  Do not advise on integral-file I/O without checking for it. A stale comment
  claiming "tens of GB of aoints per job" sent this session looking for files
  that did not exist; the actual footprint was 48 per-rank `gridpts.NN` files.
- Driver step geometries are written as `<slug>-opt.xyz-NNN.xyz`, one per step,
  written and closed individually. Unlike the `.out`, **these are reliable under
  buffering** and counting them gives a trustworthy step count for a job whose
  output was truncated.
- A 2-atom species needing 250 optimization steps is prima facie absurd (a
  diatomic has one degree of freedom) and should trigger a loud warning. That
  observation alone would have pointed at the optimizer rather than the cluster.

---

## 9. Converged is not correct: validate the molecule, not just the run

The most expensive failure found so far produced no error at all. Seven of 110
ligand families in the SRD 46 sweep were optimized from a guess geometry for a
**different molecule**, and ten of those jobs finished cleanly with
`Optimization converged`, frequencies, and properties. Slurm said `COMPLETED`,
the parser said `optimized`, and the results were headed for a training set.

The worst case: `lig-5958`, labelled "Diethylenetriamine-N(1)-acetic acid"
(C6H15N3O2), whose geometry file was DTPA (C14H23N3O10). Another,
`lig-5838`, was missing its nitrogen entirely. Nothing downstream noticed.

An MCP that writes or reads quantum chemistry inputs should treat composition
as a first-class precondition:

- **`validate_input(structure, expected)`** before writing an input deck.
  Compare the element histogram of the geometry against an independent
  declaration of what the molecule is meant to be. Refuse, do not warn, on a
  heavy-atom mismatch.
- **Derive expected composition from a structure, never from a `formula`
  string.** In this dataset the upstream `formula` field was masked to
  `********` for 1342 of 5750 entries, and the REST API returned an element set
  rather than a stoichiometry (`CO` for CO2, `AsO` for As4O6). Both would have
  passed a naive string comparison. The usable ground truth was a compressed
  ChemDraw MOL block in a different table.
- **Check charge against protonation absolutely, not relatively.** Every one of
  92 multi-form families had a perfect protonation ladder (+1 H per +1 charge
  between rungs) while entire families sat one proton off in absolute terms. A
  ladder check cannot see a uniform offset. The absolute test is: charge must
  equal minus the number of protons removed relative to the neutral parent.
- **Know the blind spot.** When a geometry differs from the reference by
  hydrogens only, composition cannot distinguish deprotonation from a different
  molecule with the same heavy-atom skeleton (here: an aromatic ring substituted
  for a 2,3-dihydro one). Flag those for a human instead of guessing.

The general rule for a status tool: `exit_code == 0` and `Optimization
converged` answer "did the program finish," which is a different question from
"did it compute the thing that was asked for." Report them as separate fields.

## 10. Do not build a Slurm array task id from `%A_%K`

For a **running** array task, `squeue`'s `%A` is the individual allocation
number Slurm assigned that task, not the array master. `%F` is the array job id.

```text
%i        %A      %K   %F
42857_2   42859   2    42857
```

`scancel 42859_2` exits 0 and cancels nothing. A tool that composes ids this way
will report success while the job keeps running. Use `%i`, or `%F` with `%K`.

For mapping a task back to what it is computing, the queue file is only valid if
it has not been rewritten since submission. The durable mapping is the process
working directory on the node (`readlink /proc/<pid>/cwd`), which is the actual
per-job directory.

---

## Suggested implementation order

1. Non-diagnostic string table plus the "classify from `.out`, never stderr"
   rule (§1). Cheapest, prevents the worst failure mode.
2. `stdbuf -oL` on launch, and `truncated_by_abort` as a distinct parse
   status (§2).
3. `diagnose_optimization` over the `@` table with per-criterion pass/fail and
   thresholds read from the output (§3).
4. COSMO noise-floor classifier with the safe/unsafe plateau split (§4).
5. Scheduler-fact corrections: exit code semantics, null accounting plugin
   detection, memory arithmetic (§5).
6. Array queue-file versioning and rsync 23/24 tolerance if the MCP grows
   batch management (§7), plus the `%A_%K` id trap (§10).
7. `validate_input` composition gate before any input deck is written (§9).
   Arguably belongs first: it is the only item here that catches a failure
   which produces no error signal at all.

## Open question

The convergence plateau explains the `maxiter` failures but not the bulk of the
2026-07-23 batch, which died at driver step 1-2 with healthy traces showing real
displacement. Node syslog for the one case inspectable at that level was
completely clean: no OOM killer, no NFS errors, no kernel messages, and the MPI
bootstrap step exited 0. Cause still unknown, not recurred. If it happens again
the thing to capture is the `errquit` block that buffering ate, which is what
§2 is for.

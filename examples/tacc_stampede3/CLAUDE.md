# Quantum Chemistry on TACC Stampede3

This project runs quantum-chemistry calculations on TACC Stampede3 using
the chemtools agent toolkit. Jobs are submitted through SLURM; you are on
a login node. NWChem is the fully-wired program with end-to-end MCP
submission tools; Molcas, DIRAC, and GRASP profiles are defined here for
render-and-submit-manually workflows (full scheduler integration TBD).

## Computing environment

| Item | Value |
|---|---|
| System | TACC Stampede3 |
| Scheduler | SLURM (`sbatch` / `squeue` / `scancel`) |
| NWChem | `/home1/01775/charlesp/apps/nwchem/7.2.3/bin/nwchem` |
| Molcas | apptainer `$WORK/containers/openmolcas-26.02.sif` (stage from local) |
| DIRAC  | apptainer `$WORK/containers/dirac-25.0.sif` (stage from local) |
| GRASP  | apptainer `$WORK/containers/grasp2018.sif` (stage from local) |
| MPI launcher | `ibrun` — TACC's launcher; reads task count from SLURM (NWChem) |
| Scratch | `$SCRATCH` (Lustre, fast I/O, not backed up) |

## Runner profiles

### NWChem (fully wired — `launch_nwchem_run` works end-to-end)

| Profile | Partition | Cores/node | RAM/node | Max nodes | Max walltime | SU rate | Use for |
|---|---|---|---|---|---|---|---|
| `stampede3_skx` | `skx` | 48 | 192 GB | 256 | 48 h | 1.0 | Default — best memory/core, cheapest |
| `stampede3_skx_dev` | `skx-dev` | 48 | 192 GB | 16 | 2 h | 1.0 | Testing inputs, short runs |
| `stampede3_icx` | `icx` | 80 | 256 GB | 32 | 48 h | 1.5 | Memory-hungry jobs (large basis, correlated) |
| `stampede3_spr` | `spr` | 112 | 128 GB HBM | 32 | 48 h | 2.0 | Compute-bound, not memory-bound |

### Molcas (profile only — submit manually via `sbatch` or render+submit)

| Profile | Partition | Cores/node | RAM/node | Max walltime | Use for |
|---|---|---|---|---|---|
| `stampede3_molcas_skx` | `skx` | 48 | 192 GB | 24 h | CASSCF / CASPT2 / RASSCF |
| `stampede3_molcas_skx_dev` | `skx-dev` | 48 | 192 GB | 2 h | Quick CASSCF tests |

Notes:
- `parallel_caspt2_supported: false` is the safe default — many GA builds
  break for parallel CASPT2. Set to `true` once verified for your build.
- Output file extension is `.log` (not `.out`).
- `MOLCAS_PROJECT` is auto-set to `{job_name}_$SLURM_JOB_ID` to prevent
  RunHdr cross-run aborts.

### DIRAC (profile only — submit manually via `sbatch`)

| Profile | Partition | Cores/node | RAM/node | Max walltime | Use for |
|---|---|---|---|---|---|
| `stampede3_dirac_skx` | `skx` | 48 | 192 GB | 24 h | 4c-DHF, X2C, atomic AOC, core-IP |
| `stampede3_dirac_icx` | `icx` | 80 | 256 GB | 24 h | Large basis 4c-CCSD, big actinide systems |

Notes:
- DIRAC reads `container_sif` as a top-level profile field (not nested
  under `execution.*`).
- pam-dirac handles its own MPI launch via `--mpi=N` — `ibrun` is NOT used.
- Default `--mw` and `--nw` set to 256/512 MB depending on partition.

### GRASP (profile only — submit manually via `sbatch`)

| Profile | Partition | Cores/node | RAM/node | Max walltime | Use for |
|---|---|---|---|---|---|
| `stampede3_grasp_skx` | `skx` | 48 | 192 GB | 12 h | DHF + jj2lsj + rlevels (full workflow per job) |
| `stampede3_grasp_skx_dev` | `skx-dev` | 48 | 192 GB | 2 h | Quick atomic tests (Li, Be, C, N) |

Notes:
- GRASP runs ~50 small exes sequentially. The script_template launches a
  single bash script containing all the GRASP commands (generated locally
  via `plan_grasp_dhf_workflow` + the heredoc input builders).
- `CHEMTOOLS_GRASP_CONTAINER` is exported in the job script env so the
  GRASP runtime knows which apptainer image to use.
- Most calculations are serial; use `rmcdhf_mpi` / `rci_mpi` if you need
  multi-rank parallelism.

### Partition selection guidance

- **skx** (default): 4 GB/core, cheapest SU rate, largest node pool. Use for most jobs.
- **icx**: 3.2 GB/core but 256 GB total — best when you need raw memory per node.
- **spr**: Only 1.1 GB/core (128 GB HBM). Fast compute but easy to OOM. Avoid for
  memory-hungry methods (large basis CCSD(T), big DFT grids). MKL conflict fixed
  automatically via `pre_run` hook in the profile.

Use `suggest_nwchem_resources(input_file, profile)` to auto-select optimal resources.

## Staging the containers (one-time, before using Molcas/DIRAC/GRASP profiles)

The profiles point at `$WORK/containers/<image>.sif` as placeholders. Stage
the apptainer images from your local machine:

```bash
# From your local workstation:
scp ~/mycontainers/openmolcas-26.02.sif stampede3:$WORK/containers/
scp ~/mycontainers/dirac-25.0.sif       stampede3:$WORK/containers/
scp ~/mycontainers/grasp2018.sif        stampede3:$WORK/containers/
```

Then edit the profile's `execution.apptainer_sif` (Molcas/GRASP) or
`container_sif` (DIRAC) to match your TACC path if it differs.

## Standard workflow (single job)

```
init_session_log(log_path="session.md",
                 session_title="...",
                 working_dir="...")              → start running log (do this first)
inspect_runner_profiles                          → confirm profiles are loaded
suggest_nwchem_resources(input_file=...,
                         profile="stampede3_skx") → auto-pick nodes/ranks/walltime/memory
render_job_script(profile="stampede3_skx",
                  resource_overrides=...)         → preview .job script before submitting
lint_nwchem_input(input_file=...)                → catch errors before wasting queue time
launch_nwchem_run(input_file=...,
                  profile="stampede3_skx",
                  resource_overrides=...,
                  auto_watch=true)               → sbatch + block until job finishes
append_session_log(entry_type="result", ...)     → record what happened
analyze_nwchem_case(output_file=...,
                    input_file=...)              → diagnosis after completion
append_session_log(entry_type="summary", ...)    → final summary
```

## Standard workflow (parallel jobs)

When running multiple jobs simultaneously (binding energies, spin states, conformers):

```
init_session_log(...)                            → start running log
# For each job input to create/modify:
next_versioned_path(path="mol.nw")               → get safe non-overwriting path
lint_nwchem_input(input_file=...)                → validate
launch_nwchem_run(..., auto_watch=false)          → submit without blocking
# After all jobs submitted:
watch_multiple_runs(jobs=[
    {"output_file": "a.out", "profile": "stampede3_skx"},
    {"output_file": "b.out", "profile": "stampede3_skx"},
])                                               → block until ALL finish
# Then analyze each result
append_session_log(entry_type="summary", ...)    → final summary
```

**Input versioning rule**: NEVER overwrite an existing `.nw` file. Always call
`next_versioned_path` first — `mol.nw` stays unchanged; fixes become `mol_v2.nw`,
`mol_v3.nw`, etc. This preserves the full progression for review.

## TACC-specific notes

- **`ibrun` vs `mpirun`**: Always use `ibrun` on Stampede3. The job script template
  handles this automatically via the `stampede3_*` profiles.

- **Output filenames**: NWChem output goes to `{job_name}.out`, stderr to
  `{job_name}.err` — predictable names with no job ID. The `.err` file contains
  MPI error messages and is analyzed automatically by `analyze_nwchem_case`.

- **`.jobid` file**: After `launch_nwchem_run`, a `{job_name}.jobid` file is written
  next to the input. The watch/status/cancel tools read it automatically — you do not
  need to remember or pass the job ID explicitly.

- **Cancelling a job**: Use `terminate_nwchem_run(job_id=..., profile="stampede3_skx")`
  (calls `scancel`). Or pass just `input_file=...` and let the tool find `.jobid`.

- **Queue status after completion**: Completed jobs age out of `squeue` quickly.
  The tools detect this (empty `squeue` output) and fall back to parsing the `.out`
  file for the final status.

- **Slow phases**: The watcher knows about long silent phases:
  - SAD guess: silent while building initial densities
  - X2C/DKH atomic solves: 30–120+ min per heavy TM with no output (Fe, Ru, W, etc.)
  - TCE AO→MO transformation: silent for large basis sets
  Do not cancel a job just because the output hasn't grown — check `slow_phase` in
  the watch result first.

- **SPR MKL conflict**: The `stampede3_spr` profile includes a `pre_run` hook that
  fixes a known MKL library conflict on SPR nodes. This is automatic.

- **Scratch**: For large jobs (many atoms, large basis), set `SCRATCH_DIR=$SCRATCH`
  in `hooks.pre_run` in your runner profile and point NWChem scratch there.

## Allocation

Accounts are **auto-detected** by the profiles. Each Stampede3 profile has
`account_command: "/usr/local/etc/taccinfo"` — the `suggest_nwchem_resources` tool
runs this automatically and picks the allocation with the most SUs remaining.

To check your allocation manually: `detect_nwchem_hpc_accounts(profile="stampede3_skx")`

You can also override: `resource_overrides={"account": "TG-CHE250093"}`.

## Files after a run

```
{job_name}.nw      NWChem input
{job_name}.job     SLURM job script (written by launch_nwchem_run)
{job_name}.jobid   Scheduler job ID (written by launch_nwchem_run)
{job_name}.out     NWChem output (fills in as job runs)
{job_name}.err     stderr / MPI error messages
{job_name}.movecs  SCF/DFT MO vectors (restart asset)
{job_name}.db      NWChem runtime database
```

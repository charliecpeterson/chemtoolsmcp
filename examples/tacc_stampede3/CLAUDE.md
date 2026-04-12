# NWChem Calculations on TACC Stampede3

This project runs NWChem quantum chemistry calculations on TACC Stampede3
using the chemtools-nwchem agent toolkit. Jobs are submitted through SLURM;
you are on a login node.

## Computing environment

| Item | Value |
|---|---|
| System | TACC Stampede3 |
| Scheduler | SLURM (`sbatch` / `squeue` / `scancel`) |
| NWChem | `/home1/01775/charlesp/apps/nwchem/7.2.3/bin/nwchem` |
| MPI launcher | `ibrun` — TACC's launcher; does **not** take `-n`; reads task count from SLURM |
| Scratch | `$SCRATCH` (Lustre, fast I/O, not backed up) |

## Runner profiles

| Profile | Partition | Cores | Max walltime | Use for |
|---|---|---|---|---|
| `stampede3_skx` | `skx` | 48 | 48 h | Most DFT/SCF/TCE jobs |
| `stampede3_skx_dev` | `skx-dev` | 48 | 2 h | Testing inputs, short runs |
| `stampede3_spr` | `spr` | 112 | 48 h | Large basis sets, more memory |

Default walltime is 24 h. Override with `resource_overrides={"walltime": "48:00:00"}`.

## Standard workflow

```
inspect_runner_profiles                          → confirm profiles are loaded
render_job_script(profile="stampede3_skx")       → preview .job script before submitting
lint_nwchem_input(input_file=...)                → catch errors before wasting queue time
launch_nwchem_run(input_file=...,
                  profile="stampede3_skx")       → sbatch; writes {name}.jobid automatically
watch_nwchem_run(output_file=...,
                 input_file=...)                 → polls squeue + tails output; job ID
                                                   auto-detected from {name}.jobid
analyze_nwchem_case(output_file=...,
                    input_file=...)              → diagnosis after completion
```

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

- **Scratch**: For large jobs (many atoms, large basis), set `SCRATCH_DIR=$SCRATCH`
  in `hooks.pre_run` in your runner profile and point NWChem scratch there.

## Allocation

Set `account` in your runner profile (or pass `resource_overrides={"account": "TG-XXX"}`)
if your project requires a specific XSEDE/ACCESS allocation.

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

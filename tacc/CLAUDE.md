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

| Profile | Partition | Cores/node | RAM/node | Max nodes/job | Max walltime | SU rate | Use for |
|---|---|---|---|---|---|---|---|
| `stampede3_skx` | `skx` | 48 | 192 GB | 256 | 48 h | 1.0 | Most DFT/SCF/TCE — best memory/core ratio |
| `stampede3_skx_dev` | `skx-dev` | 48 | 192 GB | 16 | 2 h | 1.0 | Testing inputs, short runs |
| `stampede3_icx` | `icx` | 80 | 256 GB | 32 | 48 h | 1.5 | Memory-hungry jobs (large basis, correlated) |
| `stampede3_spr` | `spr` | 112 | 128 GB | 32 | 48 h | 2.0 | Compute-bound, not memory-bound |

### Partition selection guidance

- **skx** (default): 4 GB/core, cheapest SU rate, largest node pool. Use for most jobs.
- **icx**: 3.2 GB/core but 256 GB total — best when you need raw memory per node.
- **spr**: Only 1.1 GB/core (128 GB HBM). Fast compute but easy to OOM. Avoid for
  memory-hungry methods (large basis CCSD(T), big DFT grids). MKL conflict fixed
  automatically via `pre_run` hook in the profile.

Use `suggest_nwchem_resources(input_file, profile)` to auto-select optimal resources.

## Standard workflow

```
inspect_runner_profiles                          → confirm profiles are loaded
suggest_nwchem_partition(input_file=...)          → auto-select best partition across all profiles
                                                   (checks memory fit, walltime, dev queue, SU cost,
                                                   queue availability via sinfo)
suggest_nwchem_resources(input_file=...,
                         profile="stampede3_skx") → auto-pick nodes/ranks/walltime/memory
                                                   (use if you already know the partition)
render_job_script(profile="stampede3_skx",
                  resource_overrides=...)         → preview .job script before submitting
lint_nwchem_input(input_file=...)                → catch errors before wasting queue time
launch_nwchem_run(input_file=...,
                  profile="stampede3_skx",
                  resource_overrides=...)         → sbatch; writes {name}.jobid automatically
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

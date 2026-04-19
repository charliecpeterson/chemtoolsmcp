# NWChem Calculations — Local Workstation

This project runs NWChem quantum chemistry calculations on a local workstation
using the chemtools-nwchem agent toolkit. Jobs run as foreground processes
and are monitored by PID.

## Computing environment

| Item | Value |
|---|---|
| NWChem | Set `nwchem_executable` in your runner profile |
| MPI launcher | `mpirun -np {mpi_ranks}` (or `mpiexec`, `srun` for local SLURM) |
| Cores | Set `mpi_ranks` to your available core count |

## Runner profiles

| Profile | Description |
|---|---|
| `local` | Single-process NWChem (no MPI). Good for small test jobs. |
| `local_mpirun` | MPI-parallel NWChem via `mpirun`. Set `mpi_ranks` to your core count. |

## Standard workflow

```
init_session_log(log_path="session.md",
                 session_title="...",
                 working_dir=".")                → start running log
lint_nwchem_input(input_file=...)                → catch errors before running
launch_nwchem_run(input_file=...,
                  profile="local_mpirun",
                  auto_watch=true)               → run + block until done
analyze_nwchem_case(output_file=...,
                    input_file=...)              → diagnosis after completion
append_session_log(entry_type="summary", ...)    → final summary
```

## Key differences from HPC

- Jobs run in the **foreground** — `launch_nwchem_run` with `auto_watch=true`
  blocks until the process exits.
- No job scheduler — no `.jobid` file, no queue waiting.
- Process is monitored by PID, not scheduler status.
- `terminate_nwchem_run` sends SIGTERM to the process.
- Memory is limited by your workstation's RAM — use `check_nwchem_memory_fit`
  with `node_memory_mb` set in your profile to avoid OOM.

## Input versioning

NEVER overwrite an existing `.nw` file. Always call `next_versioned_path` first.

## Files after a run

```
{job_name}.nw      NWChem input
{job_name}.out     NWChem output
{job_name}.err     stderr messages
{job_name}.movecs  SCF/DFT MO vectors (restart asset)
{job_name}.db      NWChem runtime database
```

# Local Workstation — All 4 QC Programs

This project runs quantum chemistry calculations on a local workstation
using the chemtools agent toolkit. Jobs run as foreground processes (or
inside an apptainer container) and are monitored by PID.

## Computing environment

| Program | Local execution |
|---|---|
| NWChem | `programs.nwchem` command arrays in `runner_profiles.yaml` |
| Molcas | Apptainer container at `~/mycontainers/openmolcas-26.02.sif` (or override via `CHEMTOOLS_MOLCAS_CONTAINER`) |
| DIRAC  | Apptainer container at `~/mycontainers/dirac-25.0.sif` (or override via `CHEMTOOLS_DIRAC_CONTAINER`) |
| GRASP  | Apptainer container at `~/mycontainers/grasp2018.sif` (or override via `CHEMTOOLS_GRASP_CONTAINER`) |

The read-only GRASP, Molcas, and DIRAC command builders can use their
`CHEMTOOLS_*_CONTAINER` environment variables. Live typed launch tools require
a runner profile so the execution target, allowed working directory,
resources, and program installation are explicit.

## Runner profiles

NWChem-only profiles (in `runner_profiles.yaml`):

| Profile | Description |
|---|---|
| `local` | Single-process NWChem (no MPI). Good for small test jobs. |
| `local_mpirun` | MPI-parallel NWChem via `mpirun`. Set `mpi_ranks` to your core count. |
| `local_apptainer` | NWChem inside apptainer container with `mpirun -np N`. |

Molcas and DIRAC live launches also require profile entries. Their read-only
prepare tools can still use the corresponding `CHEMTOOLS_*_CONTAINER`
environment variable.

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

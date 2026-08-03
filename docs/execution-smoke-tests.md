# Execution smoke tests

This procedure runs one five-minute H₂ SCF case through the same typed
execution, ownership, monitoring, run-registration, and artifact paths used
by the NWChem MCP launch tools.

The runner creates a timestamped directory below the supplied work root. It
writes the copied input, stdout, stderr, launch registry, and `evidence.json`
there. The evidence includes command arguments, process or job ID, terminal
state, return code, output hashes, and the parsed SCF energy. It records
environment-variable names only through the launch registry, never their
values.

## Local Apptainer

Set the container path and run from the repository root:

```bash
export NWCHEM_CONTAINER=/path/to/nwchem.sif
.venv/bin/python scripts/smoke_nwchem_execution.py \
  --profiles-path examples/local_workstation/runner_profiles.yaml \
  --profile local_apptainer \
  --expect-executor local \
  --work-root "$HOME/scratch/chemtools-execution-smoke"
```

Success requires all of the following:

- The owned process reaches `completed` with return code 0.
- Monitoring reaches a terminal state before the timeout.
- The output parser classifies the SCF calculation as `converged`.
- The output contains a numerical total energy.

## Stampede3 Slurm

Run this on a Stampede3 login node. Confirm the current allocation, commands,
NWChem executable, and project environment before submission:

```bash
hostname
qlimits
/usr/local/etc/taccinfo
command -v sbatch squeue sacct scancel ibrun
test -x /home1/01775/charlesp/apps/nwchem/7.2.3/bin/nwchem
test -x .venv/bin/python
```

The smoke command submits two ranks to `skx-dev` and waits for scheduler
accounting:

```bash
.venv/bin/python scripts/smoke_nwchem_execution.py \
  --profiles-path examples/tacc_stampede3/runner_profiles.yaml \
  --profile stampede3_skx_dev \
  --expect-executor slurm \
  --work-root "$SCRATCH/chemtools-execution-smoke" \
  --mpi-ranks 2 \
  --walltime 00:05:00 \
  --timeout-seconds 1800 \
  --poll-interval-seconds 5
```

Check the profile's allocation and executable path before submission. A real
Slurm result must come from this command or an equivalent configured target.
Render-only tests and mocked `sbatch`, `squeue`, or `sacct` calls do not count.
Stampede3 bills jobs for at least 15 minutes. At the documented `skx-dev`
charge of 1 SU per node-hour, this one-node smoke costs at least 0.25 SU even
with a five-minute walltime. See the
[Stampede3 user guide](https://docs.tacc.utexas.edu/hpc/stampede3/) for the
current queue and charging policy.

## Recorded results

| Date | Target | Program | Result | Evidence |
| --- | --- | --- | --- | --- |
| 2026-07-30 | Local linux-4090 | NWChem 7.2.2 Apptainer | Pass | `~/scratch/chemtools-execution-smoke/20260731T062859Z-c409970c/evidence.json` |
| Pending | Stampede3 Slurm | NWChem | Awaiting login-node run | Use the command above; MFA prevents remote submission from linux-4090 |

The local run used two MPI ranks and completed in 1.103 seconds with return
code 0. The owned monitor reported `completed_success`; the parser classified
the SCF task as `converged` with an energy of
`-1.116759310191 E_h`. The stdout SHA-256 is
`c09f814fb02119b36be962006208f7cdbdea03134963daca505f43e3c588e8a2`.

An initial one-rank attempt reached the configured executable but the
containerized NWChem build aborted with “ranks per node, must be at least 2.”
That failed run is recorded at
`~/scratch/chemtools-execution-smoke/20260731T062718Z-a50b0b82/evidence.json`.
The local Apptainer profile now launches the image's `mpirun` with the
configured rank count and defaults to two ranks.

## Local Quantum ESPRESSO

The `qe_local` profile on linux-4090 was verified with a two-atom Si SCF input
and the typed `launch_qe_run` path. It launches QE 7.5 as
`mpirun -np 1 pw.x -in si.in`, with 20 OpenMP threads. Its local profile must set
`I_MPI_FABRICS: "shm"`: without it, Intel MPI 2021.11 entered a one-rank
`MPI_Alltoallv` wait through the TCP/libfabric provider during QE symmetry
initialization. This is a host-specific launcher setting, not an input-deck
change.

The recorded run at
`~/scratch/chemtools-qe-final.Y6Ce07/chemtools_qe_final.out` returned zero,
converged in seven SCF iterations at `-14.54255436 Ry`, and printed `JOB DONE.`.
Chemtools' QE parser identified QE 7.5, `scf_converged: true`,
`job_done: true`, and no runtime errors. Scheduler behavior is not validated
by this local result.

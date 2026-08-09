# Chemtools on TACC Stampede3

This project uses Chemtools from a Stampede3 login node. Login nodes are for
review, drafting, submission, and monitoring. Calculations run through Slurm.
MPI calculations use `ibrun` through the configured program command.

The checked-in runner profile is a site example, not a promise that partition,
allocation, module, or executable settings are still correct. Verify those
values from the live system before submitting work:

```bash
sinfo
sacctmgr show assoc where user="$USER"
module avail nwchem
command -v sbatch squeue sacct scancel ibrun nwchem
```

## Setup

Copy and edit the profile rather than pointing Chemtools at the repository:

```bash
mkdir -p ~/.config/chemtools
cp runner_profiles.yaml ~/.config/chemtools/runner_profiles.yaml
export CHEMTOOLS_RUNNER_PROFILES="$HOME/.config/chemtools/runner_profiles.yaml"
chemtools --show-mode
chemtools --list-tools
```

Set the current allocation, partition, module commands, executable, resources,
and container paths in that copy. The resolved mode should be `hpc`, and the
default tool list should contain the eleven guided names documented in
`docs/getting-started.md`.

## Working rules

- Review an input with `review_input` before submission.
- Use `plan_calculation` when the method, basis, relativistic treatment,
  electronic state, or stage order still needs a decision.
- Call `launch_run` once to render the exact Slurm plan and approval token.
- Submit only after the user approves that plan and the second `launch_run`
  call supplies the matching token.
- Monitor only the returned `launch_id` with `monitor_run` while the same MCP
  server process remains active.
- Inspect completed output with `inspect_run`, supplying the input, stderr, and
  restart artifacts explicitly when they matter.
- Compare related states or methods with `compare_runs`; do not infer a ground
  state from energy ordering alone.
- Use `plan_recovery` only when inspection supports a retry.
- Keep large calculations and their artifacts under `$SCRATCH`, not `$HOME`.
- Never overwrite existing input, output, error, job, or restart files.

The default MCP surface does not expose arbitrary job IDs, cancellation, or
the low-level program-specific tools. Those belong to explicit developer use.

## Stampede3 constraints

- Use `ibrun`, not `mpirun`, for ordinary MPI jobs launched by Slurm.
- Treat queue limits, charging, account names, and module versions as live
  cluster state.
- Stage required Apptainer images under an allocation-owned work or scratch
  location, then record their exact paths in the private profile copy.
- Completed jobs may leave `squeue`; retained launch monitoring can query
  Slurm accounting through the configured target.

## Expected NWChem artifacts

```text
{job_name}.nw      input
{job_name}.job     reviewed Slurm script
{job_name}.jobid   submitted job identifier
{job_name}.out     primary output
{job_name}.err     standard error
{job_name}.movecs  orbital restart data when produced
{job_name}.db      runtime database when produced
```

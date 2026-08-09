# Chemtools on a local workstation

This project uses the default Chemtools guided MCP surface. Calculations run
as tracked local subprocesses through a configured runner profile. The bundled
example profiles cover NWChem; add a program entry only when that executable
is actually installed and tested on this machine.

## Setup

Copy `runner_profiles.yaml` outside the repository and edit the selected
profile's command and rank count:

```bash
mkdir -p ~/.config/chemtools
cp runner_profiles.yaml ~/.config/chemtools/runner_profiles.yaml
export CHEMTOOLS_RUNNER_PROFILES="$HOME/.config/chemtools/runner_profiles.yaml"
chemtools --show-mode
chemtools --list-tools
```

The resolved mode should be `local`, and the default tool list should contain
the eleven guided names documented in `docs/getting-started.md`.

## Working rules

- Review an existing input with `review_input` before launch.
- Use `draft_input` when starting from molecular geometry. It returns text and
  does not write a file.
- Call `launch_run` once to prepare the exact command and approval token.
- Start the calculation only after the user approves that plan and the second
  `launch_run` call supplies the matching token.
- Monitor only the returned `launch_id` with `monitor_run` while the same MCP
  server process remains active.
- Inspect the completed primary output with `inspect_run`; supply the input and
  stderr paths as explicit artifacts when they matter.
- Use `plan_recovery` only when the inspection evidence supports a retry.
- Never overwrite an existing input, output, error, or scheduler artifact.

Analysis and drafting remain available when no executable is installed. Live
launching requires a readable profile, an allowed working directory, and an
executable command that resolves on this machine.

## Expected NWChem artifacts

```text
{job_name}.nw      input
{job_name}.out     primary output
{job_name}.err     standard error
{job_name}.movecs  orbital restart data when produced
{job_name}.db      runtime database when produced
```

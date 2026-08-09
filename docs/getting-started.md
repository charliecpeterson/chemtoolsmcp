# Install and connect Chemtools

This path installs a wheel, exposes the eleven guided tools to an MCP
client, and optionally adds local or Slurm execution.

## 1. Install the wheel

Create a dedicated environment. Chemtools supports Python 3.10 and newer.

```bash
python3 -m venv ~/.venvs/chemtools
~/.venvs/chemtools/bin/python -m pip install \
  /path/to/chemtools_mcp-0.1.0-py3-none-any.whl
```

DIRAC text files work with the base install. Add the `dirac` extra only when
Chemtools must read DIRAC HDF5 checkpoints:

```bash
~/.venvs/chemtools/bin/python -m pip install \
  "/path/to/chemtools_mcp-0.1.0-py3-none-any.whl[dirac]"
```

PySCF, RDKit, Open Babel, Basis Set Exchange, QMCPACK HDF5 inspection, and
Orbitron belong in the optional companion environment. They are not required
for ordinary input review, output inspection, drafting, or recovery planning.

## 2. Check the installed command

```bash
~/.venvs/chemtools/bin/chemtools --show-mode
~/.venvs/chemtools/bin/chemtools \
  --mode analysis \
  --list-tools
```

The second command must print these eleven names:

```text
review_input
inspect_run
compare_runs
plan_recovery
plan_calculation
launch_run
monitor_run
draft_input
visualize
search_knowledge
find_reference_case
```

Analysis mode does not require a chemistry executable, container, scheduler,
or writable job directory.

`find_reference_case` defaults to scientifically reviewed
`validated_reference` cases. The currently packaged chemistry cases remain
`exploratory`, so they appear only when that status is requested explicitly.
Artifact hashes establish file identity, not scientific approval.

## 3. Connect an MCP client

Use the command inside the virtual environment. Do not set a repository
working directory.

```json
{
  "mcpServers": {
    "chemtools": {
      "command": "/absolute/path/to/.venvs/chemtools/bin/chemtools",
      "env": {
        "CHEMTOOLS_MODE": "analysis"
      }
    }
  }
}
```

Restart the MCP client after changing its configuration. Its tool list should
contain the same eleven names shown by `--list-tools`.

One representative read-only request is:

> Review `/path/to/job.nw`, inspect `/path/to/job.out` as NWChem, and explain
> any uncertainty or recovery options. Do not run a calculation or write a
> replacement input.

That request should use `review_input`, then `inspect_run`, and call
`plan_recovery` only if the inspection supports a recovery step.

### Optional Codex plugin

The repository includes a thin plugin bundle at `plugins/chemtools`. It adds
four workflow skills and starts the same installed command with
`--toolset guided`.

Install and verify the wheel before adding the plugin to a local marketplace:

```bash
chemtools --mode analysis --list-tools
```

The plugin does not contain the Python distribution, create an environment,
or install dependencies. Its `.mcp.json` deliberately calls `chemtools` from
the host environment so package installation remains independently testable.
Add the bundle to a local marketplace only after that command works, then
install it from the Codex plugin browser and start a new session. The current
Codex plugin workflow is documented in
[Build plugins](https://developers.openai.com/plugins) and
[Connect and test your plugin](https://developers.openai.com/plugins/deploy/connect-chatgpt).

The checked-in prompt contract at
`plugins/chemtools/evals/prompt-routing.yaml` covers direct and indirect skill
selection, follow-ups, unsupported requests, and launch approval behavior.

## 4. Add execution only when needed

The installed command can print profiles without access to the source tree.
Choose one template:

```bash
mkdir -p ~/.config/chemtools
~/.venvs/chemtools/bin/chemtools \
  --print-profile-example local \
  > ~/.config/chemtools/runner_profiles.json
```

```bash
mkdir -p ~/.config/chemtools
~/.venvs/chemtools/bin/chemtools \
  --print-profile-example slurm \
  > ~/.config/chemtools/runner_profiles.yaml
```

Edit the selected file before using it. For a workstation, set the NWChem
command and rank count. For Slurm, set the partition, account, module, rank
count, and program command to match the cluster.

Check mode resolution before connecting the profile to an MCP client:

```bash
CHEMTOOLS_RUNNER_PROFILES=~/.config/chemtools/runner_profiles.yaml \
  ~/.venvs/chemtools/bin/chemtools --show-mode
```

A direct profile resolves to `local`; a scheduler profile resolves to `hpc`.
Add the same `CHEMTOOLS_RUNNER_PROFILES` value to the MCP client configuration.
Remove the explicit `CHEMTOOLS_MODE=analysis` value so the profile can select
the execution mode.

The guided preset includes `launch_run`. Its first call is read-only and returns
the exact rendered plan plus an approval token. It starts nothing until a
second call supplies that token after explicit user approval. Analysis mode
can prepare plans but refuses the approved launch. Local or hpc mode is required
to start it. Existing outputs, errors, or scheduler scripts block the launch;
the guided tool never overwrites or silently archives them.

The same server process returns a `launch_id` after an approved launch.
`monitor_run` accepts only that owned identifier. It refreshes the retained
local process or target-owned Slurm job, reports recorded output and error
files, and adds scientific progress when the backend supports it. It never
submits, restarts, cancels, or accepts an arbitrary PID, job ID, or output
path. Keep the MCP server running while monitoring; active ownership is not
transferred to a replacement server process.

## 5. Troubleshoot the connection

`chemtools: command not found`

: Set the MCP command to the absolute path inside the virtual environment.
  MCP clients often start with a smaller `PATH` than an interactive shell.

The client shows hundreds of tools

: Remove `CHEMTOOLS_TOOLSET=developer` or `CHEMTOOLS_TOOLSET=full`, restart the
  client, and verify the default with `--mode analysis --list-tools`.

The server stays in analysis mode after adding a runner profile

: Remove an explicit `CHEMTOOLS_MODE=analysis` override. Run `--show-mode` with
  `CHEMTOOLS_RUNNER_PROFILES` set and read `mode_reason` in the JSON result.

The runner profile does not load

: Confirm the path is absolute and readable. Print a fresh bundled example,
  compare its YAML or JSON structure with the edited file, then run
  `--show-mode` again. Syntax and profile errors are written to stderr.

A DIRAC HDF5 tool reports that h5py is missing

: Reinstall the wheel with the `[dirac]` extra. DIRAC text parsing does not
  require h5py.

An output is not detected

: Confirm that the file is the primary program output rather than a sidecar or
  converter log. Pass the program explicitly when the content is valid but
  lacks a recognizable banner.

Companion science tools report that the runtime is unavailable

: Create the environment described in
  [environments/README.md](../environments/README.md), set
  `CHEMTOOLS_SCIENCE_PYTHON` to its interpreter, and call
  `inspect_science_runtime` before running a science operation.

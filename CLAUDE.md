# chemtoolsmcp — MCP Development Workspace

This is the source repository for the chemtoolsmcp NWChem AI agent toolkit.
Work here is about **developing and improving the MCP** — adding tools, fixing parsers, updating logic.

## Architecture

```
chemtools/           Core Python library — all parsing, analysis, and input generation
  api.py             Public API entry point (re-exports from api_*.py modules)
  api_input.py       Input drafting functions
  api_output.py      Output parsing functions
  api_strategy.py    High-level case analysis, recovery strategies
  api_runner.py      Job launch, status, watch, terminate
  api_basis.py       Basis/ECP library resolution and rendering
  nwchem_tce.py      TCE output parser, movecs binary tools, freeze count advisor
  nwchem_tasks.py    Task boundary detection and energy extraction
  nwchem_mos.py      MO analysis parser
  nwchem_input.py    Input file parsing utilities
  diagnostics.py     High-level diagnosis functions
  data/nwchem/       Bundled NWChem data (basis library — 608 files)
  mcp/
    nwchem.py        NWChem MCP server — 49 tools, thin wrappers over chemtools/
    nwchem_docs.py   NWChem documentation lookup server
    # Future: molpro.py, orca.py

tests/               Test suite
```

## MCP Tool Architecture

- Domain logic lives in `chemtools/*.py`
- Public API re-exported from `chemtools/api.py` → `chemtools/__init__.py`
- MCP handlers in `chemtools/mcp/nwchem.py` — one `@_tool(name)` decorated function per tool
- Tool naming convention: `verb_nwchem_noun` where verb ∈ {parse, analyze, draft, create, suggest, launch, get, watch, inspect, lint, find, compare, review, render, swap}
- Current tool count: 60

## How to Add a New Tool

1. Write the domain function in the appropriate `chemtools/api_*.py` (or a new module)
2. Export it from `chemtools/api.py` and `chemtools/__init__.py`
3. Add a tool definition dict to `tool_definitions()` in `chemtools/mcp/nwchem.py`
4. Add a `@_tool("tool_name")` handler function that calls the library
5. Verify: `python3 -c "from chemtools.mcp import nwchem; print(len(nwchem.tool_definitions()), 'tools')"`

## How to Improve an Existing Tool

The iterative workflow: run a NWChem job locally, see what the agent does wrong or can't do, fix the tool.

Common patterns:
- Parser misses a new output format → fix regex in `nwchem_tasks.py` or `nwchem_tce.py`
- Strategy tool gives wrong advice → improve heuristics in `api_strategy.py`
- Input drafter generates bad syntax → fix in `api_input.py`, check with lint
- New NWChem module/feature → add parser + MCP tool following the pattern above

## Key Design Rules

- **Never add `freeze atomic` to TCE inputs** — always compute and emit explicit `freeze N`
- **Always inspect orbital ordering before TCE freeze** — `parse_nwchem_movecs` first, then decide
- **Binary movecs swap required for reordering** — `vectors swap` in SCF block doesn't survive re-diagonalization
- **MCP handlers are thin** — all logic lives in `chemtools/`, handlers just translate arguments
- **Explicit basis blocks** — generate explicit per-element basis text from the library, not `library` shorthand
- **Lint after drafting** — every input tool should be followed by lint in the workflow

## Runner Profiles

Runner profiles tell the agent how to launch, monitor, and cancel NWChem jobs. They are
**per-machine configuration** (not checked into this repo). Set `CHEMTOOLS_RUNNER_PROFILES`
to point at your local YAML or JSON file.

Example files in this repo:
- `chemtools/runner_profiles.example.yaml` — canonical reference with all profile types
- `chemtools/runner_profiles.example.json` — auto-synced JSON copy (same content)
- `chemtools/runner_profiles.local.example.json` — minimal template for local customization

### Local profiles (`launcher.kind: "direct"`)

NWChem runs as a foreground subprocess on the same machine as the agent. The agent
monitors the process by PID and tails the output file.

```yaml
local_mpirun:
  launcher:
    kind: "direct"
    command: "mpirun -np {mpi_ranks} /path/to/nwchem"
  execution:
    command_template: "{launcher} {input_file} > {output_file} 2> {error_file}"
  resources:
    mpi_ranks: 14
```

### HPC / scheduler profiles (`launcher.kind: "scheduler"`)

NWChem is submitted to a queue. The agent submits via `sbatch`/`qsub`, writes
`{job_name}.jobid` alongside the input, and monitors via the scheduler's status command.

**Key fields:**

| Field | Purpose |
|---|---|
| `launcher.submit_command` | `sbatch`, `qsub`, `bsub` |
| `launcher.scheduler_type` | `slurm`, `pbs`, `lsf` — drives state mapping |
| `launcher.job_id_regex` | Regex to extract job ID from submit stdout |
| `launcher.status_command` | e.g. `squeue -j {job_id} -h -o %T` (returns state only) |
| `launcher.cancel_command` | e.g. `scancel {job_id}` |
| `execution.nwchem_executable` | Full path to the NWChem binary |
| `execution.mpi_launch` | Full MPI launch prefix: `ibrun` (TACC), `srun`, `mpirun -np 48` |
| `resources.nodes/mpi_ranks/walltime/partition/account` | Default job resources |
| `scheduler.script_template` | Shell script with `{placeholder}` substitutions |
| `scheduler.submit_script_name` | Filename for the generated script, e.g. `{job_name}.job` |
| `modules.load` | List of `module load` commands to include in the script |
| `hooks.pre_run` | Shell commands inserted before the NWChem launch line |

**Template placeholders available in `script_template`:**
`{job_name}`, `{output_file}`, `{error_file}`, `{nodes}`, `{mpi_ranks}`, `{omp_threads}`,
`{walltime}`, `{partition}`, `{account}`, `{account_line}` (the full `#SBATCH -A ...` line or
empty string), `{nwchem_executable}`, `{mpi_launch}`, `{module_block}`, `{pre_run_block}`,
`{job_dir}`, `{input_file}`.

**TACC Stampede3 example** (profiles `stampede3_skx` / `stampede3_spr` in the example file):
```yaml
stampede3_skx:
  launcher:
    kind: "scheduler"
    submit_command: "sbatch"
    scheduler_type: "slurm"
    job_id_regex: "Submitted batch job (\\d+)"
    status_command: "squeue -j {job_id} -h -o %T"
    cancel_command: "scancel {job_id}"
  scheduler:
    script_template: |
      #!/bin/bash
      #SBATCH -J {job_name}
      #SBATCH -o {output_file}
      #SBATCH -e {error_file}
      #SBATCH -p {partition}
      #SBATCH -N {nodes}
      #SBATCH -n {mpi_ranks}
      #SBATCH -t {walltime}
      {account_line}
      cd {job_dir}
      {mpi_launch} {nwchem_executable} {input_file}
    submit_script_name: "{job_name}.job"
  execution:
    nwchem_executable: "/home1/01775/charlesp/apps/nwchem/7.2.3/bin/nwchem"
    mpi_launch: "ibrun"
  resources:
    nodes: 1
    mpi_ranks: 48
    partition: "skx"
    walltime: "24:00:00"
    account: null
```

### How HPC monitoring works

1. `launch_nwchem_run` submits the job, parses the job ID, writes `{job_name}.jobid`
2. `get_nwchem_run_status` / `watch_nwchem_run` auto-detect the `.jobid` file from the
   input/output path — no need to pass `job_id` explicitly
3. Scheduler state (PENDING/RUNNING/COMPLETED/FAILED/etc.) is mapped to normalized status:
   `queued`, `running`, `completed`, `failed`, `cancelled`
4. Output file is tailed in parallel with scheduler polling — slow-phase detection works
   the same as local runs
5. `terminate_nwchem_run` accepts `job_id + profile` for HPC cancel (calls `scancel`/`qdel`)

### Agent workflow for HPC

```
inspect_runner_profiles          → verify stampede3_skx is available
render_job_script(profile=...)   → preview the .job script before submitting
lint_nwchem_input                → check input is correct
launch_nwchem_run                → sbatch; writes ferrocene.jobid
watch_nwchem_run(output_file=...)→ polls squeue + tails output; auto-reads .jobid
analyze_nwchem_case              → diagnosis after completion
```

## Development Environment

- Install in editable mode: `pip install -e .`
- Entry points: `chemtools-nwchem`, `chemtools-nwchem-docs`
- Basis library: bundled at `chemtools/data/nwchem/basis_library/` (auto-detected after install)
- Docs server requires a local NWChem docs checkout; set `NWCHEM_DOCS_ROOT` to point at it
- Runner profiles: set `CHEMTOOLS_RUNNER_PROFILES` to your local YAML/JSON file

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
  data/nwchem/       Bundled NWChem data (basis library — 608 files, docs — 29 files)
  registry.py       SQLite-backed run registry, campaigns, workflows, batch generation
  protocols.py       Pre-baked calculation protocols (thermochem, basis convergence, etc.)
  eval.py            Case evaluation framework for testing tool quality
  mcp/
    nwchem.py        NWChem MCP server — 106 tools, thin wrappers over chemtools/
    nwchem_docs.py   Standalone docs server (backward-compat; docs tools now in nwchem.py)
    # Future: molpro.py, orca.py

test_phase1/         Test suite (Phases 2–6, 244 tests)
```

## MCP Tool Architecture

- Domain logic lives in `chemtools/*.py`
- Public API re-exported from `chemtools/api.py` → `chemtools/__init__.py`
- MCP handlers in `chemtools/mcp/nwchem.py` — one `@_tool(name)` decorated function per tool
- Tool naming convention: `verb_nwchem_noun` where verb ∈ {parse, analyze, draft, create, suggest, launch, get, watch, inspect, lint, find, compare, review, render, swap, register, update, list, advance, generate, detect, estimate, compute}
- Current tool count: 106

### Tool categories (106 tools)

| Category | Count | Examples |
|----------|-------|---------|
| Input drafting | 17 | `create_nwchem_input`, `create_nwchem_dft_input_from_request`, `draft_nwchem_tce_input` |
| Output parsing | 16 | `parse_nwchem_output`, `parse_nwchem_tce_output`, `parse_nwchem_thermochem`, `parse_nwchem_freq_progress`, `parse_nwchem_tasks`, `parse_nwchem_trajectory` |
| Analysis & diagnosis | 11 | `analyze_nwchem_case`, `check_nwchem_spin_charge_state`, `summarize_nwchem_output`, `preflight_check`, `review_nwchem_progress` |
| Strategy & suggestions | 12 | `suggest_basis_set`, `suggest_nwchem_recovery`, `suggest_spin_state`, `suggest_nwchem_resources`, `suggest_resources` |
| Resource & HPC | 5 | `suggest_nwchem_resources`, `detect_nwchem_hpc_accounts`, `check_nwchem_memory_fit`, `estimate_nwchem_freq_walltime`, `render_job_script` |
| Basis & ECP | 4 | `render_nwchem_basis_setup`, `basis_library_summary` |
| Job management | 7 | `launch_nwchem_run`, `watch_nwchem_run`, `watch_multiple_runs`, `terminate_nwchem_run`, `get_nwchem_run_status`, `tail_nwchem_output` |
| Registry & campaigns | 9 | `register_nwchem_run`, `create_nwchem_campaign`, `get_nwchem_campaign_energies`, `generate_nwchem_input_batch` |
| Workflow & protocols | 7 | `plan_nwchem_calculation`, `plan_nwchem_workflow`, `get_nwchem_workflow_state`, `prepare_nwchem_next_step`, `advance_nwchem_workflow`, `create_nwchem_workflow`, `list_nwchem_protocols` |
| Geometry | 5 | `extract_nwchem_geometry`, `inspect_nwchem_geometry`, `displace_nwchem_geometry_along_mode` |
| Session & versioning | 3 | `init_session_log`, `append_session_log`, `next_versioned_path` |
| TCE (correlated methods) | 6 | `parse_nwchem_movecs`, `swap_nwchem_movecs`, `validate_nwchem_tce_setup` |
| Documentation | 6 | `search_nwchem_docs`, `lookup_nwchem_block_syntax`, `find_nwchem_examples`, `get_nwchem_topic_guide`, `read_nwchem_doc_excerpt`, `list_nwchem_docs` |
| Evaluation | 2 | `evaluate_nwchem_case`, `evaluate_nwchem_cases` |

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
- **Never overwrite input files** — always call `next_versioned_path` before writing a modified `.nw` file; the first version stays as-is and revisions become `_v2.nw`, `_v3.nw`, etc.
- **Always start a session log** — call `init_session_log` at the beginning of any multi-step workflow; append entries with `append_session_log` after each action, decision, or error; write a `summary` entry at the end
- **Parallel job monitoring** — submit jobs with `auto_watch=false`, then call `watch_multiple_runs` (not `watch_nwchem_run` in a loop) to block until all finish simultaneously
- **Register runs in the registry** — call `register_nwchem_run` when submitting jobs, `update_nwchem_run_status` after completion; this enables campaign tracking and energy tables across sessions
- **Use campaigns for related runs** — create a campaign first (`create_nwchem_campaign`), then link runs via `campaign_id`; use `get_nwchem_campaign_energies` for sorted energy tables with relative energies in kcal/mol
- **Workflow DAGs for multi-step protocols** — use `create_nwchem_workflow` for dependent steps (opt→freq), then `advance_nwchem_workflow` to find ready-to-launch steps
- **Registry is SQLite at `~/.chemtools/registry.db`** — uses stdlib `sqlite3`, no external dependency; override with `CHEMTOOLS_REGISTRY_DB` env var for testing

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
| `resources.account_command` | Shell command to discover allocations (e.g. `/usr/local/etc/taccinfo` on TACC) |
| `resources.cores_per_node` | Physical cores per node — enables auto rank selection |
| `resources.node_memory_mb` | Total RAM per node in MB — enables memory ceiling checks |
| `resources.max_nodes` | Max nodes available for jobs — enables multi-node suggestions |
| `resources.max_walltime` | Max walltime the queue allows (e.g. `"48:00:00"`) |
| `resources.cpu_arch` | CPU microarchitecture (`skx`, `icx`, `spr`, `avx2`) — tunes BF/rank |
| `scheduler.script_template` | Shell script with `{placeholder}` substitutions |
| `scheduler.submit_script_name` | Filename for the generated script, e.g. `{job_name}.job` |
| `modules.load` | List of `module load` commands to include in the script |
| `hooks.pre_run` | Shell commands inserted before the NWChem launch line |

**Template placeholders available in `script_template`:**
`{job_name}`, `{output_file}`, `{error_file}`, `{nodes}`, `{mpi_ranks}`, `{omp_threads}`,
`{walltime}`, `{partition}`, `{account}`, `{account_line}` (the full `#SBATCH -A ...` line or
empty string), `{nwchem_executable}`, `{mpi_launch}`, `{module_block}`, `{pre_run_block}`,
`{job_dir}`, `{input_file}`.

**TACC Stampede3 example** (profiles `stampede3_skx` / `stampede3_icx` / `stampede3_spr` / `stampede3_skx_dev` in the example file):
```yaml
stampede3_skx:
  launcher: { kind: "scheduler", submit_command: "sbatch", ... }
  scheduler: { script_template: "...", submit_script_name: "{job_name}.job" }
  execution:
    nwchem_executable: "/path/to/nwchem"
    mpi_launch: "ibrun"
  resources:
    # --- Defaults (overridden by suggest_nwchem_resources) ---
    nodes: 1
    mpi_ranks: 48
    partition: "skx"
    walltime: "24:00:00"
    account: null
    account_command: "/usr/local/etc/taccinfo"  # auto-detect allocation
    # --- Hardware description (static) ---
    cores_per_node: 48
    node_memory_mb: 192000
    max_nodes: 256
    max_walltime: "48:00:00"
    cpu_arch: "skx"
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

### Auto resource selection

The `suggest_nwchem_resources` tool analyzes an input file against a profile's hardware
specs and recommends optimal nodes, MPI ranks, walltime, and memory directive. This
replaces manual guessing and prevents common HPC failures (OOM, walltime exceeded).

Profiles should describe the machine with these fields in `resources`:
- `cores_per_node` — physical cores per node
- `node_memory_mb` — total RAM per node in MB
- `max_nodes` — max nodes available for jobs
- `max_walltime` — max walltime the queue allows (e.g. `"48:00:00"`)
- `cpu_arch` — CPU microarchitecture (`skx`, `spr`, `avx2`, etc.)

The advisor handles:
- **Small molecules**: reduces ranks to avoid communication overhead
- **Numerical frequencies**: estimates 6*N_atoms displacements, scales to multi-node
  if needed, warns about no checkpoint capability
- **TCE**: scales nodes for memory when correlation memory exceeds single node
- **Walltime**: task-type-aware estimates with safety margins

### Agent workflow for HPC (single job)

```
init_session_log(log_path=..., session_title=...)  → start running doc
inspect_runner_profiles                             → verify profile is available
suggest_nwchem_resources(input_file, profile)        → get optimal resource_overrides
render_job_script(profile=..., resource_overrides=.) → preview the .job script
lint_nwchem_input                                   → check input is correct
launch_nwchem_run(auto_watch=true, resource_overrides=.) → sbatch + block until done
append_session_log(entry_type="result", ...)        → record outcome
analyze_nwchem_case                                 → diagnosis
append_session_log(entry_type="summary", ...)       → final summary
```

### Agent workflow for HPC (parallel jobs)

```
init_session_log(...)                               → start running doc
# For each job:
next_versioned_path(path="mol.nw")                  → get safe output path
lint_nwchem_input                                   → validate
launch_nwchem_run(auto_watch=false)                 → submit all jobs first
# After all submitted:
watch_multiple_runs(jobs=[...])                     → block until all done
# Analyze each result
append_session_log(entry_type="summary", ...)       → final summary
```

### Agent workflow for campaigns (e.g. ligand screen)

```
init_session_log(...)                               → start running doc
create_nwchem_campaign(name="ligand_screen")         → get campaign_id
generate_nwchem_input_batch(template, vary={...})    → create all inputs
# For each generated input:
register_nwchem_run(campaign_id=..., ...)            → track in registry
lint_nwchem_input                                    → validate
launch_nwchem_run(auto_watch=false)                  → submit
# After all submitted:
watch_multiple_runs(jobs=[...])                      → block until done
# After completion:
update_nwchem_run_status(run_id=..., status=..., energy_hartree=...) → record results
get_nwchem_campaign_energies(campaign_id=...)         → sorted energy table
append_session_log(entry_type="summary", ...)         → final summary
```

### Agent workflow for multi-step protocols (e.g. opt→freq)

```
init_session_log(...)                               → start running doc
plan_nwchem_calculation(protocol="thermochem_dft")   → get step plan
create_nwchem_workflow(steps=[...])                  → create DAG
advance_nwchem_workflow(workflow_id=...)              → find ready steps
# Launch ready step, wait, update status
advance_nwchem_workflow(workflow_id=...)              → next ready steps
# Repeat until workflow is done
append_session_log(entry_type="summary", ...)         → final summary
```

## Development Environment

- Install in editable mode: `pip install -e .`
- Entry points: `chemtools-nwchem`, `chemtools-nwchem-docs`
- Basis library: bundled at `chemtools/data/nwchem/basis_library/` (auto-detected after install)
- NWChem docs: bundled at `chemtools/data/nwchem/docs/` (29 text files, always available)
- Runner profiles: set `CHEMTOOLS_RUNNER_PROFILES` to your local YAML/JSON file

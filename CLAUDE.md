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
    nwchem.py        NWChem entry point — multi-program MCP server (NWChem + Molcas tools)
    nwchem_docs.py   Standalone docs server (backward-compat; docs tools now in nwchem.py)
    tools/
      nwchem.py      NWChem tool definitions + handlers (114 tools)
      molcas.py      Molcas tool definitions + handlers (30 tools)
    # Future: molpro.py, orca.py

test_phase1/         Test suite (Phases 2–6, 244 tests)
```

## MCP Tool Architecture

- Domain logic lives in `chemtools/*.py`
- Public API re-exported from `chemtools/api.py` → `chemtools/__init__.py`
- MCP handlers in `chemtools/mcp/nwchem.py` — one `@_tool(name)` decorated function per tool
- Tool naming convention: `verb_nwchem_noun` where verb ∈ {parse, analyze, draft, create, suggest, launch, get, watch, inspect, lint, find, compare, review, render, swap, register, update, list, advance, generate, detect, estimate, compute}
- Current tool count: 144 (114 NWChem + 30 Molcas; the NWChem total includes `get_server_mode`)
- Tools are tagged with a capability (`needs=`) on the `@_tool` decorator; the active server mode filters which tools are exposed. See **Server modes** below.

### Tool categories (108 tools)

| Category | Count | Examples |
|----------|-------|---------|
| Input drafting | 17 | `create_nwchem_input`, `create_nwchem_dft_input_from_request`, `draft_nwchem_tce_input` |
| Output parsing | 16 | `parse_nwchem_output`, `parse_nwchem_tce_output`, `parse_nwchem_thermochem`, `parse_nwchem_freq_progress`, `parse_nwchem_tasks`, `parse_nwchem_trajectory` |
| Analysis & diagnosis | 11 | `analyze_nwchem_case`, `check_nwchem_spin_charge_state`, `summarize_nwchem_output`, `preflight_check`, `review_nwchem_progress` |
| Strategy & suggestions | 12 | `suggest_basis_set`, `suggest_nwchem_recovery`, `suggest_spin_state`, `suggest_nwchem_resources`, `suggest_resources` |
| Resource & HPC | 6 | `suggest_nwchem_resources`, `suggest_nwchem_partition`, `detect_nwchem_hpc_accounts`, `check_nwchem_memory_fit`, `estimate_nwchem_freq_walltime`, `render_job_script` |
| Basis & ECP | 4 | `render_nwchem_basis_setup`, `basis_library_summary` |
| Job management | 7 | `launch_nwchem_run`, `watch_nwchem_run`, `watch_multiple_runs`, `terminate_nwchem_run`, `get_nwchem_run_status`, `tail_nwchem_output` |
| Registry & campaigns | 9 | `register_nwchem_run`, `create_nwchem_campaign`, `get_nwchem_campaign_energies`, `generate_nwchem_input_batch` |
| Workflow & protocols | 7 | `plan_nwchem_calculation`, `plan_nwchem_workflow`, `get_nwchem_workflow_state`, `prepare_nwchem_next_step`, `advance_nwchem_workflow`, `create_nwchem_workflow`, `list_nwchem_protocols` |
| Geometry | 5 | `extract_nwchem_geometry`, `inspect_nwchem_geometry`, `displace_nwchem_geometry_along_mode` |
| Session & versioning | 3 | `init_session_log`, `append_session_log`, `next_versioned_path` |
| TCE (correlated methods) | 6 | `parse_nwchem_movecs`, `swap_nwchem_movecs`, `validate_nwchem_tce_setup` |
| Documentation | 7 | `search_nwchem_docs`, `lookup_nwchem_block_syntax`, `find_nwchem_examples`, `get_nwchem_topic_guide`, `read_nwchem_doc_excerpt`, `list_nwchem_docs`, `search_nwchem_forum` |
| Evaluation | 2 | `evaluate_nwchem_case`, `evaluate_nwchem_cases` |

### Molcas / OpenMolcas tools (30)

| Tool | Purpose |
|------|---------|
| `draft_molcas_input` | Render a full Molcas input deck (MOLCAS_MEM + SEWARD + SCF + RASSCF + CASPT2 chain). Methods: HF/SCF/DFT/CASSCF/RASSCF/CASPT2/RASPT2/MS-CASPT2/XMS-CASPT2/RMS-CASPT2/XDW-CASPT2 |
| `lint_molcas_input` | Validate Molcas input string. Catches block-pair issues, unknown basis libraries, RASSCF/CASPT2 Frozen mismatches, missing Nactel, suspicious LumOrb without orbital provenance |
| `compute_molcas_active_space_partition` | CAS(M,N) → per-symmetry RASSCF directives (Nactel, Frozen, Inactive, Ras1/2/3, Secondary) |
| `list_molcas_basis_sets` | Enumerate bundled basis sets; filter by element; report contractions for a (basis, element) pair |
| `parse_molcas_output` | Deep parse: per-module SCF / RASSCF / CASPT2 details + active-space summary + warnings |
| `parse_molcas_tasks` | Cheap module-boundary task index |
| `get_molcas_orbitals` | Last `++ Molecular orbitals:` block — RASSCF NOs win over SCF MOs |
| `parse_molcas_inporb` | Read INPORB / RasOrb / ScfOrb / GssOrb / LprOrb / SpdOrb files |
| `parse_molcas_frequencies` | Last `Harmonic frequencies in cm-1` block from MCLR or numerical-grad. Per-symmetry modes + IR intensity + reduced mass + per-atom displacements; imaginary modes as negative floats |
| `parse_molcas_thermochem` | Per-temperature ZPVE + S + U + H + G (kcal/mol + au); 298.15 K row hoisted under `standard_298_15` |
| `extract_molcas_geometry` | Single geometry snapshot — SLAPAF converged geometry preferred, else last `Cartesian coordinates` block |
| `parse_molcas_trajectory` | SLAPAF Energy Statistics + per-iteration geometries (cumulative table de-duplicated) |
| `parse_molcas_rassi` | RASSI state-interaction: input states, spin-free + spin-orbit eigenstates (rel + abs energies), SO composition (SF state contributions per SO state), SOC matrix elements above SOCOupling threshold, dipole oscillator strengths in SF + SO bases, NRNATO natural-orbital occupations. Includes SOC stabilization roll-up in cm-1 |
| `analyze_molcas_active_space` | NO-occupation classification → healthy/marginal/poor verdict + promote/demote orbital lists + next_actions |
| `validate_molcas_caspt2_setup` | Reference-weight, IPEA / shift, intruder-state checks → healthy/caution/unreliable verdict |
| `list_molcas_docs` | List 133 bundled OpenMolcas doc files |
| `search_molcas_docs` | Search docs (programs / tutorials / users_guide / advanced_examples) |
| `lookup_molcas_module` | Pull the docs page for a Molcas module (rasscf, caspt2, alaska, ...) |
| `read_molcas_doc_excerpt` | Read a slice of a bundled doc by relative path |
| `get_molcas_topic_guide` | Curated guidance for high-value topics (rasscf_active_space, caspt2_setup, ipea_shift, xms_caspt2, alaska_gradients, mclr_freq, rassi_state_interaction, inporb_format, scf_setup) |
| `prepare_molcas_launch` | Build safe `pymolcas` command + env. Auto-downgrades `-np` to 1 for inputs containing `&CASPT2` when `execution.parallel_caspt2_supported=False` (broken GA builds); always isolates scratch via unique `MOLCAS_PROJECT` to prevent `RunHdr%nProcs/=nProcs` cross-run aborts. Does not execute — returns the command for the caller to run. |
| `swap_molcas_inporb_orbitals` | Swap orbital pairs within a symmetry block of an INPORB / RasOrb file — swaps MO coefficients + occupation + energy + typeindex. Used for active-space tuning workflows (initial RASSCF → inspect orbital character → swap wrong-class orbital with a better candidate → re-run RASSCF with FILEORB). Bit-exact round-trip. |
| `suggest_molcas_orbital_swaps` | Character-aware swap suggester. Walks the LAST '++ Molecular orbitals:' block, classifies each orbital's space (inactive/active/secondary) from the RASSCF orbital_specs, and matches dominant AO against a target pattern (e.g. 'Cr' + '3d'). Returns suggested (active_orb, swap_with) pairs ready to feed into `swap_molcas_inporb_orbitals`. Diagnostic only — final swap choice still requires chemistry judgment. |
| `refine_molcas_active_space` | **Thick orchestrator** that closes the active-space-tuning loop in ONE call: parses an existing RASSCF .out, runs occupation + character analyses, applies the suggested swaps to the RasOrb, generates a refined input with FILEORB injected, and returns a launch plan + next_actions envelope. Replaces the manual chain `parse_molcas_output → suggest_molcas_orbital_swaps → swap_molcas_inporb_orbitals → text-edit input → prepare_molcas_launch`. Supports dry-run (`apply_swaps=False`) for inspection-only. |
| `prepare_molcas_casscf_setup` | **Thick orchestrator** for fresh CASSCF / CASPT2 / MS-CASPT2 calculations. Takes molecule + method spec + (cas_active_electrons, cas_active_orbitals) OR `chemistry_hint='valence_d'`/`'frontier_pair'`. Drafts the input, lints, computes the active-space partition, optionally writes input + builds launch plan, returns a Diagnosis envelope. Auto-corrects parity mismatches in TM hints (e.g. Cr⁰ 3d⁵ + spin-dictated 4s electron). |
| `prepare_molcas_caspt2_chain` | **Thick orchestrator** that takes a converged RASSCF .out and chains CASPT2 on top. Auto-picks SS vs MS variant from `n_roots`, sets IPEA=0.25 by default, emits imaginary shift 0.1 if active-space verdict is 'marginal', mirrors RASSCF Frozen. Short-circuits to `verdict='needs_active_space_refinement'` if RASSCF active space is 'poor' (points at `refine_molcas_active_space` first). The continuation reads previous RASSCF orbitals via FILEORB so RASSCF re-converges in ~4 iters. |
| `prepare_molcas_excited_states` | **Thick orchestrator** for multi-state excited-state workflows. Chains SEWARD + SCF + RASSCF over `n_singlets` singlets + RASSCF over `n_triplets` triplets + per-group MS-CASPT2 + optional RASSI for SOC. Handles the EMIL JobIph plumbing: `>>COPY $Project.JobIph JOBxxx` after each RASSCF, then `>>COPY JOBxxx $Project.JobIph` before each per-group CASPT2 (without this swap, all CASPT2 groups silently read the last RASSCF's wave function). Generates the right RASSI block format (no `Title` keyword — Molcas RASSI rejects it). |
| `prepare_molcas_opt_freq_workflow` | **Thick orchestrator** for geometry-optimization + analytic-frequency runs. Wraps SEWARD + (SCF on iter 1 only) + RASSCF (CASSCF mode) + ALASKA + SLAPAF in an EMIL `>>> Do while <<<` ... `>>> ENDDO <<<` loop, followed by MCKINLEY + MCLR for analytic Hessian + harmonic frequencies. Supports SCF/HF/CASSCF/RASSCF, minimum or transition-state search (`transition_state=True` → adds `TS` to SLAPAF), freq-only single points (`do_optimization=False`), numerical-gradient fallback, and `iroot_freq` for state-averaged frequencies. |
| `compute_molcas_reaction_energy` | Post-hoc reaction-energy calculator. Takes signed stoichiometric coefficients on converged outputs (products + reactants) and returns ΔE in au / kcal/mol / eV. For atomization (1 molecule reactant, N atomic products) auto-emits `binding_energy_*` and `is_bound` fields. Honors `energy_kind` (primary / scf / rasscf / caspt2 / ms_caspt2 / rassi_sf / rassi_so) so the agent can force consistent level across species. |
| `check_molcas_active_space_consistency` | Diagnostic for multireference reaction energies. Compares a molecule's CAS dimensions (n_active_electrons, n_active_orbitals) to the SUM of its dissociation-fragment CASes. Verdicts: `consistent` / `molecule_undersized` / `fragments_undersized` / `char_mismatch`. If undersized, returns `suggested_cas=(M,N)` ready to feed into `prepare_molcas_casscf_setup`. Optional character check counts e.g. 'Cr 3d' active orbitals in molecule vs. fragments. Catches the textbook "CASSCF says CrO is unbound" trap before computing the energy. |

Bundled data:
- 133 Markdown docs at `chemtools/data/molcas/docs/` (programs, tutorials, users_guide, advanced_examples, installation, overview)
- 91 basis library files at `chemtools/data/molcas/basis_library/` (3-21G, 6-31G family, ANO-RCC, ANO-L, ANO-S, ANO-XS, AUG-CC-PVxZ, etc.)

Plugin layout (`chemtools/programs/molcas/`):
- `parse/output.py` — module-boundary task index + `parse_output_full` orchestrator
- `parse/scf.py`, `parse/rasscf.py`, `parse/caspt2.py`, `parse/mos.py` — module-specific parsers
- `parse/freq.py`, `parse/thermochem.py`, `parse/geometry.py` — MCLR freq + thermochem + SLAPAF trajectory parsers
- `parse/rassi.py` — RASSI module parser (spin-free + spin-orbit eigenstates, SOC matrix, dipole strengths)
- `binary/orbitals.py` — INPORB / RasOrb file reader (named-section format)
- `strategy/active_space.py` — `analyze_active_space` and `validate_caspt2_setup`
- `input/seward.py`, `input/scf.py`, `input/rasscf.py`, `input/caspt2.py` — block-level builders
- `input/draft.py` — `draft_molcas_input` orchestrator (InputSpec → full deck)
- `input/lint.py` — `lint_molcas_input` (block pairs, basis labels, RASSCF↔CASPT2 Frozen consistency, Nactel sanity)
- `input/basis_library.py` — bundled basis library reader (default contractions, label builder)
- `docs.py` — bundled docs accessor
- `runtime.py` — launch-helper (`prepare_launch` returns safe pymolcas command + env, with CASPT2 -np guard rail and scratch isolation)
- `_plugin_parser.py`, `_plugin_binary.py`, `_plugin_drafter.py` — sub-protocol implementations

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

### Molcas / multi-reference workflow rules

- **Always check active-space quality before CASPT2** — call `analyze_molcas_active_space` after RASSCF; do not run CASPT2 on a `poor` verdict
- **Reference weight ≥ 0.70 is the trust threshold** — `validate_molcas_caspt2_setup` returns `unreliable` below that; agent should redraft the active space, not the CASPT2 input
- **Real intruders need both small denominator AND large coefficient** — large coefficient alone is normal chemistry; small denominator alone is a benign near-degeneracy
- **IPEA shift defaults to 0.25 from Molcas 6.4** — `MOLCAS_NEW_DEFAULTS=YES` switches to 0.0; `parse_molcas_output` reports the actual value used so the agent can flag mismatches with the user's intent
- **MS / XMS / RMS / XDW for state-mixing** — when SS-CASPT2 has multiple closely-spaced states, `validate_molcas_caspt2_setup` emits the `multistate_hint` warning
- **Last `++ Molecular orbitals:` block wins** — `get_molcas_orbitals` automatically returns the RASSCF NOs (which override SCF MOs that appeared earlier in the same task); use this to label active orbitals via the `dominant_aos` field
- **Energy roll-up follows SO-RASSI > RASSI SF > MS-CASPT2 > CASPT2 > RASSCF root 1 > SCF** — `parse_molcas_output` returns `energy_summary.primary_energy_hartree` with the chosen label; SO ground state wins whenever a RASSI SPINorbit run is present
- **Internal pymolcas modules are filtered** — `last_energy`, `last_atoms`, and `emil` never show up in the task list
- **CAS dimensions across reactants must match** — for atomization / binding / dissociation at CASSCF, the molecule's CAS must be at least as large as the sum of the dissociation fragments' CASes (e.g. CrO needs CAS(10,9) to dissociate into Cr CAS(6,6) + O CAS(4,3) cleanly). Run `check_molcas_active_space_consistency` BEFORE `compute_molcas_reaction_energy` — a molecule-undersized verdict explains unphysical results like CASSCF-says-CrO-unbound by 13 kcal/mol
- **DKH is required for absolute energies on transition metals** — ANO-RCC is contracted for relativistic eigenfunctions; without `Relativistic R02O02` in SEWARD, Cr atom comes out ~9 au too high. The relativistic correction mostly cancels between reactant and product for binding energies, but mixing DKH and non-DKH across species is a bug. Either both or neither.
- **TM atomic ROHF often won't converge from GuessOrb** — for Cr ⁷S(3d⁵4s¹), skip SCF and run RASSCF directly (Molcas GuessOrb starting orbitals reach the right CASSCF minimum after a few iters). Then run `suggest_molcas_orbital_swaps` to confirm 3d-character orbitals are in active before trusting the energy.

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

## Server modes

The MCP server runs in one of three modes; the mode determines which tools are exposed
at `tools/list` time and gated at `tools/call` time. This lets the same package serve a
laptop user doing post-hoc analysis, a workstation user running NWChem locally, and an
HPC user submitting to a scheduler — without the agent ever seeing tools it cannot use.

| Mode | Tools visible | Use when |
|---|---|---|
| `analysis` | 128 | No NWChem executable available; post-hoc parsing (NWChem + Molcas), drafting (incl. Molcas inputs), planning, registry tracking of runs done elsewhere |
| `local` | 141 | NWChem runs as a subprocess on this machine (profile with `launcher.kind: "direct"`) |
| `hpc` | 144 | NWChem submitted to a scheduler (profile with `launcher.kind: "scheduler"`) |

### Selecting a mode

Priority order:
1. `chemtools-nwchem --mode {analysis|local|hpc}` CLI flag
2. `CHEMTOOLS_MODE` env var
3. **Auto-detect** (default):
   - `CHEMTOOLS_RUNNER_PROFILES` not set → `analysis`
   - profiles file has any `launcher.kind: "scheduler"` → `hpc`
   - profiles file has only `direct` profiles → `local`
   - profiles file unreadable or empty → `analysis` (logged)

Auto-detect means most users never configure mode explicitly; the existing
`CHEMTOOLS_RUNNER_PROFILES` env var carries the signal.

### Capability tags

Each tool is tagged via `@_tool("name", needs="...")`. Valid tags:

| Tag | Modes exposing it | Tools |
|---|---|---|
| `none` (default) | analysis, local, hpc | 87 pure-Python tools (parsing, drafting, suggest, docs, eval) |
| `registry` | analysis, local, hpc | 9 SQLite registry/campaign/workflow tools |
| `runner_profile` | local, hpc | 2 profile inspection/validation tools |
| `executable_or_scheduler` | local, hpc | 5 resource advisors that adapt to `launcher.kind` |
| `executable` | local, hpc | 6 job-execution tools (launch, watch, terminate) |
| `scheduler` | hpc | 3 scheduler-only tools (`render_job_script`, `detect_nwchem_hpc_accounts`, `suggest_nwchem_partition`) |

To add a new tool: tag it on the decorator. `needs="none"` is the default and is
also the right answer for the majority of tools.

### CLI debugging

```
chemtools-nwchem --show-mode        # print resolved mode + reason + blocked tools, exit
chemtools-nwchem --list-tools       # print tool names visible in the resolved mode, exit
chemtools-nwchem --mode analysis    # force analysis mode (e.g. profiles configured but doing post-hoc work)
```

At runtime, agents can call `get_server_mode` to introspect which mode they are in
and which tools are blocked — useful when a tool fails with a "not available in mode"
error or before recommending a workflow that needs HPC submission.

## Development Environment

- Install in editable mode: `pip install -e .`
- Entry points: `chemtools-nwchem`, `chemtools-nwchem-docs`
- Basis library: bundled at `chemtools/data/nwchem/basis_library/` (auto-detected after install)
- NWChem docs: bundled at `chemtools/data/nwchem/docs/` (29 text files, always available)
- Runner profiles: set `CHEMTOOLS_RUNNER_PROFILES` to your local YAML/JSON file
- Server mode: set `CHEMTOOLS_MODE` or pass `--mode` (see **Server modes** above; defaults to auto-detect)

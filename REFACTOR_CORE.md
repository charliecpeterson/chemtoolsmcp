# Core-library refactor: Phase 2+

Living plan for the **next-phase** chemtoolsmcp refactor — moving from
"NWChem and Molcas each implement their own version of the same logic"
to a clear core/ vs programs/<prog>/ separation, plus a `--programs` filter
on the MCP server so a session loads only the tools it needs.

Companion to `REFACTOR.md` (the Phase 1 plugin-architecture plan, mostly
done). This doc is specifically about **logic deduplication** and
**cross-program tool parity**.

## Why now

1. We have **2 programs** with overlapping logic; adding a 3rd (Molpro)
   or 4th (ORCA) to the current structure compounds the duplication —
   tools that should be one function would have to be implemented 4×.
2. Tool count is already 154; without consolidation, 4 programs ≈ 300
   tools, eating ~150K tokens of context just for schema definitions.
3. We've now dogfooded enough patterns (atomization, recovery, reaction
   energy, case analysis, geometry inspection) to **know** which
   abstractions are real and which are forced.
4. Two programs is the right time — abstractions designed against 2 real
   implementations are well-grounded; designing against 1 is premature,
   designing against 4+ has more legacy to migrate.

## Vision

```
                            ┌──────────────────┐
                            │   MCP tools      │
                            │  (programs/      │
                            │   {nwchem,molcas │
                            │    ,molpro,…}/   │
                            │    mcp_tools.py) │
                            └────────┬─────────┘
                                     │
                                ┌────┴─────┐
                                │          │
                       ┌────────▼──┐  ┌────▼──────────┐
                       │ generic   │  │ program-      │
                       │ MCP tools │  │ specific      │
                       │           │  │ MCP tools     │
                       │ • inspect │  │ • draft_*     │
                       │   geometry│  │ • parse_*     │
                       │ • compute │  │ • prepare_*   │
                       │   reaction│  │ • lint_*      │
                       │   energy  │  │               │
                       │ • analyze │  └─────┬─────────┘
                       │   case    │        │
                       │ • suggest │        │
                       │   recovery│        │
                       └─────┬─────┘        │
                             │              │
                             └──────┬───────┘
                                    │
                       ┌────────────▼────────────┐
                       │    core/ (logic)        │
                       │  geometry • thermochem  │
                       │  reaction_energy        │
                       │  recovery rule engine   │
                       │  case_analysis          │
                       │  basis suggest          │
                       │  registry • workflow    │
                       │  session • runner       │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ programs/<prog>/        │
                       │ (thin adapters)         │
                       │  parser • drafter       │
                       │  recovery_rules         │
                       │  basis_library          │
                       │  orchestrators (only    │
                       │   genuinely program-    │
                       │   specific workflows)   │
                       └─────────────────────────┘
```

**Key principle:** logic in `core/`, syntax in `programs/<prog>/`.

## What's already done (don't break)

| Already-shared | Location | Notes |
|---|---|---|
| Plugin registry | `chemtools/core/registry.py` | Programs auto-register on import |
| Plugin protocols | `chemtools/core/program.py` | Parser/Drafter/Strategist/BinaryReader/ExamplesCorpus |
| Server-mode filter | `chemtools/mcp/modes.py` | `analysis` / `local` / `hpc` with capability tags |
| Run registry | `chemtools/core/run_registry.py` | SQLite store (NWChem-only currently) |
| Common atoms / units | `chemtools/core/common.py` | COVALENT_RADII, periodic table, read_text |
| Cube file parser | `chemtools/core/cube.py` | Gaussian cube format |
| Eval framework | `chemtools/core/eval.py` | Case evaluation |
| Runner / launcher | `chemtools/core/runner.py` | Profile-based job submission |
| 17 already-generic MCP tools | `chemtools/mcp/tools/nwchem.py` | (see "Already-generic tools" below) |

### Already-generic MCP tools (no program in the name)

These are program-agnostic — work via `registry.resolve()` to dispatch:

```
append_session_log              compute_reaction_energy
basis_library_summary           draft_initial_geometry
get_server_mode                 init_session_log
next_versioned_path             parse_cube_file
preflight_check                 render_job_script
suggest_basis_set               suggest_memory
suggest_relativistic_correction suggest_resources
suggest_spin_state              summarize_run
watch_multiple_runs
```

So the *pattern* already exists. The work is to extend it consistently.

## Target architecture

### `chemtools/core/` — pure logic

```
core/
  common.py             constants, COVALENT_RADII, atom utilities
  types.py              cross-program TypedDicts (ParsedRun, etc.)
  program.py            Plugin Protocols (Parser, Drafter, …)
  registry.py           plugin registry + resolve(path)→plugin
  runner.py             launcher / scheduler glue
  run_registry.py       SQLite store (campaigns, workflows)
  cube.py               Gaussian cube format
  eval.py               case-evaluation framework

  # New (to be factored out of programs/molcas/strategy/*)
  geometry.py           ← distance/angle/dihedral/formula/bond detection
  reaction_energy.py    ← signed-stoichiometry math + ideal-gas thermochem
  thermochem.py         ← Sackur-Tetrode + atomic mass table
  recovery.py           ← generic rule-engine + apply-recovery framework
  case_analysis.py      ← summarize/analyze dispatcher
  basis_advisor.py      ← suggest-basis chemistry rules (per-element)
  workflow.py           ← DAG engine for multi-step protocols (NWChem has the
                          NWChem-specific version of this today; promote)
  session.py            ← session log helpers
```

### `chemtools/programs/<prog>/` — thin adapters

```
programs/<prog>/
  __init__.py           plugin registration
  plugin.py             Plugin class + sub-protocol bindings
  parse/                program-specific output parsers
    output.py
    geometry.py
    freq.py
    thermochem.py
    mos.py
    …
  input/                program-specific input renderers
    draft.py
    lint.py
    seward.py            (Molcas)
    geometry.py          (NWChem)
    …
  binary/               program-specific binary readers
  strategy/             program-specific strategies that DON'T generalize
    orchestrators.py    ← genuinely workflow-level (atomization, caspt2_chain)
    recovery_rules.py   ← list of rule tuples for this program
  data/                 bundled docs + basis library
```

### `chemtools/mcp/tools/<prog>.py` — tool wrappers

```
# A program-specific tool (cannot be made generic):
@_tool("draft_<prog>_input")
def _handle_draft(...): return programs.<prog>.draft_input(...)

# A generic tool (lives ONCE in core, called from any program's MCP):
@_tool("inspect_geometry")  # no program prefix
def _handle_inspect(args):
    plugin = registry.resolve(args["output_file"])  # auto-detect program
    geom = plugin.parser.get_geometry(args["output_file"])
    return core.geometry.inspect(geom["atoms"], **args)
```

## Per-tool classification: generic vs program-specific

For each existing tool, where does it belong?

### Definitely generic (move to core, single MCP tool)

| Tool | Currently in | Notes |
|---|---|---|
| `compute_reaction_energy` | Already generic | Currently dispatches via registry |
| `inspect_geometry` | `inspect_molcas_geometry` only | NWChem version missing — make one core tool |
| `summarize_output` | `summarize_molcas_output` only | NWChem version missing |
| `analyze_case` | `analyze_molcas_case` only | NWChem `analyze_nwchem_case` exists but program-specific; merge |
| `suggest_recovery` | `suggest_molcas_recovery` only | NWChem `suggest_nwchem_recovery` exists; merge framework |
| `apply_recovery` | `apply_molcas_recovery` only | Generic regex-edit framework |
| `extract_geometry` | both | Single dispatcher via `parser.get_geometry()` |
| `parse_thermochem` | both | Same shape; single dispatcher |
| `parse_frequencies` | both | Same shape; single dispatcher |
| `parse_trajectory` | both | Different shapes today; standardize then unify |
| `init_session_log` | already generic | — |
| `append_session_log` | already generic | — |
| `next_versioned_path` | already generic | — |
| `register_run` | `register_nwchem_run` only | Generalize + add Molcas support |
| `create_campaign` | `create_nwchem_campaign` only | Generalize |
| `get_campaign_energies` | `get_nwchem_campaign_energies` only | Generalize |
| `suggest_basis_set` | already generic | (NWChem-leaning; extend for Molcas) |
| `suggest_resources` | already generic | (NWChem-leaning; extend) |
| `render_job_script` | already generic | (NWChem-leaning; extend) |

### Program-specific (keep prefixed tools)

| Tool family | Reason |
|---|---|
| `draft_*_input` | Input deck syntax is fundamentally different |
| `lint_*_input` | Tied to input syntax |
| `parse_*_output` (deep parser) | Output text format is program-specific |
| Orchestrators (`prepare_*_atomization`, `prepare_*_caspt2_chain`, etc.) | Workflow DAGs differ; some concepts don't transfer |
| Binary readers (`parse_*_movecs`, `parse_*_inporb`, …) | Each program's binary format is different |
| Docs (`search_*_docs`, `lookup_*_module`) | Per-program doc corpora |
| `evaluate_*_case` (eval framework) | Case definitions are per-program |
| `xmldump` parser | Molcas-only file |
| `parse_movecs`, `parse_inporb`, `parse_rassi` | Binary / format per program |

### Probably generic (need a small adapter from each program's parser)

| Tool | Pattern |
|---|---|
| `inspect_geometry` | Plugin's `parser.get_geometry()` → core math |
| `summarize_output` | Plugin's `parser.parse_full()` → core summary builder |
| `analyze_case` | Plugin's `parser.parse_full()` + per-program rule list → core verdict |
| `compute_reaction_energy` | Plugin's `parser.energy_summary()` → core stoichiometry |
| `suggest_recovery` | Plugin's `recovery_rules` list → core rule engine |
| `check_active_space_consistency` | Plugin's parser → core CAS-dimension comparison |
| `prepare_atomization` | Plugin's drafter for atoms + molecule, core stoichiometry |

## Cross-program parity gaps ("placeholder tools")

Tools that exist for one program but not the other — capture them so we
know what to plumb when generalizing. After refactor, each becomes a
single generic tool that works for both programs.

### NWChem has, Molcas missing

| NWChem tool | Concept | Generalize path |
|---|---|---|
| `register_nwchem_run` | Track a run in the registry | Generic `register_run(program, …)` |
| `create_nwchem_campaign` | Group related runs | Generic `create_campaign(…)` |
| `get_nwchem_campaign_energies` | Sorted energy table | Generic `get_campaign_energies(…)` |
| `update_nwchem_run_status` | Mark completed/failed | Generic |
| `generate_nwchem_input_batch` | Template-vary across N inputs | Generic batch framework + per-program input renderer |
| `plan_nwchem_calculation` | Protocol picker | Generic protocol framework + per-program protocols |
| `plan_nwchem_workflow` | DAG builder | Generic DAG + per-program step recipes |
| `advance_nwchem_workflow` | Find ready steps | Generic |
| `create_nwchem_workflow` | Create DAG | Generic |
| `evaluate_nwchem_case` | Quality test | Generic eval framework + per-program cases |
| `tail_nwchem_output` | Tail log | Generic |
| `watch_nwchem_run` | Poll status | Generic + plugin-specific progress detector |
| `launch_nwchem_run` | Submit + wait | Generic (we now have `try_molcas_run_with_recovery` — generalize) |
| `terminate_nwchem_run` | Kill | Generic |
| `get_nwchem_run_status` | Running/done/failed | Generic + plugin-specific status reader |
| `review_nwchem_progress` | "Is this run going well?" mid-flight | Generic + plugin-specific phase detector |
| `detect_nwchem_hpc_accounts` | List user's allocations | Cluster-specific; generic |
| `suggest_nwchem_partition` | Pick queue | Cluster-specific; generic |
| `check_nwchem_memory_fit` | Pre-flight memory | Generic + plugin-specific memory estimator |
| `estimate_nwchem_freq_walltime` | Time estimate | Generic + plugin estimator |

### Molcas has, NWChem missing

| Molcas tool | Concept | Generalize path |
|---|---|---|
| `inspect_molcas_geometry` | Bond/angle/contact inspector | Generic `inspect_geometry` (just-built — promote to core) |
| `summarize_molcas_output` | Single-dispatch summary | Generic `summarize_output` |
| `analyze_molcas_case` | Quality dispatcher | Generic `analyze_case` (NWChem has `analyze_nwchem_case` — merge frameworks) |
| `prepare_molcas_atomization` | Atomization orchestrator | Generic `prepare_atomization` framework + plugin's draft/parse adapters |
| `compute_molcas_reaction_energy` | Reaction energy + thermochem | Generic `compute_reaction_energy` already exists; just route Molcas through it |
| `check_molcas_active_space_consistency` | CAS-size cross-check | Generic (multi-ref concept; can apply to NWChem MCSCF too) |
| `suggest_molcas_recovery` | Failure classifier | Generic `suggest_recovery` framework + per-program rules |
| `apply_molcas_recovery` | Auto-apply fix | Generic + per-program patcher |
| `try_molcas_run_with_recovery` | Auto-retry loop | Generic `try_run_with_recovery` (executable-tagged) |
| `parse_molcas_xmldump` | XML structured dump | Genuinely Molcas-specific (no NWChem equivalent) |

### Parity items both have (already mostly aligned)

| Concept | NWChem tool | Molcas tool | Status |
|---|---|---|---|
| Deep parser | `parse_nwchem_output` | `parse_molcas_output` | Keep separate (formats differ) |
| Task index | `parse_nwchem_tasks` | `parse_molcas_tasks` | Keep separate |
| Lint | `lint_nwchem_input` | `lint_molcas_input` | Keep separate |
| Drafter | `create_nwchem_input` | `draft_molcas_input` | Keep separate |
| Geometry extract | `extract_nwchem_geometry` | `extract_molcas_geometry` | Could unify |
| Thermochem parse | `parse_nwchem_thermochem` | `parse_molcas_thermochem` | Standardize shape → unify |
| Freq parse | (NWChem has it inside parse_nwchem_output) | `parse_molcas_frequencies` | Standardize shape → unify |
| Trajectory | `parse_nwchem_trajectory` | `parse_molcas_trajectory` | Standardize shape → unify |
| Active space | `parse_nwchem_mos` (TCE-focused) | `analyze_molcas_active_space` | Different enough — keep |
| Docs framework | `search_nwchem_docs`, `lookup_nwchem_block_syntax`, `find_nwchem_examples`, `get_nwchem_topic_guide`, `read_nwchem_doc_excerpt`, `list_nwchem_docs` | `search_molcas_docs`, `lookup_molcas_module`, `read_molcas_doc_excerpt`, `get_molcas_topic_guide`, `list_molcas_docs` | Underlying functions are similar — could share search/list framework |

## Server CLI changes

```
# Current
chemtools-nwchem [--mode {analysis|local|hpc}]

# New
chemtools [--mode {analysis|local|hpc}] [--programs PROG1,PROG2,…]

# Env-var equivalents
CHEMTOOLS_MODE=analysis CHEMTOOLS_PROGRAMS=molcas chemtools
```

Filter behavior:
- `--programs nwchem` → only NWChem tools (and generic tools)
- `--programs molcas` → only Molcas tools (and generic tools)
- `--programs nwchem,molcas` → both (current default if no flag passed)
- omitted: load all installed program plugins

Generic tools (no `_<prog>_` in name) **always** load — they dispatch to
whichever plugins are loaded based on input.

## Phased migration plan

### Phase 1 — rename + program filter (small)

- Rename `chemtools-nwchem` binary → `chemtools`
- Keep `chemtools-nwchem` as a deprecated alias for one release
- Add `--programs` flag + `CHEMTOOLS_PROGRAMS` env var
- Each tool registers with `program=<name>` metadata
- Filter at `tools/list` time (same pattern as `--mode`)
- Update `chemtools-nwchem-docs` → just remove (already deprecated)
- Tests: extend test_modes.py with program-filter cases

**No tool API changes. Backward-compatible.**

### Phase 2 — promote pure math to core/ (medium)

- New `core/geometry.py`: move from `molcas/strategy/geometry_inspector.py`
- New `core/thermochem.py`: move ideal-gas + atomic mass from
  `molcas/strategy/reaction_energy.py`
- New `core/reaction_energy.py`: generic stoichiometry math, accepts an
  energy_extractor callable
- Update Molcas strategy modules to import from core
- Update NWChem's existing `compute_reaction_energy` to use the new core
- No MCP tool changes (still program-prefixed where they were)

**Tests stay green. Just deduplicates code.**

### Phase 3 — promote frameworks to core/ (medium)

- `core/recovery.py`: generic rule engine + apply-recovery framework
- `core/case_analysis.py`: generic summarize/analyze dispatcher
- `core/workflow.py`: generic DAG engine (extract from NWChem's
  `protocols.py`)
- `core/session.py`: session log (already mostly there)
- `core/basis_advisor.py`: per-element basis recommendation rules
- Each program registers its own rule list / protocols / case suite
- Old `*_nwchem_*` and `*_molcas_*` versions become thin wrappers calling
  core with the program plugin

### Phase 4 — generic MCP tools with auto-detect (medium)

- New generic MCP tools (no program prefix):
  - `inspect_geometry`
  - `summarize_output`
  - `analyze_case`
  - `suggest_recovery`
  - `apply_recovery`
  - `try_run_with_recovery`
  - `extract_geometry`
  - `parse_thermochem`
  - `parse_frequencies`
  - `parse_trajectory`
  - `register_run`
  - `create_campaign`
  - `get_campaign_energies`
  - `tail_output`
  - `watch_run`
  - `terminate_run`
  - `launch_run`
- Each auto-detects program via `registry.resolve(file_path)`
- Old `*_nwchem_*` / `*_molcas_*` tools become aliases or get
  deprecation warnings

### Phase 5 — standardize parsed-output shape (medium)

- Both `parse_nwchem_output` and `parse_molcas_output` return the same
  top-level keys:
  - `metadata` (program, version, timestamp)
  - `task_payloads` (per-module breakdown)
  - `energy_summary` (with consistent field names)
  - `geometry` (atoms + units)
  - `frequencies` (modes + ZPVE)
  - `thermochem` (per-temperature)
  - `warnings`
- This lets generic tools consume parsed outputs without per-program
  branching

### Phase 6 — fill in cross-program parity (ongoing)

For each "placeholder" cross-program gap above, build the generic
version. Each program plugin contributes its specific bits (recovery
rules, draft adapters, status detectors).

## Backward compatibility strategy

- Phase 1: full backward compat (just rename + new flag)
- Phases 2-3: full backward compat (only internal refactoring)
- Phase 4: keep old tool names as aliases for one release; emit
  deprecation warning when called; remove in the release after that
- Phase 5: parsed-output shape backward-compatible (add new keys; don't
  remove old ones for one release)

## Tool count projection

| Phase | Total tools | NWChem-prefixed | Molcas-prefixed | Generic |
|---|---|---|---|---|
| Current | 154 | 97 | 40 | 17 |
| After phase 4 | ~130 | ~80 | ~30 | ~20 |
| Phase 4 + Molpro | ~170 | ~80 | ~30 | ~20 (+ Molpro ~40) |
| Phase 4 + Molpro + ORCA | ~210 | ~80 | ~30 | ~20 (+ Molpro ~40 + ORCA ~40) |

Without refactor, the same growth would reach ~280 tools (lots of
duplication across programs).

## Conventions

- **Generic tool** = no program in the name. Auto-detects via
  `registry.resolve(file_path)`.
- **Program-specific tool** = `<verb>_<prog>_<noun>`. Takes program-
  specific inputs (input deck text, drafter spec, etc.).
- **Core module** = `chemtools/core/<topic>.py`. No imports from
  `chemtools/programs/`.
- **Program module** = `chemtools/programs/<prog>/`. Can import from
  `core` but never from another program.
- **Plugin registers itself** on import via
  `chemtools.core.registry.register(plugin)`.

## Open questions

1. **Where does workflow / protocol logic live?** NWChem's `protocols.py`
   (CASSCF + thermochem flows) is currently NWChem-only. Should the DAG
   engine + protocol framework move to `core/workflow.py`, with each
   program contributing its own protocol catalog? Yes — Phase 3.

2. **How does `registry.resolve()` handle output files?** It already
   sniffs the first 32KB looking for program markers. We need to extend
   this so it returns a structured `{program, version, parser_callable,
   plugin}` so generic tools have everything they need in one lookup.

3. **What about input files (no program markers as obvious)?** Generic
   tools that act on input files would need an explicit `program` param
   or to do their own markers (e.g., `&SEWARD` → Molcas, `geometry` +
   `end` → NWChem). Probably fine for now: input tools stay program-
   prefixed since they're tied to syntax anyway.

4. **Test infrastructure** — when a new generic tool lands, we need
   tests across at least 2 programs so the abstraction stays honest.
   Add a `test_phase1/test_generic_tools.py` that exercises each
   generic tool against an NWChem AND a Molcas fixture.

5. **MCP tool descriptions** — generic tools should mention "auto-
   detects program from the output file" so the agent knows it doesn't
   need to specify.

## Status

| Phase | Status |
|---|---|
| Phase 1: rename + filter | **DONE 2026-05-12** — `chemtools` binary + `--programs` flag + `CHEMTOOLS_PROGRAMS` env var. Legacy `chemtools-nwchem` aliased for backward compat. 12 new tests in `test_modes.py` (30 total, all passing). |
| Phase 2: pure math to core/ | **DONE 2026-05-12** — new `core/units.py` (constants), `core/thermochem.py` (atomic masses + Sackur-Tetrode), `core/geometry.py` (distance/angle/dihedral/COM/bond detection/fragments/inspect). Molcas modules thinned to adapters: `reaction_energy.py` re-exports the atomic-mass table from core; `geometry_inspector.py` drops ~170 lines of math and is now just source resolution + bohr→Å normalization. All 30 tests still pass; H2O atomization numbers (ΔE=192.98, D_0=179.54, ΔH=167.09, ΔG=157.07, ΔS=33.62) reproduce exactly. |
| Phase 3: framework promotion | **PARTIAL DONE 2026-05-12** — `core/issues.py` (IssueCollector + severity tracking), `core/recovery.py` (generic rule-walker `dispatch_rules`), `core/case_analysis.py` (`classify_imaginary_modes`, `check_charge_spin_parity`, `bond_table_for_atoms`). Molcas `recovery.py` + `case_analysis.py` thinned to thin adapters. Still pending: `core/workflow.py` (DAG engine, needs NWChem protocol-engine refactor), `core/session.py`, `core/basis_advisor.py`. |
| Phase 4: generic MCP tools | **PARTIAL DONE 2026-05-12** — 5 new generic tools landed: `extract_geometry`, `parse_thermochem`, `parse_frequencies`, `parse_trajectory`, `inspect_geometry`. Each dispatches via `registry.resolve(path)` to the appropriate plugin's `parser.get_*()` method. Tool count: 159 (was 154). Mode counts: 142/156/159 (all 5 are `needs="none"`). Molcas plugin's `get_geometry` now normalizes bohr→Å so generic dispatchers don't need program-specific unit logic. NWChem-side fixture geometries are sparse (only opt+freq outputs have atoms); validated cross-program via Molcas H2O opt+freq (formula H2O, 2 covalent O-H bonds, 1 HOH angle). Tests: 32 total (was 30) — 2 new for Phase-4 tagging + auto-dispatch. Deferred: `summarize_output`, `analyze_case`, `suggest_recovery`/`apply_recovery` generics (need NWChem-side equivalents first). |
| Phase 4b: registry generalization | **DONE 2026-05-12** — `runs` table has a `program TEXT` column (idempotent migration via ALTER TABLE for existing DBs). `core.run_registry.register_run` + `list_runs` accept a `program` kwarg/filter; `get_campaign_energies` returns program per row. New generic `register_run` MCP tool (program='generic', needs='registry'). Legacy `register_nwchem_run` pre-fills program='nwchem'. End-to-end smoke test: a single campaign with NWChem CCSD(T) Cr atom + Molcas CASPT2 CrO returns a sortable energy table with mixed-program rows; list_runs(program='molcas') correctly filters. Tool count: 160 (was 159). 33 tests pass. |
| Phase 5: standardized parse shape | partially done (energy_summary aligned) |
| Phase 6: parity fill-in | tracked above |

Last updated: 2026-05-12.

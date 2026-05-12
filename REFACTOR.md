# Multi-program refactor

Living plan for restructuring chemtoolsmcp from a flat NWChem-centric layout
into a plugin-based mono-package that serves NWChem, Molcas, Molpro, and
future programs through dedicated per-app MCP entry points.

## Goal

Be the best chemistry AI assistant for parsing hundreds of lines of QC output,
designing inputs (active spaces, MO lists, basis sets), and driving local/HPC
jobs. Lean toward **thick tools, thin LLM** — push reasoning into deterministic
Python so Haiku-class models can drive workflows.

## Architecture

**Mono-package, per-app entry points.** One repo, one install. Separate
`chemtools-<name>` CLI binaries that each load only their program's tools.
Users who want multiple programs configure multiple MCP servers in their
client. No `chemtools-all` — keep tool lists focused per session.

**Plugin Protocol composition.** Each program plugin bundles sub-protocols:

```
Program
├── parser        Parser           (text output + drill-down sections)
├── drafter       Drafter          (input creation, lint, patch)
├── strategist    Strategist       (diagnose, recovery, resources, progress)
├── binary        BinaryReader | None   (movecs, hessian, fdrst, ...)
└── examples      ExamplesCorpus | None (bundled input templates)
```

Registered via `chemtools.core.registry.register(PLUGIN)` on import of
`chemtools.programs.<name>`. Detection sniffs the first 8 KB of an output file.

## Target directory layout

```
chemtools/
  core/                            program-agnostic
    types.py                       cross-program TypedDicts
    program.py                     plugin Protocols (Parser, Drafter, ...)
    registry.py                    program plugin registry
    run_registry.py                SQLite run/campaign/workflow store
    common.py                      atoms, units, file metadata
    cube.py                        Gaussian cube parser
    eval.py                        case-evaluation framework
    runner.py                      launcher/scheduler glue   (TODO)
    workflow.py                    DAG engine                (TODO)
    session.py                     session log               (TODO)
    basis.py                       library scan / resolve    (TODO)
    geometry.py                    geometry IO / manipulation (TODO)

  programs/
    nwchem/
      __init__.py                  assembles + registers NWCHEM
      docs.py    forum.py          bundled NWChem docs / forum search
      protocols.py                 pre-baked workflow recipes
      parse/    tasks  mos  freq  input  tce
      binary/   movecs (TODO from tce split)  hessian (NEW)  fdrst (NEW)
      input/    basis  _utils  scf  dft  tce  mcscf  freq  property  ...
      strategy/ diagnose  recovery  active_space  resources  progress
      examples/ NEW corpus
      data/     docs/ basis_library/   (already in chemtools/data/nwchem/)
    molcas/
      __init__.py                  scaffold only
      parse/output                 stub
    molpro/
      __init__.py                  scaffold only
      parse/output                 stub

  mcp/
    server.py                      JSON-RPC loop                (TODO)
    decorator.py                   @_tool registry              (TODO)
    modes.py                       capability/mode filter
    tools/
      nwchem.py                    NWChem-specific tool defs    (TODO)
      molpro.py    molcas.py                                    (TODO)
      shared.py                    program-neutral tools        (TODO)

  cli/                             entry points                 (TODO)
    nwchem.py    molpro.py    molcas.py

  data/                            program-neutral data
```

## Data shape conventions

Defined in `chemtools/core/types.py` (TypedDicts, serialize to JSON over MCP).

**Thick-tool envelope.** Tools whose job is to drive agent action return:

```
{
  "verdict":      {"label", "confidence", "reasons[]"},
  "next_actions": [{"tool", "params", "reason", "confidence", "priority"}],
  "anchors":      [{"kind", "message", "line", "file"}],
}
```

A Haiku-class agent should execute `next_actions[0]` without further reasoning.

**`parse_output` small-by-default.** Default returns task summaries + flat
`derived` dict + file-level `diagnosis` — fits in agent context for huge files.
Heavy sections load via `parser.get_orbitals(path, task_index)` etc.

**`kind` × `method`.** `kind` is the operation (energy / optimize / saddle /
frequency / gradient / property / dynamics / unknown). `method` is the theory
("DFT (PBE0)", "CCSD(T)", "CASSCF(8,8)", "TDDFT/B3LYP"). In NWChem terms,
`task <method> <operation>` decomposes directly.

**`TaskOutcome` enum.** `success` / `failed` / **`incomplete`** / `unknown`,
with a separate `has_usable_data` flag. `incomplete` is the orbitron-borrowed
state — task started but no clean completion marker — that current code
collapses into "failed".

**`InputSpec` is flat.** Program-specific knobs go in `program_options: dict`,
not in nested types.

**Examples corpus is tag-based.** `find_example(task, tags, methods)` for now;
embedding search later if needed.

## Conventions

- **Absolute imports for cross-subpackage references** — `from chemtools.programs.nwchem.parse.tasks import parse_tasks` from anywhere outside `parse/`. **Relative imports inside the same subpackage** — `from .tasks import ...` within `parse/`.
- **`git mv` then `git add -A`** for moves — keeps renames detected as renames.
- **TODO markers** in moved files where a follow-up split is needed.
- **One move per commit** during the refactor — easy revert, easy review.
- **Run `test_phase1/test_phase{2..6}.py`** after each move; 244 tests should stay green.
- **No new abstractions for "what if"** — every sub-protocol exists because a real file moved into it.

## Status

### Done (Phase 1 + Phase 2 + Phase 3 + Phase 4a-4b)

**Code organization:**
- `chemtools/core/`: types, program, registry, run_registry, common (with deduped COVALENT_RADII), cube, eval, runner
- `chemtools/programs/nwchem/`: plugin scaffold + Parser/Strategist wired, docs, forum, protocols, output, runner, parse/{tasks, mos, freq, input, tce}, strategy/diagnose, input/{basis, basis_library, _utils}
- `chemtools/programs/molcas/`: full Parser (parse_output / task_index / get_orbitals), Drafter (draft_input / lint_input), parse/{output, scf, rasscf, caspt2, mos}, binary/orbitals (INPORB / RasOrb reader), strategy/active_space (analyze_active_space + validate_caspt2_setup), input/{seward, scf, rasscf, caspt2, draft, lint, basis_library}, docs.py (133 bundled OpenMolcas docs), 91 basis library files
- `chemtools/programs/molpro/`: scaffold + Parser (partial), parse/output
- `chemtools/programs/_adapter_helpers.py`: shared TaskSummary / Diagnosis adapters
- `chemtools/nwchem.py` back-compat shim removed; all callers use direct imports.
- ~17,000 LOC migrated. 244/244 smoke tests green.

**Plugin sub-protocols wired:**

| Plugin | parser | drafter | strategist | examples | binary |
|---|---|---|---|---|---|
| NWCHEM | full (8/8) | full (draft / lint / **patch** all wired) | minimal (4/4) | 8 templates (DFT energy/opt/freq, CCSD(T), open-shell Fe, MCSCF, TDDFT, COSMO) | hessian + movecs |
| MOLPRO | parse-only (3/8) | — | — | — | — |
| MOLCAS | full (7/8: parse_output / task_index / get_orbitals / get_frequency / get_thermochem / get_geometry / get_trajectory) + RASSI specialty parser | draft + lint (no patch) | — | — | inporb |

- `registry.resolve(file)` correctly dispatches NWChem / Molpro / Molcas outputs (detection window enlarged to 32KB; "echo of input deck" added as an early signal).

**Working end-to-end:**

```python
from chemtools.core import registry

plugin = registry.resolve(program=None, path=output_file)   # auto-detect

# Read existing outputs
parsed = plugin.parser.parse_output(output_file)            # ParsedRun
diagnosis = plugin.strategist.diagnose(parsed)              # Diagnosis (NWChem only)

# Draft new inputs from a 5-line spec
text = plugin.drafter.draft_input({
    "atoms": [...], "charge": 0, "multiplicity": 1,
    "method": "DFT", "functional": "b3lyp", "basis": "def2-svp",
    "task": "energy", "title": "water test",
})
issues = plugin.drafter.lint_input(text)

# Pull a curated template
template = plugin.examples.find_example(task="energy", methods=["B3LYP"])
example_text = plugin.examples.read_example(template["name"])
```

**Exposed as MCP tools (145 total: 114 NWChem + 31 Molcas):**

| Tool | What it does |
|---|---|
| `summarize_run` | One-call dispatch via `registry.resolve`. Returns combined `ParsedRun + Diagnosis` for any registered program. |
| `prepare_nwchem_tce_setup` | Thick orchestrator: parse MOs + freeze count + ordering check + swap suggestions + draft routing, with a `Diagnosis` envelope telling the agent exactly what to do next. |
| `prepare_nwchem_mcscf_setup` | Multireference analogue of the TCE orchestrator. Returns a recommended CAS(M,N) window, frontier-assessment verdict, and routed `next_actions` (draft directly, inspect more orbitals, or fix state mismatch first via vectors swap). |
| `parse_molcas_output` | Deep Molcas parse: per-module SCF / RASSCF / CASPT2 details + active-space summary + warnings. Primary energy follows MS-CASPT2 > CASPT2 > RASSCF > SCF. |
| `parse_molcas_inporb` | Greenfield reader for INPORB / RasOrb / ScfOrb / GssOrb / LprOrb / SpdOrb files (named-section format). Returns per-symmetry coefficients + occupations + orbital-energy + typeindex partitioning. |
| `analyze_molcas_active_space` | NO-occupation classification → healthy/marginal/poor verdict + promote/demote orbital lists + next_actions envelope. Accepts either output OR orbital file. |
| `validate_molcas_caspt2_setup` | Reference-weight, IPEA / shift, intruder-state checks → healthy/caution/unreliable verdict; flags real intruders (small denom AND large coeff) only. |
| `draft_molcas_input` | Render a full Molcas input deck (MOLCAS_MEM + SEWARD + SCF + RASSCF + CASPT2 chain) from a single InputSpec. Auto-derives Occupied for C1, applies basis-library defaults, includes auto-lint pass on the result. |
| `lint_molcas_input` | Validate input string: block-pair consistency, basis library/element existence, RASSCF/CASPT2 Frozen consistency, Nactel sanity, LumOrb provenance, Spin sanity. |
| `compute_molcas_active_space_partition` | Translate CAS(M,N) → per-symmetry RASSCF directives. Catches "inactive electrons not even" and "frozen+inactive+active exceeds basis" early. |

### Deferred (need real splits)

| File | LOC | Plan |
|---|---|---|
| `api.py` | 28 | Public API aggregator — touching means redesigning `chemtools.X` user-facing paths. Defer to after splits land. |
| ~~`nwchem.py` shim~~ | ~~8~~ | **DONE** — removed; COVALENT_RADII deduped to `core/common.py`; all callers use direct imports. |
| ~~`basis.py`~~ | ~~682~~ | **DONE** — whole-file move to `programs/nwchem/input/basis_library.py`. Format-neutral pieces (list_basis_sets, normalize_element_symbol, PERIODIC_SYMBOLS) marked with TODO for lift to `core/basis.py` when a second program ships a library reader. |
| ~~`runner.py`~~ | ~~1376~~ | **DONE** — whole-file move to `core/runner.py` with detailed TODO covering: (a) rename NWChem-named publics (run_nwchem → run_calculation, etc.), (b) extract `_build_nwchem_progress_summary` to `programs/nwchem/strategy/progress.py`, (c) accept `progress_summary_fn` callback. |
| ~~`api_runner.py`~~ | ~~1475~~ | **DONE** — whole-file move to `programs/nwchem/runner.py` with TODO marker that the MCP-tool wrappers should later move to `mcp/tools/nwchem.py` and the session-log helpers should lift to `core/session.py`. |
| ~~`api_output.py`~~ | ~~628~~ | **DONE** — whole-file move to `programs/nwchem/output.py` with TODO marker. Cross-program parse_tasks/parse_mos dispatchers came along; they'll lift to `mcp/tools/shared.py` once a second program has substantive coverage. |
| `api_input.py` | **4170** | Big split into `programs/nwchem/input/{scf, dft, tce, mcscf, freq, property, geometry}.py`. Multi-session. |
| `api_strategy.py` | **4353** | Big split into `programs/nwchem/strategy/{recovery, active_space, resources, progress, freq_check, geometry_check}.py`. Multi-session. |
| `mcp/nwchem.py` | **4797** | Last big split. Server framework → `mcp/server.py` + `mcp/decorator.py`; NWChem-specific tool defs → `mcp/tools/nwchem.py`; program-neutral tools → `mcp/tools/shared.py`. |

### Open TODOs (marked inline with `TODO(multi-program)`)

- ~~`programs/nwchem/parse/tce.py` — split out binary movecs IO~~ DONE (Phase 9b — binary/movecs.py).
- `programs/nwchem/protocols.py` — lift DAG engine to `core/workflow.py`. Deferred (YAGNI; no second program needs the DAG engine yet).
- ~~`core/run_registry.py::_apply_change` and `generate_input_batch`~~ DONE (Phase 9e — generate_input_batch dispatches via `registry.resolve(program).drafter.patch_input`; `_apply_change` kept as safety-net fallback).
- ~~PERIODIC_SYMBOLS dedup~~ DONE (Phase 9c — `PERIODIC_SYMBOLS = set(ATOMIC_SYMBOLS.values())`).
- `programs/nwchem/input/basis_library.py` — format-neutral file-discovery pieces (list_basis_sets, _scan_basis_library, normalize_element_symbol) lift to `core/basis.py` when a second program needs them. Deferred (YAGNI).
- `core/runner.py` — partial: ~~program-neutral aliases (run_calculation, render_calculation_run, inspect_run_status, watch_run)~~ DONE (Phase 9d). ~~progress_summary_fn callback parameter on inspect/watch~~ DONE. Full extraction of `_build_nwchem_progress_summary` + 5 helpers (316 LOC) to `programs/nwchem/strategy/progress.py` deferred — only worth doing when a second program ships its own progress builder.

## Recommended order for remaining work

1. ~~**`basis.py`**~~ — done.
2. ~~**`runner.py`**~~ — done as a whole-file move.
3. ~~**`api_runner.py` move**~~ — done as a whole-file move to `programs/nwchem/runner.py`.
4. ~~**`api_output.py` migration**~~ — done as a whole-file move to `programs/nwchem/output.py`.
5. ~~**NWChem shim removal**~~ — done; COVALENT_RADII deduped, direct imports everywhere.
6. ~~**Wire NWChem Parser sub-protocol**~~ — done (Phase 4a).
7. ~~**Wire NWChem Strategist sub-protocol**~~ — done (Phase 4a).
8. ~~**Wire Molpro/Molcas Parser sub-protocols + extract shared adapter helpers**~~ — done (Phase 4b).
9. ~~**Wire NWChem Drafter sub-protocol**~~ — done (Phase 4c). `draft_input` + `lint_input` work; `patch_input` is NotImplementedError until api_input.py splits.
10. ~~**Build NWChem ExamplesCorpus**~~ — done (Phase 4d). 4 starter templates bundled; user adds more over time.
11. **Binary readers** — partly done:
    - ~~`parse_nwchem_hessian`~~ — done (Phase 8a). NWChem `.hess` is actually ASCII (Fortran D-exponent floats, lower-triangle packed). Returns dimensions, matrix, sanity stats. Wired through new BinaryReader sub-protocol on the NWCHEM plugin.
    - ~~`compute_nwchem_harmonic_frequencies`~~ — done (Phase 8b). Companion that mass-weights, eigendecomposes, and converts to cm^-1. Bit-exact match to NWChem-printed frequencies on water / methane / ammonium / CO2 / H2O2-TS fixtures.
    - `parse_nwchem_drv_hessian` deferred — `.drv.hess` is internal driver state (BFGS update buffer), no documented format.
    - `parse_nwchem_fdrst` deferred — `.fdrst` byte layout is not externally documented; risk of misinterpretation. Defer until a workflow concretely needs it.
12. ~~**Active space design tool**~~ — done (Phase 4f). `prepare_nwchem_tce_setup` MCP tool.
13. ~~**MCP tool that dispatches through plugins**~~ — done (Phase 4e). `summarize_run`.
14. ~~**Extend MCSCF orchestrator**~~ — done (Phase 4g). `prepare_nwchem_mcscf_setup` MCP tool with frontier-aware routing.
15. ~~**Expand examples corpus**~~ — done (Phase 4h). 4 → 8 starter templates covering DFT, opt, freq, CCSD(T), open-shell Fe, MCSCF, TDDFT, COSMO.
16. ~~**`api_input.py` split**~~ — done. 11 families carved out (95% of original LOC moved):
    - cube → `programs/nwchem/input/cube.py` (5a)
    - SCF recovery + property check → `scf_recovery.py` (5b)
    - MCSCF → `mcscf.py` (5c)
    - TCE → `tce.py` (5d, biggest at 1047 LOC)
    - imaginary-mode handling → `imaginary_modes.py` (5e)
    - optimization follow-up → `opt_followup.py` (5f)
    - DFT workflow → `dft.py` (5g)
    - geometry helpers → `geometry.py` (5h)
    - lint + restart → `lint_restart.py` (5i)
    - general drafters (`create_nwchem_input`, `_variant`, `review_request`) → `general.py` (5j)
    - workflow planner (`prepare_nwchem_next_step`, `plan_nwchem_workflow`) → `programs/nwchem/strategy/workflow_planner.py` (5k)
    api_input.py: 4170 → 211 LOC (3959 moved). Remaining content is module-level imports, tiny stem-match helpers, and back-compat re-export blocks.
17. ~~**`api_strategy.py` split**~~ — done. 8 families carved out (98% of original LOC moved):
    - SCF/state recovery → `strategy/recovery.py` (6a)
    - MCSCF active-space advisor → `strategy/mcscf_active_space.py` (6b)
    - HPC resource advisors → `strategy/hpc_resources.py` (6c)
    - Resource sizing → `strategy/resources.py` (6d)
    - Plausibility checks → `strategy/plausibility.py` (6e)
    - Case review (check_spin_charge_state + summarize/review + MCSCF review) → `strategy/case_review.py` (6f)
    - Pre-job input advisors (suggest_spin_state / basis / relativistic) → `strategy/input_advisors.py` (6g)
    - Workflow state + preflight + freq restart → `strategy/workflow_state.py` (6h)
    api_strategy.py: 4353 → 89 LOC. The file is now a back-compat shim of imports + re-exports.
18. ~~**`mcp/nwchem.py` split**~~ — done. Three carves:
    - Tool decorator + shared registries → `mcp/decorator.py` (7a)
    - Protocol I/O helpers + standard arg parser → `mcp/server.py` (7b)
    - `tool_definitions()` + all 112 `@_tool` handlers + `_TOOL_ALIASES` + `dispatch_tool` + `handle_request` → `mcp/tools/nwchem.py` (7c)
    `mcp/nwchem.py`: 4983 → 105 LOC. Now just the chemtools-nwchem entry point (serve() + main() + arg parser). Per-program entry points (chemtools-molpro, chemtools-molcas) can drop in alongside it without copying any framework code.
19. **CLI entry points** — `chemtools-molpro`, `chemtools-molcas` (require `mcp/tools/<program>.py` to exist first).
20. **Thicken thin tools** — substantially done. **11 analysis tools** now carry the `next_actions[]` envelope:
    - `analyze_nwchem_frontier_orbitals` — routes on `analysis.assessment` (metal-state mismatch → swap; clean → extract energy).
    - `check_nwchem_spin_charge_state` — routes on `assessment` + state_check_assessment + SOMO count mismatch.
    - `check_nwchem_geometry_plausibility` — routes on `plausible` + `red_flags` (broken → opt followup; warnings → freq plausibility).
    - `compare_nwchem_runs` — routes on `overall_assessment` (better → confirm; worse → recovery).
    - `parse_nwchem_tce_output` — routes on `multireference_diagnostics.mr_assessment` (strong MR → MCSCF; moderate → CCSD(T); clean → thermochem).
    - `parse_nwchem_movecs` — routes on whether occupied-orbital eigenvalues are monotonic (bad → MOs + setup; sane → TCE setup).
    - `validate_nwchem_tce_setup` — routes on status + error_codes (forbidden freeze → TCE setup orchestrator; ok → launch).
    - `parse_nwchem_freq_progress` — routes on pct_complete + fdrst (≥99% → check plausibility; restart needed → prepare_freq_restart; running → watch).
    - `review_nwchem_progress` — routes on overall_status + intervention block (terminate-recommended → terminate + recovery).
    - `summarize_nwchem_electronic_structure` — routes on spin_state_consistent + metal_centers (inconsistent → state check; clean → thermochem).
    - `track_nwchem_spin_state` — routes on flip_detected (yes → vectors swap + recovery; warnings → state check; stable → freq plausibility).
    Inherited bug fixed along the way: `check_nwchem_geometry_plausibility` had been broken since the case_review carve-out — `_MAX_COORD` / `_LANTHANIDES` / `_TYPICAL_COORD` had gone to `case_review.py` by mistake. Moved back to `plausibility.py`.
    Tools still thin: pure-data parsers (parse_nwchem_thermochem, parse_nwchem_population_analysis, parse_nwchem_scf, parse_nwchem_mos, parse_cube_file, ...) — they don't have a strong "next action" signal beyond "data extracted, continue workflow"; lower priority. Most of the high-leverage analysis surface is now thick.

After all of the above:

8. **Wire the Program plugin sub-protocols** — assemble `parser`, `drafter`, `strategist` on each plugin from the now-organized submodules. MCP tools become thin dispatchers through `registry.resolve(...).parser.parse_output(...)`.
9. **`cli/<program>.py` entry points** — separate MCP CLIs that load only their program's tools.
10. **Binary readers** — `parse_nwchem_hessian`, `parse_nwchem_fdrst`. Unlocks TS-from-Hessian and intelligent freq restart.
11. **Examples corpus** — bundle Charlie's typical workflow templates with tag-based discovery.
12. **Thicken thin tools** — extend `{verdict, next_actions[]}` envelope to ~30 analysis tools still returning raw data. Biggest small-LLM payoff.
13. **Active space design tool** — `prepare_active_space(scf_output, target_method, expected_somos)` collapses 5-6 LLM reasoning steps into one call.

## Decisions log

- **Plugin pattern**: Protocol with instance (not ABC).
- **Composition over flat Program**: each plugin holds sub-protocol instances.
- **`parse_output` is small**, drill-down tools fetch heavy sections.
- **Tool list scope** per-session via per-app CLI entry, not via mode flags.
- **Cross-program tools deferred** — defer until after a second program is fully built out.
- **Examples corpus = tag-based**, no embeddings yet.
- **Mono-package**, not separate npm-style packages per program.
- **TaskKind enum is operations only**; theory goes in a separate `method` field.

## Test gate

Every move must keep these passing:

```
python3 test_phase1/test_phase2.py   # 52 tests
python3 test_phase1/test_phase3.py   # 70 tests
python3 test_phase1/test_phase4.py   # 66 tests
python3 test_phase1/test_phase5.py   # 25 tests
python3 test_phase1/test_phase6.py   # 31 tests
```

Total: 244 tests. `tests/test_chemtools_mcp.py` has pre-existing mode-filter
debt (51 failures on unmodified HEAD); not a gate.

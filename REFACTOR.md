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

### Done (Phase 1 + Phase 2 easy moves)

- `chemtools/core/`: types, program, registry, run_registry, common, cube, eval
- `chemtools/programs/nwchem/`: plugin scaffold, docs, forum, protocols, parse/{tasks, mos, freq, input, tce}, strategy/diagnose, input/{basis, _utils}
- `chemtools/programs/molcas/`: scaffold + parse/output
- `chemtools/programs/molpro/`: scaffold + parse/output
- All three plugins register on import; detection routes correctly.
- ~10,000 LOC migrated. 244/244 smoke tests green.

### Deferred (need real splits)

| File | LOC | Plan |
|---|---|---|
| `api.py` | 28 | Public API aggregator — touching means redesigning `chemtools.X` user-facing paths. Defer to after splits land. |
| `nwchem.py` shim | 8 | Remove once `COVALENT_RADII` (defined in 3 parse modules) is deduped and call sites use direct imports. |
| `basis.py` | 682 | Split: library scan → `core/basis.py`; NWChem renderers → `programs/nwchem/input/basis.py`. |
| `runner.py` | 1376 | Split: launcher/scheduler → `core/runner.py`; NWChem `_build_nwchem_progress_summary` → `programs/nwchem/strategy/progress.py`. Public symbol rename `run_nwchem` → `run_calculation`. |
| `api_runner.py` | 1475 | Paired with `runner.py`. Session log helpers → `core/session.py`; rest splits across `core/runner.py` and `mcp/tools/`. |
| `api_output.py` | 628 | Mostly migrates to `mcp/tools/nwchem.py`; thin cross-program dispatch into `mcp/tools/shared.py`. |
| `api_input.py` | **4170** | Big split into `programs/nwchem/input/{scf, dft, tce, mcscf, freq, property, geometry}.py`. Multi-session. |
| `api_strategy.py` | **4353** | Big split into `programs/nwchem/strategy/{recovery, active_space, resources, progress, freq_check, geometry_check}.py`. Multi-session. |
| `mcp/nwchem.py` | **4797** | Last big split. Server framework → `mcp/server.py` + `mcp/decorator.py`; NWChem-specific tool defs → `mcp/tools/nwchem.py`; program-neutral tools → `mcp/tools/shared.py`. |

### Open TODOs (marked inline with `TODO(multi-program)`)

- `programs/nwchem/parse/tce.py` — split out binary movecs IO into `programs/nwchem/binary/movecs.py`
- `programs/nwchem/protocols.py` — lift DAG engine to `core/workflow.py`
- `core/run_registry.py::_apply_change` and `generate_input_batch` — NWChem text rewriters belong in `programs/nwchem/input/` and dispatch via `program.drafter.patch_input`

## Recommended order for remaining work

1. **`basis.py` split** (~2 hours). Smallest remaining real split; lands `core/basis.py`.
2. **`runner.py` + `api_runner.py`** paired refactor (~half day). Public symbol renames are the design call here.
3. **`api_output.py` migration** (~half day). Mostly moves to `mcp/tools/`.
4. **NWChem shim removal** (~half day). Dedup `COVALENT_RADII`, update call sites, delete `chemtools/nwchem.py`.
5. **`api_input.py` split** (multi-session, family-by-family).
6. **`api_strategy.py` split** (multi-session, family-by-family).
7. **`mcp/nwchem.py` split** (depends on 5+6 being underway so tool defs land in clean families).

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

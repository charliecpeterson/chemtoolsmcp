# chemtools

Core Python package for the chemtoolsmcp toolkit. Lives at
`chemtools/` and is installed as the `chemtools-mcp` distribution
(top-level binary: `chemtools`).

For the full project overview see [`../README.md`](../README.md) and
[`../CLAUDE.md`](../CLAUDE.md).

## Layout

```
chemtools/
  core/                          Program-agnostic infrastructure
    registry.py                  QC-program plugin registry (auto-detect from output)
    program.py                   Parser / Drafter / Strategist / BinaryReader protocols
    runner.py                    Generic SLURM/PBS submit + render + watch + status
    run_registry.py              SQLite registry (runs, campaigns, workflows)
    eval.py                      Multi-program case evaluator
    types.py                     ParsedRun / TaskSummary / GeometryAtom typed dicts

  programs/<name>/               Per-program plugins
    nwchem/                      97 tools — input drafting, TCE, freq restart, HPC
    molcas/                      44 tools — CASSCF/CASPT2 chain, recovery rule engine
    dirac/                       38 tools — 4c/X2C, AOC + KPSELE, Cm-class workflow
    grasp/                       37 tools — multi-exe atomic workflows, hf bootstrap

  data/<name>/                   Bundled per-program data
    nwchem/basis_library/        608 basis files
    nwchem/docs/                 29 markdown docs
    molcas/basis_library/        91 basis files
    molcas/docs/                 133 markdown docs
    dirac/docs/                  179 markdown docs
    grasp/docs/                  15 markdown docs

  mcp/                           MCP server
    cli.py                       Entry point — main() / serve() / arg parsing
    server.py                    JSON-RPC transport
    dispatch.py                  tool_definitions() + dispatch_tool + handle_request
    decorator.py                 @_tool decorator + shared registries
    modes.py                     Mode + program-filter logic
    tools/                       Per-program tool definitions + handlers
      generic.py                 36 program-agnostic tools
      nwchem.py / molcas.py / dirac.py / grasp.py
```

## Entry points

Installed by `pip install -e .`:

| Binary | Module | Purpose |
|---|---|---|
| `chemtools` | `chemtools.mcp.cli:main` | Primary MCP server (all programs) |
| `chemtools-nwchem` | `chemtools.mcp.cli:main_legacy_nwchem` | Legacy alias (deprecation hint on stderr) |
| `chemtools-nwchem-docs` | `chemtools.mcp.nwchem_docs:main` | Standalone NWChem docs server (back-compat) |

## Quick smoke test

```bash
chemtools --show-mode      # mode + program filter + tool count (JSON)
chemtools --list-tools     # tool names visible under active mode + filters
chemtools --mode analysis  # force analysis mode (no execution capability)
chemtools --programs molcas  # only Molcas tools visible
```

Programmatic introspection:

```python
from chemtools.mcp.tools.nwchem import tool_definitions
defs = tool_definitions()
print(f"{len(defs)} tools registered")
```

## Plugin contract

Each program in `chemtools/programs/<name>/` registers a `Program` plugin
with `chemtools.core.registry` so generic tools (e.g. `parse_output`,
`summarize_output`) can dispatch to it. The protocol surface is in
`chemtools/core/program.py`:

| Sub-protocol | Required | Used by |
|---|---|---|
| `Parser`     | yes | `parse_output`, `summarize_output`, `extract_geometry`, etc. |
| `Drafter`    | optional | program-specific input builders |
| `Strategist` | optional | recovery / diagnosis / case-review tools |
| `BinaryReader` | optional | binary checkpoint readers (movecs, hessian, h5) |
| `Examples`   | optional | example-input registry |

Each program also has an `MCP module` at `chemtools/mcp/tools/<name>.py`
that exports `<name>_tool_definitions()` (consumed by `dispatch.py`'s
aggregator) and decorates handler functions with `@_tool(name, program=..., needs=...)`.

## Runner profiles

Job-launch behavior is governed by a profile file pointed at by the
`CHEMTOOLS_RUNNER_PROFILES` env var. Example profiles for several
machines are in [`../examples/`](../examples/):

- `local_workstation/` — direct subprocess + apptainer containers
- `tacc_stampede3/` — TACC Stampede3 SLURM profiles for all 4 programs

See [`../CLAUDE.md`](../CLAUDE.md) for the full schema and HPC workflows.

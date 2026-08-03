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
    runner.py                    Legacy render/run plus compatibility imports
    monitoring.py                Shared polling, terminal checks, and watch history
    run_records.py               SQLite run records + execution links
    run_registry.py              Compatibility facade + campaigns/workflows
    slurm.py                     Typed Slurm status results and evidence
    eval.py                      Multi-program case evaluator
    types.py                     ParsedRun / TaskSummary / GeometryAtom typed dicts

  application/                   Permission and workflow coordination
    execution_monitoring.py       Shared owned-status projection and polling
    execution_policy.py          Permission decisions and public service errors
    execution.py                 Launch, status, cancellation, and ownership
    dirac_monitoring.py           Typed DIRAC local and Slurm watching
    grasp_monitoring.py           Typed GRASP workflow local and Slurm watching
    molcas_monitoring.py          Typed Molcas local and Slurm watching
    nwchem_monitoring.py         Typed NWChem local and Slurm watching

  execution/                     Program-neutral execution adapters
    _common.py                   Command rendering, root checks, and staging
    local.py                     Local launch, status, completion, and signals
    slurm.py                     Slurm scripts, status, submission, and cancellation
    executors.py                 Compatibility imports for the split adapters
    launch_registry.py           Persistent launch state and run links
    legacy_profiles.py           Version 1 profile loading and typed conversion
    legacy_status.py             Unowned process, scheduler, and file status

  programs/<name>/               Per-program plugins
    nwchem/                      101 tools: input drafting, TCE, freq restart, HPC
    molcas/                      45 tools: CASSCF/CASPT2 chain, recovery rules
    dirac/                       39 tools: 4c/X2C, AOC, and Cm-class workflows
    grasp/                       51 tools: multi-executable atomic workflows
    qe/                          20 tools: pw.x input/output and QE-to-QMCPACK review
    qmcpack/                     14 tools: input, HDF5 metadata, and QMC analysis

  data/<name>/                   Bundled per-program data
    nwchem/basis_library/        608 basis files
    nwchem/docs/                 29 markdown docs
    molcas/basis_library/        91 basis files
    molcas/docs/                 133 markdown docs
    dirac/docs/                  179 markdown docs
    grasp/docs/                  15 markdown docs

  mcp/                           MCP server
    cli.py                       Entry point — main() / serve() / arg parsing
    catalog.py                   Built-in program and tool-module membership
    server.py                    JSON-RPC transport
    dispatch.py                  tool_definitions() + dispatch_tool + handle_request
    decorator.py                 @_tool decorator + shared registries
    modes.py                     Mode + program-filter logic
    tools/                       Per-program tool definitions + handlers
      generic.py                 56 low-level program-agnostic tools
      guided.py                  Guided cross-program workflow tools
      nwchem.py / molcas.py / dirac.py / grasp.py / qe.py
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
from chemtools.mcp.dispatch import tool_definitions
defs = tool_definitions()
print(f"{len(defs)} tools registered")
```

## Plugin contract

Each package under `chemtools/programs/<name>/` now exports a validated
`ProgramBackend` without changing global registry state. The built-in catalog
owns program membership, registers those backends at MCP composition time, and
names each program's MCP tool-definition provider.

`chemtools/core/program.py` also contains the new operation-level
`ProgramCapability` and `ProgramBackend` contract:

| Backend field | Declared operations |
|---|---|
| `parser` | Output, task, geometry, orbital, frequency, trajectory, thermochemistry, and input parsing |
| `inputs` | Input drafting, linting, and patching |
| `binary` | Program-specific binary reads and writes |
| `diagnostics` | Diagnosis and recovery advice |
| `resources` | Target-aware resource estimates |
| `progress` | In-progress output inspection |
| `consistency` | Input-output and related-artifact checks |
| `examples` | Curated example listing and reading |

Provider presence does not advertise an operation. Each backend declares the
capabilities it supports, and registration checks those declarations against
callable provider methods.

Each program also has an MCP module at `chemtools/mcp/tools/<name>.py`.
`chemtools.mcp.catalog` names its definition provider, and importing that
module registers handlers with `@_tool(name, program=..., needs=...)`.

## Runner profiles

Job-launch behavior is governed by a profile file pointed at by the
`CHEMTOOLS_RUNNER_PROFILES` env var. Example profiles for several
machines are in [`../examples/`](../examples/):

- `local_workstation/` — direct subprocess + apptainer containers
- `tacc_stampede3/` — TACC Stampede3 SLURM profiles for all 4 programs

See [`../CLAUDE.md`](../CLAUDE.md) for the full schema and HPC workflows.

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
- Current tool count: 49

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

## Development Environment

- Install in editable mode: `pip install -e .`
- Entry points: `chemtools-nwchem`, `chemtools-nwchem-docs`
- Basis library: bundled at `chemtools/data/nwchem/basis_library/` (auto-detected after install)
- Docs server requires a local NWChem docs checkout; set `NWCHEM_DOCS_ROOT` to point at it
- Runner profiles are per-machine config (not in this repo); set `CHEMTOOLS_RUNNER_PROFILES` to your local file

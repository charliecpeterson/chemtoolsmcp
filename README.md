# chemtools-mcp

AI agent toolkit for computational chemistry. Provides an MCP (Model Context Protocol) server that gives Claude structured access to quantum chemistry programs — parsing outputs, drafting inputs, managing jobs, and analyzing results.

Currently supports **NWChem**. Molpro, ORCA, and others planned.

---

## Install

```bash
pip install git+https://github.com/charliecpeterson/chemtoolsmcp.git
```

For development (changes take effect immediately):

```bash
git clone https://github.com/charliecpeterson/chemtoolsmcp.git
cd chemtoolsmcp
pip install -e .
```

---

## MCP Server

### `chemtools-nwchem` — 106 tools

Single server for all NWChem capabilities:
- DFT/SCF geometry optimization and frequency workflows
- TCE correlated methods (MP2, CCSD, CCSD(T)) with orbital ordering checks and T1/D1 diagnostics
- MCSCF active-space suggestion and convergence review
- Geometry and frequency plausibility checks
- Basis set library (bundled — 608 files, no extra download)
- NWChem documentation search and lookup (bundled — 29 doc files, always available)
- HPC job management (SLURM/PBS/LSF) with auto resource selection
- Run registry, campaigns, and workflow DAGs

---

## Claude Desktop / Claude Code Configuration

```json
{
  "mcpServers": {
    "chemtools-nwchem": {
      "command": "chemtools-nwchem",
      "env": {
        "CHEMTOOLS_RUNNER_PROFILES": "/path/to/runner_profiles.yaml",
        "CHEMTOOLS_MCP_LOG": "/tmp/chemtools-nwchem.log"
      }
    }
  }
}
```

### Env vars reference

| Variable | Required | Description |
|---|---|---|
| `CHEMTOOLS_RUNNER_PROFILES` | Yes (to launch jobs) | Path to your `runner_profiles.yaml` |
| `CHEMTOOLS_BASIS_LIBRARY` | No | Override bundled NWChem basis library path |
| `CHEMTOOLS_MCP_LOG` | No | Debug log file path |

---

## Runner Profiles

Job launch is configured via a runner profiles file (YAML or JSON) — per-machine config
that you create once and point `CHEMTOOLS_RUNNER_PROFILES` at.

Example files:
- `examples/local_workstation/` — local workstation (direct process)
- `examples/tacc_stampede3/` — TACC Stampede3 (SLURM scheduler)
- `chemtools/runner_profiles.example.yaml` — full reference with all profile types

---

## Data Layout

```
chemtools/data/
  nwchem/
    basis_library/     ← bundled NWChem basis sets (608 files)
    docs/              ← bundled NWChem documentation (29 text files)
  molpro/              ← future
  orca/                ← future
```

All data is bundled with the package. No separate downloads needed.

---

## Adding a New Program

Each program gets:
- `chemtools/<program>_input.py` — input drafting
- `chemtools/<program>_output.py` — output parsing
- `chemtools/mcp/<program>.py` — MCP server (thin wrappers)
- `chemtools/data/<program>/` — program-specific data files

Then add an entry point in `pyproject.toml`:
```toml
chemtools-molpro = "chemtools.mcp.molpro:main"
```

---

## Architecture

```
chemtools/              Python library — all parsing, analysis, input generation
  api*.py               Public API (re-exported from api.py)
  nwchem_*.py           NWChem-specific modules
  nwchem_docs.py        NWChem documentation search (bundled docs)
  common.py             Shared utilities (element tables, covalent radii, etc.)
  data/nwchem/          Bundled NWChem data (basis library + docs)
  mcp/
    nwchem.py           NWChem MCP server — 106 tools (chemtools-nwchem entry point)
    nwchem_docs.py      Standalone docs server (backward-compat entry point)
```

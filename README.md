# chemtools-mcp

AI agent toolkit for computational chemistry. Provides MCP (Model Context Protocol) servers that give Claude structured access to quantum chemistry programs — parsing outputs, drafting inputs, managing jobs, and analyzing results.

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

## MCP Servers

### `chemtools-nwchem` — NWChem tools (49 tools)

Parses NWChem output, drafts inputs, manages jobs, analyzes results. Includes:
- DFT/SCF geometry optimization and frequency workflows
- TCE correlated methods (MP2, CCSD, CCSD(T)) with orbital ordering checks and T1/D1 diagnostics
- MCSCF active-space suggestion and convergence review
- Geometry and frequency plausibility checks
- Basis set library (bundled — no extra download needed)

### `chemtools-nwchem-docs` — NWChem documentation lookup

Serves NWChem documentation for syntax reference. Requires a local copy of the NWChem docs.

---

## Claude Desktop / Claude Code Configuration

### NWChem only (most common)

```json
{
  "mcpServers": {
    "chemtools-nwchem": {
      "command": "chemtools-nwchem",
      "env": {
        "CHEMTOOLS_RUNNER_PROFILES": "/path/to/runner_profiles.json",
        "CHEMTOOLS_MCP_LOG": "/tmp/chemtools-nwchem.log"
      }
    }
  }
}
```

### NWChem + documentation server

```json
{
  "mcpServers": {
    "chemtools-nwchem": {
      "command": "chemtools-nwchem",
      "env": {
        "CHEMTOOLS_RUNNER_PROFILES": "/path/to/runner_profiles.json"
      }
    },
    "chemtools-nwchem-docs": {
      "command": "chemtools-nwchem-docs",
      "env": {
        "NWCHEM_DOCS_ROOT": "/path/to/nwchem-docs"
      }
    }
  }
}
```

### Env vars reference

| Variable | Required | Description |
|---|---|---|
| `CHEMTOOLS_RUNNER_PROFILES` | Yes (to launch jobs) | Path to your `runner_profiles.json` |
| `CHEMTOOLS_BASIS_LIBRARY` | No | Override bundled NWChem basis library path |
| `CHEMTOOLS_MCP_LOG` | No | Debug log file path |
| `NWCHEM_DOCS_ROOT` | Yes (docs server) | Path to local NWChem documentation |

---

## Runner Profiles

Job launch is configured via a `runner_profiles.json` file — this is per-machine config that you create once and point `CHEMTOOLS_RUNNER_PROFILES` at.

Example for a local workstation using Apptainer:

```json
{
  "workstation_mpi": {
    "label": "Local workstation — 15 MPI ranks via Apptainer",
    "launch_command": ["mpirun", "-np", "15", "apptainer", "exec",
                       "/path/to/nwchem.sif", "nwchem"],
    "work_dir": "/path/to/job/directory"
  }
}
```

---

## Data Layout

```
chemtools/data/
  nwchem/
    basis_library/     ← bundled NWChem basis sets (608 files)
  molpro/              ← future
  orca/                ← future
```

The basis library is bundled with the package. No separate download needed.

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
  common.py             Shared utilities (element tables, covalent radii, etc.)
  data/nwchem/          Bundled NWChem data (basis library)
  mcp/
    nwchem.py           NWChem MCP server (chemtools-nwchem entry point)
    nwchem_docs.py      NWChem docs MCP server (chemtools-nwchem-docs entry point)

tests/                  Test suite
```

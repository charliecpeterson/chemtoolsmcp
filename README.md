# chemtools-mcp

AI agent toolkit for computational chemistry. Provides an MCP (Model Context
Protocol) server that gives Claude (and other MCP clients) structured access to
quantum chemistry programs — parsing outputs, drafting inputs, managing jobs,
analyzing active spaces, recovering from failed runs.

Currently supports **NWChem** (97 program-specific tools), **OpenMolcas**
(40 tools), and **DIRAC** (29 tools: parsers, HDF5 checkpoint reader,
VECPOP per-MO j-character classifier, open-shell quality analyzer, MO
reorder block drafter + input patcher, `.inp` + `.mol` drafters,
atomic-start orchestrator with `--copy` chain, pam-dirac/apptainer
launcher, `.KPSELE` atomic-supersymmetry support for actinide AOC,
Cm-class multi-step convergence workflow scaffolding, ΔSCF core-
ionization workflow validated against DIRAC's CO 1s tutorial,
bundled 180-page docs) plus 36 program-generic tools that auto-detect
any of them. Molpro, ORCA, and others planned.

---

## Quick start

```bash
git clone https://github.com/charliecpeterson/chemtoolsmcp.git
cd chemtoolsmcp
pip install -e .
```

Verify install:

```bash
chemtools --show-mode      # prints active mode + program filter + blocked tools
chemtools --list-tools     # prints the tool names visible in this mode
```

By default the server runs in **analysis mode** (no NWChem/Molcas executable
needed) — you can parse outputs, draft inputs, look up docs, and plan
calculations without anything else installed. To launch real jobs see
[Runner profiles](#runner-profiles).

---

## MCP client setup

### Claude Desktop / Claude Code

Add to your MCP servers config:

```json
{
  "mcpServers": {
    "chemtools": {
      "command": "chemtools"
    }
  }
}
```

That's the minimum config — it gets you analysis mode with all 173 tools
exposed. The sections below show how to scope the tool list, switch modes,
and wire up a runner profile for job submission.

### Restricting to one program

Loading both NWChem and Molcas means 173 tool definitions in your agent's
context. For a session focused on one program, filter:

```json
"chemtools-molcas": {
  "command": "chemtools",
  "env": { "CHEMTOOLS_PROGRAMS": "molcas" }
}
```

`CHEMTOOLS_PROGRAMS=molcas` exposes Molcas + the 36 generic tools (76 total).
Set to `nwchem` for the NWChem-focused 133. Comma-separate for multiple
(`nwchem,molcas`).

### Server modes

| Mode | Tools visible | Use when |
|---|---|---|
| `analysis` (default if no `CHEMTOOLS_RUNNER_PROFILES`) | 156 | Post-hoc parsing, drafting, planning — no chemistry executable needed |
| `local` | 170 | NWChem / OpenMolcas installed on this machine as a subprocess |
| `hpc` | 173 | Submit to SLURM/PBS/LSF on an HPC cluster |

Mode is auto-detected from your runner profile (see below). Override with
`CHEMTOOLS_MODE=analysis` or the `--mode` flag.

---

## Runner profiles

To actually **launch** jobs (not just parse pre-existing output), point
`CHEMTOOLS_RUNNER_PROFILES` at a YAML or JSON file describing your environment.
The repo includes ready-to-copy examples:

| Example | What it shows |
|---|---|
| `chemtools/runner_profiles.local.example.json` | Minimal local-workstation profile (single direct subprocess) |
| `chemtools/runner_profiles.example.yaml` | Canonical reference covering local + SLURM/PBS HPC profiles |
| `examples/tacc_stampede3/runner_profiles.yaml` | Real TACC Stampede3 SLURM config (SKX / ICX / SPR partitions) |
| `examples/local_workstation/` | Direct-launch workstation profile |

Copy one, edit the paths to point at your NWChem / OpenMolcas binary, then:

```json
{
  "mcpServers": {
    "chemtools": {
      "command": "chemtools",
      "env": {
        "CHEMTOOLS_RUNNER_PROFILES": "/path/to/runner_profiles.yaml"
      }
    }
  }
}
```

The server auto-detects the right mode (`local` for direct profiles, `hpc` for
scheduler profiles), filters the tool surface accordingly, and exposes
`launch_nwchem_run`, `watch_nwchem_run`, `terminate_nwchem_run`, etc.

For HPC profiles, `suggest_nwchem_resources` analyzes your input against
the profile's hardware specs and recommends optimal nodes / ranks / walltime
/ memory directives — no manual guessing.

---

## What you get

**173 tools** across these areas (any tool tagged `generic` works on either
program):

| Area | NWChem | Molcas | Generic | Notes |
|---|---:|---:|---:|---|
| Parse output (basic) | ✓ | ✓ | `parse_output`, `summarize_output` | Auto-detects program |
| Parse output (deep) | `parse_nwchem_output` | `parse_molcas_output` | | Per-module rich data |
| Geometry / freq / thermo / trajectory | `*_nwchem_*` | `*_molcas_*` | `extract_geometry`, `parse_frequencies`, `parse_thermochem`, `parse_trajectory`, `inspect_geometry` | Generic versions auto-dispatch |
| Input drafting | 17 tools | `draft_molcas_input`, 6 orchestrators (CASSCF, CASPT2 chain, opt+freq, excited states, IRC, scans, atomization) | | |
| Lint | `lint_nwchem_input` | `lint_molcas_input` | | |
| Case analysis | `analyze_nwchem_case` | `analyze_molcas_case` | `analyze_case`, `summarize_output` | Auto-dispatch |
| Recovery suggestion | `suggest_nwchem_recovery` | `suggest_molcas_recovery`, `apply_molcas_recovery`, `try_molcas_run_with_recovery` | `suggest_recovery`, `apply_recovery` | Auto-dispatch |
| Active-space tools | — | `analyze_molcas_active_space`, `validate_molcas_caspt2_setup`, `refine_molcas_active_space`, `suggest_molcas_orbital_swaps` | | Multireference |
| Basis / ECP | 4 tools | `list_molcas_basis_sets` | `suggest_basis_set` | Bundled libraries |
| Documentation | 7 tools (29 docs bundled) | 7 tools (133 docs bundled) | | Plus runtime forum search for NWChem |
| HPC / resources | 6 tools | `prepare_molcas_launch` | `suggest_resources`, `render_job_script` | Scheduler-aware |
| Registry + campaigns | 9 tools (program='nwchem' default) | — | 8 tools | Cross-program SQLite registry |
| Workflow protocols | `list_nwchem_protocols`, `plan_nwchem_calculation`, `create_nwchem_workflow`, `advance_nwchem_workflow` | — | | DAG engine in core/ |

**Bundled data** (no separate downloads):
- 608 NWChem basis-set files
- 91 OpenMolcas basis-set files
- 29 NWChem documentation pages
- 133 OpenMolcas documentation pages
- 180 DIRAC documentation pages

**Optional dependencies**:
- `pip install chemtools[dirac]` adds `h5py` for reading DIRAC HDF5 checkpoints.

---

## Three-line agent workflows

The tools are designed to chain. A few worked examples the agent can drive:

**Parse a run you don't recognize** (any program):
```
parse_output(output_file)              → tasks, energies, diagnosis
summarize_output(output_file)          → high-signal narrative
analyze_case(output_file)              → verdict + issues + next_actions
```

**Recover a failed CASPT2 run** (Molcas):
```
analyze_molcas_case(output_file)       → verdict=problematic, issues list
suggest_molcas_recovery(output_file)   → failure_class + fix_recipe
apply_molcas_recovery(input_file, output_file)  → writes patched input
```

**Submit and monitor an NWChem job on HPC**:
```
suggest_nwchem_resources(input_file, profile)   → optimal nodes/ranks
launch_nwchem_run(input_file, profile, auto_watch=true)  → block until done
analyze_nwchem_case(output_file)        → quality verdict
```

**Set up a CASSCF / CASPT2 calculation from scratch**:
```
prepare_molcas_casscf_setup(molecule, cas=(M,N), method="CASPT2")  → input
prepare_molcas_launch(input_file)       → safe pymolcas command
# run the command
analyze_molcas_case(output_file)        → check verdict before trusting energy
```

---

## CLI debugging

```bash
chemtools --show-mode                          # mode + reason + program filter (JSON)
chemtools --list-tools                         # tool names visible under current filters
chemtools --mode analysis                      # force analysis mode (no executable needed)
chemtools --programs molcas                    # only Molcas + generic tools
chemtools --mode local --programs nwchem,molcas
```

Inside an agent session, call the `get_server_mode` tool to introspect at
runtime — useful when a tool fails with "not available in mode."

---

## Architecture

```
chemtools/
  core/                          program-agnostic shared infrastructure
    program.py                   plugin Protocol (Parser, Drafter, Strategist, ...)
    registry.py                  plugin registry + program auto-detection
    runner.py                    launcher / scheduler glue (no programs/ imports)
    workflow.py                  DAG engine for multi-step protocols
    basis_advisor.py             basis-set + ECP recommendation
    units.py, thermochem.py,
    geometry.py, issues.py,      shared math + helpers
    recovery.py, case_analysis.py, session.py
    run_registry.py              SQLite registry — cross-program campaigns
  programs/
    nwchem/                      NWChem plugin
      parse/                     output / input / freq / mos / tasks / tce parsers
      input/                     input file drafting + lint
      strategy/                  diagnose, recovery, case_review, progress, resources
      binary/                    movecs / hessian / fdrst readers
      data/                      bundled basis library + docs
      _plugin_*.py               sub-protocol implementations
    molcas/                      OpenMolcas plugin (mirrors nwchem/)
    molpro/                      stub for future Molpro plugin
  mcp/
    decorator.py                 @_tool registration with program / needs tags
    modes.py                     mode + program filtering
    server.py                    JSON-RPC entry point
    nwchem.py                    `chemtools` CLI entry point
    tools/
      nwchem.py                  NWChem tool definitions + handlers (133 tools)
      molcas.py                  Molcas tool definitions + handlers (40 tools)
```

Tools are tagged with `program=nwchem|molcas|generic` and
`needs=none|registry|runner_profile|executable_or_scheduler|executable|scheduler`.
The active mode + program filter decides which subset is exposed at
`tools/list` time. Generic tools auto-detect the program at call time via
`registry.resolve(program=None, path=output_file)`.

---

## Adding a new program

Each program is a plugin under `chemtools/programs/<name>/`:

1. Implement the sub-protocols from `chemtools/core/program.py`:
   `Parser`, `Drafter`, `Strategist`, optionally `BinaryReader` and
   `ExamplesCorpus`.
2. Assemble a `Program` instance and call `chemtools.core.registry.register(PLUGIN)`
   from your `chemtools/programs/<name>/__init__.py`.
3. Add MCP tool definitions in `chemtools/mcp/tools/<name>.py` —
   one `@_tool("name", program="<name>")` handler per tool.
4. Bundle docs / basis libraries under `chemtools/programs/<name>/data/`
   (or pull from `chemtools/data/<name>/` if shared).

The plugin Protocol is documented in `chemtools/core/program.py`. The
NWChem plugin is the reference implementation; the Molcas plugin shows
the "second program" pattern (lots of generics already wired).

---

## Troubleshooting

- **"No program registered for path"** — the auto-detection didn't recognize
  the output. Check that the file is a real NWChem `.out` or OpenMolcas `.log`.
  Or call the per-program tool (`parse_nwchem_output` / `parse_molcas_output`)
  directly.
- **Tool fails with "not available in mode"** — call `get_server_mode` to
  see which mode you're in. Either switch mode (`CHEMTOOLS_MODE=local`)
  or use a different tool that's available in your mode.
- **Runner profile not loading** — verify `CHEMTOOLS_RUNNER_PROFILES` points
  at a readable file; check `chemtools --show-mode` for the resolution
  result. Profile YAML / JSON syntax errors are logged on stderr.
- **NWChem job stuck** — `watch_nwchem_run` detects known output-silent
  phases (SAD X2C guess, DFT grid generation, frequency Hessian
  differentiation, TCE AO→MO transform) and reports "expected slow"
  rather than treating them as hung.

For help or to file an issue:
[github.com/charliecpeterson/chemtoolsmcp/issues](https://github.com/charliecpeterson/chemtoolsmcp/issues)

---

## License

See LICENSE file.

# Legacy chemistry-agent package audit

Audit date: 2026-08-07

`chem-agent-package/` was removed with owner approval on 2026-08-07. Its
runtime code duplicated maintained Chemtools interfaces, and no maintained
repository caller imported or launched it. Three useful NWChem policies remain
as draft knowledge cards, so the scientific judgment is not tied to the
deleted client bundle.

## Runtime and client files

| Legacy path | Finding | Maintained owner |
| --- | --- | --- |
| `mcp/chemtools_mcp_server.py` | Compatibility launcher for the old NWChem server | Installed `chemtools --toolset guided` command |
| `mcp/chemtools_nwchem_mcp_server.py` | Second compatibility launcher for the same old server | Installed `chemtools` command and catalog-driven MCP server |
| `mcp/nwchem_docs_mcp_server.py` | Separate manual server loop with source-tree path mutation and old imports | Packaged NWChem documentation tools behind the maintained MCP catalog |
| `openwebui/tools/chemtools_openwebui.py` | 1,305 lines of wrappers around top-level Chemtools imports, with hard-coded local paths | Focused Python modules and the eleven guided MCP tools |
| `openwebui/model-bundles.example.yaml` and export files | OpenWebUI-specific model and skill packaging | No replacement required for the current personal Codex workflow |
| `opencode/AGENTS.md` | Old command inventory and hard-coded checkout paths | `plugins/chemtools` skills and the maintained setup examples |

The wrappers do not own parser, input-generation, or scientific-analysis
logic. Retaining them would preserve a second client interface and the broad
top-level Python facade without preserving any unique implementation.

## Knowledge retained

### SCF recovery

The old `chem-style` and `nwchem-output-playbook` instructions require the
recent energy and density-error pattern to be classified before choosing an
SCF recovery. They distinguish slow improvement, oscillation, a plateau or
stall, and convergence to the wrong state. Increasing `maxiter` is supported
only for a slow or nearly converged trajectory. The maintained NWChem recovery
provider already implements part of this distinction, but the broad policy is
still a draft until representative tests cover each trend and wrong-state
case.

Retained card:
`nwchem.scf_recovery_requires_trend_classification`.

### Imaginary-mode interpretation

The old output playbook does not treat every imaginary frequency as the same
failure. It separates an intended transition-state mode, a small floppy or
torsional mode, and evidence that an intended minimum is a saddle. A displaced
geometry and follow-up frequency calculation may be needed to confirm the
interpretation. Current behavior-lock tests cover one significant-mode
restart, but not the full classification policy.

Retained card:
`nwchem.imaginary_modes_require_scoped_interpretation`.

### Basis coverage

The old basis policy requires explicit coverage of every element when
generating a mixed-element NWChem input. It also forbids inventing basis, ECP,
`cd basis`, `xc basis`, or auxiliary-basis choices and preserves deliberate
basis-stepping workflows. Current input drafting can accept manual basis data
without checking a configured library, so this policy remains a draft rather
than an accepted default recommendation.

Retained card:
`nwchem.basis_assignments_require_explicit_coverage`.

The general rule that normal termination does not establish scientific
success was already retained in `cross_program.silent_success`.

## Caller audit

A repository search outside `chem-agent-package/` found only planning and
audit prose that names the directory:

- `SIMPLIFICATION_PLAN.md`
- `notes/compatibility-surface-audit.md`
- this audit note

No maintained Python module, script, test, example, MCP configuration, or
plugin imports one of its files or launches one of its server scripts. This
audit cannot see notebooks or scripts outside the repository, but those would
already depend on obsolete hard-coded paths or the compatibility-only
top-level facade.

## Removal record

The owner confirmed that no external OpenWebUI or OpenCode setup still needed
the directory. All 16 tracked files were deleted together. The model bundles,
generated skill exports, hard-coded paths, and duplicated tool inventory were
not migrated.

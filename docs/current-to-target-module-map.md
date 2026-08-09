# Current-to-target module map

Chemtools already has useful program-specific packages, parsers, chemistry
rules, and MCP transport code. The main architectural problem is ownership:
application orchestration and execution are spread across MCP handlers, the
NWChem-heavy public facade, `core`, and per-program scheduler wrappers.

This map assigns current modules to the owners defined by ADRs 001 through
005 and records the compatibility seams that remain after the ownership
migration.

Snapshot date: 2026-08-09

Governing decisions:

- [ADR 001](adr/001-optional-program-capabilities-and-builtin-catalog.md)
- [ADR 002](adr/002-execution-targets-replace-behavior-modes.md)
- [ADR 003](adr/003-runs-are-artifact-collections-with-provenance.md)
- [ADR 004](adr/004-public-mcp-alias-and-deprecation-policy.md)
- [ADR 005](adr/005-reference-corpus-boundaries.md)

## Target dependency direction

```text
CLI composition root
  |
  +-- loads built-in backend catalog
  +-- loads target configuration
  +-- constructs application services
  +-- registers MCP adapters
          |
          v
MCP transport and tool adapters
          |
          v
Application services
  |
  +-- review and inspection
  +-- workflow planning
  +-- execution coordination
  +-- run comparison
  +-- reference lookup
          |
          v
Core models and ports
  ^             ^              ^               ^
  |             |              |               |
Program      Execution      Persistence      Integration
backends     executors       adapters         adapters
```

Dependencies point inward:

- MCP code may depend on application services.
- Application services may depend on core models and declared ports.
- Program, execution, persistence, and external integration adapters may
  depend on core models.
- Core models do not import MCP, concrete programs, executors, SQLite, or
  Orbitron.
- The CLI composition root is the one place allowed to import every concrete
  adapter needed to assemble the server.

The owner names in this document are responsibilities. A target owner does
not need its own package until code with that responsibility has at least one
real consumer.

## Target owners

| Owner | Responsibility |
| --- | --- |
| MCP transport | Official SDK connection handling, initialization, and protocol envelopes |
| MCP adapter | Request validation, public tool metadata, stable result translation |
| Composition root | Load configuration and concrete adapters, then construct the server |
| Application service | Coordinate several domain ports for one user intent |
| Core domain | Program-neutral models, result types, chemistry rules, and interfaces |
| Program backend | Program syntax, parsing, diagnostics, artifacts, and launch planning |
| Execution adapter | Local or Slurm process control with no chemistry-program rules |
| Persistence adapter | SQLite storage and retrieval for runs, artifacts, and provenance |
| Integration adapter | Versioned boundary to an optional external tool such as Orbitron |
| Reference data | Manifests, committed fixtures, datasets, and curation metadata |
| Compatibility facade | Preserve old Python, CLI, and MCP names during migration |

## Current package map

### Composition and public Python API

| Current module | Current responsibility | Target owner | Disposition |
| --- | --- | --- | --- |
| `chemtools/mcp/cli.py` | Parses CLI options, loads schema-2 named targets, creates one server state, and starts the SDK stdio server | Composition root | Keep target loading and explicit execution permission here while legacy mode remains only for tool visibility and profile migration. |
| `chemtools/__init__.py` | Re-exports a large NWChem-oriented API plus registry and evaluation functions | Compatibility facade | Keep public imports working. Stop using this facade inside MCP, core, and backend implementations. Add new public entry points from application services, then retire old exports under ADR 004. |
| `chemtools/api.py` | Aggregates NWChem parsing, drafting, strategy, execution, registry, and workflow functions | Compatibility facade | Preserve behavior while callers move to direct owners. Do not add QE or QMCPACK to this file. |
| `chemtools/api_input.py` | NWChem input and follow-up orchestration | NWChem backend and application services | Leave program syntax in the NWChem backend. Move only operations that coordinate parsing, review, file creation, and later execution into an application service. |
| `chemtools/api_strategy.py` | NWChem diagnosis and resource-advice facade | NWChem backend and application services | Keep deterministic NWChem rules in `programs/nwchem/strategy`. Replace this file with compatibility exports after callers use the backend catalog. |

The top-level facade remains only for compatibility with existing Python
callers. Supported direct Python use goes through the module that owns each
operation.
`application/evaluation.py` and `chemtools/mcp/tools/generic.py` now import
those owners directly, and the
generic MCP module no longer changes `sys.path` to prefer a source checkout.
The NWChem and Molcas documentation accessors and the Molcas basis accessor
also resolve data relative to their owning modules instead of importing the
top-level package for its path. An AST contract test prevents these migrated
areas from returning to the public facades. All remaining NWChem program
callers now import their focused input or strategy owner, including lazy
imports retained to avoid module cycles. The public `chemtools`, `api_input`,
and `api_strategy` exports remain exact aliases of those owners. The
post-`v0.1.0` tree removes the broad `_nwchem_base.py` MCP facade and the
`chemtools.mcp.tools.nwchem` aggregator. All five handler families import their
focused owners directly. The built-in catalog uses `_nwchem_provider.py` to
compose those families and their schemas. NWChem-specific decorator
registration and basis-data path resolution remain small shared modules.

### MCP transport, dispatch, and tool registration

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/mcp/sdk_server.py` | Low-level official SDK server, stdio connection, typed tool definitions, and result translation | MCP transport and MCP adapter | Keep protocol behavior in the SDK. Preserve serialized text alongside structured results until clients no longer require the compatibility representation. |
| `chemtools/mcp/server.py` | Validated image result type, compatibility JSON-RPC envelopes, and common CLI arguments | Compatibility facade and MCP adapter | Keep the image boundary and shared arguments. Remove dictionary response helpers after direct callers of `handle_request` are retired. |
| `chemtools/mcp/dispatch.py` | Catalog startup, validated alias resolution, state-bound tool filtering, and handler dispatch | MCP adapter | Keep transport-independent filtering and handler invocation here. The dictionary-based `handle_request` remains only for direct Python compatibility callers. |
| `chemtools/mcp/compatibility.py` | Hidden alias metadata, pure argument adapters, and registration validation | Compatibility registry | Keep aliases out of canonical tool discovery. Recover historical schemas and effects before marking contracts verified or hiding advertised legacy definitions. |
| `chemtools/mcp/state.py` | Immutable mode, program filter, tool filter, and process-owned execution service for one server | Composition state | Keep this object transport-neutral. The CLI creates it once and dispatch binds it while a registered handler runs. |
| `chemtools/mcp/decorator.py` | Mutable handler registries, program tags, mode tags, server identity, logging, and request-state binding | MCP adapter | Keep the decorator during compatibility work. Old direct Python setters replace one fallback state object; CLI servers do not use those setters or separate module globals. Do not let backend code import it. |
| `chemtools/mcp/modes.py` | Hard-coded program names, tool filtering, runner-profile inspection, execution-mode selection | Legacy compatibility facade | The catalog owns built-in program membership, while this adapter retains old mode and `needs` behavior through the ADR 002 window. |
| `chemtools/mcp/inventory.py` | Reads the live registries and emits contract metadata | MCP adapter | Keep it as the machine-readable contract ledger. Change its source to the catalog and compatibility registry as those become authoritative. |
| `chemtools/mcp/tools/guided.py` | Eight contract-bound guided MCP adapters calling application services | MCP adapter | Keep argument and protocol translation here. Each `_handle_<tool name>` function now derives its registration name from the matching declarative contract. Scientific interpretation belongs in application services and declared backend providers. |
| `chemtools/mcp/tools/_guided_definitions.py` | Public descriptions, input schemas, output schemas, and annotations for eight guided adapters | MCP contract metadata | Keep declarative contracts separate from runtime adapters. The catalog receives them only after `guided.py` verifies one exact handler per definition. |
| `chemtools/mcp/tools/_nwchem_provider.py` | Imports five focused NWChem handler families and exposes their schemas to the built-in catalog | MCP composition provider | Keep regular startup independent from the legacy NWChem Python facade. |
| Other `chemtools/mcp/tools/*.py` modules | Schemas and handlers for program-specific and generic tools | MCP adapter | Preserve names, schemas, and results. Handlers should gradually call application services or backend capabilities instead of importing concrete implementation functions. |
| `chemtools/mcp/nwchem.py` | Legacy NWChem MCP entry point | Compatibility facade | Keep until the CLI deprecation ledger permits removal. |
| `chemtools/mcp/nwchem_docs.py` | Standalone NWChem documentation CLI | Compatibility facade or NWChem backend CLI | Keep independently from MCP alias decisions. |

The transport boundary currently performs too much composition. It imports
all tool modules explicitly, and the tool modules import program
implementations directly. This is acceptable as a compatibility path, but it
must not become the route used to add QE and QMCPACK.

The NWChem handler families use focused imports, and catalog-driven MCP startup
uses `_nwchem_provider.py`. The former `_nwchem_base.py` and `nwchem.py`
compatibility modules were removed after the `v0.1.0` tag; the current tree has
no wildcard NWChem MCP namespace or dynamic handler lookup.

Inventory schema version 3 reports canonical definitions, advertised
legacy definitions, hidden MCP aliases, executable aliases, and Python import
shims separately, plus every advertised input schema and stable guided output
schema. Hidden aliases are owned by a validated ADR 004 registry
that rejects missing targets, collisions, chains, and broader availability.
Their historical schemas and effects remain explicitly unverified, and no
versioned removal metadata is set without a tagged release boundary.

The CLI now creates one immutable `ServerState` after resolving its arguments.
That state travels through `serve`, `handle_request`, and `dispatch_tool`, so a
handler sees the same filters and execution-service instance that gated its
request. A context-local binding preserves the existing zero-argument handler
shape. The former setters remain only for direct Python compatibility calls;
they replace one fallback state and are not used by the CLI. Tests pin filter
isolation and distinct execution ownership for two server states in one
process.

The eight contracts separated into `_guided_definitions.py` are indexed once
when `guided.py` loads. Its decorator derives each tool name from the existing
`_handle_<tool name>` function, rejects a handler without a definition or a
second handler for the same definition, and refuses to return the definition
list if any handler is missing. This removes the second hand-written tool name
from every guided adapter. `visualize` and `search_knowledge`, the other two
members of the guided preset, already keep their definitions beside their
handlers in their owning modules. `find_reference_case`, the eleventh member,
keeps its MCP contract in `tools/reference.py` and calls the bounded
`application/reference_case_search.py` service. Its two searchable manifests
live under packaged data so the same lookup works from a wheel.

### Program model and registration

| Current module | Current responsibility | Target owner | Disposition |
| --- | --- | --- | --- |
| `chemtools/core/program.py` | Broad parser, drafter, strategist, binary, and example protocols | Core domain | Operation-level capabilities and optional providers are present. Remove the broad compatibility protocol only after every generic caller uses capability checks. |
| `chemtools/core/registry.py` | Validated, unique runtime program registry and file detection | Core domain service | Keep runtime lookup and detection here without importing the catalog or MCP metadata. |
| `chemtools/programs/<name>/__init__.py` | Exposes a validated backend without registry side effects | Program backend | Keep backend construction here. Built-in membership and registration belong to the composition catalog. |
| `chemtools/programs/_adapter_helpers.py` | Cross-program conversion helpers for current parser adapters | Core domain or local adapter helper | Keep while two or more backends use the same result conversion. Move only genuinely program-neutral result logic into core. |

The built-in catalog now owns program membership, registration order, MCP tool
modules, and inventory order. MCP decorators still tag individual tools with a
program name, but they do not define which programs are built in.

`registry.register` rejects duplicate names, so import order cannot silently
replace a backend.

### Program packages

The internal `parse`, `input`, `strategy`, `binary`, `docs`, and `examples`
directories are sound ownership boundaries. Keep them in their program
packages.

| Current package | Keep | Change |
| --- | --- | --- |
| `chemtools/programs/nwchem` | Declared backend, parsers, input rendering, binary readers, diagnosis, resource models, docs, examples | Keep the validated declaration. Reassign runtime and cross-run orchestration as described below. |
| `chemtools/programs/molcas` | Declared partial backend, parsers, input modules, orbital handling, active-space rules, recovery, docs | Keep the tested capability set. `launch.py` owns typed launch plans, while `runtime.py` remains a direct Python compatibility path. |
| `chemtools/programs/dirac` | Declared partial backend, parsers, HDF5 reader, basis data, atomic and core-ionization input builders | Keep the tested capability set. Preserve `.inp` and `.mol` staging rules in its launch-plan provider. |
| `chemtools/programs/grasp` | Declared parser backend, multi-artifact parsers, bounded radial-wavefunction and paired mixing/CSF inspection, explicit orbital merging, heredoc builders, workflow knowledge, diagnosis | Keep binary inspection, CSF interpretation, and first-donor-wins merging with the GRASP backend. Keep binary writes limited to the atomic, no-clobber merge contract. Model the working directory as an artifact collection. Convert direct executable calls into launch plans before exposing them through the common launch service. |
| `chemtools/programs/qe` | Declared periodic backend with input review, UPF inspection, output diagnosis, consistency checks, normalized geometry, and trajectory parsing | Keep program syntax and scientific interpretation here. `_elements.py` owns species-label normalization. `_coordinates.py` owns shared PWSCF output-card parsing and unit conversion; `input_geometry.py` normalizes the supported input coordinate forms; `geometry.py` selects one usable output snapshot; `trajectory.py` assembles optimization history; `trajectory_analysis.py` owns bounded periodic metrics and molecular structural checks shared by input and output review. |

Program packages may import core models and utilities. They must not import
MCP transport, public tool names, or scheduler implementations.

The scientific-ownership audit moved the legacy NWChem action table, recovery-mode
aggregation, multiplicity inference, and SCF syntax; DIRAC occupation,
summary, and spinor-filter rules; QE-to-QMCPACK readiness precedence; and
Molcas task, geometry, and RASSI selection below MCP. Generic geometry
normalization and recovery source agreement now belong to application
services. Compatibility imports remain where direct Python callers still need
them, but the normal MCP path imports the program owner directly. The final
branch disposition is recorded in
`notes/mcp-scientific-ownership-audit.md`.

One cross-program exception remains for direct Python compatibility:
`chemtools/programs/nwchem/output.py` imports the Molcas output parser and
dispatches `parse_tasks` itself. New composition uses backend capabilities;
the old dispatcher can leave with its compatibility callers.

### Execution and scheduling

| Current module | Current responsibility | Target owner | Disposition |
| --- | --- | --- | --- |
| `chemtools/application/execution_policy.py` | Immutable execution decisions plus disabled, status, and cancellation errors | Application policy | Keep policy result shapes independent of process, scheduler, and persistence adapters. The old `application.execution` imports remain compatible. |
| `chemtools/application/execution.py` | Default-off permission checks, read-only rendering, asynchronous and synchronous launch, read-only ownership resolution, owned local and Slurm status, terminal-state recording, and registry-bound cancellation | Application service | Status polling requires ownership but no second execution permission decision. Resolve configured targets at composition time before removing compatibility target adapters. |
| `chemtools/execution/targets.py` | Loads schema-2 YAML or JSON into immutable named local and Slurm targets with explicit execution permission and default selection | Target configuration adapter | Keep arbitrary commands and roots in host-owned configuration. Guided NWChem, Molcas, DIRAC, GRASP workflow, QE, and QMCPACK can use this path without loading a version 1 profile. |
| `chemtools/application/dirac_execution.py` | Applies DIRAC pairing and profile policy, coordinates typed execution, and translates legacy MCP responses | DIRAC compatibility application adapter | Keep `.inp/.mol` validation and exact output archival here while the public DIRAC tools remain. Add live checkpoint staging only after its destination and overwrite rules are explicit. |
| `chemtools/application/dirac_monitoring.py` | Combines owned local or Slurm execution status with legacy DIRAC file inspection | DIRAC monitoring application adapter | Keep DIRAC file interpretation here. Scientific-run linking and artifact observations remain separate work. |
| `chemtools/application/execution_monitoring.py` | Refreshes program-matched owned identifiers, projects typed process and Slurm evidence into compatibility responses, and polls owned status readers | Shared monitoring application support | Keep program parsing, scientific status, artifact recording, and scheduler subprocess calls out. |
| `chemtools/application/grasp_execution.py` | Coordinates typed GRASP workflow-script and synchronous per-executable launches and translates legacy MCP responses | GRASP execution compatibility adapter | Keep workflow path validation, exact output archival, capture files, session logs, launch IDs, and owned scheduler cancellation here while public response shapes remain. |
| `chemtools/application/grasp_monitoring.py` | Combines owned local or Slurm workflow status with legacy GRASP file inspection | GRASP monitoring application adapter | Keep synchronous per-executable results outside the watcher. Scientific-run linking and artifact observations remain separate work. |
| `chemtools/application/legacy_execution.py` | Projects typed local and Slurm results into existing MCP response dictionaries | Compatibility application adapter | Keep shared launch IDs, effective argv, submitted scripts, `.jobid` writes, timeout fields, and cancellation fields consistent while program-specific public tools remain. |
| `chemtools/application/molcas_execution.py` | Applies Molcas launch policy, coordinates typed execution, and translates legacy MCP responses | Molcas compatibility application adapter | Keep the CASPT2 guard and exact `.log` archival here while the low-level Molcas tools and version 1 preview remain. |
| `chemtools/application/molcas_monitoring.py` | Combines owned local or Slurm execution status with legacy Molcas file inspection | Molcas monitoring application adapter | Keep process and scheduler ownership in the execution service. Molcas scientific-run linking and artifact observations remain separate work. |
| `chemtools/application/nwchem_execution.py` | Converts version 1 NWChem profiles and legacy responses to typed calls; verifies and registers owned launches; synchronizes local or Slurm completion with linked runs and output observations | NWChem compatibility application adapter | Keep MCP response translation, launch/run registration checks, and NWChem artifact kinds here while the public tools remain aliases. Move the completion pattern to a program-neutral service only after another backend needs it. |
| `chemtools/application/nwchem_monitoring.py` | Combines owned execution status with NWChem output inspection and runs typed local or Slurm watch requests | NWChem monitoring application adapter | Keep chemistry progress, linked-run synchronization, and artifact observations in the NWChem path. Use the shared application helper only for execution response fields and polling. |
| `chemtools/application/run_monitoring.py` | Refreshes one process-owned launch ID and normalizes execution, recorded artifact, and declared backend progress evidence for the guided tool | Guided monitoring application service | Keep arbitrary PIDs, scheduler IDs, and paths out. Reuse typed executor status and backend progress providers; do not add cancellation or restart effects. |
| `chemtools/core/execution.py` | Immutable launch plans, target-owned entrypoints, stdin and timeout intent, rendered commands, launch records, launch/run links, and asynchronous, status, or synchronous result models | Core domain | Keep this module free of process, scheduler, and SQLite calls. Store stdin digest and size in launch records, never stdin content. |
| `chemtools/core/monitoring.py` | Polling, adaptive intervals, compact history, timeouts, and terminal detection for calculation-status readers | Core service | Keep scheduler commands, persistence, and program parsing out. Typed `not_found` results must never imply completion. |
| `chemtools/core/slurm.py` | Typed Slurm status states, query evidence, job exit code, signal, and elapsed time | Core domain | Keep scheduler subprocess calls and state persistence out. Preserve raw scheduler state beside normalized status. |
| `chemtools/execution/_common.py` | Command rendering, allowed-root validation, and copy or symlink staging shared by execution adapters | Shared execution adapter support | Keep process handles, scheduler calls, program rules, and permission decisions out. |
| `chemtools/execution/local.py` | Captured synchronous execution, asynchronous launch, live-handle status, and live-handle cancellation | Local execution adapter | Status and cancellation must use retained process handles rather than arbitrary operating-system PIDs. |
| `chemtools/execution/slurm.py` | Slurm script rendering, submission, job-ID parsing, queue and accounting status, and target-command cancellation | Slurm execution adapter | Keep scheduler commands and script policy target-owned. Empty queue and accounting results must remain unknown rather than imply completion. |
| `chemtools/execution/executors.py` | Re-exports the local executor, Slurm executor, and work-root error | Compatibility facade | Preserve existing Python imports while callers move to `chemtools.execution` or the focused modules. |
| `chemtools/execution/profiles.py` | Loads version 1 profile files, merges defaults, and converts shared resource, hardware, module, program-installation, direct-command, and Slurm fields | Target configuration adapter | Keep program argument syntax and chemistry rules out. The standard `programs.<name>` installation block wins over old field locations. |
| `chemtools/execution/legacy_archive.py` | Timestamped, collision-safe archival of existing compatibility-launch outputs | Legacy output policy | Application adapters import this focused owner. Preserve exact imports from `legacy_runner.py` until its direct Python surface is removed. |
| `chemtools/execution/resource_inspection.py` | Local CPU and memory budgeting plus Slurm and PBS partition discovery | Target resource inspection | Keep scheduler discovery separate from chemistry advice and version 1 launch rendering. Replace dictionary results only when a typed target inventory has a real caller. |
| `chemtools/execution/legacy_runner.py` | Version 1 script rendering, launch behavior, and neutral compatibility imports | Legacy render and launch adapter | QE and QMCPACK callers are gone. Keep implementation out of core and remove it after the remaining NWChem, Molcas, DIRAC, GRASP, and direct Python compatibility callers leave. |
| `chemtools/execution/external_status.py` | Read-only file inspection and explicit external Slurm attachment with optional output interpretation | External-run inspection adapter | Keep process probing, cancellation, PBS, LSF, `.jobid` inference, and program imports out. |
| `chemtools/programs/nwchem/external_status.py` | Adds the NWChem progress reader to external file and Slurm status | NWChem external-run adapter | Keep NWChem interpretation in the backend while sharing file and Slurm evidence. |
| `chemtools/persistence/launches.py` | SQLite persistence and state-transition checks for execution launch records, including staging manifests, terminal metadata, and launch/run link lookup | Persistence adapter | Keep command and staging intent separate from artifact bytes. Local and Slurm NWChem completion use the link to synchronize the run; other programs still need the same integration. |
| `chemtools/execution/launch_registry.py` | Exact imports from `persistence/launches.py` | Compatibility facade | Preserve direct Python imports through the final compatibility release. No persistence implementation belongs here. |
| `chemtools/core/runner.py` | Exact imports from the neutral execution owner and NWChem status adapter | Compatibility facade | Keep old Python imports stable through the final compatibility release. No implementation belongs here. |
| `chemtools/programs/nwchem/runner.py` | NWChem launch wrappers, progress chemistry, intervention advice, structure-drift analysis, comparison, and follow-up review | NWChem backend plus application services | Keep NWChem progress and chemistry assessment in the backend. Move launch coordination and cross-run comparison behind application services. |
| `chemtools/programs/nwchem/launch.py` | Builds NWChem launch plans and adapts version 1 NWChem profiles into typed targets | NWChem launch-plan provider and compatibility adapter | Keep NWChem arguments, filenames, and artifact expectations here. Remove the profile adapter after the version 1 compatibility window. |
| `chemtools/programs/nwchem/_plugin_launcher.py` | Prepares guided NWChem plans from schema-2 named targets or the version 1 migration adapter | NWChem guided launch provider | Keep guided preparation independent of `execution/legacy_runner.py`. Named targets are current; profiles remain only as the migration fallback. |
| `chemtools/programs/qe/launch.py` | Builds typed `pw.x` plans and adapts direct or Slurm version 1 profiles into targets | QE launch-plan provider and compatibility adapter | Keep `-in`, output artifacts, and QE installation selection here. Remove the profile adapter after the version 1 compatibility window. |
| `chemtools/programs/qe/_plugin_launcher.py` | Prepares guided QE plans from schema-2 named targets or the version 1 migration adapter | QE guided launch provider | Keep guided preparation independent of `execution/legacy_runner.py`. Named targets are current; profiles remain only as the migration fallback. |
| `chemtools/programs/qmcpack/launch.py` | Builds typed QMCPACK plans and adapts direct or Slurm version 1 profiles into targets | QMCPACK launch-plan provider and compatibility adapter | Keep input arguments, initialization-only `--dryrun`, and output artifacts here. Remove the profile adapter after the version 1 compatibility window. |
| `chemtools/programs/qmcpack/_plugin_launcher.py` | Prepares guided ordinary or initialization-only QMCPACK plans from schema-2 named targets or the version 1 migration adapter | QMCPACK guided launch provider | Keep guided preparation independent of `execution/legacy_runner.py`. Named targets are current; profiles remain only as the migration fallback. |
| `chemtools/programs/molcas/runtime.py` | Builds the read-only legacy command preview and owns the shared CASPT2 detection and rank guard | Molcas compatibility facade and runtime rules | Keep `prepare_molcas_launch` stable while typed calls use the same guard through the launch-plan provider. |
| `chemtools/programs/molcas/launch.py` | Builds typed Molcas plans and adapts direct or Slurm version 1 profiles into targets | Molcas launch-plan provider and compatibility adapter | Keep pymolcas arguments, protected Molcas environment values, CASPT2 allocation changes, output rules, and dynamic Slurm project identity here. |
| `chemtools/programs/molcas/_plugin_launcher.py` | Prepares guided Molcas plans from schema-2 named targets or the version 1 migration adapter | Molcas guided launch provider | Keep named-target CASPT2 policy conservative and expose any rank adjustment in the reviewed plan. Profiles remain the path for verified parallel-CASPT2 installations during migration. |
| `chemtools/programs/dirac/runtime.py` | Builds the read-only advanced `pam-dirac` preview and owns shared argument construction | DIRAC compatibility facade and runtime rules | Keep `prepare_dirac_launch` stable for `--copy`, `--put`, `--get`, and `--outcmo` previews while typed launch plans use the same argument builder. |
| `chemtools/programs/dirac/launch.py` | Builds typed DIRAC plans and adapts direct or Slurm version 1 profiles into targets | DIRAC launch-plan provider and compatibility adapter | Keep paired input names, `pam-dirac` MPI and memory arguments, output rules, and container installation data here. Add checkpoint staging after its live MCP contract is defined. |
| `chemtools/programs/dirac/_plugin_launcher.py` | Prepares guided paired-input DIRAC plans from schema-2 named targets or the version 1 migration adapter | DIRAC guided launch provider | Keep both input identities approval-bound. Named targets use installation memory defaults; profiles retain explicit `--mw` and `--nw` values during migration. |
| `chemtools/programs/grasp/launch.py` | Builds typed shell-workflow and interactive plans, declares reviewed GRASP entrypoints, and adapts direct or Slurm version 1 profiles into container targets | GRASP launch-plan provider and compatibility adapter | Keep executable selection, stdin, arguments, and workflow artifact expectations here. Remove the committed default container path with named target configuration. |
| `chemtools/programs/grasp/_plugin_launcher.py` | Prepares guided GRASP workflow plans from schema-2 named targets or the version 1 migration adapter | GRASP guided launch provider | Keep container commands target-owned and the exact workflow script approval-bound. Interactive stdin-driven entrypoints remain on their typed low-level path. |
| `chemtools/programs/grasp/runtime.py` | Resolves the compatibility container, retains the direct Python runner, and formats session notes | GRASP compatibility runtime | MCP execution no longer calls its subprocess runner. Keep session formatting stable until session logs become artifact observations, then remove the direct executor after the compatibility window. |
| `chemtools/programs/grasp/strategy/runner.py` | Coordinates structured workflow steps through an injected runner and applies contained copy actions | GRASP workflow application service | MCP workflows inject the typed execution service. Record step outputs and the working directory as artifact observations when launch records link to scientific runs. |
| `chemtools/programs/{molcas,dirac,grasp}/scheduler.py` | Thin public wrappers around version 1 rendering plus external file and Slurm status | Compatibility facade | MCP status and watch calls use typed monitoring for owned launches. External attachment requires an explicit Slurm profile and job ID. |
| `chemtools/programs/nwchem/strategy/hpc_resources.py` | Scheduler discovery mixed with NWChem resource advice | NWChem resource provider plus target inspection | Keep basis and method sizing in NWChem. Move account, partition, and hardware queries to target inspection. |

`execution/legacy_runner.py` owns `run_calculation` and
`render_calculation_run`. Profile loading lives in `execution/profiles.py`;
read-only file and external Slurm status live in
`execution/external_status.py`. Molcas, DIRAC, and GRASP scheduler modules
import those focused owners. The former QE and QMCPACK low-level MCP and
application callers are gone. The old NWChem run and render names remain
direct aliases. `programs/nwchem/external_status.py` injects the NWChem
progress reader, and `core/runner.py` re-exports the retained owners for old
direct imports. Execution has no program-package imports.

Guided NWChem, Molcas, DIRAC, GRASP workflow, QE, and QMCPACK preparation no
longer call the legacy renderer. They read resolved migration profiles through
`execution/profiles.py` or accept a schema-2 target, build program-owned typed
plans, and let the selected executor render the exact command or Slurm script
used for approval.

Version 1 profiles now use one program installation shape:
`programs.<name>.launcher_argv` plus `executable_argv`. Molcas CASPT2
capability and DIRAC MPI and memory defaults live beside those arrays. The
four adapters still accept their previous field locations at lower
precedence, and the legacy renderer exposes `{program_command}` for templates
that use the standard block.

The typed execution path separates three operations:

1. A program backend builds a `LaunchPlan` containing program arguments.
2. An executor combines target-owned launcher and executable arrays with
   those arguments, then renders or runs the result for a named target.
3. An application service coordinates permission, persistence, status, and
   artifact observations.

The execution gate is enforced by the application service as well as MCP
dispatch. GRASP MCP handlers no longer call the direct Python
`run_grasp_exe` runtime, though that function remains as a compatibility API.
The same service resolves staging and output paths under target roots, records
synchronous terminal states, and limits default cancellation to runs launched
through its registry and target. NWChem status uses the retained local process
handle or target-owned Slurm queries, persists terminal launch metadata,
updates the linked scientific run, and records immutable stdout and stderr
observations. Molcas, DIRAC, and asynchronous GRASP workflow status and watch
also use owned local handles or target-owned Slurm queries through a shared
execution projection. These programs do not yet create scientific-run links,
so they do not record run-level artifact observations.

The NWChem, Molcas, DIRAC, and GRASP launch handlers now use that service.
Their compatibility adapters keep dry runs on the read-only renderer and
translate typed launch results back to existing response keys. Cancellation
accepts only a PID or job ID retained by the same MCP process. Molcas carries
its CASPT2 rank guard into the typed resource request. DIRAC keeps
`pam-dirac` MPI, paired input names, and memory flags in its launch plan
without adding a scheduler MPI prefix. GRASP keeps shell workflows intact
for asynchronous direct or Slurm launch and routes interactive steps through
the synchronous local contract. Advanced DIRAC checkpoint flags, PBS, LSF,
and alternate working directories still need target or adapter support.

### Workflow and application coordination

| Current module | Current responsibility | Target owner | Disposition |
| --- | --- | --- | --- |
| `chemtools/core/workflow.py` | Builds workflow dictionaries containing MCP tool names and parameters | Application workflow service | Keep DAG planning logic. Replace MCP tool names with typed actions or backend operations. MCP translates those actions to public names. |
| `chemtools/programs/nwchem/protocols.py` | NWChem protocol library, dynamic step generation, NWChem tool-name mapping | NWChem workflow provider | Keep calculation recipes and program steps. Remove public MCP names from the provider after typed actions exist. |
| `chemtools/programs/molcas/strategy/orchestrators.py` | Active-space and multi-step workflow coordination | Molcas backend and application services | Keep Molcas chemistry choices in the backend. Move filesystem writes, execution decisions, and multi-run state coordination outward when those services exist. |
| `chemtools/programs/grasp/strategy/workflows.py` | GRASP executable sequence and atomic-workflow knowledge | GRASP workflow provider | Keep the executable sequence and chemistry rules. Emit steps and expected artifacts instead of shell execution. |
| `chemtools/core/case_analysis.py` | Shared charge, spin, geometry, and imaginary-mode checks | Core chemistry rules | Keep program-neutral deterministic checks in core. |
| `chemtools/core/recovery.py` and `chemtools/core/issues.py` | Shared recovery and issue result structures | Core domain | Keep shared types and rules. Program-specific diagnoses remain in program packages. |

The current workflow engine returns actions such as
`check_nwchem_freq_plausibility` and `launch_nwchem_run`. That makes a
program-neutral module aware of the public MCP surface. The target plan
returns operations such as `output.frequencies`, `input.patch`, and
`execution.launch`; the MCP adapter chooses the public name.

### Runs, artifacts, and persistence

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/core/artifacts.py` | Immutable artifact identity, observations, expectations, provenance snapshots, freshness evidence, and versioned JSON conversion | Core domain | Keep filesystem observation, classification, and SQLite persistence outside this module. |
| `chemtools/core/artifact_classification.py` | Classifies caller-supplied paths from exact launch expectations and the selected backend's filename declarations | Core domain service | Keep the operation bounded and free of filesystem access. Content inspection and directory discovery belong in explicit application or backend operations. |
| `chemtools/persistence/artifacts.py` | Stores and loads normalized artifact collections, observations, expectations, and provenance metadata | Persistence adapter | Keep artifact bytes outside SQLite. Preserve global artifact identity, exact metadata conflicts, and append-only run membership. |
| `chemtools/application/legacy_artifacts.py` | Projects legacy input, output, and parent-run columns into backend-aware artifact candidates | Compatibility application service | Create artifact identities only for unambiguous kinds. Keep recorded paths out of observations and parent IDs out of provenance until exact snapshots exist. |
| `chemtools/persistence/sqlite.py` | Owns the shared SQLite connection and schemas for runs, artifacts, provenance, execution launches, and launch/run links | Persistence adapter | Keep schema changes here so all persistence callers use one migration path. |
| `chemtools/core/systems.py` | Immutable molecular and periodic system identity, geometry, lattice, k-points, pseudopotentials, charge, and spin | Core domain | Keep method, cutoff, smearing, executable, and scheduler choices in calculation or execution models. |
| `chemtools/persistence/runs.py` | Run CRUD, portable IDs, status fields, restart-chain lookup, and atomic execution-launch linking | Persistence adapter | Keep this module limited to scientific run rows and their execution links. |
| `chemtools/application/run_registry.py` | Re-exports run-record functions and owns campaigns, workflow state, input-batch generation, and an NWChem patch fallback | Compatibility application service | Move campaigns, workflow coordination, and batch generation along separate seams when another concrete need justifies the split. |
| `chemtools/core/{artifact_registry,registry_db,run_records,run_registry,legacy_artifacts}.py` | Exact imports from the new persistence or application owners | Compatibility facades | Preserve direct Python imports through the final compatibility release. No implementation belongs in these files. |
| `chemtools/core/session.py` | Markdown session-log writes and versioned output paths | Application support and artifact provenance | Keep compatibility functions. Later record written logs and renamed paths as artifacts and provenance events. |
| `chemtools/core/types.py` | Shared `TypedDict` result shapes | Core domain | Keep as the current interchange boundary. Add frozen, versioned models beside it and adapt old dictionaries during migration. |
| `chemtools/core/cube.py` | Program-neutral cube parsing | Core artifact parser | Keep if QE and other backends can consume the same contract. Do not attach program ownership without a format-specific reason. |

`persistence/runs.py` is the narrow storage module for scientific runs.
`application/run_registry.py` still contains
`generate_input_batch`, which reads and writes inputs, resolves the NWChem
backend, applies a fallback patcher, and registers runs. Batch generation
belongs in an application service backed by the catalog, an input adapter,
and run-record persistence. The former core paths are exact import facades.

### Evaluation and reference corpus

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/application/evaluation.py` | Discovers case files and contains separate NWChem, Molcas, DIRAC, and GRASP evaluators | Reference evaluation service | Keep case orchestration outside core. Program-specific checks still need provider ownership if this legacy evaluator survives compatibility cleanup. |
| `chemtools/core/eval.py` | Exact imports from `application/evaluation.py` | Compatibility facade | Remove after the final compatibility release if no direct caller remains. |
| `references/orbitron_contract_cases.json` | Eight external Orbitron cases with pinned hashes | Reference data | Keep the pinned compatibility cases, resolved Molcas vibration case, QE geometry comparisons, and failed-relaxation provenance case. |
| `chemtools/integrations/orbitron_contract.py` | Manifest loading, external-file verification, raw reference parsing, Orbitron comparison, reporting, and CLI behavior | Reference evaluation plus Orbitron integration | Move general manifest and bounded-access logic to the reference owner. Keep Orbitron invocation and field comparison in the integration contract. |
| `chemtools/integrations/orbitron.py` | Fixed-argument, versioned, read-only Orbitron subprocess boundary | Integration adapter | Keep this boundary. Application services consume it as optional evidence. |
| `chemtools/data/fblock` | Versioned GRASP, ATSP2K, and DIRAC scientific data | Committed scientific dataset | This is the canonical package-data copy with versioned metadata and typed validation. |
| `chemtools/reference/fblock_lookup.py`, `fblock_plan.py`, and `fblock_donors.py` | Exact state retrieval, ATSP2K recipe validation, donor dependency planning, consumer-scoped alias validation, and GRASP reference inputs | Reference application boundary | Keep lookup and planning read-only. Resolve only catalog state slugs; preserve external donor aliases until a reviewed mapping exists. |
| Other `notes/` material | Working scientific notes and lessons | Curation source | Keep outside runtime logic until a lesson has scope, status, evidence, and tests. |

Case evaluation now lives in `application/evaluation.py`. First-party callers
use that owner directly; `core/eval.py` keeps only exact compatibility imports.

### Shared scientific utilities

These modules already sit close to their intended owner:

| Current module | Target owner | Guidance |
| --- | --- | --- |
| `chemtools/core/units.py` | Core domain | Keep conversions explicit and tested. |
| `chemtools/core/geometry.py` | Core chemistry rules | Keep operations program-neutral and unit-aware. |
| `chemtools/core/thermochem.py` | Core chemistry rules | Keep equations independent of parser formats. |
| `chemtools/core/basis_advisor.py` | Core chemistry rules or application advice | Keep shared selection rules here; program-specific basis syntax stays in backends. |
| `chemtools/core/common.py` | Small shared utilities | Keep file and number helpers. Remove the hand-written program detector after registry detection covers all callers. |

`core/common.detect_program` recognizes only NWChem and Molcas, while
`core.registry` can detect all four registered programs. NWChem modules still
call the smaller detector. Remove it after those compatibility callers use
registry detection.

## Remaining compatibility seams

- `core/common.detect_program` still covers fewer programs than the backend
  catalog, and several NWChem compatibility functions call it directly.
- `core/workflow.py` returns low-level MCP action names. It is retained for
  the older protocol API; guided calculation planning uses backend-owned
  stage descriptions instead.
- `application/run_registry.py` still combines campaign operations, legacy
  workflow coordination, and NWChem batch generation. Its focused persistence
  owner is `persistence/runs.py`.
- `programs/nwchem/output.py` still routes one compatibility parser call to
  Molcas. New parser composition uses backend capabilities.
- Molcas, DIRAC, and GRASP scheduler modules repeat thin wrappers over the
  legacy execution engine. Owned guided launches use typed targets and
  execution services.
- Low-level Orbitron analysis tools remain integration-specific. The guided
  surface exposes only the bounded `visualize` operation.

Resolved migration tasks were removed from this section. The remaining seams
have active compatibility callers or explicit removal gates in
`SIMPLIFICATION_PLAN.md`.

## Refactor guardrails

- Preserve the generated MCP inventory and golden cases after each step.
- Keep refactors separate from new chemistry features.
- Add a target owner when a real caller needs it; do not create empty package
  scaffolding.
- Split large files along the responsibilities in this map, never by line
  count alone.
- Keep compatibility wrappers thin and give them removal criteria.
- Make the composition root own concrete imports and configuration.
- Reject reverse imports from core into MCP, public facades, or concrete
  program packages.
- Keep program-specific scientific judgment in the program backend even when
  an application service coordinates the workflow.
- Keep external corpus access read-only and manifest-selected.

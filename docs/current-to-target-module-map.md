# Current-to-target module map

Chemtools already has useful program-specific packages, parsers, chemistry
rules, and MCP transport code. The main architectural problem is ownership:
application orchestration and execution are spread across MCP handlers, the
NWChem-heavy public facade, `core`, and per-program scheduler wrappers.

This map assigns current modules to the owners defined by ADRs 001 through
005. It describes migration seams. It does not require an immediate directory
reorganization, and it does not authorize behavior changes during Phase 0.

Snapshot date: 2026-07-30

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
| MCP transport | JSON-RPC framing, MCP initialization, request and response envelopes |
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

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/mcp/cli.py` | Parses CLI options, selects modes and programs, starts the server loop | Composition root | Keep as the composition root. In Phase 1, own the built-in backend catalog here or in an adjacent composition module. Load backend objects into core separately from MCP tool metadata. In Phase 3, load the execution gate and named targets here. |
| `chemtools/__init__.py` | Re-exports a large NWChem-oriented API plus registry and evaluation functions | Compatibility facade | Keep public imports working. Stop using this facade inside MCP, core, and backend implementations. Add new public entry points from application services, then retire old exports under ADR 004. |
| `chemtools/api.py` | Aggregates NWChem parsing, drafting, strategy, execution, registry, and workflow functions | Compatibility facade | Preserve behavior while callers move to direct owners. Do not add QE or QMCPACK to this file. |
| `chemtools/api_input.py` | NWChem input and follow-up orchestration | NWChem backend and application services | Leave program syntax in the NWChem backend. Move only operations that coordinate parsing, review, file creation, and later execution into an application service. |
| `chemtools/api_strategy.py` | NWChem diagnosis and resource-advice facade | NWChem backend and application services | Keep deterministic NWChem rules in `programs/nwchem/strategy`. Replace this file with compatibility exports after callers use the backend catalog. |

The top-level facade is useful for existing Python callers, but it is a poor
internal dependency. `chemtools/mcp/tools/generic.py` currently imports it,
which makes generic MCP startup load NWChem-specific code. `core/eval.py`
imports the same facade back into `core`, creating an inward dependency on a
public compatibility layer.

### MCP transport, dispatch, and tool registration

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/mcp/server.py` | Content-Length and JSON-lines framing, JSON-RPC envelopes, common CLI arguments | MCP transport | Keep framing and envelope code here. Move mode-specific arguments out when ADR 002 replaces behavior modes. |
| `chemtools/mcp/dispatch.py` | Eager tool imports, manual definition concatenation, aliases, request dispatch, initialization | MCP transport and MCP adapter | In Phase 1, derive program tool imports and definitions from the built-in catalog. In Phase 4, move aliases to the compatibility registry from ADR 004. Keep protocol negotiation in the transport boundary. |
| `chemtools/mcp/decorator.py` | Mutable handler registries, program tags, mode tags, server identity, logging, active global filters, and the process-owned execution service | MCP adapter and composition state | Keep the decorator during compatibility work. It now creates one default-off service for analysis mode and one enabled service for local or HPC mode so launch ownership survives across MCP calls. Move this construction into a dedicated composition object when behavior modes are retired. Do not let backend code import it. |
| `chemtools/mcp/modes.py` | Hard-coded program names, tool filtering, runner-profile inspection, execution-mode selection | Legacy compatibility facade | Phase 1 derives program names from the catalog. Phase 3 replaces mode and `needs` logic with `requires_target` and `changes_execution_state`. Retain a legacy adapter through the ADR 002 window. |
| `chemtools/mcp/inventory.py` | Reads the live registries and emits contract metadata | MCP adapter | Keep it as the machine-readable contract ledger. Change its source to the catalog and compatibility registry as those become authoritative. |
| `chemtools/mcp/tools/*.py` | Schemas and handlers for 274 advertised tools | MCP adapter | Preserve names, schemas, and results. Handlers should gradually call application services or backend capabilities instead of importing concrete implementation functions. |
| `chemtools/mcp/nwchem.py` | Legacy NWChem MCP entry point | Compatibility facade | Keep until the CLI deprecation ledger permits removal. |
| `chemtools/mcp/nwchem_docs.py` | Standalone NWChem documentation CLI | Compatibility facade or NWChem backend CLI | Keep independently from MCP alias decisions. |

The transport boundary currently performs too much composition. It imports
all tool modules explicitly, and the tool modules import program
implementations directly. This is acceptable as a compatibility path, but it
must not become the route used to add QE and QMCPACK.

The NWChem MCP surface has a second temporary facade:
`chemtools/mcp/tools/_nwchem_base.py` imports most of `chemtools` and exposes
that namespace through wildcard imports. The split category modules are
smaller, but their dependency surface still comes from one broad file.
Phase 4 should move one handler family at a time to explicit application or
backend calls. A wholesale rewrite would make compatibility review difficult.

### Program model and registration

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/core/program.py` | Broad parser, drafter, strategist, binary, and example protocols | Core domain | Phase 1 adds operation-level capabilities and optional providers from ADR 001. Remove the broad protocol only after every generic caller uses capability checks. |
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
| `chemtools/programs/molcas` | Declared partial backend, parsers, input modules, orbital handling, active-space rules, recovery, docs | Keep the tested capability set. Convert `runtime.py` into a launch-plan provider in Phase 3. |
| `chemtools/programs/dirac` | Declared partial backend, parsers, HDF5 reader, basis data, atomic and core-ionization input builders | Keep the tested capability set. Preserve `.inp` and `.mol` staging rules in its launch-plan provider. |
| `chemtools/programs/grasp` | Declared parser backend, multi-artifact parsers, bounded radial-wavefunction and paired mixing/CSF inspection, explicit orbital merging, heredoc builders, workflow knowledge, diagnosis | Keep binary inspection, CSF interpretation, and first-donor-wins merging with the GRASP backend. Keep binary writes limited to the atomic, no-clobber merge contract. Model the working directory as an artifact collection. Convert direct executable calls into launch plans before exposing them through the common launch service. |
| `chemtools/programs/qe` | Declared periodic backend with input review, UPF inspection, output diagnosis, consistency checks, normalized geometry, and trajectory parsing | Keep program syntax and scientific interpretation here. `_elements.py` owns species-label normalization. `_coordinates.py` owns shared PWSCF output-card parsing and unit conversion; `input_geometry.py` normalizes the supported input coordinate forms; `geometry.py` selects one usable output snapshot; `trajectory.py` assembles optimization history; `trajectory_analysis.py` owns bounded periodic metrics and molecular structural checks shared by input and output review. |

Program packages may import core models and utilities. They must not import
MCP transport, public tool names, or scheduler implementations.

One current cross-program exception needs removal during Phase 1:
`chemtools/programs/nwchem/output.py` imports the Molcas output parser and
dispatches `parse_tasks` itself. The backend registry should own that
selection. NWChem output code should parse NWChem only.

### Execution and scheduling

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/application/execution_policy.py` | Immutable execution decisions plus disabled, status, and cancellation errors | Application policy | Keep policy result shapes independent of process, scheduler, and persistence adapters. The old `application.execution` imports remain compatible. |
| `chemtools/application/execution.py` | Default-off permission checks, read-only rendering, asynchronous and synchronous launch, read-only ownership resolution, owned local and Slurm status, terminal-state recording, and registry-bound cancellation | Application service | Status polling requires ownership but no second execution permission decision. Resolve configured targets at composition time before removing compatibility target adapters. |
| `chemtools/application/dirac_execution.py` | Applies DIRAC pairing and profile policy, coordinates typed execution, and translates legacy MCP responses | DIRAC compatibility application adapter | Keep `.inp/.mol` validation and exact output archival here while the public DIRAC tools remain. Add live checkpoint staging only after its destination and overwrite rules are explicit. |
| `chemtools/application/dirac_monitoring.py` | Combines owned local or Slurm execution status with legacy DIRAC file inspection | DIRAC monitoring application adapter | Keep DIRAC file interpretation here. Scientific-run linking and artifact observations remain separate work. |
| `chemtools/application/execution_monitoring.py` | Refreshes program-matched owned identifiers, projects typed process and Slurm evidence into compatibility responses, and polls owned status readers | Shared monitoring application support | Keep program parsing, scientific status, artifact recording, and scheduler subprocess calls out. |
| `chemtools/application/grasp_execution.py` | Coordinates typed GRASP workflow-script and synchronous per-executable launches and translates legacy MCP responses | GRASP execution compatibility adapter | Keep workflow path validation, exact output archival, capture files, session logs, launch IDs, and owned scheduler cancellation here while public response shapes remain. |
| `chemtools/application/grasp_monitoring.py` | Combines owned local or Slurm workflow status with legacy GRASP file inspection | GRASP monitoring application adapter | Keep synchronous per-executable results outside the watcher. Scientific-run linking and artifact observations remain separate work. |
| `chemtools/application/legacy_execution.py` | Projects typed local and Slurm results into existing MCP response dictionaries | Compatibility application adapter | Keep shared launch IDs, effective argv, submitted scripts, `.jobid` writes, timeout fields, and cancellation fields consistent while program-specific public tools remain. |
| `chemtools/application/molcas_execution.py` | Applies Molcas launch policy, coordinates typed execution, and translates legacy MCP responses | Molcas compatibility application adapter | Keep the CASPT2 guard and exact `.log` archival here while the public Molcas tools remain. Move to the program-neutral launch service after named target configuration replaces version 1 profiles. |
| `chemtools/application/molcas_monitoring.py` | Combines owned local or Slurm execution status with legacy Molcas file inspection | Molcas monitoring application adapter | Keep process and scheduler ownership in the execution service. Molcas scientific-run linking and artifact observations remain separate work. |
| `chemtools/application/nwchem_execution.py` | Converts version 1 NWChem profiles and legacy responses to typed calls; verifies and registers owned launches; synchronizes local or Slurm completion with linked runs and output observations | NWChem compatibility application adapter | Keep MCP response translation, launch/run registration checks, and NWChem artifact kinds here while the public tools remain aliases. Move the completion pattern to a program-neutral service only after another backend needs it. |
| `chemtools/application/nwchem_monitoring.py` | Combines owned execution status with NWChem output inspection and runs typed local or Slurm watch requests | NWChem monitoring application adapter | Keep chemistry progress, linked-run synchronization, and artifact observations in the NWChem path. Use the shared application helper only for execution response fields and polling. |
| `chemtools/core/execution.py` | Immutable launch plans, target-owned entrypoints, stdin and timeout intent, rendered commands, launch records, launch/run links, and asynchronous, status, or synchronous result models | Core domain | Keep this module free of process, scheduler, and SQLite calls. Store stdin digest and size in launch records, never stdin content. |
| `chemtools/core/monitoring.py` | Polling, adaptive intervals, compact history, timeouts, and terminal detection for calculation-status readers | Core service | Keep scheduler commands, persistence, and program parsing out. Typed `not_found` results must never imply completion. |
| `chemtools/core/slurm.py` | Typed Slurm status states, query evidence, job exit code, signal, and elapsed time | Core domain | Keep scheduler subprocess calls and state persistence out. Preserve raw scheduler state beside normalized status. |
| `chemtools/execution/_common.py` | Command rendering, allowed-root validation, and copy or symlink staging shared by execution adapters | Shared execution adapter support | Keep process handles, scheduler calls, program rules, and permission decisions out. |
| `chemtools/execution/local.py` | Captured synchronous execution, asynchronous launch, live-handle status, and live-handle cancellation | Local execution adapter | Status and cancellation must use retained process handles rather than arbitrary operating-system PIDs. |
| `chemtools/execution/slurm.py` | Slurm script rendering, submission, job-ID parsing, queue and accounting status, and target-command cancellation | Slurm execution adapter | Keep scheduler commands and script policy target-owned. Empty queue and accounting results must remain unknown rather than imply completion. |
| `chemtools/execution/executors.py` | Re-exports the local executor, Slurm executor, and work-root error | Compatibility facade | Preserve existing Python imports while callers move to `chemtools.execution` or the focused modules. |
| `chemtools/execution/legacy_profiles.py` | Loads version 1 profile files, merges defaults, and converts shared resource, hardware, module, program-installation, direct-command, and Slurm fields | Legacy target compatibility adapter | Keep program argument syntax and chemistry rules out. The standard `programs.<name>` installation block wins over old field locations. Remove this module with the version 1 profile format. |
| `chemtools/execution/legacy_status.py` | Inspects unowned PIDs, legacy Slurm, PBS, and LSF jobs, files, output tails, scheduler cancellation, and optional NWChem progress | Legacy status compatibility adapter | Keep typed owned execution out. Retain this path for unowned identifiers and direct Python callers until compatibility removal. |
| `chemtools/execution/launch_registry.py` | SQLite persistence and state-transition checks for execution launch records, including staging manifests, terminal metadata, and launch/run link lookup | Persistence adapter | Keep command and staging intent separate from artifact bytes. Local and Slurm NWChem completion use the link to synchronize the run; other programs still need the same integration. |
| `chemtools/core/runner.py` | Legacy resource inspection, script rendering, launch behavior, and direct compatibility imports for split profile and status modules | Compatibility facade plus legacy render and launch adapter | Keep neutral and NWChem-named Python imports stable during the compatibility window. Move the remaining render and launch implementation only when version 1 profiles are retired or need independent maintenance. |
| `chemtools/programs/nwchem/runner.py` | NWChem launch wrappers, progress chemistry, intervention advice, structure-drift analysis, comparison, and follow-up review | NWChem backend plus application services | Keep NWChem progress and chemistry assessment in the backend. Move launch coordination and cross-run comparison behind application services. |
| `chemtools/programs/nwchem/launch.py` | Builds NWChem launch plans and adapts version 1 NWChem profiles into typed targets | NWChem launch-plan provider and compatibility adapter | Keep NWChem arguments, filenames, and artifact expectations here. Remove the profile adapter after the version 1 compatibility window. |
| `chemtools/programs/molcas/runtime.py` | Builds the read-only legacy command preview and owns the shared CASPT2 detection and rank guard | Molcas compatibility facade and runtime rules | Keep `prepare_molcas_launch` stable while typed calls use the same guard through the launch-plan provider. |
| `chemtools/programs/molcas/launch.py` | Builds typed Molcas plans and adapts direct or Slurm version 1 profiles into targets | Molcas launch-plan provider and compatibility adapter | Keep pymolcas arguments, protected Molcas environment values, CASPT2 allocation changes, output rules, and dynamic Slurm project identity here. |
| `chemtools/programs/dirac/runtime.py` | Builds the read-only advanced `pam-dirac` preview and owns shared argument construction | DIRAC compatibility facade and runtime rules | Keep `prepare_dirac_launch` stable for `--copy`, `--put`, `--get`, and `--outcmo` previews while typed launch plans use the same argument builder. |
| `chemtools/programs/dirac/launch.py` | Builds typed DIRAC plans and adapts direct or Slurm version 1 profiles into targets | DIRAC launch-plan provider and compatibility adapter | Keep paired input names, `pam-dirac` MPI and memory arguments, output rules, and container installation data here. Add checkpoint staging after its live MCP contract is defined. |
| `chemtools/programs/grasp/launch.py` | Builds typed shell-workflow and interactive plans, declares reviewed GRASP entrypoints, and adapts direct or Slurm version 1 profiles into container targets | GRASP launch-plan provider and compatibility adapter | Keep executable selection, stdin, arguments, and workflow artifact expectations here. Remove the committed default container path with named target configuration. |
| `chemtools/programs/grasp/runtime.py` | Resolves the compatibility container, retains the direct Python runner, and formats session notes | GRASP compatibility runtime | MCP execution no longer calls its subprocess runner. Keep session formatting stable until session logs become artifact observations, then remove the direct executor after the compatibility window. |
| `chemtools/programs/grasp/strategy/runner.py` | Coordinates structured workflow steps through an injected runner and applies contained copy actions | GRASP workflow application service | MCP workflows inject the typed execution service. Record step outputs and the working directory as artifact observations when launch records link to scientific runs. |
| `chemtools/programs/{molcas,dirac,grasp}/scheduler.py` | Thin public wrappers around program-neutral legacy-profile runner functions | Compatibility facade | MCP status and watch calls use typed monitoring for owned launches. Retain these Python entry points for unowned identifiers and direct Python callers during the compatibility window. |
| `chemtools/programs/nwchem/strategy/hpc_resources.py` | Scheduler discovery mixed with NWChem resource advice | NWChem resource provider plus target inspection | Keep basis and method sizing in NWChem. Move account, partition, and hardware queries to target inspection. |

`core/runner.py` continues to export `run_calculation`,
`render_calculation_run`, `inspect_run_status`, and `watch_run`. Profile
loading now lives in `execution/legacy_profiles.py`, while status and watch
implementations live in `execution/legacy_status.py`. Molcas, DIRAC, and
GRASP scheduler modules import the neutral names. The old NWChem names are
direct aliases for Python compatibility. NWChem progress parsing remains an
optional legacy behavior; non-NWChem wrappers currently use file, process,
and scheduler evidence only.

Version 1 profiles now use one program installation shape:
`programs.<name>.launcher_argv` plus `executable_argv`. Molcas CASPT2
capability and DIRAC MPI and memory defaults live beside those arrays. The
four adapters still accept their previous field locations at lower
precedence, and the legacy renderer exposes `{program_command}` for templates
that use the standard block.

Phase 3 should separate three operations:

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

| Current module | Current responsibility | Target owner | Migration |
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
| `chemtools/core/artifact_registry.py` | Stores and loads normalized artifact collections, observations, expectations, and provenance metadata | Persistence adapter | Keep artifact bytes outside SQLite. Preserve global artifact identity, exact metadata conflicts, and append-only run membership. |
| `chemtools/core/legacy_artifacts.py` | Projects legacy input, output, and parent-run columns into backend-aware artifact candidates | Compatibility facade | Create artifact identities only for unambiguous kinds. Keep recorded paths out of observations and parent IDs out of provenance until exact snapshots exist. |
| `chemtools/core/registry_db.py` | Owns the shared SQLite connection and schemas for runs, artifacts, provenance, execution launches, and launch/run links | Persistence adapter | Keep schema changes here so all persistence callers use one migration path. |
| `chemtools/core/systems.py` | Immutable molecular and periodic system identity, geometry, lattice, k-points, pseudopotentials, charge, and spin | Core domain | Keep method, cutoff, smearing, executable, and scheduler choices in calculation or execution models. |
| `chemtools/core/run_records.py` | Run CRUD, portable IDs, status fields, restart-chain lookup, and atomic execution-launch linking | Persistence adapter | Keep this module limited to scientific run rows and their execution links. Application code should import it directly; the old facade remains for compatibility. |
| `chemtools/core/run_registry.py` | Re-exports run-record functions and still owns campaigns, workflow state, input-batch generation, and an NWChem patch fallback | Compatibility facade plus application services | The file is below the size ceiling after the first split. Move campaigns, workflow coordination, and batch generation along separate seams without wrapping the re-exported run functions. |
| `chemtools/core/session.py` | Markdown session-log writes and versioned output paths | Application support and artifact provenance | Keep compatibility functions. Later record written logs and renamed paths as artifacts and provenance events. |
| `chemtools/core/types.py` | Shared `TypedDict` result shapes | Core domain | Keep as the current interchange boundary. Add frozen, versioned models beside it and adapt old dictionaries during migration. |
| `chemtools/core/cube.py` | Program-neutral cube parsing | Core artifact parser | Keep if QE and other backends can consume the same contract. Do not attach program ownership without a format-specific reason. |

`run_records.py` is now the narrow storage module for scientific runs.
`run_registry.py` remains a compatibility facade and still contains
`generate_input_batch`, which reads and writes inputs, resolves the NWChem
backend, applies a fallback patcher, and registers runs. Batch generation
belongs in an application service backed by the catalog, an input adapter,
and run-record persistence.

### Evaluation and reference corpus

| Current module | Current responsibility | Target owner | Migration |
| --- | --- | --- | --- |
| `chemtools/core/eval.py` | Discovers case files and contains separate NWChem, Molcas, DIRAC, and GRASP evaluators | Reference evaluation service | Move out of core after the ADR 005 manifest model exists. Resolve backends through the catalog and keep program-specific checks with their owners. |
| `references/orbitron_contract_cases.json` | Eight external Orbitron cases with pinned hashes | Reference data | Keep the Phase 0 compatibility cases, resolved Molcas vibration case, QE geometry comparisons, and failed-relaxation provenance case. Migrate through ADR 005 without changing the current checker first. |
| `chemtools/integrations/orbitron_contract.py` | Manifest loading, external-file verification, raw reference parsing, Orbitron comparison, reporting, and CLI behavior | Reference evaluation plus Orbitron integration | Move general manifest and bounded-access logic to the reference owner. Keep Orbitron invocation and field comparison in the integration contract. |
| `chemtools/integrations/orbitron.py` | Fixed-argument, versioned, read-only Orbitron subprocess boundary | Integration adapter | Keep this boundary. Application services consume it as optional evidence. |
| `chemtools/data/fblock` | Versioned GRASP, ATSP2K, and DIRAC scientific data | Committed scientific dataset | Phase 5 moved the single canonical copy into package data, added versioned metadata and typed validation, and removed the old notes-tree copy. |
| `chemtools/reference/fblock_lookup.py`, `fblock_plan.py`, and `fblock_donors.py` | Exact state retrieval, ATSP2K recipe validation, donor dependency planning, consumer-scoped alias validation, and GRASP reference inputs | Reference application boundary | Keep lookup and planning read-only. Resolve only catalog state slugs; preserve external donor aliases until a reviewed mapping exists. |
| Other `notes/` material | Working scientific notes and lessons | Curation source | Keep outside runtime logic until a lesson has scope, status, evidence, and tests. |

`core/eval.py` currently imports `chemtools.api`, which imports NWChem modules
and the NWChem runner. This is the clearest dependency inversion in `core`.
The evaluation code is useful, but its owner is reference testing rather than
the domain foundation.

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
call the smaller detector. Phase 1 should make registry detection the single
path and remove program selection from `core/common.py`.

## Boundary violations to fix in migration order

1. Program membership has several authorities. The built-in catalog from ADR
   001 fixes this first.
2. Generic MCP code imports the NWChem-heavy `chemtools` facade. Generic
   handlers should call catalog-backed operations or application services.
3. Shared execution uses NWChem names and contains one NWChem-only progress
   path. ADR 002 separates launch plans, executors, and monitoring.
4. Program detection exists in both `core/common.py` and `core/registry.py`.
   The smaller detector has incomplete program coverage.
5. `core/eval.py` imports the public API, pulling program and execution code
   into core.
6. `core/workflow.py` emits MCP tool names instead of program-neutral
   operations.
7. `core/run_registry.py` still combines campaign persistence, input
   generation, and workflow coordination behind its compatibility facade.
8. `programs/nwchem/output.py` imports Molcas parsing instead of using the
   backend registry.
9. Per-program scheduler modules repeat one shared wrapper pattern.
10. The Orbitron contract combines general corpus access with
    integration-specific comparisons.

These are migration targets, not a request for one large refactor. The first
five ADRs deliberately place the work in separate phases.

## Phase ownership

### Phase 1: backend catalog

- Add the backend and capability models beside the current protocol.
- Add one explicit built-in catalog at the composition boundary.
- Derive CLI program choices, program loading, tool-module loading, and
  inventory ordering from that catalog.
- Convert generic parser handlers to capability checks.
- Remove import-time program registration only after catalog tests pass.
- Remove the NWChem-to-Molcas parser dispatch from
  `programs/nwchem/output.py`.

Do not move parser files or rename MCP tools in this phase.

### Phase 2: run and artifact models

- Add run, step, artifact, observation, and provenance models.
- Extend SQLite through additive migrations.
- Move input-batch generation and workflow advancement behind application
  services as those services adopt the new models.
- Record exact artifact observations for parser and comparison results, and
  use snapshot references for both provenance inputs and outputs.

Do not copy artifact bytes into the registry.

### Phase 3: execution targets

- Adapt legacy profiles into named targets.
- Convert one NWChem local path and one Slurm path to typed launch plans.
- Put generic process behavior behind local and Slurm executors.
- Convert Molcas, DIRAC, and GRASP runtime rules into launch-plan providers.
- Enforce execution permission, resolved-root containment, and registry-bound
  cancellation in the application service.
- Keep the NWChem-named runner functions and per-program scheduler functions
  as compatibility wrappers.
- Replace mode tags only after golden tool and execution contract tests pass.

### Phase 4: MCP application boundary

- Add guided application services one intent at a time.
- Make MCP handlers validate, call one service, and translate the result.
- Move aliases into the compatibility registry.
- Preserve each alias's old program and availability scope, and apply
  protocol-specific error and metadata rules.
- Reduce `_nwchem_base.py` by handler family rather than replacing it at once.
- Keep public result shapes stable until their declared migration boundary.

### Phase 5: knowledge and references

- Add the general reference manifest loader from ADR 005.
- Keep the canonical f-block dataset in package data with typed access, exact
  lookup, and provenance-aware ATSP2K/GRASP planning.
- Move reference discovery and evaluation out of `core/eval.py`.
- Keep exploratory and shelved cases out of default recommendations.

### Phase 6 and later

- Use Orbitron through the existing integration adapter.
- Move parser ownership only after differential evidence supports the move.
- Add QE and QMCPACK through the catalog, artifact model, and reference
  corpus. Do not copy the current NWChem execution shape into either backend.

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

## Phase 0 completion check

This map is complete enough for the Phase 0 gate when review agrees that:

- The target dependency direction is correct.
- Existing program parser, input, strategy, and binary directories remain in
  place.
- The built-in catalog is the first structural migration.
- Execution splits into launch planning, executors, and coordination.
- MCP handlers become adapters without a single large rewrite.
- SQLite is a persistence adapter rather than a workflow engine.
- Reference evaluation does not remain a dependency of core.
- QE and QMCPACK enter through the new boundaries instead of extending the
  NWChem facade.

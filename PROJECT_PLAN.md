# Chemtools MCP Project Plan

Status: QE input, output, conversion, and local pw.x execution support are complete; the configured linux-4090 local QE smoke has passed. QMCPACK analysis, conversion, and local execution scope are complete; scheduler profiles remain deferred. The optional companion science runtime is installed on linux-4090 with RDKit molecular preflight, inspected Open Babel SMILES/MOL conversion, Orbitron Python periodic-electronic-structure and structure-identity inspection, and bounded PySCF single-point execution. Other Orbitron Python API methods remain deferred until they have a bounded response contract and a passing owned fixture.
Last updated: 2026-08-02

## Purpose

Chemtools MCP should help an AI assistant reason about computational
chemistry calculations the way an experienced practitioner does. It should
review inputs before execution, inspect the complete set of run artifacts,
recognize scientifically suspicious results, explain uncertainty, and suggest
the next useful check.

This plan covers the architectural cleanup needed to support that goal across
multiple chemistry programs, local machines, and HPC systems. It also defines
how the existing notes become testable domain knowledge, how Orbitron fits
into the system, how Quantum ESPRESSO and QMCPACK should be added, and how an
optional companion scientific Python runtime can provide portable chemistry
utilities and small reference calculations.

This is a living plan. Work proceeds through explicit gates. Later phases
should be revised when evidence from earlier phases changes an assumption.

## Project objective

Build a maintainable MCP server with:

- A small default set of chemistry-aware workflow tools.
- Program backends with truthful, optional capabilities.
- One execution model for local processes and schedulers.
- First-class support for multi-file and multi-step calculations.
- Deterministic checks backed by notes, fixtures, and source references.
- Curated reference cases that let the assistant retrieve a proven workflow,
  a known failure, or an explicitly unresolved calculation.
- Structured output that tells an AI assistant what was checked, what remains
  uncertain, and what to do next.
- An optional Orbitron integration for canonical chemistry data, advanced
  parsing, and visualization.
- An optional, isolated scientific Python companion runtime for PySCF, RDKit,
  Open Babel, and Orbitron's Python API.
- A path for adding Quantum ESPRESSO and QMCPACK without adding another
  program-specific architecture.

## Non-goals

The refactor will not:

- Rewrite the working runner algorithms without evidence that they need it.
- Hide every chemistry program behind an identical feature set.
- Turn the raw notes into executable rules automatically.
- Make Orbitron a required dependency for core input review or execution.
- Make the companion runtime a required dependency for core input review,
  execution, or analysis.
- Expose arbitrary Python, package imports, or shell commands through the MCP.
- Copy or scan the full `/home/charlie/input_examples` tree during normal MCP
  operation.
- Expose arbitrary shell execution through the MCP.
- Add every program mentioned in the notes during the first migration.
- Preserve every internal Python interface. The public MCP compatibility
  surface receives stronger protection than internal implementation details.
- Treat an exit code of zero as sufficient evidence of scientific success.

## Current state

The repository already contains useful domain logic, parsers, workflow tools,
and execution code. The main problem is that several architectural ideas grew
in parallel and no longer form one clear model.

### Tool inventory

The live registry currently exposes 328 tools across six registered backends:

| Tool group | Count |
| --- | ---: |
| NWChem | 101 |
| Molcas | 45 |
| DIRAC | 39 |
| GRASP | 51 |
| Quantum ESPRESSO | 20 |
| QMCPACK | 14 |
| Generic | 58 |
| Total | 328 |

The generated inventory in `docs/tool-inventory.json` and
`docs/tool-inventory.md` now records the live totals, input schemas,
capabilities, program ownership, mode visibility, and compatibility aliases.
README totals come from that inventory.

Most tools have no execution capability tag. The current capability inventory
is:

| Capability | Count |
| --- | ---: |
| Executable | 40 |
| Executable or scheduler | 5 |
| Registry | 18 |
| Runner profile | 4 |
| Scheduler | 3 |
| No capability tag | 242 |

### Program support

Program support is uneven, which is normal for chemistry software. The current
interface does not represent that unevenness clearly.

| Program | Parser | Input drafting | Strategy | Binary lookup | Examples |
| --- | --- | --- | --- | --- | --- |
| NWChem | Yes | Yes | Yes | Yes | Yes |
| Molcas | Yes | Yes | No | Yes | No |
| DIRAC | Yes | No | No | Yes | No |
| GRASP | Yes | No | No | Yes | No |
| Quantum ESPRESSO | Yes | No | Yes | No | No |

`Program` currently requires parser, drafter, and strategist fields, while
implementations fill unsupported slots with `None`. Capability checks should
become part of the interface instead of relying on nullable required fields.

### Structural problems

- Program lists and imports are repeated in the MCP mode and dispatch layers.
- Dispatch imports every program eagerly and concatenates tool definitions by
  hand.
- Analysis, local, and HPC are presented as three behavior modes even though
  local and HPC are execution targets.
- The generic runner is exposed through NWChem-named public functions.
- DIRAC and GRASP scheduler code call those NWChem-named functions.
- Runner profile shapes differ between programs.
- The input model is molecule-shaped and cannot represent periodic systems
  cleanly.
- A run is often modeled as an input and an output file, which is insufficient
  for GRASP, DIRAC, Quantum ESPRESSO, and QMCPACK.
- The default MCP surface exposes many narrow functions without a smaller
  guided workflow surface.
- Valuable lessons in `notes/` are readable by a person but are not yet linked
  to checks, fixtures, or retrieval records.

### What should be preserved

The runner has useful internal separation for subprocess execution, scheduler
rendering, staging, and status handling. Existing run-layer notes also conclude
that its basic structure is sound. The refactor should preserve those
algorithms while replacing confusing names, public boundaries, and profile
shapes.

The existing design principle of thick deterministic tools and a thin language
model remains appropriate. Chemistry judgments that can be stated as rules
belong in Python with tests. The assistant should combine and explain those
results rather than reconstruct the checks from prose on every call.

### Available reference material

Two different reference sources are available and should remain distinct.

The in-repository f-block atomic library is a versioned scientific dataset. It
covers 31 elements and 633 atomic states across GRASP2018, ATSP2K, and selected
DIRAC workflows. The current v2 GRASP references were rebuilt on 2026-07-28.
Only 36 states converge from the default estimate; 597 require a donor,
multi-donor merge, ATSP seed, or staged orbital birth.

`/home/charlie/input_examples` is a broad local development corpus. It
currently contains about 58 GB and 7,953 files. The program areas most relevant
to this project include:

| Program area | Files | Size |
| --- | ---: | ---: |
| NWChem | 2,172 | 4.6 GB |
| OpenMolcas | 946 | 37 GB |
| DIRAC | 97 | 73 MB |
| GRASP2018 | 201 | 5.3 MB |
| Quantum ESPRESSO | 1,092 | 11 GB |
| QMCPACK | 2,009 | 857 MB |

The local corpus is a working tree with committed, modified, deleted, and
untracked material. It also contains third-party tutorials, pseudopotentials,
binary checkpoints, and large scratch artifacts. Chemtools should not copy it
wholesale or assume one Git revision describes its current contents.

## Design principles

### Chemistry owns the verdict

Execution status, parser status, and chemistry status are separate facts. A
calculation can exit successfully and still be unusable because it converged
to the wrong state, wrote stale artifacts, used inconsistent
pseudopotentials, or passed a weak same-family comparison.

### Program support is capability-based

A backend advertises only the operations it can perform. Adding a parser
should not require placeholder implementations for drafting, strategy, or
execution.

### Programs and execution targets are independent

A program backend describes the software, files, chemistry checks, and launch
requirements. An execution target describes where and how commands run.
NWChem on a workstation and NWChem under Slurm should use the same program
backend with different targets.

### Runs are artifact collections

Scientific interpretation must consider checkpoints, converted files,
orbitals, pseudopotentials, restart state, provenance, and sidecar data.
Primary input and output files are entries in that collection, not the entire
run.

### High-level tools return evidence

Workflow tools should report observations, checks performed, verdicts,
uncertainties, source references, and next actions. A verdict without the
evidence used to reach it is too weak for tricky calculations.

### Compatibility is deliberate

Current MCP tool names remain available as aliases during migration. New
guided tools become the documented default. Alias removal requires a separate
decision, usage evidence, and a deprecation period.

### Raw notes are source material

The notes are not executable specifications. Each lesson must be curated,
given scope and confidence, then connected to a deterministic rule, a
regression fixture, a workflow recipe, or a searchable knowledge record.

### Reference cases have declared status

A calculation becomes a reference only after its expected facts, provenance,
and validation status are recorded. A difficult or interesting calculation is
not automatically a good example. Failed and unresolved cases remain valuable
when they are labeled accurately.

## Target architecture

```text
MCP tools
  |
  +-- review_input
  +-- inspect_run
  +-- plan_calculation
  +-- launch_run
  +-- monitor_run
  +-- compare_runs
  +-- visualize
  +-- search_knowledge
  |
Application services
  |
  +-- Program backends
  +-- Cross-program workflows
  +-- Chemistry rules
  +-- Execution targets
  +-- Artifact store and provenance
  +-- Reference corpus
  +-- Orbitron bridge
```

The MCP layer should validate requests and format structured responses.
Application services should coordinate program, chemistry, execution, and
artifact components. Program-specific code should not import MCP transport
details or scheduler implementations.

## Core domain models

ADRs 001 through 003 define the initial Python boundaries. Implementation may
refine names without changing their ownership or contract.

### Program backend

```python
@dataclass(frozen=True)
class ProgramBackend:
    name: str
    capabilities: frozenset[ProgramCapability]
    artifact_kinds: Mapping[str, ArtifactKindSpec]
    detector: ProgramDetector
    parser: OutputAdapter | None = None
    inputs: InputAdapter | None = None
    binary: BinaryAdapter | None = None
    diagnostics: DiagnosticAdapter | None = None
    resources: ResourceAdvisor | None = None
    progress: ProgressAdapter | None = None
    consistency: RunConsistencyAdapter | None = None
    examples: ExamplesCorpus | None = None
```

The fields describe real seams, not a requirement to create eight classes for
every program. A small backend may implement several capabilities in one
module. The catalog exposes what exists and rejects unsupported operations
with a structured capability error.

One built-in backend catalog at the composition boundary replaces repeated
program tuples, eager imports, and manual registry concatenation. Core
receives backend objects without importing MCP metadata.

### Scientific system

```python
ScientificSystemSpec = MolecularSystemSpec | PeriodicSystemSpec
```

`MolecularSystemSpec` retains charge, multiplicity, atoms, and molecular
metadata. `PeriodicSystemSpec` adds lattice vectors, periodic dimensions,
k-point sampling, pseudopotential assignments, and periodic spin settings.

Quantum ESPRESSO fields must not be stored as an untyped bag in
`program_options`.

### Run artifacts

```python
@dataclass(frozen=True)
class RunArtifacts:
    run_uid: str
    artifacts: tuple[ArtifactRef, ...]
    observations: tuple[ArtifactObservation, ...]
    expectations: tuple[ExpectedArtifact, ...]
    provenance: tuple[ProvenanceRecord, ...]
```

ADR 003 defines stable artifact IDs, point-in-time observations, role and kind
separation, evidence-based freshness, and append-only transformation lineage.
The model supports artifacts outside the working directory and compound
directory artifacts. Seed wavefunctions retain the producing program, method,
configuration, exact snapshots, and donor lineage. This is required for
ATSP2K to GRASP2018 workflows where the same GRASP stdin sequence can mean
different things depending on how the seed was built.

### Launch plan

```python
@dataclass
class LaunchPlan:
    program: str
    program_arguments: list[str]
    environment: dict[str, str]
    working_directory: Path
    staged_files: list[StagedFile]
    expected_artifacts: list[ExpectedArtifact]
    resources: ResourceRequest
    progress_detector: ProgressDetector | None
```

The program runtime adapter supplies program arguments and artifact rules. The
selected target owns launcher and executable arrays. An executor assembles
`launcher_argv + executable_argv + program_arguments` and runs it. This keeps
chemistry software syntax out of scheduler implementations and target command
prefixes out of program adapters.

### Execution target

```python
@dataclass
class ExecutionTarget:
    name: str
    executor: ExecutorKind
    allowed_work_roots: tuple[Path, ...]
    scheduler: SchedulerDefaults | None
    hardware: HardwareDescription
    programs: dict[str, ProgramInstallation]
```

Initial executor kinds should be `local` and `slurm`. PBS and LSF can be added
when there is a real target and test case.

### Review result

```python
@dataclass
class ReviewResult:
    observations: list[Observation]
    checks_performed: list[CheckRecord]
    verdict: Verdict
    uncertainties: list[Uncertainty]
    knowledge_refs: list[str]
    next_actions: list[RecommendedAction]
```

The result should distinguish `pass`, `warning`, `failure`, `inconclusive`,
and `unsupported`. A parser refusal or missing artifact should not be
reported as scientific agreement.

## Execution model

The current three-way analysis, local, and HPC mode should be replaced with
two independent choices:

1. Whether execution is enabled.
2. Which named execution target is selected.

The default remains analysis-only:

```toml
[chemtools]
enable_execution = false
default_target = "linux4090"

[targets.linux4090]
executor = "local"
allowed_work_roots = ["/path/to/calculations"]

[targets.linux4090.programs.nwchem]
executable_argv = ["/path/to/nwchem"]
launcher_argv = ["mpirun", "-np", "{mpi_ranks}"]

[targets.stampede3]
executor = "slurm"
allowed_work_roots = ["/path/to/scratch"]

[targets.stampede3.programs.nwchem]
executable_argv = ["/path/to/nwchem"]
launcher_argv = ["ibrun"]

[targets.stampede3.programs.qmcpack]
executable_argv = ["/path/to/qmcpack"]
launcher_argv = ["ibrun"]
```

The exact configuration format should follow the repository's existing config
approach. This example records the intended separation.

Execution tools must remain absent or return a clear disabled response when
`enable_execution` is false. An MCP client cannot select an arbitrary command.
It selects a configured program, operation, and target.

The application execution service repeats the permission check below MCP,
resolves all working and output paths under allowed roots, and limits default
cancellation to runs launched through the same registry and target.

### Local development paths

Machine-specific paths belong in environment variables or ignored local
configuration, never in committed portable defaults.

The initial Orbitron adapter should use this resolution order:

1. An explicit adapter argument for tests and callers with local config.
2. `CHEMTOOLS_ORBITRON_CLI`.
3. `orbitron` on `PATH`.
4. Report the integration as unavailable.

The selected-target lookup belongs in the execution-target phase because that
configuration model does not exist yet. It will supply the explicit adapter
argument without changing the subprocess boundary.

On `linux-4090`, the current development setting is:

```bash
export CHEMTOOLS_ORBITRON_CLI=/home/charlie/projects/orbitron/target/release/orbitron
export CHEMTOOLS_REFERENCE_CORPUS=/home/charlie/input_examples
```

The adapter must run `orbitron --version` before its first operation and retain
the reported version and commit in provenance. The inspected binary currently
reports `orbitron-cli 0.4.0 (20e81d225b4c)`.

`CHEMTOOLS_REFERENCE_CORPUS` enables opt-in corpus tests and local reference
lookup. The MCP must not recursively scan the directory on startup. It reads a
curated manifest and resolves only entries listed there.

### Executor responsibilities

`LocalExecutor` should:

- Run an approved launch plan without a shell.
- Enforce working-directory and timeout boundaries.
- Capture stdout, stderr, exit status, and expected artifact state.
- Support cancellation and progress inspection where the program permits it.

`SlurmExecutor` should:

- Render and submit a batch script from the same launch plan.
- Apply target defaults and explicit resource requests.
- Record job ID, script, submission command, and scheduler state.
- Keep cluster-specific launchers in target configuration.
- Avoid assuming that all Slurm systems use `srun` or `mpirun`.

## MCP tool surface

The default server should expose a small guided set:

| Tool | Responsibility |
| --- | --- |
| `review_input` | Parse and check an input before execution |
| `inspect_run` | Classify artifacts, parse results, and assess scientific status |
| `plan_calculation` | Recommend method, steps, resources, and checks |
| `launch_run` | Execute an approved plan on a configured target |
| `monitor_run` | Report scheduler, process, artifact, and scientific progress |
| `compare_runs` | Compare states, methods, structures, or repeated calculations |
| `visualize` | Request a supported Orbitron inspection or render operation |
| `search_knowledge` | Retrieve scoped rules, traps, and workflow guidance |
| `find_reference_case` | Find validated, failing, or unresolved calculations similar to a proposed run |

The final set may contain up to twelve tools if a real workflow does not fit
these nine. A new high-level tool requires a distinct user intent, not merely
a distinct Python function.

Existing program-specific tools remain available in focused developer
toolsets. They are useful for testing, direct control, and compatibility, but
should not dominate the default AI-facing surface.

## Knowledge pipeline

The knowledge system has three layers:

```text
notes/                 raw observations and working records
knowledge/             curated, scoped knowledge cards
chemtools/.../checks   deterministic rules with tests
references/manifest    curated example and artifact metadata
```

Every hard-won lesson should produce at least one durable project artifact:

- A deterministic check.
- A regression fixture.
- A workflow recipe.
- A searchable knowledge card.
- A reference case with pinned expected facts.

### Knowledge card shape

```yaml
id: qmcpack.jastrow_vmc_energy_gate
programs: [qmcpack]
workflows: [jastrow_optimization]
kind: validation
status: accepted
confidence: high
applies_when:
  wavefunction: optimized_jastrow
  trial_scf_energy: matched
claim: A post-optimization VMC mean must not exceed its matched trial-SCF energy.
check:
  tool: check_qmcpack_vmc_energy_gate
  criterion: VMC LocalEnergy <= trial SCF energy
failure:
  severity: error
  action: Re-optimize or replace the Jastrow seed before starting DMC.
sources:
  - notes/fblock/qe-qmcpack-oncvpsp.md#44-jastrow-optimization-on-hostile-landscapes
tests:
  - tests/test_qmcpack_scalar.py::test_check_vmc_energy_gate_compares_matching_hartree_energies
```

Cards must state when a claim applies. Program version, method, relativistic
treatment, artifact family, and workflow stage may all affect scope.

### First curated lessons

The first knowledge milestone should encode these cross-program lessons:

1. Silent success is common, so require positive independent verification.
2. Artifacts produced by the same upstream step do not independently validate
   one another.
3. Agreement between starting guesses from the same method family is one
   measurement, not independent confirmation.
4. Cheap sign, ordering, or monotonicity checks can expose a wrong basin.
5. Optimizer failure sentinels must lose to every valid objective value for
   both minimization and maximization.

Each lesson needs at least one fixture showing a pass and one showing the
failure it is intended to catch.

## Reference corpus

The reference corpus has three storage tiers. [ADR 005](docs/adr/005-reference-corpus-boundaries.md)
separates storage, redistribution permission, case purpose, and scientific
review status.

### Committed fixtures

Small, permission-clean text fixtures live in this repository and run in every
test environment. They should include the minimum input, output excerpt, and
expected facts needed to prove a parser or chemistry check.

### Committed scientific datasets

Maintained first-party scientific data lives in a versioned repository
directory with declared scope, provenance, validation rules, and
redistribution metadata.

The f-block atomic library belongs in this tier. Phase 5 moved its one
canonical copy to `chemtools/data/fblock`, where installed packages can access
it. Its `grasp/fblock-all.json` file is the source for typed lookup and
validation.

### External development corpus

Large or locally maintained cases stay under
`/home/charlie/input_examples`. Tests that need them run only when
`CHEMTOOLS_REFERENCE_CORPUS` is configured.

A committed JSON manifest should identify selected cases without copying
their artifacts:

```json
{
  "id": "qe.fe_bcc.scf",
  "programs": ["qe"],
  "status": "validated_reference",
  "purposes": ["differential_contract", "workflow_recipe"],
  "artifacts": [
    {
      "id": "primary_output",
      "roles": ["primary_output"],
      "path": "qe/Fe/fe.scf.out",
      "storage_tier": "external_reference",
      "size_bytes": 12345,
      "sha256": "<pinned hash>",
      "redistribution": "review_required",
      "source": "local-research",
      "attribution": "Charles Peterson",
      "license": {
        "identifier": null,
        "terms": null
      },
      "permission_evidence": null
    }
  ],
  "expected": {
    "atoms": 2,
    "calculation": "scf",
    "total_energy_ry": -495.637697
  },
  "tags": ["periodic", "magnetic", "metal", "orbitron"]
}
```

Each manifest entry needs:

- Stable ID and program or workflow.
- Per-artifact storage tier, source, attribution, license evidence, and
  redistribution status.
- Case purpose, recorded separately from scientific status.
- Relative artifact paths, exact sizes, roles, and content hashes.
- Status: `validated_reference`, `regression_failure`, `exploratory`, or
  `shelved`.
- Expected parsed facts and scientific checks.
- Provenance and any license restriction.
- Required companion files.
- Tags for method, element, property, failure mode, and execution shape.

The first manifest should contain a small pilot set:

- GRASP Si serial and MPI equivalence.
- GRASP Cf high-Z HF bootstrap.
- GRASP Th serial SCF plus parallel RCI.
- NWChem one wrong-state case and its corrected run.
- OpenMolcas one complete workflow and one deliberate parity failure.
- DIRAC U heavy-element or X2C versus 4c case.
- QE Fe SCF and FeO relaxation.
- One QMCPACK determinant-only and Jastrow comparison.

The Tm3+ and Md3+ LS-coupling study remains `shelved`. It is useful as a
methodology warning and future research case, but it must not be returned as a
validated recipe.

## Orbitron integration

Orbitron should own broad file detection, canonical chemistry data, advanced
parsing, and visualization. Chemtools should own calculation planning, input
review, chemistry judgments, run diagnosis, recovery advice, execution, and
cross-artifact consistency.

| Concern | Chemtools | Orbitron |
| --- | --- | --- |
| Input review and drafting | Owns | May provide structure conversion |
| Program-specific diagnostics | Owns | Supplies parsed facts |
| Workflow planning | Owns | No |
| Local and scheduler execution | Owns | CLI is a configured subprocess |
| Canonical structures and trajectories | Consumes | Owns |
| Orbitals and volumetric data | Consumes | Owns |
| Rendering and interactive inspection | Requests | Owns |
| Scientific verdict | Owns | Supplies evidence |

### Integration boundary

Add an optional subprocess adapter at
`chemtools/integrations/orbitron.py`. It should:

- Discover a configured Orbitron executable.
- Check the CLI and schema versions.
- Invoke a fixed set of operations without a shell.
- Parse versioned JSON into Chemtools models.
- Preserve source and provenance fields.
- Return `unsupported` when Orbitron lacks an operation or format.

Orbitron now offers versioned JSON for `info`, `inspect`, and its `analyze`
subcommands. Geometry, molecular-orbital, atomic-population, and vibrational
summaries are exposed through fixed read-only MCP operations.

Initial integration should use Orbitron as a second parser and differential
check. Parsing ownership should move only after corpus tests show agreement
and document intentional differences.

Orbitron remains optional. Core Chemtools operations need a clear degraded
behavior when the CLI is absent or its schema version is unsupported.

### Current Orbitron integration findings

The local release binary is usable now for an early contract adapter, before
the full visualization phase. The adapter currently permits only:

- `--version`
- `info <source> --json`
- `inspect <source> --json`
- `analyze geometry <source> --json`
- `analyze orbitals <source> --json --frontier 3`
- `analyze populations <source> --json --top 8`
- `analyze vibrations <source> --json --top 10`
- `render <source> --output <ephemeral PNG> --width 1024 --height 768`

Orbitron 0.4.0 at commit `20e81d225b4c` provides the current machine contract
needed by the Phase 0 adapter:

1. `info --json` uses `orbitron.info/2`.
2. `inspect --json` uses `orbitron.inspect/2`.
3. Both envelopes include producer name, version, commit, and structured
   warnings.
4. The QE Fe output returns a nonzero 2.8660238 angstrom cubic cell.
5. The deliberate Molcas parity failure returns `run.outcome = "failed"` with
   the source diagnostic and module sequence.
6. Orbital and population analysis use their `/2` schemas with explicit spin,
   occupancy-policy, expected-charge, and residual fields.
7. Geometry analysis uses `orbitron.analyze.geometry/3` with explicit angstrom
   distance units, one of four geometry roles, and a source description.
8. Vibration analysis uses `orbitron.analyze.vibrations/4` with explicit mode
   set, units, scaling, total and per-mode displacement availability,
   thermochemistry units, and the role and source of the analyzed geometry.

The original QE Fe zero-cell and relaxation task-energy defects are fixed and
pinned by exact-value differential cases. The Molcas HCN calculation now
selects the final complete MCLR frequency block. The eight-case differential
contract reports eight agreements with no disagreement, refusal, or missing
reference at commit `20e81d225b4c`. Three geometry cases compare Si SCF, Fe
SCF, and converged FeO `vc-relax` atom counts, elements, Cartesian bounds,
periodic flags, cell vectors, and provenance.

The failed benzene relaxation is the eighth case. Orbitron labels its geometry
as `last_attempted` and reports `step 8 of 8; the run stopped without
converging`. Chemtools keeps that geometry as diagnostic evidence and does not
promote it to the common converged-output geometry contract.

The `inspect` command's CLI description also lists DIRAC, NWChem, Molpro, and
Molcas while the user guide describes periodic and QE summaries. The help,
implementation contract, and JSON schema should name the same support.

### Early feedback loop

Orbitron contract testing moves into Phase 0 because both projects are under
active development. Full visualization and parser-ownership decisions remain
in the later Orbitron phase.

The early loop should:

1. Run the configured binary against a small manifest subset.
2. Validate the JSON envelope and required invariants.
3. Compare overlapping Chemtools and Orbitron facts.
4. Write a short machine-readable discrepancy report.
5. Fix Orbitron defects in the Orbitron repository, then pin the passing
   commit in the report.

Chemtools should not silently compensate for an Orbitron parser defect unless
the workaround is version-scoped and covered by a regression test.

## Companion scientific Python runtime

Chemtools should support one optional, isolated `chemtools-science` runtime.
It is a configured execution dependency, not a dependency of the MCP server
itself. The first runtime contains PySCF, RDKit, Open Babel, Basis Set
Exchange, h5py, Orbitron's
Python API, and a small `chemtools-science-runner` entry point. Conda or micromamba is
the supported installation mechanism because this set includes native Python
extensions and external libraries. A plain virtual environment may be useful
for a PySCF-only installation, but is not the portable baseline for the shared
runtime.

The MCP server should invoke the runner through a named execution profile. A
request is a versioned JSON document, and each invocation writes a versioned
JSON result, a bounded text log, and declared artifacts. The runner must not
accept caller-provided Python source, module names, function names, shell
fragments, output locations, or package-install requests.

| Component | Initial responsibility | Boundary |
| --- | --- | --- |
| RDKit | Molecular input validation and explicit molecular-property evidence | Does not silently repair a submitted chemistry model |
| Open Babel | Declared format conversion | Converted connectivity, charge, aromaticity, and stereochemistry remain conversion evidence, not scientific truth |
| Basis Set Exchange | Offline rendering of an explicit, selected basis block | Does not choose a basis/ECP or validate relativistic and angular conventions |
| h5py | Fixed QMCPACK artifact-layout metadata inspection | Does not expose a general HDF5 browser or decode coefficients, density grids, walkers, or estimator values |
| Orbitron Python API | Canonical structure and orbital-data operations supported by its versioned contract | The existing optional CLI bridge remains valid and independent |
| PySCF | Small molecular RHF, UHF, RKS, and UKS reference calculations | A converged calculation does not validate a production result or replace relativistic, multireference, periodic, or QMC workflows |

Every result must retain the request, runner schema version, environment lock
digest, Python and package versions, executable versions where applicable,
execution argv, and artifact hashes. A companion result without that provenance
is diagnostic evidence only and cannot be promoted to a curated reference.

The first PySCF scope is molecular single-point SCF and DFT. Periodic PySCF,
geometry optimization, MP2, multireference methods, custom Python snippets,
GPU acceleration, and QMCPACK trial-wavefunction generation remain deferred.
Geometry optimization adds an explicit optimizer dependency and restart
semantics; periodic calculations require a separate pseudopotential and
Hamiltonian-comparability decision rather than being treated as a QE fallback.

### Companion-runtime delivery sequence

1. [x] Define the versioned request/result schemas and a read-only runtime
   health probe. The probe reports availability and versions without installing
   or modifying the environment.
2. [x] Commit the declarative environment specification. Installation is an
   explicit developer operation, never an MCP side effect. The linux-4090
   `chemtools-science` environment currently has Python 3.12.13, PySCF 2.13.1,
   RDKit 2025.09.5, Open Babel 3.1.0, Basis Set Exchange 0.12, h5py 3.16.0, and an editable Orbitron 0.4.0 bridge
   built from commit `d913197bdb35`. The exercised linux-64 Conda resolution
   is now bundled as package data and every science-runner response records
   its SHA-256, interpreter, package versions, and Orbitron native-extension
   hash when available.
3. [x] Add fixed RDKit molecular preflight with provenance and refusal
   behavior for unsupported inputs.
4. [x] Add fixed Open Babel SMILES/MOL conversion with independent RDKit
   inspection before and after conversion. The converted text, hashes, package
   versions, molecular evidence, and named differences are returned together.
   SMILES-to-MOL does not generate coordinates and marks its zero-coordinate
   output as connectivity-only evidence. Unsupported formats, caller flags,
   and outputs that RDKit cannot inspect are refused.
   The reviewed seven-case fixture corpus now pins neutral, charged, aromatic,
   disconnected, radical, chiral, and MOL-to-SMILES outcomes against Open
   Babel 3.1.0 and RDKit 2025.09.5. Its opt-in checker treats the known chiral
   canonical-SMILES difference as expected evidence, not a passing round trip.
5. [x] Add a fixed Orbitron Python periodic electronic-structure summary. It
   reads one absolute local source through the configured companion runtime,
   records source hash and package provenance, and returns compact Fermi,
   gap, band-dimension, and DOS-dimension evidence with projections omitted.
   The committed synthetic VASP fixture pins the initial end-to-end response.
6. [x] Add fixed QMCPACK HDF5 layout inspection. It recognizes only the
   pw2qmcpack electronic-structure wavefunction, variational-parameter,
   walker-configuration, and statistics layouts from one absolute local path.
   It returns named bounded metadata and cannot browse arbitrary datasets or
   decode coefficients, grids, walkers, or estimator values. A real oxygen
   `pwscf.h5` in the external example corpus confirms the conversion layout.
7. [x] Add the PySCF molecular single-point runner. It accepts RHF, UHF, RKS,
   and UKS requests with typed Cartesian atoms, records process completion
   separately from SCF convergence, and rejects custom Python, periodic,
   geometry-optimization, and multireference requests. The reviewed fixture
   corpus covers closed-shell RHF/RKS, open-shell UHF/UKS, a completed but
   unconverged SCF, and an electron-spin-inconsistent runtime refusal.
8. [x] Add an opt-in comparison report for deliberately matched PySCF and
   production-code calculations. It reports matched and unmatched settings;
   it does not choose a correctness winner from an energy difference alone.
   `compare_pyscf_reference_calculation` accepts one completed PySCF result
   and caller-declared reference evidence, retaining geometry, settings,
   electron-count, convergence, energy, and optional CUBE comparisons. The
   first reusable field-comparison component is
   `compare_cube_densities`: it compares caller-declared density CUBE files
   only on a common grid and nuclear geometry, with explicit density-value
   units and no resampling. The bounded PySCF runner can write a derived-name
   total-density CUBE after converged SCF, with its hash and value unit in the
   result. `compare_cube_orbitals` now adds phase-aligned normalized overlap
   for one caller-matched non-degenerate orbital CUBE pair under that same
   strict geometry and grid contract. `compare_cube_orbital_subspaces` now
   adds phase- and rotation-invariant principal-angle comparison for two
   caller-declared equal-dimension orbital sets, refusing rank-deficient
   sampled fields. The PySCF runner can now write up to eight selected
   restricted or spin-resolved orbital CUBEs with derived names, MO energy,
   occupation, and hash provenance. `draft_nwchem_pyscf_reference` now
   provides the first NWChem adapter: it extracts only explicitly-unit-normalized Cartesian
   geometry, unambiguous library-basis, charge, multiplicity, and converged
   SCF/energy evidence. PySCF method, density fitting, and effective electron
   count remain caller declarations because generic NWChem text cannot map
   those fields safely. `run_nwchem_pyscf_matched_reference` now composes a
   comparison-ready NWChem draft, one bounded local PySCF execution, and the
   existing evidence-only report. It refuses execution for incomplete NWChem
   evidence and requires caller-declared PySCF XC for DFT. It can also attach
   one caller-declared, pre-written NWChem total-density CUBE and request the
   bounded PySCF density CUBE in the same run. The existing comparator keeps
   incompatible grids as `not_comparable`. The NWChem CUBE drafters now derive
   the matching PySCF 3-bohr-margin box and endpoint grid from one explicit-unit
   Cartesian geometry when the caller supplies `pyscf_compatible_grid_points`.
   The reviewed NWChem/PySCF fixture corpus now records NWChem 7.2.2 Apptainer
   captures for H2 RHF, H2O RHF, triplet O2 UHF, H2 B3LYP, and an intentional
   H2O SCF failure that proves no PySCF run starts from incomplete evidence.
9. [x] Curate PySCF knowledge cards only from reviewed fixtures and documented
   failure modes. The initial cards distinguish process completion from SCF
   convergence and an electron-spin-inconsistent runtime refusal from an
   unconverged SCF.
10. [x] Attach one bounded runtime-provenance record to every
   `chemtools-science-runner` response, including request refusals. It binds
   the fixed operation and request SHA-256 to the result, records interpreter
   and package versions, hashes the bundled resolved linux-64 Conda lock, and
   records the editable Orbitron native-extension hash when available.

## GRASP2018 and ATSP2K f-block references

ATSP2K should initially be modeled as a seed-producing backend used by a
cross-program atomic workflow. It does not need a broad MCP tool surface before
that workflow exists.

The first service should answer:

```text
plan_fblock_atomic_state(element, charge, configuration)
  -> reference state
  -> J blocks and CSF counts
  -> seed class
  -> donor lineage
  -> ATSP2K input when required
  -> GRASP2018 workflow
  -> expected configuration-average energy
  -> checks and known traps
```

The in-repository atomic library supplies the reference data. The service must
preserve these distinctions:

- Cold start.
- Single donor.
- Multi-donor merge.
- ATSP2K HF seed converted for GRASP2018.
- Staged birth of one or both relativistic orbital components.

The Th neutral false-vacuum case should be the first scientific regression. A
Thomas-Fermi start reaches a stationary solution that passes internal checks
but is about 9 eV above the accepted basin. The test should prove that internal
convergence alone cannot mark the state as validated.

The U multi-donor case should be the first lineage regression. Its 5f and 6d
orbitals must come from separate donors because neither can be born reliably
in the other's presence. The generated GRASP stdin does not encode that
history, so artifact provenance must.

The f-block catalog is production reference data, but individual claims still
need method metadata. The current GRASP values are DC plus Breit
configuration averages from the 2026-07-28 v2 rebuild. They should not be
presented as experimental levels or correlated spectroscopic predictions.

## Quantum ESPRESSO plan

Quantum ESPRESSO should be the first new backend because it forces the
periodic-system and multi-artifact boundaries and Orbitron already has useful
QE parsing support.

The first vertical slice covers:

- `pw.x` input parsing, linting, and drafting.
- SCF and relaxation output diagnosis.
- Pseudopotential assignment and elemental coverage.
- Cutoff, k-point, occupation, smearing, charge, and spin consistency checks.
- `pw.x`, `ph.x`, and `pw2qmcpack.x` workflow steps.
- Expected output and restart artifacts.
- Local and Slurm launch plans.
- Orbitron-backed structure, trajectory, and periodic-data inspection.

The backend should identify unsupported calculation types and companion-program
inputs rather than accept them through a generic input dictionary. `bands.x`,
`dos.x`, `pp.x`, and `projwfc.x` are currently identified by their top-level
namelist and declined as unsupported rather than being misread as incomplete
`pw.x` decks.

## QMCPACK plan

QMCPACK should be a separate backend added after the Quantum ESPRESSO slice is
stable. The cross-program workflow service should connect them.

The initial XML-input review is now registered. It parses `simulation` and
`qmcsystem` files, records includes, HDF5 sidecars, particle sets,
pseudopotential references, Hamiltonians, and QMC blocks, and rejects malformed
or incomplete reference syntax. Explicit non-positive DMC counts and timesteps
are rejected, and unresolved numeric controls or unrecognized `nonlocalmoves`
values are kept as review warnings. Both `warmupSteps` and legacy
`warmupsteps` are accepted as nonnegative controls but must agree when both
appear. The deprecated `nonlocalpp` parameter is also reported because QMCPACK
ignores it. Simultaneous `targetWalkers` and `total_walkers` declarations must
agree.
`determinantset` entries that use `twistnum` without `twist` are reported as
ambiguous, and legacy inline `slaterdeterminant` entries are flagged for
migration to `sposet_collection`.
It resolves referenced HDF5 sidecars relative to the input, verifies their
supported HDF5 superblock signature, and reports missing, invalid, or older
files. A missing or invalid variational-parameter override sidecar is an error
because the note-backed workflow treats it as the authoritative optimized
parameter store. It lints each present included XML file within the existing
file and size limits, but does not merge includes, decode HDF5 datasets, or run
QMCPACK. If an override sidecar shares an XML file with inline `coefficients`,
the review warns that the arrays may be stale display values and preserves the
sidecar as the parameter source to carry forward. Primary-log inspection records
version, exact completion, a timing-only legacy completion evidence state,
final reported execution time, unique warnings, and
top-level VMC, DMC, or linear-optimization sections, but does not interpret
their scientific impact. Legacy VMC and DMC driver names normalize to those
section identities, and their execution times remain timing evidence rather
than completion. For a log with multiple numeric QMCPACK banners, version,
completion, timing, warnings, sections, optimizer diagnostics, project labels, and
particle pools describe the trailing run; its starting line is retained as evidence.
Matching QMCPACK input and output compare declared `linear`,
`vmc`, and `dmc` method presence only, treating repeated optimizer sections as
internal iterations. When both files name a project, they also compare the XML
project ID with the primary log's printed project label. This is not input-control
or output-provenance evidence. Repeated identical labels remain usable; distinct
labels are retained as runtime evidence and are not compared. Supported particle-pool summaries retain runtime
particle-set and group counts, including whether the listed groups sum to the
printed set count. Direct XML particle sets with an explicit set count or complete
group counts are compared with matching runtime sets. Included XML remains unmerged,
so unresolved sets are not compared. Matching named groups are also compared
independently, so a matching total can still expose a different partition. Anonymous
legacy groups are not compared. Ambiguous repeated direct XML or runtime-set names
are reported as `not_checked`. Missing runtime groups and runtime group totals that
disagree with the printed particle-set total also report each affected direct XML
group as `not_checked`. A direct XML particle set without an unambiguous
matching runtime set is reported as `not_checked`. Explicit `minwalkers` threshold warnings retain their
occurrence count, printed threshold, and minimum immediately preceding observed
effective weight without becoming a convergence verdict.
An explicit QMCPACK warning that a non-positive input parameter was replaced
with a positive value is an exception: Chemtools retains the requested and
replacement values with their QMC section and source lines and returns an
`input_parameter_auto_corrected` review verdict. A completed process then
means the altered controls ran, not that the requested deck did.
The QMCPACK parser has small corpus-derived fixtures for a modern oxygen DMC
run, a concatenated legacy hydrogen log, and a legacy linear-optimizer run with
effective-walker failures. They preserve the output shapes without vendoring
full calculations.
Explicit invalid-cost and parameter-reversion records are retained as
optimizer-review verdicts even when process completion is present. They do not
mark task execution failed or establish a scientific result.
The generic inspection result exposes those compacted records as
`qmcpack:optimization_messages`, retaining their codes, occurrence counts, line
ranges, and every affected QMC section when the log supplies section starts.
Failed and good linear-method candidate steps retain separate occurrence counts
and largest reported parameter changes. They describe trial history, not
optimization or scientific convergence, and do not replace the completion
assessment.
Legacy effective-walker recovery messages retain their occurrence count and
smallest printed value. Those messages and the historical `Revertting to old
Parameters` spelling are optimizer recovery evidence, not convergence or
population-control thresholds.
Direct
and nested include paths are checked with explicit file and size limits. HDF5
references in included XML are resolved relative to their declaring file.
Included XML content remains unmerged and its wavefunction semantics are out
of scope. Scalar block files can be summarized, and explicitly labelled DMC
series can be reblocked after a chosen warmup discard and fit to zero time step.
A scalar file with both `LocalEnergy` and `LocalEnergy_sq` reports whether every
recorded block satisfies the second-moment bound within a scale-aware numerical
tolerance. This is estimator-consistency evidence, not an uncertainty or
convergence result. An `AcceptRatio` column also reports whether every value is
within `[0, 1]` without recommending an acceptance-rate target.
Scalar filenames matching `project.sNNN.scalar.dat` retain their project label
and series index as filename identity only; that convention does not establish
the source QMC block or its controls.
When such a scalar file is supplied as a related `inspect_run` artifact, its
filename project label is compared with the primary log label. This also does
not establish source-run or QMC-block lineage. If the primary log lacks an
unambiguous project label, or the scalar filename lacks a recognized project
label, the comparison is explicitly `not_checked`.
`BlockWeight` evidence records whether every value is positive, without changing
the unweighted scalar analyses.
When scalar output also has `Kinetic` and `LocalPotential`, it compares their
reported sum with `LocalEnergy` using a print-precision-aware tolerance. This
is a reported-field balance check, not Hamiltonian-completeness evidence.
A semilocal pseudopotential inspector records the expected `hartree`/`r*V`
encoding, linear channel grids, declared grid counts, local-channel presence,
recognized angular labels, and unique angular-momentum/spin channel pairs as
structural evidence without claiming transferability.
A DMC population record can report retained walker count, living fraction, and
diffusion efficiency against an input-derived target, with source block-index
continuity and excluded-row counts by malformed, non-finite, and
non-integral-index cause. If no valid row remains, it identifies those causes
in its refusal. The series tool keeps
T-move and no-T-move points separate and requires input-derived time-step
metadata because sequential scalar file names do not encode controls. Neither
tool treats its summary as a convergence or autocorrelation result. A
caller-supplied `potential_label` for every series point reports uniform or
mixed potential identity; omitted labels remain `not_assessed`. Confirmed
mixed identity preserves points but withholds the combined time-step fit. Both
the series and paired locality-shift results report excluded scalar rows by
malformed, non-finite, and non-integral-index cause, warn when block indices
have gaps or restarts, mark a bounded reader as incomplete, and retain an
inconsistent scalar second-moment bound or out-of-bounds acceptance ratio as a
quality warning. Non-positive block weights are reported separately. A VMC energy
gate compares a post-optimization scalar mean against the matching trial SCF
energy in Hartree and rejects a Jastrow that raises the VMC energy. The gate
preserves malformed, non-finite,
and non-integral-index row warnings alongside block-index continuity and
bounded-read warnings, plus an inconsistent scalar second-moment bound or
out-of-bounds acceptance ratio or non-positive block weight, without treating
them as a convergence verdict.
`analyze_qmcpack_dmc_input_series` now extracts the DMC controls from selected
direct blocks in the primary XML input, so callers no longer need to copy them
into the time-step request. It does not merge included XML. Callers still
supply the scalar-file-to-QMC-block association because scalar data does not
contain the source block identity; the response preserves that as caller-supplied
provenance rather than treating it as established lineage.
When both paths provide project labels, input-bound DMC tools also compare the
input project ID with the scalar filename label. This is a mismatch check, not
source-block provenance, and a mismatch produces a top-level binding warning.
`compare_qmcpack_tmove_locality_shift_from_input` applies the same boundary to
one T-move/no-T-move pair and rejects selected input blocks with the wrong
`nonlocalmoves` setting.
`inspect_qmcpack_dmc_population_from_input` also reads the walker target from
a selected DMC input block while preserving the caller-supplied population-file
binding as unverified provenance.
`inspect_qmcpack_referenced_pseudopotentials` now exposes bounded deck-level
semilocal-card inspection outside the QE conversion workflow and rejects a
declared `elementType` that disagrees with the card header symbol. It does not
claim pseudopotential-family equivalence or transferability.
A determinant-only VMC offset inspection compares at least two caller-labelled
states with matching trial SCF energies. It preserves their supplied state
order and reports positive-offset evidence and the observed strict offset trend
alongside scalar-input quality warnings, without setting a small-offset threshold
or establishing Hamiltonian consistency.
A matched T-move/no-T-move comparison reports the signed locality shift with
the control's separate walker-count provenance. Supplied potential labels must
match; unlabeled pairs preserve the comparison while marking that identity as
not assessed. Both the paired comparison and time-step series require distinct
scalar artifacts for every supplied run.
Input review also records DMC campaign structure, including the requested
time-step ladder, block-count ordering, declared walker targets from either
`targetWalkers` or `total_walkers`, and a matched no-T-move control when present.
The ladder records whether its time steps match the four-point f-block reference,
with its optional fifth fine point.
With at least three distinct T-move time steps, it also records whether every
no-T-move control matches an interior ladder point; shorter ladders remain
`not_assessed` for that check. It also reports whether the control count matches
the one-repeat f-block reference protocol. At a shared time step, it compares
declared blocks, steps, warmup, walker target, move, and checkpoint settings.
When linear optimization, VMC, and DMC blocks coexist, it records whether their
order matches the f-block production reference without rejecting partial decks.
For a complete production sequence, it separately checks that every linear
block is enclosed by a `loop` with a positive `max` from 6 through 8.
It retains named linear `cost` entries and compares `MinMethod`, energy cost,
and unreweighted-variance cost with the f-block reference recipe.

The first QMCPACK slice covers:

- XML input parsing and validation.
- Wavefunction, determinant, Jastrow, particle-set, and pseudopotential
  references.
- HDF5 sidecar discovery and freshness checks.
- Determinant-only VMC reference runs.
- Jastrow variational checks.
- Optimization failure detection.
- Local and Slurm launch plans.
- Structured scalar and estimator summaries.
- Input-labelled DMC time-step series with reblocking and separate T-move fits.
- Input-bound DMC time-step series that retains caller-supplied file bindings.
- DMC population records with input-derived walker-count comparison.
- Post-optimization VMC energy gate before DMC continuation.
- Matched T-move/no-T-move locality-shift comparison.
- Input-review evidence for DMC ladder and control structure.

The `qe_to_qmcpack` workflow should validate:

- Consistent structure, electron count, charge, spin, and boundary conditions.
- The expected `pw2qmcpack.x` HDF5 artifact exists and is current.
- Pseudopotential families match across DFT and QMC representations.
- Semilocal QMC potentials are present when required.
- Authoritative HDF5 coefficients take precedence over stale XML copies.
- A determinant-only reference is retained as an independent check.

The first artifact-lineage step now exists as
`inspect_qe_qmcpack_conversion_artifacts`. It checks declared QE input,
completed SCF output, and `.pwscf.h5` paths without guessing artifact
locations. It validates the HDF5 signature at the supported superblock offsets
without decoding HDF5 datasets. The companion
`inspect_qe_qmcpack_conversion_deck` follows the bounded QMCPACK XML include
graph and requires an orbital reference to that exact artifact; a
variational-parameter sidecar does not satisfy the check. The companion
`inspect_qe_qmcpack_conversion_pseudopotentials` inspects every declared QMC
pseudopotential for the supported semilocal XML evidence. The companion
`inspect_qe_qmcpack_conversion_electrons` compares QE electron evidence with
the QMCPACK Hamiltonian target's particle groups. The companion
`inspect_qe_qmcpack_conversion_atoms` compares QE's declared atom count with
QMCPACK non-electron particle-set sizes. The companion
`inspect_qe_qmcpack_conversion_geometry` compares explicit QE periodic cell
and atomic-position evidence with one explicit QMCPACK `bohr` simulation cell
and ion particle set, modulo periodic translations. It deliberately returns
review for other boundary conditions, coordinate units, or ambiguous particle
sets, and `not_ready` for non-finite or singular periodic cells. QE-UPF
equivalence and physical charge and spin interpretation still need a separate
evidence model. The companion `inspect_qe_qmcpack_conversion_charge` compares
QE UPF valence accounting and `tot_charge` with QMCPACK ion `valence`
parameters and electron-particle counts. It leaves incomplete particle or
valence declarations at review. Remaining cross-code consistency checks still
need a separate evidence model. The companion
`inspect_qe_qmcpack_conversion_species` compares QE atomic-species elements
with QMCPACK pseudopotential `elementType` declarations, but does not claim
pseudopotential family equivalence. The companion
`inspect_qe_qmcpack_conversion_valence` compares QE UPF `z_valence` with
QMCPACK XML `zval` when each element has one unambiguous value. Bounded UPF
headers retain the declared local channel and total projector count, and expose
per-channel counts only when the declared `PP_BETA` entries are all visible in
the bounded preamble. `inspect_qe_qmcpack_conversion_projectors` activates for
DMC blocks in the primary or bounded included QMCPACK XML, preserving each
source path and local block index. It requires bounded `PP_BETA` evidence only
for nonlocal `NC` sources, because native `SL` UPFs are already semilocal. It
returns review for a repeated QE projector channel or a source UPF type other
than `NC` or `SL`, prompting confirmation of a separately generated semilocal
QMC potential. Scattering and family equivalence still need a separate
evidence model; the projector counts do not establish DMC compatibility.
`inspect_qe_qmcpack_conversion_spin` now compares a QE `nspin=2` input with an
explicit fixed `tot_magnetization` against QMCPACK `u` and `d` electron groups.
It leaves unconstrained, noncollinear, and spin-orbit cases at review rather
than inferring a physical spin state.

`inspect_qe_qmcpack_conversion_ion_species` now compares QE atomic element
counts with explicitly sized QMCPACK ion groups without requiring positions,
covering species evidence that sits between atom-count and geometry checks.

`plan_qe_qmcpack_conversion` now declares the SCF, converter, and deck
validation handoff against a caller-supplied `.pwscf.h5` path. It records the
same QE preflight evidence as the readiness checker, but intentionally leaves
the converter command line empty because execution and converter-option
selection remain outside the supported scope.

`draft_pw2qmcpack_input` now renders the repeated `&inputpp` form from the
local QMCPACK examples. It copies only explicit QE `prefix` and `outdir`
values and holds missing paths at review rather than guessing QE defaults.
The generic input review recognizes the bounded converter form when
`write_psir` is present and no other options appear. Missing `prefix` or
`outdir` remains a converter-specific handoff warning; other `&INPUTPP` inputs
remain in the unsupported `pp.x` scope.

The QE output adapter also recognizes the `pw2qmcpack.x` banner and records
reported `esh5 create` HDF5 paths and converter timing. It reports artifact
evidence without promoting older logs to a clean completion. A newer log is
successful only when it reports an HDF5 artifact and the terminal `JOB DONE.`
marker.
It diagnoses the recorded collected-wavefunction and gamma-trick converter
failures without guessing a preceding QE input correction from the log alone.
When both converter artifacts are supplied, the input-output check compares
the demonstrated `outdir/prefix.pwscf.h5` handoff without reading HDF5 data or
claiming converter completion.
With an explicit sidecar, it also compares the artifact path with the reported
output path, resolving a relative report from the converter output directory.
The matching sidecar must be at least as new as the converter input. This
timestamp comparison does not decode or validate HDF5 contents or converter
completion because the converter can append its log after creating the sidecar.
An explicit `.pwscf.h5` sidecar is also classified as a QE binary checkpoint
and wavefunction artifact, so generic inspection records its metadata without
reading its contents.

`draft_ph_x_input` now drafts one explicit q-vector phonon calculation from
the same QE paths. It preserves the caller's job title and q-vector, leaves
advanced phonon settings unset, and returns a Gamma-point advisory rather than
selecting `epsil` on the user's behalf.

The generic `review_input` path now recognizes the same bounded `&INPUTPH`
form. It parses the job identifier, namelist, and one q-vector, checks that
explicit path provenance is present, and leaves q grids and lists at limited
review rather than misclassifying them as invalid single-q decks.

`inspect_qe_qmcpack_conversion` returns all supported conversion evidence in
one response, so MCP clients can obtain an overall readiness verdict before
using the granular checks to diagnose individual failures. When the optional
science runtime is configured, the aggregate also compares recognized
`pw2qmcpack` HDF5 atom count, per-element atom counts, total electron count,
and two-spin populations against the QMCPACK deck. An absent companion runtime
is explicitly non-gating; a readable but unrecognized or incomplete HDF5
layout remains review-required. It does not decode coefficients, density grids,
walker coordinates, estimator values, or arbitrary datasets.
`inspect_qe_qmcpack_conversion_execution` adds explicit converter input and
output evidence. It requires `pw2qmcpack.x` to report completion, then composes
the QE-to-converter `prefix` and `outdir` handoff with the existing
converter-input, reported-HDF5-path, supplied-sidecar, QMCPACK-deck, and fixed
HDF5-metadata checks without decoding coefficients or arbitrary datasets, or
making an energy claim. Converter options outside the
documented `&INPUTPP` form remain review-required.
The aggregate conversion checks share the QE input, QE output, converter input,
and HDF5 artifact evidence. XML-reference, pseudopotential, particle-count,
geometry, spin, and charge checks are separate cross-checks on that handoff;
they do not independently establish electronic-structure validity.

### QE-to-QMCPACK tool structure

`chemtools.mcp.tools.qe` is the stable catalog entry point and re-exports the
conversion handlers from `chemtools.mcp.tools.qe_qmcpack`. The handler module
now owns only MCP transport and static schemas. QE preconditions and
cross-program particle-count evidence live in `chemtools.programs.qe.qmcpack`;
QMCPACK HDF5-reference and pseudopotential evidence live with their respective
QMCPACK program modules. Keep the current handler imports and tool names
stable. HDF5 decoding remains deferred until the separate conversion-evidence
model defines the supported `.pwscf.h5` structure and its bounds.

ONCVPSP or `ld1.x` support should be a later pseudopotential-generation
backend. It should not be hidden inside the Quantum ESPRESSO backend because
generation, solid-state validation, and QMC conversion have separate artifacts
and failure modes.

## Compatibility policy

The migration should keep existing user workflows working while internals
change.

- Capture the current MCP tool names, schemas, and representative outputs.
- Preserve tool names as compatibility aliases where a guided service replaces
  them.
- Mark aliases in generated inventory and documentation.
- Emit deprecation metadata only after the replacement is functionally
  equivalent.
- Do not remove an alias in the same phase that introduces its replacement.
- Record intentional output changes in an architecture decision record and
  migration note.
- Keep analysis-only startup as the safe default.

Internal modules can change when tests cover the behavior and no documented
Python API promises otherwise.

## Testing strategy

### Behavior lock

Before structural changes:

- Generate a machine-readable inventory of tools, input schemas,
  capabilities, and backend ownership.
- Add golden request and response cases for representative tools.
- Pin registry totals by group.
- Capture representative success, warning, failure, inconclusive, and
  unsupported responses.

### Unit tests

Pin exact classifications, parsed values, generated launch arguments, and
diagnostic codes. Assertions should verify expected values rather than mere
existence.

### Fixture tests

Maintain small, attributable artifact sets for each program and workflow.
Fixtures should include:

- Normal completion.
- Parser refusal.
- Truncated output.
- Exit success with scientific failure.
- Restart or sidecar inconsistency.
- Stale converted artifact.
- Unsupported feature.

### Differential tests

Compare Chemtools and Orbitron on a real corpus for fields both systems claim
to parse. Every case must end as:

- Agreement.
- Documented intentional difference.
- Tool refusal.
- Defect.

Refusals must remain distinct from agreement.

External corpus tests must verify pinned sizes before hashing or parsing. A
size or hash mismatch marks the case as changed and skips comparison until its
expectations are reviewed.

### Execution contract tests

Use fake executables and scheduler clients to test launch-plan behavior without
running expensive chemistry. Real smoke tests should be small and explicitly
selected for each configured machine or cluster.

### Scientific regression tests

Each knowledge rule needs a positive and negative example. Where a rule depends
on a numerical tolerance, the test should state the physical or numerical
reason for that tolerance.

## Roadmap

### Phase 0: Behavior lock and decisions

Deliverables:

- Generated live tool and capability inventory.
- README counts reconciled against that inventory.
- Golden tests for representative MCP behavior.
- Architecture decision records for program capabilities, execution targets,
  artifacts, compatibility, and reference-corpus boundaries.
- A [current-to-target module map](docs/current-to-target-module-map.md) from
  existing modules to intended owners.
- Orbitron CLI resolution and version smoke test.
- A pinned differential case for the QE Fe zero-cell defect.
- Reference-manifest schema and a three-case pilot.

Exit criteria:

- Current public behavior can be compared before and after a refactor.
- No program list must be counted by hand for documentation.
- The five load-bearing architecture decisions are accepted.
- Orbitron JSON compatibility is measured instead of assumed.
- External corpus cases are accessed only through reviewed manifest entries.
- Existing tests pass from the repository environment.

### Phase 1: Backend catalog and truthful capabilities

Deliverables:

- `ProgramBackend` capability model.
- One built-in backend catalog.
- NWChem, Molcas, DIRAC, and GRASP adapted without feature additions.
- Structured unsupported-capability errors.
- Dispatch and mode code read from the catalog.

Progress:

- [x] Add the operation-level capability enum, frozen backend model, structured
      unsupported-capability error, and provider validation.
- [x] Add the composition-owned built-in catalog for the four current programs,
      with exact provider and tool-definition parity tests.
- [x] Make CLI program choices and MCP tool aggregation read from the catalog.
- [x] Adapt the four current plugins to validated backend declarations.
- [x] Gate generic backend-provider handlers through declared capabilities and
      return the stable `unsupported_capability` MCP error object.
- [x] Remove import-time self-registration after catalog-backed loading passes
      the compatibility suite.

Exit criteria:

- Adding a test backend requires one registration point.
- No implementation uses required interface slots filled with `None`.
- Current MCP tools still resolve through compatibility aliases.
- Program inventory and tests derive from the catalog.

### Phase 2: Artifact and system models

Deliverables:

- `RunArtifacts` and provenance types.
- `MolecularSystemSpec` and `PeriodicSystemSpec`.
- Artifact classification for representative existing program runs.
- Compatibility adapters for current molecule and input models.

Progress:

- [x] Add immutable artifact, observation, expectation, provenance, and
      freshness types with versioned JSON round-trip tests.
- [x] Make backend artifact declarations use the shared `ArtifactRole` enum.
- [x] Add molecular and periodic system specifications with typed lattice,
      k-point, pseudopotential, charge, coordinate, and spin fields.
- [x] Add a compatibility adapter for the current molecular `InputSpec`.
- [x] Add bounded, declaration-driven artifact classification for representative
      NWChem, Molcas, DIRAC, and GRASP paths, including explicit ambiguity.
- [x] Add read-only compatibility projection from legacy `input_file`,
      `output_file`, and `parent_run_id` records without inventing observations
      or snapshot provenance.
- [x] Add stable UUID run IDs beside local integer IDs, including migration
      of existing registries.
- [x] Add normalized SQLite storage for artifact metadata, observations,
      expectations, ordered run membership, and provenance snapshots.

Exit criteria:

- DIRAC and GRASP examples can be represented without pretending each run is
  one input and one output.
- Periodic systems have typed lattice, k-point, and pseudopotential fields.
- Artifact freshness and producing step can be expressed.

### Phase 3: Execution boundary

Deliverables:

- `LaunchPlan`, `ExecutionTarget`, `LocalExecutor`, and `SlurmExecutor`.
- One consistent target configuration shape.
- NWChem migrated first as the reference backend.
- Molcas, DIRAC, and GRASP migrated only after NWChem contract tests pass.
- Existing runner functions retained temporarily as adapters.

Progress:

- [x] Add immutable launch-plan, resource, installation, target, scheduler,
      staging, and rendered-command models.
- [x] Add schema-2 named target YAML and JSON loading at the MCP composition
      root. Keep execution permission explicit and independent from legacy
      tool-visibility mode. Guided NWChem and Quantum ESPRESSO launch can select
      a configured target or the server default without reading a version 1
      profile; equivalent local and Slurm targets render the same
      approval-bound plan as the migration adapter. The QE migration passed
      140 focused checks and all 1,919 tests with the external corpus; isolated
      base and DIRAC-extra wheel installs retained the provider.
- [x] Add render-only local and Slurm executors with argument-array assembly,
      expected stdout and stderr paths, and allowed-root checks.
- [x] Add a schema 1.0 legacy-profile adapter for NWChem direct and Slurm
      targets, while leaving PBS and LSF on the legacy path.
- [x] Prove that one NWChem launch plan renders for local MPI and Slurm and
      compare the new local command boundary with the current renderer.
- [x] Add a default-off application permission service with exact `launch`
      and `cancel` decisions, structured refusal data, and ungated read-only
      rendering.
- [x] Add local process launch and Slurm submission behind the permission
      service, using argument arrays without a shell and refusing to overwrite
      existing output or script files.
- [x] Separate immutable execution contracts from stateful local and Slurm
      adapters before adding status and cancellation.
- [x] Split the execution adapter into shared rendering and staging helpers,
      live local-process control, and Slurm script and scheduler control.
      Keep `execution/executors.py` as a direct compatibility import for the
      three previously exported names.
- [x] Persist launch ownership, effective argv, environment keys, resources,
      resolved paths, process IDs, scheduler job IDs, timestamps, and state
      transitions.
- [x] Restrict local and Slurm cancellation to active launches from the same
      service instance and target. Local cancellation also requires the live
      process handle retained by that service instance.
- [x] Route the `launch_nwchem_run` and `terminate_nwchem_run` MCP handlers
      through one process-owned execution service. Preserve dry-run fields,
      live response fields, output archival, scheduler timeout results,
      `.jobid` files, and local `term` and `kill` requests while rejecting
      cancellation of unrecorded process and job IDs.
- [x] Add copy and symlink staging with resolved source and destination
      containment, whole-plan preflight, optional-source handling, overwrite
      and output-collision refusal, partial-transfer rollback, and persistent
      launch manifests. Exact artifact hashes and copy provenance remain part
      of the artifact-observation integration.
- [x] Link execution launch records to scientific run records after successful
      NWChem auto-registration. Store the one-to-one relationship in a
      foreign-keyed table and create the run plus link in one transaction.
      The NWChem application adapter verifies service ownership, program, and
      input identity before registration.
- [x] Poll local NWChem processes through the live handle owned by the
      execution service. Persist `started` to `completed` or `failed`
      transitions with return code and elapsed time, synchronize the linked
      scientific-run status, and record SHA-256 observations for terminal
      stdout and stderr files. Repeated status calls reuse the terminal launch
      record and do not duplicate artifact observations. The other program
      adapters remain on their compatibility paths.
- [x] Query owned Slurm jobs through target-owned argument arrays. Use
      `squeue` for active state and fall back to `sacct` for allocation state,
      exit code, signal, and elapsed time after a job leaves the queue.
      Preserve timeout and out-of-memory outcomes in linked NWChem runs and
      record terminal output observations. An empty result from both commands
      remains `not_found` and does not complete the launch. The other program
      adapters remain on their compatibility paths.
- [x] Route NWChem scheduler auto-watch through the owned typed status path.
      Extract the existing polling, adaptive interval, timeout, stall, and
      compact-history behavior into `core/monitoring.py` so legacy watchers
      keep their response contract. Typed `not_found` results continue polling
      and terminal results synchronize the launch, linked run, and output
      observations without a second scheduler interpretation.
- [x] Route explicit NWChem scheduler watch requests through the same typed
      path when the current service owns the job. Resolve ownership without a
      scheduler call, retain the legacy watcher for unowned jobs, and expose
      the final overall status so follow-up actions distinguish success from
      incomplete or failed output.
- [x] Route explicit NWChem local watch requests through the retained process
      handle when the current service owns the PID. Treat a live process as
      authoritative over a partial output parse, persist terminal state and
      artifacts once, and retain the legacy PID watcher for unowned processes.
- [x] Extract run CRUD and atomic launch linking from the oversized
      `core/run_registry.py` into `core/run_records.py`. Preserve the old
      import path as a direct re-export while campaigns, workflows, and batch
      generation await their own focused moves.
- [x] Adapt Molcas direct and Slurm launch calls. Preserve read-only legacy
      previews, apply the CASPT2 parallelism guard to the command and Slurm
      allocation, protect `MOLCAS_PROJECT` and `MOLCAS_NPROCS`, retain dynamic
      Slurm scratch identity, archive exact `.log` outputs, and restrict
      scheduler cancellation to jobs owned by the same MCP process.
- [x] Route explicit Molcas status and watch requests through retained local
      process handles or target-owned Slurm queries when the current service
      owns the identifier. Persist terminal launch metadata and keep one-call
      legacy fallback for unowned identifiers. Molcas scientific-run
      registration and artifact lineage remain separate work.
- [x] Adapt DIRAC direct and Slurm launch calls. Keep `pam-dirac` MPI,
      `.inp/.mol` pairing, and master and node memory flags in the program
      plan; keep containers, modules, hooks, and scheduler commands in the
      target; preserve legacy previews; and restrict scheduler cancellation
      to jobs owned by the same MCP process.
- [x] Route explicit DIRAC status and watch requests through retained local
      process handles or target-owned Slurm queries for identifiers owned by
      the current service. Share only the execution-state projection and
      polling boundary across NWChem, Molcas, and DIRAC; keep file parsing and
      scientific interpretation in each program adapter. Retain external file
      and explicit Slurm inspection after the compatibility release.
- [x] Adapt GRASP workflow-script launch calls for direct and Slurm profiles.
      Run the ordered shell workflow inside the target-owned container,
      preserve legacy previews, archive exact scheduler output paths, and
      restrict cancellation to jobs owned by the same MCP process.
- [x] Route explicit GRASP workflow status and watch requests through retained
      local process handles or target-owned Slurm queries for identifiers
      owned by the current service. Retain external file and explicit Slurm
      inspection. Keep synchronous per-executable calls outside this watcher
      because they already return terminal execution results.
- [x] Make `run_calculation` and `render_calculation_run` the version 1 runner
      entry points. Keep the old NWChem names as direct compatibility aliases,
      and move external status and watch to a focused read-only owner after
      `v0.1.0`.
- [x] Split the oversized legacy runner by responsibility. Move version 1
      profile loading and default merging into `execution/profiles.py`; retain
      `execution/legacy_profiles.py` as an exact compatibility import path
      through `v0.1.0`, then remove the unused facade on the breaking
      development line;
      move compatibility-launch output archival into
      `execution/legacy_archive.py`;
      move local and scheduler hardware discovery into
      `execution/resource_inspection.py`;
      isolate the former unowned status behavior through `v0.1.0`, then retain
      only file and explicit Slurm inspection in `execution/external_status.py`;
      and keep `core/runner.py` as the direct compatibility import path.
- [x] Normalize version 1 installation fields under `programs.<name>` with
      explicit `launcher_argv` and `executable_argv` arrays. Move Molcas
      CASPT2 capability and DIRAC MPI and memory defaults into their program
      blocks. Keep old field locations as lower-priority compatibility inputs,
      teach the dry-run renderer the neutral `{program_command}` placeholder,
      and migrate the bundled local and Stampede3 profiles.
- [x] Add a profile-driven NWChem smoke runner with a committed H2 input,
      bounded monitoring, cleanup, parsed-energy verification, and JSON
      evidence. Record a real local NWChem 7.2.2 Apptainer pass on linux-4090.
      The real Slurm run remains pending because it must be launched from an
      MFA-authenticated Stampede3 login node. The preflight, bounded command,
      minimum charge, and evidence requirements are documented in
      `docs/execution-smoke-tests.md`.
- [x] Move synchronous `run_grasp_exe` and `run_grasp_workflow` MCP calls
      behind a typed local execution contract. Targets own an explicit GRASP
      entrypoint allowlist; plans carry stdin and timeouts; executors capture
      stdout and stderr without a shell; and launch records persist stdin
      digest and size, return code, elapsed time, and completed, failed, or
      timed-out state. Preserve optional capture files and
      `grasp_session.md` through the compatibility adapter. Replace structured
      workflow pre-step and post-step shell calls with contained `cp` actions.
      Exact session-log and output observations remain part of the artifact
      integration item above.

Typed NWChem execution currently accepts direct and Slurm version 1 profiles
whose working directory is the input directory. PBS, LSF, alternate working
directories, and scheduler submission without writing a script remain
unsupported at this boundary. The standalone legacy Python runner still
exists during the compatibility window, but MCP live launch does not bypass
the execution service for those cases.

Typed Molcas execution has the same direct and Slurm profile limits. Slurm
profiles must put module setup in `modules` and scratch setup in
`hooks.pre_run`; the Stampede3 example now shows that split. The existing
`prepare_molcas_launch` tool remains a read-only command preview. Local Molcas
launches return a PID, and owned local or Slurm status and watch requests use
typed execution state. The current Molcas termination tool still accepts
scheduler job IDs only. Molcas launches do not yet create scientific-run links,
so monitoring does not claim run-level artifact observations.

Typed DIRAC execution also accepts direct and Slurm version 1 profiles. The
`.inp` and `.mol` files must exist in the same working directory. Profile
`programs.dirac.default_mpi`, `default_mw`, and `default_nw` values become reviewed
`pam-dirac` arguments, while Slurm does not add a second MPI launcher.
Advanced `--copy`, `--put`, `--get`, and `--outcmo` commands remain available
through the read-only `prepare_dirac_launch` tool; live checkpoint staging
needs its own destination and overwrite contract. Local launches return a
PID, and owned local or Slurm status and watch requests use typed execution
state. The current DIRAC termination tool still accepts scheduler job IDs
only. DIRAC launches do not yet create scientific-run links, so monitoring
does not record run-level artifact observations.

Typed GRASP workflow execution treats the shell script as an ordered
calculation recipe rather than a single-program input file. The target runs
`apptainer exec <sif> bash <workflow>` so plain GRASP commands inside the
script resolve to the container installation. The workflow script must share
the launch working directory, and `programs.grasp` must declare the container
prefix and `bash` executable arrays.
The script is executable content supplied by the caller and should be
reviewed before live launch. Per-executable MCP tools now use a synchronous
local target whose reviewed entrypoint map prevents arbitrary executable
selection. The launch registry stores only the stdin digest and byte count,
while the MCP response carries captured output and the compatibility adapter
maintains `grasp_session.md`. Owned direct or Slurm workflow status and watch
requests use typed execution state. GRASP launches do not yet create
scientific-run links, so monitoring does not record run-level artifact
observations. Direct Python calls to the old runtime remain available during
the compatibility window but are no longer the canonical MCP path.

Exit criteria:

- The same NWChem launch plan can be rendered for local and Slurm execution.
- Scheduler code has no NWChem-named generic dependency.
- No program-specific profile stores execution fields in a unique location.
- Execution remains disabled by default.
- At least one real local and one real Slurm smoke test have documented
  results.

### Phase 4: Guided MCP services

Deliverables:

- High-level workflow tools and structured review results.
- Focused developer toolsets for low-level functions.
- Compatibility aliases for current public tools.
- Updated client examples and Codex development instructions.

The real Stampede3 smoke test remains deferred at the user's request. Phase 4
continues with analysis-only services that do not depend on runner profiles.

- [x] Add `inspect_run` as the first guided service. It classifies one output
      artifact, uses the selected backend parser, prefers a declared scientific
      diagnosis adapter, and labels task-outcome fallbacks and artifact
      ambiguity as uncertainty. Preserve partial parse evidence when optional
      diagnosis fails. Retain supported NWChem total-energy records from
      truncated fragments that omit the input-module header, while leaving
      those tasks incomplete without completion evidence. Cover NWChem,
      Molcas, DIRAC, and GRASP through exact service tests and the JSON-RPC
      golden path.
      Retain incomplete NWChem SCF and DFT energy tasks from their module
      banners even when the file ends before the first total-energy record.
      Convert the explicit NWChem input-error message into a failed unknown
      task when parsing stops before a scientific module, and abstain from
      method or operation comparison in that case.
- [x] Identify standalone NBO analysis output before `inspect_run` invokes a
      selected program parser. Return `unsupported_output_format` with the
      detected `nbo` format for both automatic detection and an explicit
      program override, while accepting NWChem output that embeds an NBO
      section after its own run banner.
- [x] Reject explicit `inspect_run` program overrides when one or more other
      registered backends positively identify the output content and the
      selected backend does not. Report `program_content_mismatch` with every
      detected program. Keep explicit overrides valid for detector-negative
      sparse or truncated files, and recognize the short `OpenMolcas` product
      header as Molcas output.
- [x] Make automatic backend resolution reject ambiguous content instead of
      selecting the first registered match. Preserve first-match behavior in
      the low-level `detect_from_file` compatibility helper, add a typed
      `ProgramDetectionAmbiguous` error to `registry.resolve`, and return
      `program_detection_ambiguous` with candidate names from guided, generic,
      recovery, and legacy dispatch paths. Explicit program selection remains
      the resolution mechanism for intentionally mixed output.
- [x] Audit detector collisions and dispatch mismatches against a bounded
      408-file corpus. Replace loose program-name substrings with output-shaped
      signatures, reject explicit generic overrides that conflict with
      positive detector evidence, and compare recovery inputs with the
      selected program before any patch is written. Route GRASP hyperfine,
      isotope-shift, LSJ transition, and RCI summary files through the shared
      parser adapter; classify `.sum` and `.csum` as distinct artifact kinds.
      Separate lossless detection probes from the low-level compatibility
      helpers so authoritative dispatch reports detector crashes and
      source-read errors instead of treating them as ordinary misses.
- [x] Extend `inspect_run` with an explicit related-artifact set. Preserve the
      primary output as the only parser source, deduplicate caller-supplied
      paths, classify files by the selected backend, and report missing,
      non-file, ambiguous, or unmatched artifacts as uncertainty. Do not scan
      directories or parse binary checkpoints implicitly.
- [x] Include a bounded 16 KiB tail excerpt for explicitly supplied stderr
      artifacts. Declare the `.err` artifact kind for every current backend,
      record truncation and UTF-8 replacement as uncertainty, and leave inputs,
      checkpoints, and generic auxiliary outputs as metadata-only evidence.
- [x] Add `content_kind` to `ArtifactKindSpec` and classification candidates.
      Require each built-in artifact kind to declare `text`, `binary`, or
      `unknown`, default third-party declarations to `unknown`, and gate stderr
      reading on both its role and an explicit `text` declaration.
- [x] Split Molcas formatted `INPORB` from binary `JOBIPH` classification.
      Preserve `molcas.orbitals` for the formatted file, add
      `molcas.jobiph` with checkpoint, orbital, and wavefunction roles, and
      give both artifact kinds truthful content declarations.
- [x] Add bounded excerpts for every explicitly supplied related artifact
      declared as text. Keep stderr tail-only, use whole or UTF-8-safe head and
      tail segments for other text files, cap each artifact at 16 KiB, and cap
      one inspection at 64 KiB. Report truncation, replacement decoding, and
      skipped text after budget exhaustion as uncertainty.
- [x] Add an optional `run.consistency` backend capability and use it when
      `inspect_run` receives exactly one explicit primary input. Implement the
      first NWChem checks for normalized echoed input, task method and
      operation, explicit charge and multiplicity, atom count, invariant
      geometry distances, per-task state, and external restart references.
      Track sequential charge, module multiplicity, named `set geometry`
      selection, and geometry provenance across optimization follow-up tasks.
      Track explicit and standard-library ECP core-electron replacements and
      derive the expected electron count from each task's active geometry and
      molecular charge.
      Check electron-count and spin-multiplicity parity independently for the
      input-derived state and output-reported state. Pair this state with
      charge, multiplicity, atom-count, electron-count, parity, alpha/beta
      occupations, wavefunction class, AO basis mode, ECP replacement,
      and available geometry evidence inside each output task boundary. Track
      explicit RHF, ROHF, UHF, ODFT, and RODFT reference choices, including SCF
      references reused by correlated tasks. Track explicit named DFT `xc`
      aliases across repeated DFT blocks and compare B3LYP, PBE0, SCAN, BHLYP,
      and M06-2X with the runtime XC Method label for DFT and TDDFT tasks.
      Keep TDDFT's internal DFT ground-state step inside its top-level task,
      report the printed target-state energy, and preserve each completed
      DPLOT section as a property task. Classify the explicit TDDFT gradient
      module as a gradient operation without losing the TDDFT method or target
      excited-state energy. Keep weighted and component expressions as
      `not_checked` until coefficient-aware comparison exists.
      Preserve explicit NWChem Property Module sections as property operations
      in the generic task model, and treat the property handler's explicit
      energy-failure marker as a failed task. Carry an open-shell DFT reference
      activated by one task into later tasks in the same persistent database,
      including a later singlet fragment that remains spin-polarized.
      Normalize NWChem Raman input operations to the shared frequency task kind
      while retaining the raw Raman operation and label. Use resolved task
      state for coarse operation comparisons so omitted operations receive
      their NWChem defaults.
      Store an explicit TCE model separately from the `tce` execution module,
      normalize MBPT2 to MP2, and use that model when pairing input tasks with
      CCSD, CCSD(T), or MP2 output evidence. Leave unrecognized TCE models
      unresolved rather than guessing.
      Report public task boundaries as 1-based inclusive line ranges while
      retaining separate raw byte offsets for internal NWChem diagnostics.
      Accept the total-energy delimiters emitted by SCF, DFT, MP2, CCSD, and
      CCSD(T), and keep optimization summaries on the highest-level recognized
      method when lower-level reference energies appear in the same task.
      Populate the compact task basis only for one unambiguous echoed library
      family. Leave manual and mixed-family cases unresolved there and keep
      their details in the per-element runtime basis evidence.
      Follow spherical or Cartesian AO basis selection and named-basis
      indirection, compare the mode with each runtime basis summary, and report
      NWChem's shell and function counts as evidence. Preserve the populated
      summary rows and verify that every element in the selected geometry has
      runtime AO basis coverage. Compare explicit `nelec` values with printed
      ECP replacement counts when available. Do not derive runtime basis size
      from library input because canonical orthogonalization can reduce the
      functions used.
      Resolve standard ECP assignments against the bundled NWChem library and
      retain the family and library file as evidence for each derived `nelec`.
      Store named ECP definitions independently, follow `ecp basis` selection
      between tasks, and let a later unnamed ECP block restore the default
      active entry instead of treating every repeated ECP block as ambiguous.
      Keep sparse and ambiguous fields as `not_checked`, including ECP families
      absent from the bundle, external library files, unresolved selected ECP
      names, restart state, and explicit center charges; report mismatches
      without discarding parsed run evidence.
- [x] Add `review_input` with conservative input-format detection and a stable
      parse, lint, assessment, uncertainty, and edit-action envelope. Use only
      backend-declared capabilities: NWChem parses and lints, Molcas lints,
      DIRAC parses, and GRASP reports that single-file review is unsupported.
      Treat a clean lint result as implemented checks passing rather than proof
      that the complete program grammar is valid. Reject an explicit program
      override when positive content evidence identifies another program, but
      keep explicit selection available for detector-negative fragments. Use
      filename extensions only as an automatic-detection fallback, not as
      mismatch evidence. The bounded 301-file NWChem, Molcas, and DIRAC input
      corpus produced no mixed-signature collisions; 27 sparse Molcas
      fragments remained extension-only matches.
- [x] Add a `guided` toolset preset containing only `review_input` and
      `inspect_run`. Keep the existing low-level generic and program tools
      available when the preset is unset.

Exit criteria:

- A new user can review an input and inspect a run without choosing among
  hundreds of narrow tools.
- High-level results include evidence, uncertainty, and next actions.
- Low-level tools remain available for direct testing.

### Phase 5: Curated knowledge

Deliverables:

- Versioned knowledge-card schema.
- Cards and tests for the first five cross-program lessons.
- Search and filtering by program, workflow, claim type, and confidence.
- Links from cards to checks, fixtures, and raw notes.
- Move the canonical f-block atomic catalog into package data and add typed
  access.
- `find_reference_case` backed by the curated manifest.
- GRASP2018 and ATSP2K seed-lineage checks for Th and U.

Phase 5 starts with the curation boundary, before MCP search exposes any
claims:

- [x] Add the versioned `chemtools.knowledge-card/1` model and a bounded YAML
      loader for installed package data. Reject unknown fields, unsafe source
      paths, duplicate IDs, and accepted cards without source and test links.
- [x] Add the silent-success lesson as the first traceable `draft` card. Keep
      it out of accepted recommendations until its positive and negative
      scientific fixtures exist.
- [x] Add positive-convergence and zero-exit cycle-limit RMCDHF fixtures. Teach
      the shared parser that `Convergence not reached` is an explicit failure,
      and require positive log evidence before its task outcome is successful.
      Keep the cross-program claim in `draft` because one GRASP regression does
      not establish every program's behavior.
- [x] Add an `accepted`, GRASP-scoped card for the tested RMCDHF rule. Link it
      to both fixtures through the exact parser regression.
- [x] Encode the same-producer lesson as a program-neutral direct-provenance
      check. Treat outputs from the same `run_uid` and `step_id` as correlated,
      accept distinct direct producers only when both records are `recorded`,
      and abstain when provenance is missing, ambiguous, declared, or inferred.
- [x] Encode the same-starting-class lesson as a program-neutral provenance
      check. Require a canonical `starting_guess_class` on each distinct,
      recorded producer; count repeated classes as one measurement and abstain
      instead of inferring missing or malformed classes.
- [x] Add deterministic sign and monotonicity checks with explicit,
      caller-supplied expectations and synthetic pass/failure fixtures. Keep
      the cross-program basin-detection card in `draft` until scoped chemistry
      rules cite recorded reference values; the checks do not infer scientific
      direction from a program or workflow name.
- [x] Encode the optimizer-sentinel lesson against declared valid objective
      bounds. Require the failure value to rank strictly worse than the full
      interval for minimization and maximization. Reject non-finite sentinels
      and unbounded validity claims because MCP JSON and optimizer support for
      infinities is not portable.
- [x] Add deterministic card search and one read-only MCP tool. Default to
      `accepted`, require an explicit single status for incomplete or rejected
      material, support text, program, workflow, kind, confidence, and bounded
      result filters, and return scope, sources, checks, and tests with every
      match. Include it in the three-tool `guided` preset.
- [x] Curate four accepted QMCPACK cards from the f-block reference notes:
      determinant-only VMC offsets, the post-Jastrow VMC gate, authoritative
      variational-parameter sidecars, and the heavy-open-f-ion DMC reference
      protocol. Link each claim to a deterministic regression.
- [x] Move the canonical f-block atomic library into `chemtools/data/fblock`
      without leaving a duplicate. Add versioned adjacent metadata, immutable
      element and state models, pinned payload integrity and coverage checks,
      component-level scientific status, and package-resource access.
- [x] Expose exact f-block element and state lookup through a read-only,
      GRASP-scoped MCP tool. Return review status, method scope, catalog hash,
      limitations, J/CSF structure, and complete donor lineage without file
      access or fuzzy configuration matching.
- [x] Add exact f-block atomic planning over the reviewed catalog and ATSP2K
      recipe tables. Emit the recorded 13-line ATSP2K input, static-nucleus
      GRASP input chain through DC+Breit RCI, expected J/CSF structure and
      energies, ordered donor prerequisites, and explicit manual requirements.
      Preserve unresolved donor aliases and refuse to synthesize the 17 Y
      states whose GRASP prompt fields were never recorded.
- [x] Add a catalog-hash-bound donor alias manifest. Inventory all 132
      consumer-scoped occurrences, reject coverage drift and guessed targets,
      and expose their scientific-review status in atomic plans. No exact
      target is claimed until a donor artifact or explicit review identifies
      it.
- [x] Add bounded GRASP radial-wavefunction inspection from the format in
      `rwfntotxt.f90` and `rwfnrelabel.f90`. Validate record framing, orbital
      identities, array lengths, finite values, and radial grids without
      returning the arrays. Check the reader against GRASP's converter and
      all 251 available GRASP wavefunction files before enabling writes.
- [x] Add an explicit first-donor-wins radial-wavefunction merge. Validate all
      donors through the bounded reader, preserve orbital records byte for
      byte, reject mixed byte order and later donors with no new orbitals,
      protect donor files, default to no-clobber output, commit atomically,
      and inspect the completed file before returning.
- [x] Add bounded GRASP `.m` and `.cm` mixing-coefficient inspection from the
      RMCDHF and RCI writers and the shared `getmixblock.f90` reader. Validate
      all block totals, level indices, energies, coefficients, and vector
      norms while returning only a caller-bounded level summary. In the local
      86-file corpus, accept 81 `G92MIX` binaries, identify four ASCII Octave
      exports by content, and reject one file missing its final record marker.
- [x] Connect GRASP mixing coefficients to an explicitly supplied `.c` file.
      Parse the source-defined three-line CSF records and ` *` block markers,
      verify electron, subshell, CSF, block, and symmetry agreement, and map
      each returned dominant coefficient to its block-local configuration and
      coupling lines. All 81 valid local binary/list pairs agree. Confirm the
      first real mapping independently with `rmixextract` at a 0.94 cutoff.
- [x] Return a caller-bounded leading-component list for each included mixing
      level while still validating every coefficient. Report the weight kept
      and omitted, preserve the dominant-component fields for compatibility,
      and map every returned component through the matching `.c` file. The
      first five components of the real Cf `5f10` ground state match sorted
      `rmixextract` output at a 0.24 cutoff, including signs and CSF text.

Exit criteria:

- Each first-round lesson has a durable project artifact and regression test.
- Search results state scope and source.
- Raw notes are never presented as accepted rules without curation status.
- Shelved and exploratory calculations cannot be returned as validated
  recipes.
- F-block responses state method, rebuild version, and seed provenance.

### Phase 6: Orbitron bridge

Progress:

- [x] Expose the existing optional adapter through one read-only
      `inspect_with_orbitron` MCP tool. Accept only a local file path and call
      the fixed `inspect --json` operation without a shell, remote target, or
      caller-supplied command arguments.
- [x] Harden the adapter before MCP exposure. Accept Orbitron's documented
      `-dirty` build suffix, validate every structured warning, enforce a 2 GiB
      source limit and 2 MiB JSON limit, and preserve unavailable, incompatible,
      and tool-refused outcomes.
- [x] Run the eight-case differential contract against Orbitron 0.4.0 at commit
      `20e81d225b4c`. QE task, periodic geometry, and failed-relaxation
      provenance checks plus Molcas failure diagnosis and HCN frequency
      analysis all agree with the independent references.
- [x] Map selected Orbitron evidence into canonical Chemtools models. Preserve
      Orbitron as an `external_tool` producer and classify supported output,
      NWChem `.movecs` and `.hess`, and DIRAC `.h5` subjects through the
      existing backend artifact declarations. Report unsupported programs and
      subjects explicitly. Do not construct `MolecularSystemSpec` or
      `PeriodicSystemSpec` from `inspect/2`: it reports counts, not coordinates
      or a complete periodic-system specification.
- [x] Expose the fixed `analyze geometry <source> --json` operation through
      `analyze_geometry_with_orbitron`. Validate count agreement, finite bounds
      and vectors, bond-statistic consistency, the angstrom distance unit, and
      the unit-cell shape before returning evidence. The `/3` contract also
      validates `input`, `single_point`, `converged_final`, and
      `last_attempted` roles plus a non-empty source description. Mark a
      last-attempted structure as diagnostic uncertainty. Accept no output path
      or analysis flags.
- [x] Expose `analyze orbitals <source> --json` with an explicit frontier
      count of three. Validate orbital-count partitioning, finite values,
      Hartree-to-eV conversions, gap arithmetic, unique frontier labels, and
      HOMO/LUMO membership. Validate the `/2` occupancy threshold and restricted
      or alpha/beta channel partitions, including channel-local gaps and
      frontier membership. Keep periodic band analysis out of this tool.
- [x] Expose `analyze populations <source> --json` with an explicit top-charge
      count of eight. Validate atom counts, finite charges, descending absolute-
      charge order, derived totals and extrema, mean absolute charge, per-atom
      maps, top-charge membership, and method warnings. Validate the `/2`
      expected-charge source and residual arithmetic. Mark the expected charge
      unknown when the source establishes neither a declared charge nor
      nonzero formal charges.
- [x] Add the fixed raw-mode `analyze vibrations <source> --json --top 10`
      operation and expose it through `analyze_vibrations_with_orbitron`.
      Validate mode and imaginary counts, statistics, sorted samples, frequency
      magnitudes, unique indices, raw-mode identity, `cm^-1` units, the 1.0
      scaling policy, total and per-mode displacement availability, and unit-
      labelled thermochemistry. The `/4` contract also requires a geometry role
      and source description; a last-attempted geometry is explicit uncertainty.
      The HCN transition-state reference agrees at 9 modes and one imaginary
      mode. Missing pressure or imaginary-mode displacements are explicit
      uncertainties.
- [x] Add bounded visualization through `visualize` (formerly
      `render_with_orbitron`, now a hidden compatibility alias). The MCP
      transport returns JSON provenance followed by a PNG content item under
      the negotiated 2024-11-05 contract. Render at fixed 1024 by 768 pixels
      into an ephemeral sibling directory, validate PNG signature, dimensions,
      and an 8 MiB size ceiling, then delete the temporary output. Do not
      expose caller-controlled destinations, cameras, appearances, diagrams,
      or dimensions.
- [x] Add two fixed companion-runtime Python API summaries while preserving
      the versioned CLI as the broad parser and renderer boundary. Periodic
      electronic-structure inspection returns only Fermi, gap, band, and DOS
      dimensions from a source-hashed local file. Structure identity returns
      formula and independently available InChI, InChIKey, and SMILES evidence
      plus atom, bond, and canonical `Dative` bond counts. Both operations
      preserve source and environment provenance, use pinned Orbitron-owned
      fixtures, and make no scientific verdict about a parsed bonding model.
- [x] Add fixed, bounded Natural Bond Orbital inspection through Orbitron's
      Python API after standalone NBO parsing was repaired upstream. Return
      type counts, occupancy range, per-atom entry counts, and at most twelve
      `BD`, `BD*`, `LP`, or `LP*` samples. Keep the NBO7 UO₂ source in
      Orbitron's own corpus, pin its hash, and record an explicit opt-in bridge
      check instead of duplicating third-party program output in Chemtools.

Deliverables:

- Stable Orbitron JSON contract for required operations.
- Optional Chemtools subprocess adapter.
- Schema and CLI version negotiation.
- Differential corpus tests.
- `visualize` integration for supported data.

Exit criteria:

- Chemtools works when Orbitron is absent.
- Supported Orbitron output maps to canonical Chemtools artifacts with
  provenance.
- Parser comparisons distinguish agreement, intentional differences,
  refusals, and defects.
- No MCP request can pass an arbitrary Orbitron command.
- Known periodic-cell invariants pass on the selected corpus.

### Phase 7: Quantum ESPRESSO vertical slice

Progress:

- [x] Register a QE backend with explicit input, output, task-index, and run
      diagnosis capabilities; `.in`, `.out`, and `.err` artifact roles; and
      conservative `pw.x` input and PWSCF output detection.
- [x] Parse scalar `pw.x` namelist assignments, indexed keys, Fortran `D`
      exponents, inline comments, and the observed bare, parenthesized, and
      braced card options. Normalize species, positions, cells, k-point grids,
      spin setup, occupations, smearing, and cutoffs into `qe-pw-input/1`.
- [x] Review SCF, relax, and vc-relax inputs through the existing generic
      `review_input` MCP tool. Check required namelists and cards, `nat` and
      `ntyp` agreement, species references, `ibrav` and cell consistency,
      positive `ecutwfc`, automatic k-point shape and shifts, and smearing
      requirements.
- [x] Scan all 39 supplied `pw.x` inputs under
      `/home/charlie/input_examples/qe`. The 19 SCF and two relaxation inputs
      pass the current structural checks. The 13 bands and five NSCF inputs
      parse and report the intended limited-review warning.
- [x] Read bounded UPF 2 preambles and expose `PP_HEADER` identity, type,
      relativistic treatment, functional, valence charge, selected flags, and
      positive cutoff suggestions. Resolve relative `pseudo_dir` from the input
      directory as a labelled review assumption. Report missing or invalid
      files, species and UPF element mismatches, and input cutoffs below the
      hardest positive suggestion. Keep `convergence_established` false even
      when the suggestions are met.
- [x] Parse all 47 UPF files in the supplied QE corpus. Thirty-four of the 39
      `pw.x` inputs resolve every referenced file from their input directory.
      Five Bi2Se3 inputs reference fully relativistic Bi and Se files that are
      absent there; Chemtools reports the missing references without treating
      the input-directory assumption as the execution directory.
- [x] Add deterministic charge and spin review. Reject contradictory
      `nspin`, `noncolin`, `lspinorb`, fixed-magnetization, and species-index
      settings; report missing magnetic seeds; compare spin-orbit requests
      with inspected UPF flags; and sum UPF valence electrons before and after
      `tot_charge` when every species is resolved. Keep physical charge and
      spin-state selection outside the linter. The 39-input corpus produces no
      charge or spin errors; its nonmagnetic Bi2Se3 spin-orbit inputs retain
      time-reversal symmetry as informational evidence.
- [x] Add k-point evidence and convergence planning. Parse and validate all
      documented `K_POINTS` forms, explicit point counts, band paths,
      tetrahedron-grid compatibility, shifts, and symmetry controls. Propose a
      labelled three-stage refinement heuristic for automatic meshes while
      keeping dimensionality assumptions, user tolerances, irreducible counts,
      and `convergence_established` explicit. The supplied 39-input corpus
      passes the expanded card-shape and calculation-type checks.
- [x] Add compact `pw.x` SCF and relaxation output parsing and diagnosis.
      Keep converged bang-prefixed energies separate from ordinary iteration
      energies; retain Ry and Hartree values; parse SCF cycles, BFGS status,
      final enthalpy, force, stress, and native final coordinates; and require
      scientific convergence in addition to `JOB DONE`. Deduplicate repeated
      MPI errors and identify missing pseudopotentials from `readpp`. All 33
      PWSCF outputs in the supplied QE corpus parse without exceptions: 15
      converged SCF runs, 14 completed bands/NSCF runs, one truncated bands
      run, one SCF failure during relaxation, one missing-pseudopotential
      failure, and one converged variable-cell relaxation. QE-suite
      post-processing banners for `bands.x`, `dos.x`, `pp.x`, and `projwfc.x`
      are rejected before PWSCF parsing, while bannerless launcher-abort
      fragments remain unassigned.
- [x] Add QE input-output consistency to `inspect_run`. Compare calculation
      mode, atom and atomic-type counts, UPF-derived electron count,
      `ecutwfc`, and explicit or defaulted `ecutrho`. Compare gamma-only
      sampling with the runtime count, but keep requested grids and
      symmetry-processed PWSCF counts separate. All 33 paired outputs in the
      supplied corpus were inspected: 32 have matching available evidence,
      while the early Bi2Se3 `readpp` failure correctly leaves all seven
      comparisons unchecked because no runtime summary was printed.
- [x] Add periodic output geometry extraction. Normalize SCF runtime cells and
      positions from `alat`, and normalize converged relaxation final blocks
      from `angstrom`, `bohr`, `alat`, or `crystal` coordinates. Expose the
      common output-geometry capability while refusing failed relaxations whose
      only coordinates are attempted structures. Differential checks against
      Orbitron agree for Si SCF, Fe SCF, and FeO `vc-relax`.
- [x] Add the QE output-trajectory capability without weakening converged
      geometry extraction. Preserve the initial geometry and each distinct
      normalized update with cell and source provenance. Label final frames as
      `converged_final` or `last_attempted`, distinguish completed
      non-convergence from truncated output, and leave SCF energies unpaired
      when QE's extra final SCF makes positional matching ambiguous. The real
      FeO and benzene cases match Orbitron at 19 and 8 frames respectively.
      Split shared card parsing and unit conversion into `_coordinates.py`,
      single-snapshot selection into `geometry.py`, and history assembly into
      `trajectory.py` before adding further coordinate diagnostics.
- [x] Add bounded structural analysis to QE trajectories. Compute exact
      periodic minimum-image distances for accepted cells and abstain on cells
      too ill-conditioned for bounded analysis. Apply molecular bond-network
      checks only when every frame has at least 5 angstrom of vacuum in each
      lattice direction and complete covalent-radius coverage. Extended solids
      receive metrics and cell-volume observations without a molecular
      topology verdict. The supplied benzene geometry is already concerning in
      its input frame, with 18 sub-0.6-angstrom contacts and 12 overcoordinated
      main-group atoms. The FeO case receives a 53 percent cell-contraction
      observation without applying molecular connectivity rules.
- [x] Surface compact QE trajectory evidence through `inspect_run`. Keep full
      frames in `parse_trajectory`, attach per-finding input or trajectory
      origins to the run assessment, and return that tool as the next action
      when structural review is needed. Bound automatic analysis to 16 MiB of
      output, 512 frames, and 250,000 atom-pair evaluations. The supplied
      benzene case reports mixed input and trajectory findings; FeO retains a
      converged run verdict with a separate cell-contraction observation.
- [x] Reuse the periodic structural checker during QE input review. Normalize
      explicit `ibrav=0` cells in alat, angstrom, or bohr, plus QE's documented,
      deprecated unitless-cell and position defaults and every nonzero Bravais
      form documented for QE 7.5 from either `celldm` or conventional `A/B/C`
      and cosine parameters. Normalize alat, angstrom, bohr, and crystal
      positions. Accept symmetry-expanded `crystal_sg` records without
      mislabelling their non-numeric coordinate forms, require their
      `space_group` number, and leave them
      structurally `not_assessed` until symmetry expansion is available. Treat sub-0.6-angstrom contacts as errors, main-group
      overcoordination as a warning, and periodic solids as metrics-only. The
      current 64-input scan flags only the five benzene inputs and gives
      periodic metrics to all 34 nonzero-`ibrav` inputs. The other 25 inputs
      lack a `pw.x` geometry and remain `not_assessed`. Sort guided input-edit
      actions by priority so the benzene geometry error precedes its cutoff and
      coordination warnings. Evaluate documented no-space coordinate arithmetic
      with a bounded numeric-only expression tree, while rejecting names,
      function calls, malformed expressions, and excessive powers.
- [x] Add a bounded QE-to-QMCPACK artifact handoff plan. It preserves the
      caller-declared `.pwscf.h5` path and delegates validation to the existing
      conversion inspectors without inventing converter options or execution.
- [x] Draft the supported `pw2qmcpack.x` `&inputpp` form from explicit QE
      `prefix` and `outdir` values, retain review for missing paths, and parse
      the bounded converter form through generic input review.
- [x] Parse bounded `pw2qmcpack.x` output evidence, retaining the reported
      HDF5 artifact path and requiring `JOB DONE.` before inferring terminal
      completion.
- [x] Diagnose the recorded collected-wavefunction and gamma-trick converter
      failures without inferring a correction from output evidence alone.
- [x] Compare the bounded converter input's expected HDF5 path with the
      output's reported artifact path when both are supplied.
- [x] Compare an explicit `.pwscf.h5` sidecar with the converter's reported
      path without reading the binary file.
- [x] Reject a matching explicit `.pwscf.h5` sidecar when its timestamp
      predates the converter input.
- [x] Classify explicit `.pwscf.h5` converter products as metadata-only QE
      checkpoint and wavefunction artifacts.
- [x] Draft one explicit-q `ph.x` input from explicit QE `prefix` and `outdir`
      values, preserving caller-owned title and q-vector choices.
- [x] Add typed `pw.x` launch rendering and execution through named runner
      profiles. The plan passes the input with QE's `-in` flag, requires the
      input directory as the working directory, records QE output and error
      artifacts, and keeps process completion separate from SCF convergence.
- [x] Verify a completed local QE smoke calculation for the configured host
      installation. The linux-4090 `qe_local` profile runs QE 7.5 through
      `mpirun -np 1` with `I_MPI_FABRICS=shm`; the Si SCF completed with return
      code 0, seven SCF iterations, `-14.54255436 Ry`, and `JOB DONE.`. The
      shared-memory provider avoids an Intel MPI TCP/libfabric collective stall
      observed during the one-rank initialization. Scheduler profile behavior
      remains a separate, verified-launcher follow-up. Periodic Orbitron
      inspection remains a separate scope decision.

Deliverables:

- Periodic input and output support for the defined `pw.x` scope.
- Pseudopotential, cutoff, k-point, spin, and occupation checks.
- `ph.x` and `pw2qmcpack.x` workflow artifacts.
- Local and Slurm launch plans.
- Orbitron-backed periodic inspection where supported.

Exit criteria:

- A representative SCF and relaxation can be reviewed, launched, inspected,
  and diagnosed through guided tools.
- Invalid or inconsistent periodic inputs produce specific checks.
- The QE to QMCPACK conversion artifacts are represented with provenance.

### Phase 8: QMCPACK and cross-program workflow

Deliverables:

- QMCPACK input, artifact, output, and diagnostic support for the defined
  scope.
- Determinant-only and Jastrow validation workflow.
- `qe_to_qmcpack` consistency service.
- Local launch plans and execution through a named QMCPACK target profile.
  Scheduler launch plans remain deferred until a verified site launcher
  contract is configured.

Exit criteria:

- The complete QE to QMCPACK chain can be inspected without losing artifact
  lineage.
- Stale HDF5, pseudopotential mismatch, and variational-gate failures have
  regression fixtures.
- The workflow explains which checks are independent and which share upstream
  evidence.

### Phase 9: Companion scientific Python runtime

Deliverables:

- A versioned `chemtools-science` request/result contract and read-only health
  probe.
- A committed Conda or micromamba environment specification with platform
  lock files for supported developer systems.
- Fixed, provenance-preserving RDKit, Open Babel, and Orbitron Python API
  operations.
- Molecular PySCF RHF, UHF, RKS, and UKS single-point execution through a
  named runner profile.
- Closed-shell and open-shell fixtures, including successful convergence,
  non-convergence, unsupported input, and conversion-disagreement cases.

Exit criteria:

- Chemtools starts and retains its analysis-only behavior when the companion
  runtime is absent.
- A companion request cannot execute arbitrary Python, install packages, or
  select an unapproved executable.
- Each result identifies its environment and all scientific transformations.
- A PySCF result distinguishes process completion, SCF convergence, and
  method-comparison scope.
- Conversion and perception changes are surfaced as evidence rather than
  silently applied to the caller's submitted model.

## First milestone backlog

Work should begin here and stop at the gate before changing the program or
runner architecture.

- [x] Add a script that emits the live tool inventory as JSON and Markdown.
- [x] Add a test that pins current totals by group and capability.
- [x] Reconcile README tool counts and explain how they are generated.
- [x] Select representative golden cases for each existing program and generic
      tool group.
- [x] Add golden cases without rewriting existing response shapes.
- [x] Write ADR 001: optional program capabilities and one backend catalog.
- [x] Write ADR 002: execution targets replace local and HPC behavior modes.
- [x] Write ADR 003: runs are artifact collections with provenance.
- [x] Write ADR 004: public MCP alias and deprecation policy.
- [x] Write ADR 005: local and committed reference corpus boundaries.
- [x] Map current program, runner, scheduler, and dispatch modules to the
      [target boundaries](docs/current-to-target-module-map.md).
- [x] Add `CHEMTOOLS_ORBITRON_CLI` resolution and a read-only version probe.
- [x] Add a three-case external corpus manifest with pinned sizes and hashes.
- [x] Add the QE Fe Orbitron discrepancy as a differential regression.
- [x] Confirm the Orbitron QE zero-cell fix with an exact-value check.
- [x] Pin the versioned JSON envelopes used by Orbitron machine output.
- [x] Pin supported MCP protocol versions and negotiate initialization instead
      of echoing an arbitrary client version.
- [x] Record the [current test collection and historical recovery
      queue](docs/testing-baseline.md).
- [x] Run the full committed suite and save the command in contributor docs.
- [x] Review the five ADRs before starting Phase 1.
- [x] Resolve the eleven items in the
      [Phase 0 ADR review](docs/adr/phase-0-review.md), then accept ADRs 001
      through 005.

Phase 0 should avoid moving modules or renaming public tools. Its purpose is to
make later changes measurable.

## Architecture decisions and phase gates

The first eight decisions are accepted through ADRs 001 through 005. Later
phase gates retain the current direction until their implementation work
needs a more specific decision.

| Decision | Needed by | Current direction |
| --- | --- | --- |
| Backend discovery | Phase 1 | [ADR 001](docs/adr/001-optional-program-capabilities-and-builtin-catalog.md): explicit built-in catalog, no dynamic plugin loading yet |
| Capability representation | Phase 1 | [ADR 001](docs/adr/001-optional-program-capabilities-and-builtin-catalog.md): operation-level string enum |
| Execution target model | Phase 3 | [ADR 002](docs/adr/002-execution-targets-replace-behavior-modes.md): named local or Slurm targets |
| Execution approval boundary | Phase 3 | [ADR 002](docs/adr/002-execution-targets-replace-behavior-modes.md): explicit global gate plus configured target operations |
| Artifact identity and lineage | Phase 2 | [ADR 003](docs/adr/003-runs-are-artifact-collections-with-provenance.md): stable IDs, point-in-time observations, and append-only provenance |
| Canonical artifact serialization | Phase 2 | [ADR 003](docs/adr/003-runs-are-artifact-collections-with-provenance.md): frozen dataclasses with an explicit versioned JSON envelope |
| Public tool aliases | Phase 4 | [ADR 004](docs/adr/004-public-mcp-alias-and-deprecation-policy.md): one canonical definition, hidden validated aliases, and versioned removal gates |
| Reference corpus storage, curation, and redistribution | Phase 0 | [ADR 005](docs/adr/005-reference-corpus-boundaries.md): three storage tiers with independent permission, purpose, and scientific status |
| Configuration library and file format | Phase 3 | Reuse current project conventions where possible |
| Default guided tool list | Phase 4 | Start with the nine intent names above, add only for distinct intents |
| Knowledge-card storage | Phase 5 | Human-readable YAML with schema validation |
| Orbitron JSON envelope | Phase 0 | Schema ID, producer version, data, warnings |
| Orbitron advanced inspection contract | Phase 6 | Add `inspect --json` |
| Parser ownership transfer criteria | Phase 6 | Corpus agreement and documented differences |
| ATSP2K program scope | Phase 5 | Seed-producing backend for GRASP before a wider tool surface |
| Initial QE calculation subset | Phase 7 | SCF and relaxation first |
| Initial QMCPACK estimator subset | Phase 8 | Decide from available fixtures and workflow needs |
| Companion runtime package boundary | Phase 9 | Optional Conda or micromamba runtime, fixed JSON runner contract, no arbitrary Python |

Dynamic third-party backend loading is deferred. The project needs a clean
built-in catalog before it needs a plugin mechanism.

## Risks and controls

| Risk | Control |
| --- | --- |
| Refactor changes public behavior silently | Behavior inventory, golden cases, compatibility aliases |
| Capability model becomes a large framework | Implement only capabilities used by current programs |
| Guided tools hide needed expert control | Keep focused developer toolsets |
| Orbitron and Chemtools duplicate parsers indefinitely | Differential corpus and explicit ownership gates |
| Orbitron returns structurally valid but unphysical data | Cross-field invariants such as nonzero periodic-cell volume |
| Raw notes become overconfident rules | Scope, confidence, sources, and curation status |
| External examples change underneath tests | Exact sizes, content hashes, and an opt-in manifest |
| Third-party or research artifacts are copied without review | Per-case provenance and redistribution status |
| Exit success is reported as scientific success | Separate execution, parsing, and chemistry verdicts |
| Local and cluster configuration drift | Shared launch plan and named target schema |
| Periodic support contaminates molecular models | Typed molecular and periodic system variants |
| QMCPACK artifacts lose lineage | First-class conversion steps and provenance |
| Companion packages drift across machines | Platform lock files, runtime health probe, and per-result environment provenance |
| Conversion changes are mistaken for chemical facts | Preserve source artifact and report each perception or conversion change explicitly |
| Documentation counts become stale again | Generate inventory from the live catalog |
| Refactor scope expands into feature work | Phase gates and behavior-preserving exit criteria |

## Definition of done

The architectural program is complete when:

- Existing public tools have working replacements or an accepted reason to
  remain low-level.
- Program registration and capabilities have one source of truth.
- Local and Slurm execution share launch plans and target configuration.
- NWChem, Molcas, DIRAC, GRASP, Quantum ESPRESSO, and QMCPACK expose truthful
  supported capabilities.
- Molecular and periodic systems are represented without untyped escape
  fields for core concepts.
- Multi-step artifact lineage is retained across parsing, conversion,
  execution, and diagnosis.
- The default MCP surface supports input review, run inspection, planning,
  execution, monitoring, comparison, visualization, and knowledge search.
- The first curated lessons from the notes are linked to sources and tests.
- The f-block catalog can produce a provenance-aware GRASP2018 and ATSP2K seed
  plan without losing donor history.
- External reference cases are selected through a size-and-hash-pinned manifest and
  clearly labeled as validated, failing, exploratory, or shelved.
- Orbitron integration is optional, versioned, restricted to known operations,
  and covered by differential tests.
- The companion scientific runtime is optional, versioned, restricted to fixed
  operations, and records environment provenance for every result.
- README setup, development, architecture, tool inventory, and program support
  match the live implementation.
- The committed test suite covers compatibility, scientific checks, executor
  contracts, and representative cross-program workflows.

## Evidence used for this plan

Repository sources:

- The committed backend catalog, ADRs, and current implementation, which
  supersede the earlier refactor sketches.
- `chemtools/core/program.py`, current required program interface.
- `chemtools/mcp/modes.py`, current program list and three-way modes.
- `chemtools/mcp/dispatch.py`, eager program imports and manual registry
  composition.
- `chemtools/core/runner.py`, generic execution internals behind
  NWChem-oriented public names.
- `notes/run-layer-hardening.md`, runner assessment and known boundary issues.
- `notes/detector-collision-audit-2026-07-31.md`, verified detector, dispatch,
  recovery, and GRASP output-routing defects plus the bounded corpus result.
- `notes/fblock/README.md`, scope of the f-block notes and element/state data.
- `notes/fblock/qe-qmcpack-oncvpsp.md`, the QE, pseudopotential, and QMCPACK
  workflow and failure cases.
- `chemtools/data/fblock/README.md`, the 31-element, 633-state seed
  catalog and its seeding classes.
- `chemtools/data/fblock/grasp/fblock-all.json`, the machine-readable
  GRASP2018 reference catalog.
- `/home/charlie/input_examples/grasp/README.md`, validated GRASP workflows and
  MPI constraints.
- `/home/charlie/input_examples/grasp/Tm_Md_LS_coupling/README.md`, the
  explicitly shelved LS-coupling study.

Orbitron sources inspected in `/home/charlie/projects/orbitron`:

- `website/user-guide/cli.qmd`, machine-readable CLI operations.
- `io/pipelines/src/tasks.rs`, program task summaries and Quantum ESPRESSO
  adapter.
- `automation/cli/src/cli.rs`, current `info` and `inspect` flags.
- `automation/cli/src/handlers/reports.rs`, the versioned `info` and `inspect`
  response envelopes.
- `io/pipelines/tests/qe/scf.rs`, QE periodic-cell regression assertions.
- The canonical document model, including source, provenance, capabilities,
  payload, attachments, and extras.

Current registry counts are generated in
[`docs/tool-inventory.md`](docs/tool-inventory.md). The working-tree suite
passed with 869 tests on 2026-08-02 using `.venv/bin/python -m pytest -q`; the
collection and its historical recovery queue are recorded in
[`docs/testing-baseline.md`](docs/testing-baseline.md).

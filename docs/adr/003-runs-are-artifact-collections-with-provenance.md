# ADR 003: Runs are artifact collections with provenance

Status: Accepted

Date: 2026-07-30

Chemtools will represent a run as one execution attempt containing ordered
steps, an artifact collection, and append-only provenance. An input path and
an output path remain useful compatibility fields, but they will no longer
define the run.

Artifact identity will be separate from filesystem location. Each artifact
will have a stable ID and one or more point-in-time observations. Copies,
conversions, merges, restart selection, and sidecar authority will be recorded
as relationships rather than inferred from filenames.

## Context

The current SQLite registry stores one `input_file`, one `output_file`, and
one `parent_run_id` per run. It also stores selected parsed values such as
energy, thermochemistry, and imaginary-mode count directly on the run row.
That shape fits a simple NWChem job but loses information needed by several
existing workflows:

- GRASP produces a working set containing nuclear data, CSFs, angular
  integrals, radial orbitals, SCF summaries, mixing coefficients, level
  tables, and interactive input records. Its plugin already states that no
  single input or output represents the calculation.
- A DIRAC atomic-start chain retrieves or renames an HDF5 checkpoint, then
  supplies that checkpoint to a later calculation. The current plan records
  the filenames but not the exact checkpoint snapshot consumed.
- A Molcas active-space refinement creates a new `RasOrb` and input from an
  earlier output and orbital file. Their relationship lives in the
  orchestrator response rather than the registry.
- NWChem restart discovery chooses files by stem and modification time. TCE
  restart preparation copies amplitude files and reports source and
  destination as display strings, with no durable lineage.
- Quantum ESPRESSO and QMCPACK require input decks, pseudopotentials, save
  directories, converted HDF5 files, XML files, scalar data, and optimized
  parameter sidecars to be considered together.

The notes contain cases where file presence and timestamps would give the
wrong answer:

- A blanket sync can overwrite live output with an older local copy.
- `copy2` preserves timestamps while creating a new artifact location.
- QMCPACK optimization XML can contain stale display coefficients while a
  referenced `vp.h5` contains the authoritative parameters.
- ATSP2K consumers can read months-old orbitals from an otherwise active
  working directory.
- GRASP multi-donor seeds combine selected orbitals from several parents.
  The final stdin sequence does not preserve that history.
- Two apparently independent results may descend from the same flawed
  upstream artifact.

A single parent run cannot express multiple donors, a conversion chain, or a
sidecar dependency. A path cannot distinguish a live output from an older
file later written at the same name.

## Decision

### A run is one execution attempt

A `RunRecord` represents one attempt that is launched, submitted, or imported
as a unit:

```python
@dataclass(frozen=True)
class RunRecord:
    run_uid: str
    program: str
    origin: RunOrigin
    target: str | None
    working_directory: Path | None
    steps: tuple[RunStep, ...]
    artifacts: RunArtifacts
    status: RunStatus
```

`run_uid` is a UUID string generated with the Python standard library. The
current integer database ID remains a local lookup key and compatibility
field. It is not used as a portable identity because integer IDs collide when
records move between machines or registries.

A run may contain several ordered `RunStep` records when one monitored unit
invokes several executables. This supports a GRASP batch script without
pretending that `rnucleus`, `rwfnestimate`, and `rmcdhf` are one parser task.
It also supports a scheduler job that deliberately chains related program
commands.

A workflow connects runs through declared dependencies. A restart is a new
run, never an update that erases the earlier attempt. Imported historical
directories use `origin="imported"` and may lack target, command, or scheduler
metadata.

Execution state, artifact state, parser state, and chemistry verdict remain
separate. A completed run may contain a missing or stale checkpoint. A fresh
artifact may still represent the wrong electronic state.

### Run artifacts use roles and program-specific kinds

Replace category-specific lists with one collection:

```python
@dataclass(frozen=True)
class RunArtifacts:
    run_uid: str
    artifacts: tuple[ArtifactRef, ...]
    observations: tuple[ArtifactObservation, ...]
    expectations: tuple[ExpectedArtifact, ...]
    provenance: tuple[ProvenanceRecord, ...]


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    roles: frozenset[ArtifactRole]
    kind: str
    producing_step: StepRef | None
    metadata: Mapping[str, JsonValue]
```

An artifact may have several roles. A QMCPACK HDF5 file can be both an
auxiliary input and a wavefunction. A Molcas `RasOrb` can be an orbital
result, a restart checkpoint, and a later run's input.

The core role set stays small:

- `primary_input`
- `primary_output`
- `auxiliary_input`
- `auxiliary_output`
- `stdout`
- `stderr`
- `checkpoint`
- `orbital`
- `wavefunction`
- `wavefunction_seed`
- `pseudopotential`
- `volumetric_data`
- `scheduler_script`

`kind` carries program or format meaning such as `nwchem.movecs`,
`molcas.rasorb`, `dirac.checkpoint_h5`, `grasp.radial_wfn`,
`qe.save_directory`, or `qmcpack.vp_h5`. New program formats add kinds without
expanding the core role enum.

An `ExpectedArtifact` comes from a launch plan or workflow step. It records
the intended role, kind, location rule, required status, and producing step.
After execution, an expectation resolves to an artifact ID or a structured
missing or ambiguous result. A file with a plausible suffix does not
silently satisfy an expectation for another step.

Artifacts may be files or directories. This is required for compound
artifacts such as a Quantum ESPRESSO save directory. Version 1 will not add a
general nested collection model. A directory observation records directory
metadata and an optional versioned manifest hash when a program adapter
supplies a stable inventory.

### Artifact identity is separate from location and observation

An artifact ID identifies a logical produced or imported artifact. It is not
a path and it is not a content hash.

```python
@dataclass(frozen=True)
class ArtifactLocation:
    path: Path
    entry_type: Literal["file", "directory"]
    root_name: str | None
    relative_path: Path | None


@dataclass(frozen=True)
class ArtifactObservation:
    observation_id: str
    artifact_id: str
    observed_at: datetime
    location: ArtifactLocation
    exists: bool
    size_bytes: int | None
    modified_ns: int | None
    sha256: str | None
    hash_status: Literal["not_requested", "verified", "unavailable"]
    directory_manifest_schema: str | None
    directory_manifest_sha256: str | None
```

A growing output or checkpoint retains its artifact ID while Chemtools records
new observations. A parser result refers to the exact observation it read.
If a later run reuses the same path, it gets a new artifact ID because it has
a new producer. Historical data does not silently follow the path.

A copy receives a new artifact ID and a provenance record pointing to the
source observation. Equal SHA-256 values show equal bytes, but they do not
collapse the records. The copies may have different roles, locations, and
downstream consumers.

A pure move or rename retains the artifact ID, records a `moved` provenance
event, and adds an observation at the new location. This preserves DIRAC
checkpoint renames without treating unchanged bytes as a new scientific
result.

Locations retain both a normalized local path and, when possible, a named root
plus relative path. Local SQLite records may contain absolute paths. Portable
exports prefer root-relative paths and do not claim that a path is valid on
another machine. Symlink paths and resolved targets are recorded separately
when symlinks matter; the observer does not silently replace one with the
other.

SHA-256 is the canonical content hash because the reference manifest already
uses it and it requires no new dependency. Every observation records cheap
file metadata. Hashing policy is selective:

- Hash committed fixtures and reference-manifest artifacts.
- Hash artifacts before a recorded copy, conversion, merge, or seed operation
  when practical.
- Hash small final artifacts used in scientific comparisons.
- Permit `sha256=null` for large or changing files, with the missing hash
  stated through `hash_status`.

Chemtools will not use partial-file hashes as if they were full content
identity. Large files can remain unverified until a workflow requests an
exact snapshot.

Directory manifest hashes are reproducible only within their declared schema.
The first schema sorts normalized POSIX relative paths bytewise and hashes
each entry's type, relative path, size, and full file SHA-256 without
following symlinks. Changing those rules requires a new schema identifier.

### Provenance is an append-only event graph

Provenance records transformations and use of artifact snapshots:

```python
@dataclass(frozen=True)
class ProvenanceRecord:
    event_id: str
    event_type: str
    occurred_at: datetime
    actor: ProducerIdentity
    run_uid: str | None
    step_id: str | None
    inputs: tuple[ArtifactSnapshotRef, ...]
    outputs: tuple[ArtifactSnapshotRef, ...]
    evidence: Literal["recorded", "declared", "inferred"]
    parameters: Mapping[str, JsonValue]
```

The initial event types are:

- `generated`
- `copied`
- `moved`
- `converted`
- `merged`
- `selected_as_seed`
- `referenced`
- `parsed`
- `imported`
- `superseded`

The set is string-valued so program adapters can introduce a qualified event
such as `grasp.staged_orbital_birth` when the generic types lose important
meaning. A new type needs a consumer and a test.

`actor` records the responsible program, Chemtools component, external tool,
or manual import, plus version or commit when known. Transformation parameters
record the scientifically relevant choices. A GRASP merge must name every
donor snapshot and its duplicate-selection rule. A QE to QMCPACK conversion
must name the input save data, pseudopotentials, converter version, and output
HDF5 artifact.

A transformation event is appended only after its outputs have observations,
so both sides refer to exact snapshots. A failed attempt that produces no
observable output records an execution failure instead of a provenance output
that cannot be identified.

`evidence` prevents discovery from becoming invented history:

- `recorded` means Chemtools observed or performed the operation.
- `declared` means a launch plan, workflow, or trusted manifest specified it.
- `inferred` means a classifier or filename heuristic proposed the
  relationship.

Missing provenance remains unknown. Importing a directory does not invent a
producer from a matching stem or modification time.

The graph supports multiple inputs and outputs. This replaces
`parent_run_id` as the scientific lineage model. The compatibility parent
field may still point to a previous attempt, but a restart is only
well-specified when the consumed checkpoint or seed snapshots are recorded.

### Freshness is an assessment with evidence

Freshness is relational. It asks whether an artifact snapshot is suitable for
a particular consumer or expected producer. It is not a boolean property of a
path.

```python
@dataclass(frozen=True)
class FreshnessAssessment:
    verdict: Literal["current", "stale", "changed", "missing", "unknown"]
    artifact_id: str
    observation_id: str | None
    compared_with: tuple[str, ...]
    evidence: tuple[str, ...]
```

The verdicts mean:

- `current`: recorded lineage identifies the snapshot required by the
  consumer, or its exact hash matches a declared required snapshot, and
  available observations do not contradict it.
- `stale`: the snapshot comes from an earlier or superseded producer, or a
  required newer dependency is known.
- `changed`: the current path no longer matches the recorded snapshot.
- `missing`: the expected location or artifact is absent.
- `unknown`: available metadata cannot establish the relationship.

Modification time alone cannot produce `current`. It may support a stale
warning, but copies can preserve timestamps and syncs can move old content
forward. Exact hashes and recorded transformation events are stronger
evidence.

Program adapters also declare authoritative dependencies. When a QMCPACK XML
file references `vp.h5`, the HDF5 snapshot is authoritative for the optimized
parameters. Parsing coefficients printed in XML must report that they are
display copies and must not use them as a seed while the sidecar reference is
active.

### Provenance informs independence without claiming validity

Review and comparison services will trace artifact ancestors before calling
two checks independent. Shared ancestry is reported as correlated evidence.
For example, a GRASP level count and configuration-average check derived from
the same truncated CSF artifact do not independently validate that artifact.

Provenance alone does not establish scientific validity. A fully traced
Thomas-Fermi seed can still converge to the wrong basin. Validation status and
review verdicts remain separate records that cite the artifact snapshots and
checks they evaluated.

### Discovery is bounded and program-aware

Each backend may provide an artifact classifier. Classification starts from:

1. Expected artifacts declared by a launch plan or workflow.
2. Files explicitly named by inputs, outputs, and sidecar references.
3. Program-specific bounded filename and content rules.

Classifiers do not recursively scan arbitrary trees at server startup. They
return the proposed role, kind, producing step when known, and evidence level.
Ambiguous candidates remain separate and are reported as ambiguous.

`inspect_run` may accept a run UID, working directory, or artifact path. A UID
loads the recorded collection. A path without a record creates an imported
run candidate and invokes the bounded classifier. The user or a deterministic
rule must confirm relationships that cannot be established from file content.

### Serialization is versioned JSON; SQLite stores metadata

The portable form uses an explicit envelope:

```json
{
  "schema": "chemtools.run-artifacts/1",
  "run": {},
  "artifacts": [],
  "observations": [],
  "provenance": []
}
```

Core Python models will be frozen dataclasses with explicit JSON conversion.
Chemtools will not use pickle, implicit `dataclasses.asdict` output, or
unversioned database rows as a public interchange format.

SQLite remains the local metadata index. It stores runs, steps, artifacts,
observations, provenance events, and derived-result references. Artifact bytes
stay in their existing filesystems. Chemtools will not create a content store
or copy multi-gigabyte checkpoints into `~/.chemtools`.

The current wide `runs` columns remain compatibility projections during
migration. A projected energy or geometry records the source observation and
parser identity. Parsed-result caches are keyed by source observation,
parser version, and result schema. If the source path changes after parsing,
the old result remains historical and is not returned as current.

Reference-case selection, redistribution status, and licensing belong to ADR
005. A reference manifest may point to these artifact snapshots, but reference
status is not part of runtime artifact identity.

## Migration

Phase 2 will introduce the model without rewriting the runner:

1. Add frozen run, step, artifact, observation, freshness, and provenance
   types with exact serialization round-trip tests, including observation
   membership and snapshot-valued provenance outputs.
2. Add stable `run_uid` values beside current integer run IDs.
3. Add normalized SQLite tables for artifact metadata and provenance while
   retaining current run columns.
4. Add bounded artifact classifiers for representative NWChem, Molcas, DIRAC,
   and GRASP fixtures.
5. Adapt current `input_file`, `output_file`, and `parent_run_id` records into
   artifact collections with `evidence="inferred"` where history is absent.
6. Record lineage in existing write paths that copy or create restart files,
   `RasOrb` files, DIRAC checkpoints, and GRASP seeds.
7. Connect ADR 002 launch-plan expectations to the artifact collection when
   execution migration begins.
8. Key parsed summaries and review results to artifact observations before
   treating registry scalar columns as caches.
9. Remove single-input and single-output assumptions only after golden MCP
   cases cover compatibility responses.

The first lineage regressions will cover:

- A GRASP seed merged from two donor artifacts.
- An ATSP2K artifact converted before GRASP use.
- A DIRAC checkpoint renamed and consumed by a later step.
- A Molcas `RasOrb` transformed and supplied through `FILEORB`.
- A QMCPACK XML file whose referenced `vp.h5` is authoritative.
- A path overwritten after a parser observation.

## Consequences

Chemtools can represent existing GRASP and DIRAC workflows without choosing a
fake primary file. Quantum ESPRESSO save trees, pseudopotentials, conversion
outputs, and QMCPACK sidecars fit the same collection.

Recorded lineage gives chemistry checks the information needed to identify
stale dependencies, multiple donors, and shared upstream evidence. Unknown
history stays visible as uncertainty.

The model adds IDs, observations, and graph records around files that users
already manage. Keeping bytes in place and using SQLite for metadata limits
the operational cost. Selective hashing avoids reading tens of gigabytes for
routine inspection.

Program adapters must classify artifacts and record transformations
truthfully. That is extra work, but the alternative is to keep critical
scientific state in filenames, response prose, and working-directory
conventions.

## Alternatives rejected

### Keep one input and one output plus an attachments list

An untyped attachments list still loses roles, producing steps, authority,
and transformation lineage. It cannot express a multi-donor seed or distinguish
an optional visualization from a required checkpoint.

### Use the path as the artifact ID

Paths are reused, renamed, copied, synced, and moved across machines. A live
output can change after parsing. Path identity would make historical parser
results appear to describe new content.

### Use the content hash as the artifact ID

Large and growing artifacts may not have a hash. Byte-identical copies can
play different roles and have different downstream histories. Hash equality
is evidence about content, not a replacement for artifact identity.

### Infer freshness from modification time

Copies may preserve timestamps, old data may be synced onto a live path, and
filesystem clocks may differ. Modification time is useful evidence but cannot
prove that the expected producer created the file.

### Copy all artifacts into a managed content store

The local corpus includes tens of gigabytes of checkpoints and scratch data,
and cluster runs may produce much more. A managed blob store adds quota,
cleanup, transfer, and duplication problems without being required for
lineage.

### Store provenance as free-form notes

Notes cannot answer which exact snapshot a later run consumed or whether two
checks share an ancestor. The core relationships must be structured; human
notes can remain attached metadata.

## Acceptance checks

Phase 0 accepted this decision with these conditions:

- A run is one attempt and may contain several ordered executable steps.
- Artifact IDs, locations, and point-in-time observations are distinct.
- Roles remain cross-program while kinds preserve program-specific meaning.
- Provenance records exact input snapshots and supports multiple donors.
- Provenance outputs also identify exact observed snapshots.
- Directory manifest hashes declare the algorithm schema that produced them.
- Unknown or inferred history is not presented as recorded fact.
- Freshness uses evidence and never relies on modification time alone.
- Parsed results identify the artifact observation and parser that produced
  them.
- SQLite stores metadata while scientific artifact bytes remain in place.
- The first schema is versioned JSON without a new serialization dependency.

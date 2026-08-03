# ADR 005: Reference corpus boundaries

Status: Accepted

Date: 2026-07-30

Chemtools will keep reference material in three storage tiers: committed test
fixtures, committed scientific datasets, and an external development corpus.
Storage tier, redistribution permission, case purpose, and scientific review
status are separate fields. A file's location does not establish whether it
may be copied or whether its chemistry is a validated reference.

Committed tests will remain self-contained. External cases will be opt-in,
read-only, selected through committed JSON manifests, and verified by exact
size and SHA-256 before any parser or external tool reads them. Chemtools will
not scan the external corpus at startup or copy it into a managed data store.

## Context

Chemtools has two materially different sources of reference data.

The repository contains small NWChem examples, MCP golden cases, and an
f-block atomic library. The atomic library currently contains 106 files and
about 704 KB of data. Its largest file,
`chemtools/data/fblock/grasp/fblock-all.json`, is about 461 KB. It
records 31 elements and 633 states, including GRASP2018 v2 results rebuilt on
2026-07-28, ATSP2K inputs, and selected DIRAC cases. This is maintained
scientific data, rather than an ordinary parser fixture.

The local development corpus under `/home/charlie/input_examples` is about
61.5 GB (57.2 GiB) and contains 7,953 files. It includes NWChem, OpenMolcas,
DIRAC, GRASP2018, Quantum ESPRESSO, and QMCPACK material. It also contains:

- Large outputs, binary checkpoints, save directories, and scratch files.
- Modified, deleted, and untracked research data.
- Third-party tutorials and pseudopotentials with separate license terms.
- Successful calculations, deliberate failures, unresolved work, and shelved
  studies.
- Files whose useful meaning depends on companions or their production
  history.

Copying that tree into the repository would blur ownership, inflate clones,
and freeze transient work as if it were reviewed reference data. Recursively
scanning it would also make server startup depend on a private, changing
filesystem.

The Phase 0 Orbitron contract has already established a safer initial pattern.
Its committed manifest names three relative paths and pins their SHA-256
hashes. The checker resolves paths under `CHEMTOOLS_REFERENCE_CORPUS`, rejects
root escapes, verifies each hash, and keeps `agree`, `disagree`,
`tool_refused`, and `no_reference` outcomes separate.

That manifest is intentionally small, but its current `status` values mix two
ideas. `validated` describes review status, while `known_failure` describes
why a case exists. A known failure can be fully reviewed as a regression
case, yet it must never be recommended as a scientifically valid workflow.

The project plan also needs a policy for promoting a small external case into
the repository. Permission, attribution, sensitive paths, expected facts,
companion artifacts, and scientific scope must be checked before copying
bytes. Git history makes an accidental copy difficult to retract.

## Decision

### Storage has three tiers

Every reference artifact belongs to one storage tier:

| Tier | Location | Default use |
| --- | --- | --- |
| `committed_fixture` | Repository test or example directory | Deterministic tests in every environment |
| `committed_dataset` | Versioned repository data directory | Typed scientific lookup and dataset validation |
| `external_reference` | Configured local corpus root | Opt-in differential, integration, and research cases |

A committed fixture is the smallest artifact set that proves one parser,
diagnostic, or workflow contract. It should contain exact expected values and
enough surrounding content to retain the parser behavior under test.

A committed dataset is maintained scientific content with its own version,
scope, provenance, and validation rules. It may support many lookups and
tests. It is not duplicated into fixture directories for convenience.

An external reference stays in its existing filesystem. Chemtools stores
metadata and expectations in Git, while the artifact bytes remain outside the
repository.

Storage tier does not imply scientific status. A committed fixture may encode
a malformed input. An external case may be the best validated workflow in the
corpus.

### The f-block atomic library is a committed dataset

Phase 5 moved the single canonical f-block dataset to
`chemtools/data/fblock`, rather than copying it. Installed packages read it
through package resources, and the old notes path retains no second copy.
Documentation, metadata, and validation tests live with the dataset.

Before the dataset becomes a public lookup source, its machine-readable
envelope or adjacent metadata must record:

- Dataset schema and version.
- Rebuild date and producing program versions when known.
- Hamiltonian and method scope.
- Element and state coverage.
- Seed and donor provenance.
- Validation procedure and known limitations.
- Redistribution statement for first-party and third-party components.

The GRASP v2 values remain method-scoped references. Their presence in Git
does not turn convergence into proof that a state is physically correct.
DIRAC failure cases in the same library retain their failure purpose and
cannot be returned as successful recipes.

### Location, permission, purpose, and status are independent

The manifest keeps artifact custody separate from case meaning:

1. Each artifact records `storage_tier`, `redistribution`, `source`,
   `attribution`, `license`, and `permission_evidence`.
2. Each case records `purposes` and `status`.

A case may mix tiers and redistribution terms. For example, a committed input
fixture may refer to an external licensed pseudopotential and an external
output. A case-level storage or custody default is allowed only when every
artifact has the same value; normalization expands defaults onto each
artifact before validation.

The redistribution values are:

- `allowed`: ownership or license permits the intended committed use.
- `restricted`: the bytes must remain outside the repository.
- `review_required`: permission has not been established.

`allowed` needs an attribution and license basis. Repository availability,
public download access, or local possession is not sufficient evidence.
Pseudopotentials and tutorial files receive their own artifact-level
redistribution record because their terms may differ from the calculation
that uses them.

Manifest validation ties the evidence fields to redistribution:

- `allowed` requires attribution, a license identifier or recorded terms, and
  permission evidence covering the intended committed use.
- `restricted` requires attribution, recorded terms or a restriction basis,
  and evidence explaining why the bytes remain external.
- `review_required` may leave evidence unresolved, but it cannot pass
  promotion validation or be copied into Git.

Initial purpose values are:

- `parser_contract`
- `differential_contract`
- `scientific_regression`
- `workflow_recipe`
- `failure_diagnosis`
- `methodology_warning`

A case may have more than one purpose. New purpose names require a consumer
or test; the list is not an extension mechanism for free-form tags.

Scientific status is one of:

- `validated_reference`: expected scientific facts and scope were reviewed.
- `regression_failure`: a reviewed failure or defect with expected detection
  behavior.
- `exploratory`: useful evidence exists, but scientific conclusions or
  workflow guidance remain unresolved.
- `shelved`: retained for history or future work and excluded from ordinary
  retrieval and testing.

`validated_reference` requires positive expected facts, a named reviewer,
review date, and method scope. It does not claim experimental truth unless
the case explicitly includes that comparison.

`regression_failure` requires the observed failure and expected diagnostic to
be pinned. It may be a valid parser contract. It cannot serve as a positive
workflow example.

`find_reference_case` will default to `validated_reference`. Diagnostic
queries may include `regression_failure`. `exploratory` and `shelved` require
an explicit request and will carry their status in the response. The Tm3+ and
Md3+ LS-coupling study remains `shelved`.

### Committed JSON manifests select external cases

Phase 0 manifests remain versioned JSON so the existing standard-library
loader can read them without another dependency. Manifests live under
`references/` and may be split by program or workflow after the case count
makes one file difficult to review. A database is not needed.

The next manifest schema will describe cases and their artifact collections:

```json
{
  "schema": "chemtools.reference-corpus/1",
  "cases": [
    {
      "id": "qe.fe_bcc.scf",
      "programs": ["qe"],
      "status": "validated_reference",
      "purposes": ["differential_contract", "workflow_recipe"],
      "artifacts": [
        {
          "id": "primary_output",
          "roles": ["primary_output"],
          "kind": "qe.stdout",
          "path": "qe/Fe/fe.scf.out",
          "storage_tier": "external_reference",
          "size_bytes": 12345,
          "sha256": "<pinned hash>",
          "required": true,
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
        "total_energy": {
          "value": -495.637697,
          "unit": "Ry",
          "absolute_tolerance": 0.000001
        }
      },
      "review": {
        "reviewed_by": "charlie",
        "reviewed_at": "2026-07-30",
        "scope": "Parser and periodic-cell contract"
      },
      "tags": ["periodic", "magnetic", "metal", "orbitron"]
    }
  ]
}
```

Each case has:

- A stable ID and schema version.
- Program or workflow ownership.
- One scientific status and one or more purposes.
- Explicit artifact roles, kinds, relative paths, exact sizes, SHA-256
  hashes, and required status.
- Artifact-level storage tier, source, attribution, license,
  permission evidence, and redistribution status.
- Expected facts with units and explicit tolerances where exact equality is
  inappropriate.
- Review identity, date, and scope.
- Required companion artifacts.
- Tags for method, element, property, failure mode, and execution shape.

The manifest stores exact `size_bytes`. User interfaces may derive `small`,
`medium`, or `large` labels, but access decisions use byte counts rather than
an unverified label.

Expected values are part of the reviewed contract. A tool output does not
become the expected value merely because it was the first result recorded.
When the expected value comes from Chemtools or Orbitron, the manifest records
that producer and version so circular comparisons remain visible.

### External access is bounded and read-only

`CHEMTOOLS_REFERENCE_CORPUS` names the single external root for the initial
implementation. Committed runtime defaults will not contain
`/home/charlie/input_examples` or another user's absolute path. A command-line
override may select a different root for development.

Paths resolve according to each artifact's tier. Committed fixtures and
datasets resolve only under their declared repository or installed package
resource roots. External artifacts resolve only under
`CHEMTOOLS_REFERENCE_CORPUS`. A mixed case does not change those boundaries.

For every selected artifact, the loader will:

1. Require a relative manifest path.
2. Resolve it under the configured root.
3. Reject `..`, absolute paths, and symlink targets outside that root.
4. Check that the entry type and exact size match the manifest.
5. Enforce the command's byte budget before hashing or parsing.
6. Verify SHA-256.
7. Pass the verified path to the parser or read-only external tool.

The loader will not recursively scan the root, search for replacement files,
or accept a same-named file after a hash mismatch. Missing files, size
changes, and hash changes produce `no_reference` with a specific reason.
They do not become parser disagreements.

There is a known interval between verification and the consumer's final read
during which an external file could change. Version 1 accepts that limitation
for a local, read-only corpus. If it causes a real failure, the loader will add
a post-use observation and hash before changing the baseline access path.

Manifests do not grant permission to execute chemistry programs, modify
artifacts, or write into the corpus. Execution follows ADR 002. An integration
may write reports or temporary conversion products only to an explicit output
directory outside the source corpus.

Compound artifacts such as a QE save directory must list required companion
files or provide a versioned directory inventory. A directory name alone is
not a stable reference. General recursive directory hashing is deferred until
a real compound case requires it.

### Size is checked before content is read

Committed fixtures have a default ceiling of 1 MiB per artifact. A larger
fixture needs an explicit, reviewed exception explaining why a smaller
excerpt would change the behavior under test. Committed datasets use a
documented dataset budget instead of the fixture ceiling.

External commands have a declared total byte budget. A manifest case that
exceeds the command's default is refused before hashing. An explicit
large-case option may raise the budget for a selected case. This prevents a
small contract command from reading a multi-gigabyte checkpoint by accident.

The manifest records every required companion's size, so selection can
calculate the total before opening files. Unknown sizes are invalid in a
committed external manifest.

### Tests use the tier appropriate to their contract

Unit tests and the default CI suite depend only on committed fixtures and
committed datasets. They do not require an environment variable, network
download, private checkout, or local corpus.

External tests are opt-in and select named manifest cases. They do not scan
for whatever happens to be available. Their machine outcomes remain:

- `agree`: all selected contract checks passed.
- `disagree`: at least one selected field differed.
- `tool_refused`: the external parser or tool could not complete the case.
- `no_reference`: the root, artifact, expected size, or expected hash was
  unavailable.

These outcomes describe comparison mechanics. Scientific disposition remains
in the manifest. A `regression_failure` can produce `agree` when a tool
correctly detects the pinned failure. `agree` does not promote the case to
`validated_reference`.

A disagreement is reviewed before changing either the expected facts or the
parser. A documented intentional difference belongs in a named comparison
rule with its scope and tolerance. A changed file cannot be accepted by
updating its hash alone.

Committed regression fixtures pin exact parsed values and diagnostic codes.
Existence-only assertions are insufficient. Numerical tolerances state their
units and reason.

### Promotion into Git is a review operation

To promote material from the external corpus:

1. Select the smallest artifact set that preserves the behavior.
2. Record hashes and expected facts against the original source artifacts.
3. Establish ownership, license, attribution, and redistribution permission
   for every artifact.
4. Remove credentials and private operational data, including tokens,
   scheduler accounts, private hostnames, and unnecessary absolute paths.
5. Preserve enough context for parser position, cross-file consistency, and
   scientific meaning.
6. Give the copy a stable case ID, storage tier, purpose, status, and review
   record.
7. Record the source artifact hash and the transformation used to create any
   excerpt or redacted copy.
8. Add exact tests before treating the copy as a fixture.

An excerpt or redacted copy is a new artifact with provenance under ADR 003.
Its hash must differ from the source when its bytes differ. The external
source is never modified as part of promotion.

If permission is uncertain, the bytes stay external and
`redistribution="review_required"`. If a case needs a large binary only for
one local experiment, it remains external even when redistribution is
allowed.

### Reference metadata and runtime provenance stay separate

ADR 003 defines runtime artifact identity, observations, and transformation
history. A reference manifest can identify or import those snapshots, but
reference status does not become an artifact ID or runtime freshness verdict.

The same bytes may appear in several cases with different purposes, while one
case may contain several artifact snapshots. Manifests should refer to shared
artifact metadata instead of copying expectations only after a third concrete
reuse makes that indirection worthwhile.

Orbitron is a comparison participant, not an automatic source of scientific
truth. Its executable version, commit, output schema, and warnings are
recorded in each report. Agreement between Chemtools and Orbitron may still
reflect a shared assumption or parser rule.

## Migration

Phase 0 will keep the existing three-case Orbitron contract working while the
general corpus model is introduced:

1. Accept this ADR before changing manifest semantics.
2. Add a versioned general reference-manifest loader under `references/`.
3. Keep the current Orbitron schema reader as a compatibility adapter.
4. Map `status="validated"` to `validated_reference`.
5. Map `status="known_failure"` to `regression_failure` and add
   `failure_diagnosis` to its purposes.
6. Add exact byte sizes, artifact roles, per-artifact storage tier, source,
   attribution, license, permission evidence, redistribution status, expected
   facts, and review scope to the three pilot cases.
7. Separate `no_reference` reasons for missing root, missing file, size
   change, hash change, and path rejection without changing the top-level
   outcome or exit-code contract.
8. Enforce the byte budget before SHA-256 calculation.
9. Add manifest validation tests for absolute paths, parent traversal,
   symlink escape, duplicate IDs, invalid hashes, unknown sizes, missing
   companions, invalid mixed-tier resolution, missing redistribution
   evidence, and missing status-required review fields.
10. During Phase 5, move the canonical f-block library to
    `chemtools/data/fblock`, add dataset metadata and validation tests, update
    every old-path reference, and test access from an installed package.
11. Add more programs only through reviewed cases from the pilot list in
    `PROJECT_PLAN.md`.
12. Implement status-filtered `find_reference_case` during Phase 5.

The current checker continues to return exit code 0 for full agreement, 1 for
a disagreement, 2 for tool refusal, and 3 for unavailable reference data.

## Consequences

The repository remains small enough to clone and test without access to a
private research tree. Local development can still use large, realistic
artifacts through stable, reviewable manifests.

Scientific curation becomes visible. A failing calculation can be a
high-quality regression case without appearing as a recommended workflow.
The assistant can state whether a claim comes from a validated reference,
diagnostic failure, exploratory case, or shelved study.

Permission review becomes an explicit gate before bytes enter Git. This is
especially important for pseudopotentials, tutorials, and licensed program
outputs whose terms may differ from first-party research data.

Manifests add upkeep when external artifacts change. That is intentional:
size or hash drift stops the comparison until the expectations and provenance
are reviewed.

The first implementation needs a general manifest validator and clearer
unavailable reasons. It does not need a database, network downloader, content
store, or recursive indexer.

## Alternatives rejected

### Commit every useful case

The corpus contains tens of gigabytes, private working state, binaries,
third-party material, and scratch data. Repository size would grow quickly,
and Git history would retain accidental or unlicensed copies.

### Keep every reference outside Git

CI would lose deterministic parser and chemistry regressions. Small
permission-clean fixtures and maintained first-party datasets are worth
versioning with the code that consumes them.

### Treat every committed file as validated

Many useful fixtures encode malformed input, parser refusal, stale artifacts,
or known scientific failures. Storage location cannot carry a scientific
verdict.

### Use one `validated` boolean

A boolean cannot distinguish a positive reference from a reviewed failure,
an unresolved experiment, or a deliberately shelved study. It also cannot
state the scope of the review.

### Search the corpus dynamically

Discovery by filename or recursive scan would make results depend on local
working state and could select an unreviewed file. Stable IDs and listed paths
make selection reproducible.

### Download missing references automatically

The corpus includes private data and artifacts with varied license terms.
Automatic downloads would add network, authentication, integrity, and
retention policy before there is a concrete need.

### Put large files in Git LFS now

Git LFS would still require storage, bandwidth, access control, cleanup, and
license decisions. The external manifest solves the current testing need
without adding that service.

### Let a hash define reference identity

The same bytes may have different roles, review scopes, or provenance.
Transformed excerpts also need a relationship to their source. Hashes verify
content; stable case and artifact records carry meaning.

## Acceptance checks

Phase 0 accepted this decision with these conditions:

- Committed fixtures, committed datasets, and external references are
  distinct storage tiers.
- Storage, redistribution, purpose, and scientific status are independent.
- Storage, source, attribution, license, permission evidence, and
  redistribution are recorded per artifact, including mixed-tier cases.
- The four scientific statuses control testing and retrieval behavior.
- The f-block atomic library remains one versioned committed dataset.
- Phase 5 moves that dataset into package data without leaving a duplicate.
- External paths are relative, root-contained, size-checked, and hash-checked.
- The loader never scans the external corpus or substitutes a nearby file.
- Default tests are self-contained and external tests are explicit.
- Missing or changed reference data cannot become agreement or disagreement.
- Promotion into Git requires permission, attribution, provenance, and exact
  expectations.
- Large external artifacts require an explicit byte-budget override.
- No database, downloader, content store, or Git LFS dependency is added in
  Phase 0.

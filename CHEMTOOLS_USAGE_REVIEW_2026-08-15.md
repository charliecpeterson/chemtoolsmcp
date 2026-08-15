# Chemtools usage review: NWChem state and provenance inspection

Date: 2026-08-15

This note records changes suggested by using Chemtools during the
`scfdiffusion` PuCl4 reference calculation. The highest-value change is to
surface numerical rank reduction from NWChem output. The next two are to keep
execution outcome separate from state interpretation and to bind orbital
evidence to the task that produced it.

The recommendations are based on one real multi-task NWChem archive and one
matched PySCF follow-up. They are deliberately narrow. They do not call for a
new parser framework or a general external-results abstraction.

## Case that exposed the gaps

The archived PuCl4 output contains six successful tasks: two fragment single
points, a full-system single point, optimization, frequency, and property.
Chemtools correctly extracted all six, the final energy, the lack of
significant imaginary modes, and the exact normalized input/output deck match.

The current guided inspection returned:

```json
{
  "label": "wrong_state_convergence",
  "confidence": 0.6,
  "reasons": [
    "singly occupied orbitals do not match expected metal state",
    "stage: property"
  ]
}
```

It returned no uncertainty. The output itself contains a numerical warning
that was more important for the immediate cross-code calculation:

```text
WARNING : Found     1 linear dependencies
S eigenvalue threshold:     1.00000E-05
Smallest S eigenvalue :     3.27149E-06
```

NWChem removed that overlap eigenvector. The first matched PySCF run retained
all 223 AOs, did not converge in 150 cycles, and produced an independent
orbital-gradient norm of 27.42 with nonsensical Pu populations. Both isolated
fragments converged cleanly with the expected electron counts and Pu 5f
population. The missing rank reduction is therefore the first protocol
mismatch to control, though the rerun is still needed before assigning it as
the sole cause.

Source artifacts:

- NWChem output:
  `/home/charlie/scratch/scfdiffusion_attempt70_intake/295444.pucl4-b3lyp-stu97-vtz_pucl4-b3lyp-stu97-vtz.out`
- NWChem input:
  `/home/charlie/scratch/scfdiffusion_attempt70_intake/pucl4-b3lyp-stu97-vtz.nw`
- PySCF summary:
  `/home/charlie/projects/scfdiffusion/results/attempt70_h2/summary.json`

## 1. Parse and surface overlap-rank reduction

This is the first change I would make. NWChem prints enough information to
capture the condition without reading a binary artifact:

```json
{
  "kind": "linear_dependence",
  "task_index": 2,
  "removed_vector_count": 1,
  "overlap_eigenvalue_threshold": 1e-5,
  "smallest_removed_eigenvalue": 3.27149e-6,
  "line_range": [7031, 7039]
}
```

Put this under structured diagnostics or a small `numerical_conditioning`
record. It should be visible in guided `inspect_run` evidence and summarized in
uncertainty when the effective orbital rank matters for restart or cross-code
work. A warning alone should not change a successful task into a failed task.

The current task parser only promotes errors into diagnostics. The relevant
work belongs near
[`parse/tasks.py`](chemtools/programs/nwchem/parse/tasks.py) and should preserve
the task boundary and source lines.

Acceptance tests:

- A fixture containing the PuCl4 warning returns count `1`, threshold `1e-5`,
  and smallest eigenvalue `3.27149e-6`.
- The diagnostic is attached to the full-system task, not either fragment.
- An output without this warning has no inferred rank reduction.
- Repeated warnings from optimization steps remain task-scoped and do not get
  collapsed into one global value.

## 2. Separate execution outcome from state identity

The current diagnosis converts a medium-confidence frontier-orbital heuristic
into `wrong_state_convergence` in
[`diagnose.py`](chemtools/programs/nwchem/strategy/diagnose.py#L525), then
[`_plugin_strategist.py`](chemtools/programs/nwchem/_plugin_strategist.py#L46)
prefers that failure class over the successful task outcome. This makes the
top-level verdict stronger than the evidence warrants.

The guided result should carry separate axes, for example:

```json
{
  "execution": {
    "outcome": "success"
  },
  "numerical_quality": {
    "assessment": "rank_reduction_applied"
  },
  "state_identity": {
    "assessment": "suspicious",
    "confidence": 0.6,
    "evidence_grade": "canonical_orbital_coefficients",
    "density_validated": false
  },
  "structure": {
    "assessment": "minimum_by_reported_frequencies"
  }
}
```

If `chemtools.inspect-run/1` must retain one compatibility verdict, use a label
such as `success_with_state_uncertainty` while preserving the more specific
state assessment below it. `wrong_state_convergence` should require stronger
evidence, such as a density-derived population or an explicit occupation check
against a saved wavefunction.

There is already a useful distinction in
[`recovery.py`](chemtools/programs/nwchem/strategy/recovery.py#L265): a
ligand-dominated SOMO pattern with metal-centered total spin can be a covalent
ligand-hole candidate rather than a bad state. Guided inspection should not
discard that nuance before recovery planning sees the case.

Acceptance tests:

- Six successful tasks remain `execution.outcome = success` even when state
  identity is suspicious.
- A SOMO-only mismatch records the evidence grade and
  `density_validated = false`.
- A saved-wavefunction or AO-population contradiction can produce a stronger
  state verdict than canonical-orbital coefficients alone.
- Existing callers that read the compatibility verdict receive a documented,
  stable label during the schema transition.

## 3. Make orbital and state evidence task-scoped

The NWChem parser accepts `task_index`, but orbital parsing currently ignores
it. [`_plugin_parser.py`](chemtools/programs/nwchem/_plugin_parser.py#L122)
states that MO parsing treats the file as one aggregate section, and
[`parse/mos.py`](chemtools/programs/nwchem/parse/mos.py#L114) keeps the latest
section for each spin across the complete file.

That is unsafe for fragment-guess workflows. A single output can contain
different geometries, charges, multiplicities, and orbital spaces. Each
orbital, population, and state assessment should record at least:

- task index and task kind;
- geometry or system identity when available;
- source line range;
- which evidence block supplied the state claim.

The parser can use the task boundaries it already emits. The requested
`task_index` should slice the text or filter blocks by those boundaries before
frontier analysis. The default can remain the selected primary task, but the
selection must be explicit in the result.

Acceptance tests:

- A synthetic two-task output with different SOMO characters returns different
  results for `task_index=0` and `task_index=1`.
- The Pu fragment, Cl4 fragment, and full PuCl4 system cannot share one orbital
  assessment accidentally.
- A requested task with no MO block returns `unavailable` for that task rather
  than borrowing another task's block.

## 4. Report input-declared artifacts that were not supplied

The PuCl4 input declares `m.movecs`, `ln.movecs`, and
`pucl4-b3lyp-stu97-vtz.movecs`. None was supplied to guided inspection. The
result nevertheless had no uncertainty about the absence of density-level
state evidence.

This can fit the existing explicit-artifact policy without directory scanning.
When an input is passed in `artifact_files`, parse output declarations such as
`vectors ... output NAME` and return an expectation with one of these states:

- `supplied`;
- `declared_not_supplied`;
- `supplied_missing`;
- `ambiguous`.

`declared_not_supplied` means only that the caller did not attach the artifact.
It must not claim the file is absent on disk. Add an uncertainty such as
`state_identity_not_density_validated` when the missing attachment limits the
scientific claim.

This extends the artifact-expectation model already specified in
[`ADR 003`](docs/adr/003-runs-are-artifact-collections-with-provenance.md),
especially its explicit input, output, and sidecar references. It does not need
a second artifact model.

Acceptance tests:

- `vectors output final.movecs` in a supplied input creates one expectation.
- An explicitly supplied matching path satisfies it.
- An unsupplied declaration reports `declared_not_supplied`, without probing
  the output directory.
- Fragment input and full-system output vector declarations remain distinct.

## 5. Let guided inspection consume the existing PySCF result schema

Chemtools already defines `chemtools.pyscf-single-point-result/1` in
[`science_runner.py`](chemtools/science_runner.py) and validates it in
[`pyscf_comparison.py`](chemtools/core/pyscf_comparison.py). The useful next
step is an adapter from that result into the guided inspection shape.

The adapter should map the existing fields for completion, SCF convergence,
energy, electron counts, method, geometry, provenance, and written CUBE
artifacts. It should reject unknown schemas and incomplete results explicitly.
This keeps PySCF out of the base environment and avoids parsing arbitrary
stdout from user-written PySCF scripts.

The project-specific Attempt 70 JSON should remain outside this interface for
now. A generic external-result envelope would be premature based on one custom
workflow. Add that only after another concrete result format needs the same
boundary.

Acceptance tests:

- A completed `chemtools.pyscf-single-point-result/1` maps to guided evidence
  without importing PySCF.
- An unconverged result remains a completed execution with failed SCF quality.
- Unknown schema versions and malformed results fail with typed errors.
- Existing artifact hashes and companion-runtime provenance survive the map.

## Suggested implementation order

1. Parse linear-dependence warnings and add the PuCl4 fixture.
2. Split execution, numerical quality, and state identity in the guided
   assessment.
3. Honor `task_index` for MO, population, and state parsing.
4. Connect input-declared wavefunction files to the existing artifact
   expectation model.
5. Add the adapter for the existing PySCF result schema.

The first item is small and immediately useful. Items 2 and 3 should probably
land together because task provenance determines how much confidence a state
claim deserves. The PySCF adapter is useful, but it did not cause the PuCl4
misdiagnosis and can wait until the NWChem evidence path is sound.

## Behavior worth keeping

- Exact normalized input/output deck comparison caught no provenance mismatch.
- Multi-task extraction correctly found all six NWChem tasks and their energy
  records.
- Program version fallback worked on the old NWChem 7.0.0 output.
- Frequency parsing distinguished near-zero modes from significant imaginary
  modes.
- Related artifacts are caller-supplied and bounded. The recommendations above
  preserve that rule.

# Non-NWChem reference review queue

This queue selects a small cross-program slice from the external corpus. It
does not approve scientific expectations. Every case in
`chemtools/data/reference_cases/non_nwchem_review_cases.json` is `exploratory`,
and each recorded
`current_inspection_verdict` is only the result produced by the current
backend on 2026-08-07.

The selection uses two cases per backend. Each pair covers different behavior
without attempting to catalog the full directory:

| Candidate | Current verdict | Pinned artifacts | Review focus |
| --- | --- | ---: | --- |
| `molcas.nactel_parity_failure` | `failed` | 2 | Confirm the deliberate active-electron/spin parity failure and bounded recovery advice. |
| `molcas.hcn_transition_state_frequency` | `completed` | 2 | Confirm transition-state provenance, raw frequencies, imaginary-mode meaning, and thermochemistry. |
| `dirac.h2o_x2c_4c_comparison` | `converged` | 5 | Establish comparability before pinning an X2C versus four-component energy difference. |
| `dirac.uranium_open_shell` | `converged` | 3 | Confirm the uranium configuration, occupations, relativistic method, basis, and expected energy fields. |
| `grasp.thorium_relativistic_limit` | `completed` | 4 | Confirm the altered speed-of-light limit, matching state content, and required derived level files. |
| `grasp.lithium_e1_transition` | `completed` | 3 | Confirm state identities, biorthogonal provenance, transition units, and E1 rate fields. |
| `qe.feo_vc_relaxation` | `relaxation_converged` | 2 | Confirm pseudopotentials, magnetic state, trajectory and cell changes, and final geometry. |
| `qe.iron_spin_scf` | `scf_converged` | 2 | Confirm pseudopotential, metallic occupations, magnetic moment, charge accounting, and energy. |
| `qmcpack.hydrogen_vmc_statistics` | `incomplete` | 3 | Resolve tutorial provenance, incomplete-log classification, and expected scalar statistics. |
| `qmcpack.oxygen_dmc_autocorrection` | `input_parameter_auto_corrected` | 4 | Identify the generated run input and confirm autocorrection, series roles, pseudopotential dependency, and provenance. |

Artifact verification establishes containment, size, and SHA-256 only. It
does not establish that an input and output belong together, that two runs are
scientifically comparable, or that a parser verdict is correct. Promotion to
`validated_reference` requires named review, a review date, an exact scope,
approved expected facts, and every artifact needed to support those facts.

The review order should start with the cases whose intent is already explicit
in their inputs or scripts: Molcas active-electron parity, QE FeO relaxation,
DIRAC X2C versus four-component, GRASP thorium limit, and QMCPACK oxygen DMC.
The second case for each backend can follow once that backend's first review
establishes the expected-fact format.

`find_reference_case` now defaults to `validated_reference`, so this queue is
absent from ordinary retrieval until a case is promoted. An explicit
`scientific_status=exploratory` search can find these candidates for curation
without presenting them as validated examples.

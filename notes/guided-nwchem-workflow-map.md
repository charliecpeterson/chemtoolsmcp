# Guided NWChem workflow map

Mapping date: 2026-08-07

This note checks the guided interface against the five pinned NWChem cases in
`$CHEMTOOLS_REFERENCE_CORPUS/nwchem/hard_cases`. It records which scientific
decision each tool can support now and where provenance or a specialized
workflow still requires manual review.

## Implemented guided intents

| Tool | Representative prompt | Expected use |
| --- | --- | --- |
| `review_input` | "Review this NWChem input before I run it." | Parse and lint one existing input without changing it |
| `draft_input` | "Draft a quintet B3LYP/def2-TZVP FeO energy input from these coordinates." | Render one new NWChem or OpenMolcas input from a common molecular specification, then lint the rendered text |
| `inspect_run` | "Did this calculation finish, and what scientific warning should I act on?" | Inspect one output plus explicitly supplied related artifacts |
| `compare_runs` | "Compare these triplet and quintet outputs and tell me which parsed energy is lower." | Check available comparability evidence before reporting conditional energy ordering |
| `plan_recovery` | "Plan the next calculation for this run, using doublet Fe(III) as the target state." | Separate current-run evidence from the intended state and return read-only candidate inputs when supported |
| `search_knowledge` | "What NWChem traps should I remember for an open-shell transition-metal calculation?" | Retrieve curated rules without inspecting a file |

`draft_input` intentionally covers one ordinary calculation at a time. It does
not claim to reconstruct fragment-guess chains, orbital swaps, projected-basis
restarts, or imaginary-mode displacements from the common specification.
Those are recovery workflows with additional provenance and state decisions.

## Case routing

| Case | Guided sequence | Current result | Missing guided decision |
| --- | --- | --- | --- |
| Fe(CN)6 low-spin fragment | `review_input` on both inputs, `inspect_run` on both outputs, `plan_recovery` with the doublet target, then `compare_runs` | Both calculations are individually successful. The candidate is conditionally 0.50538692213 Ha lower, with multiplicity 6 to 2. Recovery planning identifies five observed SOMOs against the one-SOMO target and requires a fresh doublet input rather than a same-state swap | A general fragment-guess builder is absent, so the tool reports the manual rebuild boundary instead of fabricating fragment inputs |
| Hexaaquairon swap chain | `review_input`, `inspect_run`, and `plan_recovery` for the default, fragment, and swap stages | The default wrong-state pair can prepare one reviewed swap candidate. The fragment result remains a successful but unstable multistage calculation. The supplied swap input and output disagree on restart provenance, so recovery returns `source_consistency_required` with no draft | Scientific ordering remains blocked because the notes, filenames, energies, and current state diagnoses disagree. The first local recovery candidate does not validate the cross-stage narrative |
| FeO spin comparison | `draft_input` for each proposed multiplicity, `review_input` before execution, `inspect_run` for each result, then `compare_runs` | The full decision path is covered. The quintet is conditionally 0.10209484695 Ha lower than the triplet | `plan_calculation` still requires the caller to enumerate the multiplicity set |
| Ferrocene basis stepping | `inspect_run`, then `plan_recovery` on the atomic-guess, standalone small-basis, and projected-and-damped outputs | The first two successful runs retain their +16.7 and +17.2 Ha SCF excursions. Recovery planning keeps those results usable with warnings and prepares a damped restart from each converged checkpoint. The controlled run needs no hardening | Building a smaller-basis seed from scratch remains manual because the pinned small-basis run proves that a smaller basis is not automatically stable |
| Cr(CO)6 Bailar twist | `review_input`, `inspect_run`, then `plan_recovery` on the saddle and displaced minimum | The saddle reports one significant imaginary mode at -234.96 cm-1. Recovery planning selects that mode and returns plus/minus reoptimization inputs. The displaced structure is a minimum | Running both candidates and comparing their resulting minima remains outside the read-only recovery call |

## Pinned application contract

All five cases remain exploratory. The facts below are regression evidence,
not approved chemical conclusions.

| Case | Pinned files and facts | Guided uncertainty | Expected next action | `draft_input` role |
| --- | --- | --- | --- | --- |
| Fe(CN)6 | `failed.nw/.out` and `solution.nw/.out`; -1819.496662763585 and -1820.002049685715 Ha; multiplicities 6 and 2; five versus one SOMO | Geometry and composition remain unchecked in the energy comparison. A doublet rebuild needs a manually reviewed fragment guess. The multistage solution cannot be hardened automatically | Rebuild for the target state, review fragment initialization, then compare state character | Not used for the fragment chain; the common drafter cannot preserve fragment provenance |
| Hexaaquairon | `hexaaquairon`, `_frag`, and `_swap` input/output pairs; current parsed energies -2093.070628221697, -2093.022067822851, and -2093.070929682629 Ha | The swap pair reports `input_output_mismatch` for restart artifacts. Cross-stage provenance is unresolved | Confirm source artifacts before another swap; do not claim an ordering across the three stages | Not used; this is an occupation-steering recovery sequence |
| FeO | Triplet `failed.nw/.out` and quintet `solution.nw/.out`; quintet conditionally lower by 0.10209484695 Ha | Geometry remains unchecked and multiplicity is the comparison axis | Review state character and extend the multiplicity comparison when chemically needed | Applicable. The exact quintet B3LYP/def2-TZVP draft is pinned in `tests/test_input_drafting.py` |
| Ferrocene | `failed`, `small_basis`, and `solution` input/output pairs; +16.7 and +17.2 Ha transient excursions; controlled result -1650.9693626187 Ha | The controlled input/output pair lacks explicit restart-artifact provenance, so verification retains `input_output_mismatch` without preparing a recovery | Accept successful results with the stability warning, and review a damped checkpoint restart only for the unstable simple inputs | Not used; basis projection and checkpoint reuse belong to recovery planning |
| Cr(CO)6 | Saddle `failed.nw/.out` and minimum `solution.nw/.out`; one -234.96 cm-1 significant imaginary mode versus none | The saddle interpretation remains exploratory even though the parser/recovery behavior is pinned | Review the plus/minus displacement drafts, save one reviewed candidate, then run and compare the resulting minima | Not used; normal-mode displacement requires output-derived recovery evidence |

`tests/test_nwchem_behavior_lock.py` pins the review verdict, inspection
verdict, uncertainty codes, and next-action names for every listed input/output
pair. It also pins recovery behavior for Fe(CN)6, hexaaquairon, ferrocene, and
Cr(CO)6. `tests/test_run_comparison.py` and `tests/test_input_drafting.py` pin
the FeO comparison and ordinary draft path.

## Recovery boundary decision

Keep `inspect_run` responsible for evidence and the current-run verdict. Add a
separate `plan_recovery` intent for requests that need a new calculation. The
recovery service should consume an inspection plus the explicit input and
expected-state context, then return a bounded plan before any file is written.

This separation matters for successful but scientifically unsuitable runs.
`inspect_run` cannot infer the desired electronic state from one output, while
`plan_recovery` can accept the missing target and propose a fragment guess,
orbital swap, stabilization restart, or imaginary-mode displacement. The
current low-level tool names remain implementation details behind that plan.

## Implemented recovery contract

`plan_recovery` now covers the two pinned starting behaviors. For Cr(CO)6 it
returns plus/minus displacement drafts for the -234.96 cm-1 mode. For the
Fe(CN)6 high-spin input, an explicit doublet target produces a target-state
rebuild plan and blocks a misleading automatic vectors swap. The matching
low-spin solution has the requested one-SOMO state, but its own 7.35 Ha SCF
excursion still produces a separate manual hardening warning. Its multi-stage
fragment input is not rewritten automatically.

For ferrocene, completed but unstable paths now return optional stability
hardening rather than being mislabeled as failed. Simple inputs get a
read-only restart candidate with 70 percent damping for 25 cycles and the
declared converged vectors. Smaller-basis projection remains a manual fallback,
and its seed calculation must pass the same SCF-path inspection.

Recovery planning now runs the same explicit input/output consistency check as
run inspection. A confirmed mismatch blocks automatically prepared candidate
text while leaving verification-only or already-manual plans intact with an
uncertainty record. Hexaaquairon remains excluded from positive cross-stage
scientific assertions until its provenance discrepancy is resolved.

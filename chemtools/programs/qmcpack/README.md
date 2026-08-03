# QMCPACK support

This first QMCPACK slice reviews XML input files with a `simulation` or
`qmcsystem` root. It records project metadata, includes, HDF5 sidecars,
particle sets, pseudopotential references, Hamiltonians, and QMC blocks.

The linter rejects malformed XML, unsupported roots, empty include or
variational-parameter references, incomplete pseudopotential references,
missing QMC methods, and non-positive explicit particle-group sizes. It also
rejects explicit non-positive `blocks`, `steps`, `targetWalkers`, and
`total_walkers`, plus non-positive or non-finite `timestep` values. Modern
`warmupSteps` and legacy `warmupsteps` are accepted as nonnegative values and
must agree when both are present. Nonnumeric values and unrecognized
`nonlocalmoves` controls are reported for review rather than rejected because
they may be template tokens resolved before launch. `targetWalkers` and
`total_walkers` may coexist only when they agree.
The deprecated `nonlocalpp` parameter is a warning because it no longer affects
execution. It reports missing or invalid HDF5 sidecars, verifies the supported
HDF5 superblock signature, and flags a sidecar older than its XML input for
review. A missing or invalid `override_variational_parameters` `vp.h5` is an
error: that file holds the optimized parameters. The signature check does not
decode HDF5 datasets or prove that a sidecar has the right wavefunction data. A
`determinantset` that uses `twistnum`
without an explicit `twist` is flagged as ambiguous. Legacy inline
`slaterdeterminant` setups are flagged for migration to `sposet_collection`.
When an authoritative variational-parameter sidecar appears with inline
`coefficients`, the linter warns that the XML values may be stale display
values. The sidecar remains the source to carry forward after optimization.

Direct XML includes are resolved relative to the reviewed input and missing
files are errors. Chemtools follows includes recursively, up to 64 files and
8 MiB per included XML file, and resolves HDF5 references relative to the XML
that declares them. It reports cycles, malformed included XML, and review
limits. Each present include also receives the supported structural lint checks,
with findings labelled by include path. It does not merge XML trees or validate
full wavefunction semantics.

The configured `launch_qmcpack_run` path runs QMCPACK through the shared
execution service and records the effective command and expected artifacts.
It does not merge includes or validate full wavefunction semantics. Primary-log
inspection records only version, the exact completion
marker, final reported execution time and its line, completion evidence,
line-anchored unique warnings, and
top-level VMC, DMC, or linear-optimization sections. It does not turn those
warnings into a scientific diagnosis by default. An explicit warning that
QMCPACK replaced a non-positive control with a positive value returns an
`input_parameter_auto_corrected` review verdict with the affected QMC section
and source line. Repeated optimizer-recovery messages retain their code,
occurrence count, line range, and, when the log provides section starts, every
affected QMC section. For a log with multiple numeric QMCPACK
banners, version, completion, timing, warnings, sections, optimizer diagnostics,
project labels, and particle pools describe the trailing run; its starting line
is retained as evidence. When the matching QMCPACK input is
supplied, it compares declared `linear`, `vmc`, and `dmc` method presence with
those sections. Repeated optimizer sections count as internal iterations, not
distinct declared QMC blocks. When both files name a project, it also compares
the XML project ID with the primary log's printed `Project = ...` label. This
does not establish input controls or output provenance. Repeated identical labels
remain usable; distinct labels are retained as runtime evidence and are not
compared. Supported particle-pool
summaries retain runtime particle-set and group counts, including whether the
listed groups sum to the printed set count. Direct XML particle sets with an
explicit set count or complete group counts are compared with matching runtime
sets. Matching named groups are also compared independently, so a matching
total can still expose a different partition. Included XML remains unmerged,
so unresolved sets are not compared. Anonymous groups are not compared, while
ambiguous repeated direct XML or runtime-set names are reported as `not_checked`.
Legacy offset summaries retain anonymous group sizes rather
than invented labels. Missing runtime groups and runtime group totals that
disagree with the printed particle-set total also report each affected direct
XML group as `not_checked`.
A direct XML particle set without an unambiguous matching runtime set is
reported as `not_checked`.
Explicit `minwalkers` threshold warnings also receive a separate occurrence count
and threshold list. When an
effective-weight record immediately precedes one, Chemtools reports the
smallest such observed value. Explicit invalid-cost and parameter-reversion
records produce an optimizer-review verdict even when the process reports
completion. They do not mark task execution failed or establish a scientific
result.
When several correction or optimizer-recovery conditions occur together, the
verdict label follows the highest-priority condition and the reasons retain all
of them.
`Failed Step. Largest LM parameter change: ...` and `Good Step. Largest LM
parameter change: ...` records retain separate counts and largest reported
changes. They describe line-minimizer trial history, not optimization or
scientific convergence, and do not replace the run-completion assessment.
`ERROR CostFunction-> Number of Effective Walkers is too small` is condensed
into an occurrence count and the smallest printed value. That message and
`Reverting` or legacy `Revertting to old Parameters` are recovery evidence,
not convergence or population-control thresholds.
`completion_evidence=total_execution_time_only` records a legacy timing footer
without treating it as a successful run.
Legacy `VMCSingleOMP` and `DMCOMP` driver starts normalize to VMC and DMC;
their following `QMC Execution time` records remain attached to the matching
section. A legacy optimizer keeps its enclosing
`QMCFixedSampleLinearOptimize` section and receives the same timing evidence
without adding a duplicate nested VMC task.

`inspect_qmcpack_scalar` reads one QMCPACK `.scalar.dat` file. It preserves the
observed columns, invalid-row count, and block-index continuity evidence with
compact summaries for each estimator. Non-integer block-index rows are
excluded. The `LocalEnergy` weighted mean is available only when every
`BlockWeight` is positive. When `LocalEnergy_sq` is also present, the response
checks its per-block second-moment bound against `LocalEnergy` with a
scale-aware numerical tolerance. It does not estimate uncertainty or
convergence. When `AcceptRatio` is present, it also checks that the recorded
values remain within `[0, 1]` without recommending an acceptance-rate target.
It records whether `BlockWeight` values are positive, while leaving unweighted
analyses unchanged.
When a filename follows `project.sNNN.scalar.dat`, it also reports the project
label and series index as filename identity. That does not establish the source
QMC input block or its controls.
When such a scalar file is supplied as a related `inspect_run` artifact, its
filename project label is compared with the primary log label. This also does
not establish source-run or QMC-block lineage. If the primary log lacks an
unambiguous project label, or the scalar filename lacks a recognized project
label, the comparison is explicitly `not_checked`.
When `Kinetic` and `LocalPotential` are present, it also checks their reported
sum against `LocalEnergy` with a print-precision-aware tolerance. This does not
establish Hamiltonian completeness.

`inspect_qmcpack_determinant_vmc_offsets` compares determinant-only VMC scalar
means with matching trial SCF energies for at least two caller-labelled states.
It reports the offsets, whether they are all positive, and their strict trend
in the supplied state order, plus cause-specific scalar-input quality warnings.
It does not define a small-offset threshold or establish Hamiltonian
consistency.

`inspect_qmcpack_pseudopotential` reads a semilocal QMCPACK pseudopotential
XML card. It reports the header, grid, local channel, data-point count per
channel, and the final `r*V` values against `−zval`. It also checks the
declared `hartree` and `r*V` encoding, linear channel grids, grid-count
agreement, presence of the declared local channel, recognized angular labels,
and unique angular-momentum/spin channel pairs. Those are structural checks
only, not transferability or DMC-compatibility evidence.

`inspect_qmcpack_referenced_pseudopotentials` follows bounded XML includes to
inspect each pseudopotential card declared by a QMCPACK deck. It compares every
declared `elementType` with the card header symbol in addition to the supported
semilocal structural checks. It does not establish pseudopotential-family
equivalence or transferability.

`analyze_qmcpack_dmc_series` accepts DMC scalar files labelled with their
recorded input controls, discards a leading block fraction, reblocks the
remaining `LocalEnergy` values, and fits T-move and no-T-move points separately
to zero time step. The explicit labels are required because a scalar series
number tracks a sequential QMC section, not its time step or nonlocal-move
setting. Each supplied run must name a distinct scalar file. The result is a
reblocked statistical fit, not a convergence decision
or an autocorrelation estimate. A caller-supplied `potential_label` for every
point produces uniform or mixed identity evidence; omitted labels remain
`not_assessed`. Confirmed mixed identity preserves the points but withholds the
combined time-step fit. The result reports excluded scalar rows by malformed,
non-finite, and non-integral-index cause, warns about block-index gaps or
restarts, marks a bounded reader as incomplete, and retains an inconsistent
`LocalEnergy_sq` second-moment bound or out-of-bounds `AcceptRatio` as a
scalar-quality warning. It also reports non-positive `BlockWeight` values.
An unbalanced reported `LocalEnergy`, `Kinetic`, and `LocalPotential` record is
also retained as a warning.

`analyze_qmcpack_dmc_input_series` obtains the time step, `nonlocalmoves`, and
walker target from direct DMC blocks in the supplied primary QMCPACK XML. The
caller still binds each scalar file to a zero-based QMC-block index because
scalar files do not contain that provenance. When both paths provide a project
label, it compares the scalar filename label with the input project ID without
claiming that the selected block produced the file. A mismatch also produces a
top-level binding warning. Included XML is not merged. The
tool rejects a non-DMC selection or a DMC block without explicit time-step and
`nonlocalmoves` controls, then delegates the statistical work to the ordinary
DMC series analysis.

`inspect_qmcpack_dmc_population` reads a DMC population record and summarizes
the retained walker count, living fraction, and diffusion efficiency. An
optional input-derived target walker count adds observed mean and final
deviations. It preserves source block-index continuity and reports excluded
rows by malformed, non-finite, and non-integral-index cause, alongside
discontinuous-index and bounded-read warnings. The response contains those
measurements without declaring a population-control threshold. If no valid row
remains, the failure uses the same cause detail.

`inspect_qmcpack_dmc_population_from_input` gets the target walker count from
a selected direct DMC block in the primary QMCPACK XML. Included XML is not
merged. The caller supplies the population file's QMC-block index because the
file does not contain its source block identity.

`check_qmcpack_vmc_energy_gate` compares a post-optimization VMC scalar-file
mean with a matching trial SCF reference in Hartree. A VMC value above the trial
energy fails the gate. The result does not estimate autocorrelation or establish
statistical convergence, but it preserves scalar-input quality warnings.

`compare_qmcpack_tmove_locality_shift` compares a T-move run with its matched
no-T-move control at one time step. It reports no-T-move minus T-move in
Hartree, propagated reblocked uncertainty, and whether supplied target walker
counts match. It rejects different time steps. When both runs supply a
potential label, the labels must match; unlabeled pairs report that potential
identity was not assessed. The comparison also warns about excluded malformed
rows or truncated scalar input. Its two controls must name distinct scalar
files.

`compare_qmcpack_tmove_locality_shift_from_input` gets those controls from
selected direct DMC blocks in the primary QMCPACK XML. Included XML is not
merged. The selected T-move block must enable `nonlocalmoves`, and the selected
control must disable it. Scalar-file-to-block bindings remain caller-supplied
evidence because scalar files do not contain their source block identity.

Input parsing adds a DMC campaign summary whenever an XML file contains DMC
blocks. It records time steps, requested blocks, `nonlocalmoves`, and declared
walker targets. Both `targetWalkers` and the production-deck spelling
`total_walkers` populate that count. Explicit T-move ladders report whether
smaller time steps have more requested blocks, and no-T-move blocks report
whether they match a T-move time step. This is input evidence, not a calculation
verdict. The ladder also records whether its time steps match the four-point
f-block reference, with its optional fifth fine point. With at least three
distinct T-move time steps, it also records whether
all no-T-move controls match an interior ladder point; shorter ladders remain
`not_assessed` for that check. If linear optimization, VMC, and DMC blocks all
occur, it also records whether their order matches the f-block production
reference. It also reports whether the control count matches that reference's
one-repeat protocol. Partial and continuation decks remain `not_assessed` for
the production-order comparison. For a complete production sequence, it also
checks that each linear block is enclosed by a `loop` with a positive `max`
within the f-block reference range of 6 through 8. It retains named `cost`
entries and compares `MinMethod`, energy cost, and unreweighted-variance cost
with the f-block reference recipe.

At a shared T-move time step, the no-T-move control comparison reports agreement
or missing evidence for declared blocks, steps, warmup, walker target, move,
and checkpoint settings. It does not infer values omitted from the XML.

The supported reference shape comes from the f-block QMCPACK examples in
`notes/fblock/examples/qmcpack/` and the workflow notes in
`notes/fblock/qe-qmcpack-oncvpsp.md`.

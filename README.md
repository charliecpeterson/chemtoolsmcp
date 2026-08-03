# chemtools-mcp

AI agent toolkit for computational chemistry. Provides an MCP (Model Context
Protocol) server that gives Claude (and other MCP clients) structured access to
quantum chemistry programs — parsing outputs, drafting inputs, managing jobs,
analyzing active spaces, recovering from failed runs.

The live MCP registry currently covers six chemistry programs. Support is
uneven by design: each program exposes the parsing, drafting, diagnosis, and
execution operations it actually implements.

| Program | Tools | Highlights |
|---|---|---|
| **NWChem** | 101 | Input drafting, TCE/MCSCF parsers, frequency restart, full HPC submission, runner-profile auto-resource sizing, 29 bundled docs |
| **OpenMolcas** | 45 | CASSCF/CASPT2 chain orchestrators, active-space refinement loop, recovery rule engine (11 failure modes), 133 bundled docs |
| **DIRAC** | 39 | 4c/X2C atomic + molecular SCF, AOC + KPSELE for actinides, Cm-class workflow, basis browser (Dyall), 179 bundled docs |
| **GRASP2018** | 51 | Multi-exe DHF workflow (rnucleus → rmcdhf → jj2lsj → rlevels), exact f-block reference planning, bounded radial-wavefunction inspection, leading mixing components mapped to matching CSFs, first-donor-wins orbital merging, hf-bootstrap for high-Z, non-rel limit, 15 bundled docs |
| **Quantum ESPRESSO** | 20 | `pw.x` SCF, relax, and vc-relax input review plus output diagnosis, local or scheduler launch rendering and execution through a named profile, single-q phonon and converter-input drafters, a declared QE-to-QMCPACK artifact handoff plan, conversion-readiness, artifact-lineage, deck-reference, semilocal-card, pseudopotential and ion species, valence, DMC projector evidence, electron-count, atom-count, periodic-geometry, fixed-moment spin, charge-accounting, aggregate conversion, and completed-converter chain checks |
| **QMCPACK** | 14 | XML input review, runner-profile launch preview and execution, semilocal pseudopotential inspection, referenced-pseudopotential inspection, fixed-layout HDF5 metadata inspection, primary-log completion and warning inspection, scalar summaries, determinant-only VMC offset inspection, DMC population inspection, input-bound DMC population inspection, time-step analysis, input-bound time-step analysis, a VMC energy gate, a T-move control comparison, and an input-bound T-move control comparison |

Plus 56 program-generic tools (auto-detect supported inputs and outputs)
and a multi-program eval framework with 15 reference cases.

**Total: 326 MCP tools.** Counts, capability tags, mode visibility, aliases,
and input schemas come from the generated
[MCP tool inventory](docs/tool-inventory.md).

---

## Quick start

```bash
git clone https://github.com/charliecpeterson/chemtoolsmcp.git
cd chemtoolsmcp
pip install -e .
```

Verify install:

```bash
chemtools --show-mode      # prints active mode + program filter + blocked tools
chemtools --list-tools     # prints the tool names visible in this mode
```

By default the server runs in **analysis mode** (no NWChem/Molcas executable
needed) — you can parse outputs, draft inputs, look up docs, and plan
calculations without anything else installed. To launch real jobs see
[Runner profiles](#runner-profiles).

---

## MCP client setup

### Claude Desktop / Claude Code

Add to your MCP servers config:

```json
{
  "mcpServers": {
    "chemtools": {
      "command": "chemtools"
    }
  }
}
```

The minimum config exposes all 272 analysis-mode tools. The sections below
show how to scope the tool list, switch modes, and wire up a runner profile
for job submission.

### Codex development

For development, use an editable install in a project-local virtual
environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

The [testing baseline](docs/testing-baseline.md) records the current
collection, external-test boundaries, and the recovery queue for useful tests
removed from earlier revisions.

Then create `.codex/config.toml` with the absolute path to this checkout:

```toml
[mcp_servers.chemtools_dev]
command = "/absolute/path/to/chemtoolsmcp/.venv/bin/chemtools"
cwd = "/absolute/path/to/chemtoolsmcp"
startup_timeout_sec = 20
tool_timeout_sec = 60

[mcp_servers.chemtools_dev.env]
CHEMTOOLS_MODE = "analysis"
CHEMTOOLS_PROGRAMS = "nwchem"
```

Start a new Codex session in the repository, then use `/mcp` or `codex mcp
list` to confirm that `chemtools_dev` is connected. The editable install
means Python source changes are picked up when Codex starts the next MCP
server process; restart the session after changing server code.

The development configuration intentionally starts in `analysis` mode and
limits the initial tool surface to NWChem plus generic tools. Remove the
program filter when working on another backend. Add
`CHEMTOOLS_RUNNER_PROFILES=/path/to/runner_profiles.yaml` only when you are
ready to expose local or scheduler-backed launch tools. See the [Codex MCP
configuration guide](https://developers.openai.com/codex/mcp) for available
server settings.

For the three-tool guided surface, add this environment value:

```toml
CHEMTOOLS_TOOLSET = "guided"
```

That preset exposes `review_input`, `inspect_run`, and
`search_knowledge_cards`. Leave it unset when developing or directly testing
the lower-level tools.

Representative MCP requests live in [`tests/golden/mcp`](tests/golden/mcp).
They cover NWChem, Molcas, DIRAC, GRASP, Quantum ESPRESSO, and generic
auto-detection through the real `tools/call` dispatch path:

```bash
.venv/bin/python -m pytest -q tests/test_mcp_golden.py
```

### Quantum ESPRESSO input and output review

The generic `review_input` tool recognizes `pw.x` inputs and reviews the
current SCF, relax, and vc-relax scope. It identifies `bands.x`, `dos.x`,
`pp.x`, and `projwfc.x` inputs by their top-level namelist and returns one
explicit unsupported-program finding instead of applying the `pw.x` card
checks. When `pseudo_dir` is explicit, the QE backend reads each referenced
UPF `PP_HEADER`, checks that the file matches the atomic species, preserves
declared local-channel and total-projector metadata, and reports projector
counts by angular channel when bounded UPF evidence is complete. It then
compares `ecutwfc` and the effective `ecutrho` with the largest positive
suggestions in the referenced set. Relative `pseudo_dir` values are resolved
from the input directory for review and labelled as that assumption in the
response.

`inspect_run` handles PWSCF output. It recognizes `bands.x`, `dos.x`, `pp.x`,
and `projwfc.x` banners and returns `unsupported_output_format` before the
PWSCF parser can misclassify them. An MPI abort fragment without a QE program
banner remains an unassigned output rather than a guessed QE run.

`check_qe_qmcpack_conversion_ready` reviews a QE SCF input before invoking
`pw2qmcpack`. It checks `disk_io`, the explicit crystal gamma-point form, and
the Martyna-Tuckerman isolation setting. It does not run QE or the converter,
inspect written orbitals, or establish an energy comparison.

`plan_qe_qmcpack_conversion` records the declared handoff from one QE SCF
input to a caller-supplied `.pwscf.h5` path and then to QMCPACK deck
validation. It keeps the converter command line unset because execution and
converter-option selection remain outside the supported scope.

`draft_pw2qmcpack_input` renders the demonstrated `&inputpp` namelist from
explicit QE `prefix` and `outdir` values, with `write_psir = .false.`. It
returns review instead of guessing missing QE defaults or converter options.
`review_input` recognizes the bounded converter shape when it contains
`write_psir` and no options outside the demonstrated form. It warns when
`prefix` or `outdir` is missing. Other `&INPUTPP` decks remain unsupported
`pp.x` inputs, since Chemtools does not yet validate their options.
`inspect_run` recognizes the corresponding `pw2qmcpack.x` banner and records
each reported `esh5 create` HDF5 path. It reports a successful conversion only
when that artifact evidence accompanies `JOB DONE.`. Older logs without the
terminal marker retain their artifact evidence with unresolved completion.
Two known converter failures receive explicit diagnoses: wavefunctions not in
collected format and the unsupported QE gamma-only reduced G-space form. The
output identifies the failure but does not establish which preceding QE input
setting needs correction.
When both converter input and output are supplied, it compares the reported
path with the bounded `outdir/prefix.pwscf.h5` handoff. That checks lineage,
not the HDF5 contents or converter completion.
When the `.pwscf.h5` sidecar is also supplied, it compares that path with the
converter report. Relative reported paths are resolved from the converter
output directory, an explicit inspection assumption. A matching sidecar must
also be at least as new as the converter input. This timestamp check does not
validate HDF5 contents or converter completion, since the converter may append
its log after creating the sidecar.
An explicitly supplied `.pwscf.h5` file is classified as a QE binary
checkpoint and wavefunction artifact. Generic inspection records its metadata
without reading its contents.

`draft_ph_x_input` renders a single-q `ph.x` deck from the same explicit QE
paths, a caller-supplied job title, and a caller-supplied q-vector. It leaves
grid, dielectric, Raman, and electron-phonon settings unset. Gamma-point
drafts include an advisory to decide whether the non-analytic LO-TO term is
needed before setting `epsil`.

`review_input` also recognizes this bounded `ph.x` form. It checks the title,
the `&INPUTPH` closure, explicit QE paths, one finite q-vector, and the
documented `epsil` restriction at finite q. Grid and q-list phonon inputs stay
at limited review rather than being treated as malformed single-q decks.

`inspect_qe_qmcpack_conversion` runs the full supported conversion review in
one request. It combines artifact lineage, bounded XML and pseudopotential
evidence, pseudopotential and ion species, valence, particle counts, periodic
geometry, fixed collinear spin, and charge accounting. With the optional
science runtime configured, it also compares recognized `pw2qmcpack` HDF5
atom, species, electron, and spin metadata against the QMCPACK deck. The
individual conversion tools remain available when a failed check needs closer
inspection.

`inspect_qe_qmcpack_conversion_execution` adds explicit `pw2qmcpack.x` input
and output evidence to that review. It requires the converter to report
completion, requires its explicit `prefix` and `outdir` to match the QE input,
and compares its declared HDF5 path with both the converter input and supplied
sidecar. Converter options outside Chemtools' documented `&INPUTPP` form,
including `write_psir=.true.`, require review. It uses only the aggregate
report's fixed-layout metadata cross-check and does not decode coefficients or
arbitrary HDF5 datasets, or establish a valid energy comparison.

The conversion checks share the QE input, QE output, converter input, and HDF5
artifact evidence. XML references, pseudopotential cards, particle counts,
geometry, spin, and charge provide separate cross-checks on that handoff, but
they do not independently establish that the underlying electronic-structure
calculation is valid.

`inspect_qe_qmcpack_conversion_artifacts` extends that preflight to explicitly
supplied QE input, QE output, and `.pwscf.h5` paths. It requires a clean,
converged SCF output and an HDF5-signature-bearing artifact at least as new as
the input and output. It accepts signatures after a legal HDF5 user block. It
does not decode HDF5, run `pw2qmcpack`, inspect a QMCPACK deck, or compare
energies.

`inspect_qe_qmcpack_conversion_deck` adds a QMCPACK XML path and follows the
existing bounded XML include graph to confirm that the deck references the
exact declared `.pwscf.h5` artifact through a QMCPACK orbital reference. It
does not treat a variational-parameter sidecar as an orbital reference, merge
XML, decode HDF5, compare particle or pseudopotential semantics, or compare
energies.

`inspect_qe_qmcpack_conversion_pseudopotentials` also inspects every declared
QMCPACK pseudopotential file in that bounded graph. It requires the supported
semilocal XML structure, including the expected units, `r*V` format, linear
channel grids, and declared local channel. It does not prove equivalence to
the QE UPF or establish transferability.

`inspect_qe_qmcpack_conversion_species` compares QE atomic-species elements
with QMCPACK pseudopotential `elementType` declarations, including bounded XML
includes. It detects a missing or extra declared element, but does not prove
pseudopotential family or valence equivalence.

`inspect_qe_qmcpack_conversion_valence` compares parsed QE UPF `z_valence`
with the QMCPACK XML pseudopotential header `zval` for each unambiguous
element. It detects header-valence disagreement, but does not establish
pseudopotential family or scattering equivalence.

`inspect_qe_qmcpack_conversion_projectors` reviews bounded `PP_BETA`
projector counts for nonlocal `NC` UPFs when the primary or bounded included
QMCPACK XML declares DMC. Native `SL` UPFs do not require KB-projector
evidence. A
multi-projector angular channel returns review, prompting confirmation that the
QMCPACK card came from a separately generated semilocal potential. A source
UPF type other than `NC` or `SL` also returns review. It does not prove
pseudopotential family equivalence, card provenance, or DMC compatibility.

`inspect_qe_qmcpack_conversion_electrons` compares the QE runtime electron
count, and complete UPF-based valence accounting when available, with the
particle-group sizes in the QMCPACK Hamiltonian target. It returns review when
either side lacks enough evidence. It does not determine a physical charge or
spin state.

`inspect_qe_qmcpack_conversion_atoms` compares QE's declared `nat` with the
sizes of QMCPACK particle sets outside the Hamiltonian target. It returns
review when those sizes are not explicit. It does not compare elements, cells,
or coordinates.

`inspect_qe_qmcpack_conversion_ion_species` compares QE atomic element counts
with explicitly sized QMCPACK ion groups. It works without ion coordinates and
returns review for missing sizes or unrecognized labels. It does not compare
pseudopotential identity, charge, or spin.

`inspect_qe_qmcpack_conversion_geometry` compares a QE periodic cell and atom
positions with the QMCPACK `simulationcell` lattice and ion-particle
positions. It accepts the supported explicit form: one Hamiltonian target, one
non-electron ion geometry, a `bohr` lattice, and `p p p` boundary conditions.
Atomic positions are compared modulo periodic lattice translations. It does not
compare pseudopotentials, spin, or energy conventions. A non-finite or
zero-volume cell returns `not_ready` with the observed cell volumes.

`inspect_qe_qmcpack_conversion_spin` compares a QE `nspin=2` input with an
explicit `tot_magnetization` against QMCPACK `u` and `d` electron-group sizes.
It returns review when that fixed collinear evidence is absent. It does not
compare spin densities, establish a physical spin state, or support
noncollinear or spin-orbit calculations.

`inspect_qe_qmcpack_conversion_charge` compares QE UPF valence accounting and
`tot_charge` with QMCPACK ion-group `valence` parameters and the selected
electron-particle count. Missing valence or particle-size evidence returns
review. It does not establish pseudopotential-family equivalence or a physical
charge state.

### QMCPACK XML input review

`inspect_qmcpack_pseudopotential` reads a QMCPACK semilocal pseudopotential
card. It reports header metadata, the local channel, channel data counts, and
the final `r*V` values against `−zval`. It also checks the declared `hartree`
and `r*V` encoding, linear channel grids, grid-count agreement, presence of
the declared local channel, recognized angular labels, and unique
angular-momentum/spin channel pairs. This is structural evidence, not a
transferability or DMC-compatibility verdict.

`inspect_qmcpack_referenced_pseudopotentials` follows bounded XML includes to
inspect every declared pseudopotential card in a QMCPACK deck. It checks each
card's supported semilocal structure and compares declared `elementType` values
with the card header symbol. It does not establish pseudopotential-family
equivalence or transferability.

The same `review_input` tool recognizes XML files rooted at `simulation` or
`qmcsystem`. It reports project metadata, includes, HDF5 sidecar references,
particle sets, pseudopotential references, Hamiltonians, and QMC blocks. The
current linter checks XML well-formedness and the required reference and QMC
method attributes. Explicit `blocks`, `steps`, `targetWalkers`, and
`total_walkers` values must be positive integers. Both supported warmup
spellings, modern `warmupSteps` and legacy `warmupsteps`, may be zero and must
agree if both are present. A positive finite `timestep` is required. Unresolved
numeric template tokens and unrecognized `nonlocalmoves` values are warnings
because Chemtools cannot determine the caller’s final control values. The
deprecated `nonlocalpp` parameter is also a warning
because it does not affect execution. It resolves referenced
HDF5 sidecars relative to the input file, verifies their supported HDF5
superblock signature, reports missing or invalid files, and flags sidecars
older than the XML for review. A `determinantset` that uses `twistnum`
without an explicit `twist` is also flagged as ambiguous. Legacy inline
`slaterdeterminant` setups are flagged for migration to `sposet_collection`.
A missing `vp.h5` variational-parameter override is an error because that
sidecar is authoritative. Direct XML includes receive the same path check.
If a file has both that override and inline `coefficients`, Chemtools warns
that the XML values may be stale display values and that the sidecar is the
parameter source to carry forward after optimization.
The review follows nested includes within bounded limits and resolves each
HDF5 reference from its declaring XML file. Each present include also receives
the supported structural lint checks, with findings labelled by include path.
It does not decode HDF5 datasets, merge XML trees, validate full wavefunction
semantics, or launch QMCPACK.

`inspect_qmcpack_scalar` reads one `.scalar.dat` block file and returns its
observed columns, valid-row count, excluded-row causes, and estimator summaries.
When its filename follows `project.sNNN.scalar.dat`, it also records that
project label and series index as filename identity. This does not establish
the source QMC input block or its controls.
When such a scalar file is supplied as a related `inspect_run` artifact, its
filename project label is compared with the primary log label. This also does
not establish source-run or QMC-block lineage. If the primary log lacks an
unambiguous project label, or the scalar filename lacks a recognized project
label, the comparison is explicitly `not_checked`.
For `LocalEnergy`, it adds a BlockWeight-weighted mean only when every weight is
positive. Rows with non-integer block indices are excluded, and the response
reports gaps or nonincreasing index transitions without assuming they are safe
to combine. When both `LocalEnergy` and `LocalEnergy_sq` are present, it also
checks the recorded per-block second-moment bound with a scale-aware numerical
tolerance. When `AcceptRatio` is present, it checks the recorded values remain
within `[0, 1]`. Neither check estimates uncertainty, convergence, or an ideal
acceptance rate.
It also records whether `BlockWeight` values are positive, so an unavailable
weighted mean has an explicit cause; unweighted analyses remain unchanged.
When `Kinetic` and `LocalPotential` are also present, it checks their reported
sum against `LocalEnergy` with a print-precision-aware tolerance. That does not
establish Hamiltonian completeness.

`analyze_qmcpack_dmc_series` discards a specified leading block fraction,
reblocks the recorded `LocalEnergy` values, and fits each T-move mode separately
to zero time step. The caller must provide each time step, `nonlocalmoves` value,
and, when available, `targetWalkers` value from the matching input block. Scalar
series suffixes identify sequential QMC sections, not their input controls. The
fit requires each input to name a distinct scalar file and does not establish
convergence or validate autocorrelation beyond the chosen reblocking. Supplying one `potential_label`
for every point yields uniform or mixed potential-identity evidence; omitted
labels remain `not_assessed`. A confirmed mixed identity returns the individual
points but withholds the combined time-step fit. The result reports excluded
scalar rows by malformed, non-finite, and non-integral-index cause, warns about
block-index gaps or restarts, marks a bounded reader as incomplete, and retains
an inconsistent `LocalEnergy_sq` second-moment bound or out-of-bounds
`AcceptRatio`, and records non-positive `BlockWeight` values as a
scalar-quality warning. It also retains an unbalanced reported energy-component
record as a warning.

`analyze_qmcpack_dmc_input_series` gets those controls from selected direct DMC
blocks in the primary QMCPACK XML. The caller supplies each scalar file and its
zero-based QMC-block index because scalar data does not record the source block.
When both paths provide a project label, it also compares the scalar filename
label with the input project ID and reports a match or mismatch. That comparison
does not establish the source block or its controls. A mismatch also produces a
top-level binding warning.
It rejects a selected non-DMC block or a DMC block without explicit time-step
and `nonlocalmoves` settings. Included XML is not merged; the association
remains caller-supplied evidence.

`inspect_qmcpack_dmc_population` reads one `.dmc.dat` record and summarizes
the retained `NumOfWalkers`, `LivingFraction`, and `DiffEff` values. Supplying
the matching input block's `targetWalkers` value adds observed mean and final
population deviations. It reports the measurements without imposing a
population-control threshold. It also preserves source block-index continuity
and reports excluded rows by malformed, non-finite, and non-integral-index
cause, alongside discontinuous-index and bounded-read warnings. If no valid
row remains, it refuses the report with the same cause detail.

`inspect_qmcpack_dmc_population_from_input` gets the walker target from a
selected direct DMC block in the primary XML input. Included XML is not merged.
The DMC-file-to-QMC-block link remains caller-supplied evidence because the
population file does not record its source block.

`check_qmcpack_vmc_energy_gate` compares a post-optimization VMC scalar-file
mean with the matching trial SCF energy in Hartree. It passes only when the VMC
energy is at or below the trial energy. The response deliberately leaves
autocorrelation and statistical convergence unresolved, while preserving
malformed, non-finite, and non-integral-index row warnings alongside
index-continuity, bounded-read warnings, and an inconsistent `LocalEnergy_sq`
second-moment bound, out-of-bounds `AcceptRatio`, or non-positive `BlockWeight`.
An unbalanced reported `LocalEnergy`, `Kinetic`, and `LocalPotential` record is
also retained as a warning.

`inspect_qmcpack_determinant_vmc_offsets` compares determinant-only VMC scalar
means with their matching trial SCF energies across at least two caller-labelled
states. It reports whether all offsets are positive and their strict trend in
the supplied state order, plus cause-specific scalar-input quality warnings. It
does not set a small-offset threshold or establish Hamiltonian consistency.

`compare_qmcpack_tmove_locality_shift` compares matched T-move and no-T-move
DMC scalar files at one time step. It returns the signed difference as
no-T-move minus T-move in Hartree, a propagated reblocked uncertainty, and
target-walker comparability. Different time steps are rejected. When both
runs supply a potential label, the labels must match; unlabeled pairs report
that potential identity was not assessed. The two controls must name distinct
scalar files.

`compare_qmcpack_tmove_locality_shift_from_input` gets those controls from
selected direct T-move and no-T-move DMC blocks in the primary XML input. It
rejects an incorrectly selected `nonlocalmoves` setting. Included XML is not
merged. Scalar-file-to-QMC-block bindings remain caller-supplied evidence
because scalar files do not record their source block.

QMCPACK input review also extracts the declared DMC campaign. For explicitly
labelled T-move blocks, it reports whether time steps decrease while requested
block counts increase, lists declared walker targets from either `targetWalkers`
or `total_walkers`, and identifies any
no-T-move block that shares a T-move time step. These fields describe the input
plan; they do not establish a completed calculation or statistical convergence.
The ladder also records whether its time steps match the four-point f-block
reference, with its optional fifth fine point.
With at least three distinct T-move time steps, the review also records whether
all no-T-move controls match an interior ladder point; shorter ladders remain
`not_assessed` for that check. It also reports whether the control count matches
the one-repeat f-block reference protocol. At a shared time step, it compares
declared blocks, steps, warmup, walker target, move, and checkpoint settings.
When linear optimization, VMC, and DMC blocks are all present, the review also
records whether they follow the f-block production order. Partial and
continuation decks remain `not_assessed` for that comparison. It separately
checks that each linear block is enclosed by a `loop` whose positive `max` is
within the reference range of 6 through 8, and compares `MinMethod`, energy
cost, and unreweighted-variance cost with the f-block reference recipe.

The generic `inspect_run` tool recognizes QMCPACK primary logs. It records the
QMCPACK version, exact success marker, final reported execution time, and
line-anchored unique warnings. It also indexes top-level VMC, DMC, and linear
optimization sections when their start and execution-time records are present.
For a log with multiple numeric QMCPACK banners, version, completion, timing,
warnings, sections, optimizer diagnostics, project labels, and particle pools
describe the trailing run; its starting line is retained as evidence.
When the matching QMCPACK input is supplied, it compares declared `linear`,
`vmc`, and `dmc` method presence with those log sections. Repeated optimizer
sections count as internal iterations, not distinct declared QMC blocks.
When both files name a project, it also compares the XML project ID with the
primary log's printed `Project = ...` label. This check does not establish
input controls or output provenance. Repeated identical labels remain usable;
distinct labels are retained as runtime evidence and are not compared.
Primary logs using the supported particle-pool summary also retain runtime
particle-set and group counts, and report whether the listed groups sum to the
printed set count. Legacy offset summaries retain anonymous group sizes rather
than invented labels. When direct XML declares a particle-set count or all of
its group counts, Chemtools compares it with the matching named runtime set.
It also compares matching named group counts, so a matching total can still
expose a different partition. Included XML remains unmerged. Anonymous groups
are not compared, while ambiguous repeated direct XML or runtime-set names are
reported as `not_checked`. Missing runtime groups and runtime group totals that
disagree with the printed particle-set total also report each affected direct
XML group as `not_checked`.
A direct XML particle set without an unambiguous matching runtime set is
reported as `not_checked`.
Explicit `minwalkers` threshold warnings receive a separate occurrence count
and threshold list. When an effective-weight record immediately precedes one,
Chemtools also reports the smallest such observed value. This records QMCPACK's
warning, not a population-control or convergence verdict.
When QMCPACK reports that it replaced a non-positive input parameter with a
positive value, Chemtools records the requested and replacement values with
their QMC section and line locations and returns an
`input_parameter_auto_corrected` review verdict. The process may have
completed, but it did not run with the requested control values.
Explicit `Cost Function is Invalid` and `Reverting to old Parameters` records
produce an optimizer-review verdict even when the process-completion marker is
present. They do not mark task execution failed or establish a scientific
result. The `qmcpack:optimization_messages` evidence keeps each message's code,
occurrence count, line range, and every affected QMC section when the log
provides section starts.
`Failed Step. Largest LM parameter change: ...` and `Good Step. Largest LM
parameter change: ...` records retain separate counts and largest reported
changes. They describe line-minimizer trial history, not optimization or
scientific convergence, and do not replace the run-completion assessment.
Legacy `Revertting to old Parameters` and repeated `CostFunction` effective-
walker recovery messages receive the same treatment. The latter retains an
occurrence count and smallest printed value, without applying a threshold.
A missing success marker is incomplete, even if the log contains timings or
partial output. `completion_evidence=total_execution_time_only` preserves a
legacy timing footer and its line without upgrading that result to success. The
result is a completion summary, not a scientific
interpretation of the calculation.

The same review checks the internal charge and spin setup. It rejects
conflicting `nspin` and `noncolin` controls, `lspinorb` outside the
noncollinear path, simultaneous `tot_magnetization` and
`starting_magnetization`, and indexed starting values outside `1..ntyp`. A
spin-polarized input with no magnetic seed is reported, while a nonmagnetic
spin-orbit setup is retained as informational evidence because its zero seed
preserves time-reversal symmetry. When every referenced UPF supplies
`z_valence`, the response also reports the valence-electron total before and
after `tot_charge`.

Input review normalizes explicit `ibrav=0` cells in alat, angstrom, or bohr and every
nonzero Bravais form documented for QE 7.5. It accepts either `celldm` or
conventional `A/B/C` and cosine parameters. Positions may use alat, angstrom,
bohr, or crystal coordinates. Coordinate fields accept QE's restricted
arithmetic expressions, including `1/3` and powers, without evaluating names
or function calls. Unitless `CELL_PARAMETERS` input is normalized using QE's
deprecated default and is labelled `implicit_alat` or `implicit_bohr`; unitless
`ATOMIC_POSITIONS` input is labelled `implicit_alat`. `crystal_sg` position
records require a positive `space_group` number and remain structurally
`not_assessed` until Chemtools can expand them. It applies the same periodic distance and molecule-versus-
solid boundary used for output trajectories. Contacts below 0.6 angstrom are
errors; main-group overcoordination is a warning. The current 64-input corpus
scan finds these issues only in the five benzene inputs, each of which contains
18 close contacts and 12 overcoordinated atoms. The 34 Bravais inputs receive
periodic metrics without molecular findings. The remaining 25 inputs lack a
`pw.x` geometry and remain `not_assessed`.

K-point evidence records automatic meshes, shifts, requested full-grid counts,
explicit points, band paths, and the active symmetry flags. For an automatic
mesh, Chemtools proposes the current grid plus two parity-preserving
refinements while keeping the shift fixed. Axes sampled with one point remain
at one and are called out as a dimensionality assumption. The series is a
starting point: the response requires user-selected energy, force, or stress
tolerances and leaves both the irreducible k-point count and
`convergence_established` unresolved until outputs are available. The linter
also rejects tetrahedron occupations without an automatic grid and checks the
shape and calculation type of explicit band paths. The syntax and symmetry
behavior follow the QE 7.5
[`INPUT_PW`](https://www.quantum-espresso.org/Doc/INPUT_PW.html) reference.

UPF suggestions are a starting point for convergence tests. A passing review
always returns `convergence_established: false`; Chemtools does not turn header
metadata into a claim that energy, forces, or stress are converged. This follows
the QE [UPF field specification](https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format)
and its [pseudopotential guidance](https://www.quantum-espresso.org/pseudopotentials/).

The generic `inspect_run` tool parses PWSCF SCF and BFGS relaxation output. It
keeps converged `! total energy` records separate from unconverged iteration
energies, converts Ry to Hartree for the cross-program task summary, and retains
the original Ry values with line anchors. Variable-cell relaxation output also
keeps final enthalpy, force, stress, BFGS step counts, and final coordinates in
their native units. The generic `inspect_geometry` tool returns Cartesian atoms
and a periodic cell in angstroms from the PWSCF runtime summary. For a
relaxation, it returns only the final geometry after BFGS convergence. It
refuses failed and incomplete relaxations rather than relabelling the last
attempted coordinates as a converged structure.

The generic `parse_trajectory` tool provides the complementary relaxation
evidence. It returns the initial structure and each distinct geometry update
with normalized atoms, the current periodic cell, source lines, and an explicit
role. The supplied FeO run has 19 frames ending at `converged_final`; the failed
benzene run has 8 frames ending at `last_attempted`. Chemtools does not pair SCF
energies with frames by list index because PWSCF can print an extra final SCF
energy for the already-relaxed structure.

The trajectory response also reports periodic minimum-image distances, cell
volume, and a bounded structural-health assessment. Molecular connectivity
checks require at least 5 angstrom of empty periodic space along every lattice
direction in every frame and complete covalent-radius coverage. Extended solids
receive metrics without a molecular topology verdict. In the supplied corpus,
the benzene input already contains 18 atom pairs closer than 0.6 angstrom and
12 overcoordinated main-group atoms. The FeO cell contracts by 53 percent, which
is reported for review without treating molecular connectivity rules as
applicable to the solid.

QE `inspect_run` responses include the compact trajectory assessment without
copying every frame into the result. Structural findings retain separate
`input_geometry` and `trajectory` origins, and mixed cases keep both. The
assessment adds those findings, or a large cell-change observation, to the run
verdict reasons and points to `parse_trajectory` for the full history.
Automatic trajectory inspection is limited to 16 MiB of output, 512 frames,
and 250,000 atom-pair evaluations. Exceeding a limit produces an explicit
`not_assessed` result.

Diagnosis requires scientific completion markers as well as `JOB DONE`. An SCF
failure during relaxation remains a failed calculation even when pw.x shuts down
cleanly. Repeated MPI errors are deduplicated, and a missing pseudopotential from
`readpp` is reported separately from a generic runtime error. Bands and NSCF
outputs receive limited completion inspection because their output markers do
not distinguish the two input modes.

Supplying the corresponding `.in` file in `artifact_files` enables QE
input-output consistency checks. Chemtools compares calculation mode, atom and
atomic-type counts, UPF-derived electron count, and the wavefunction and density
cutoffs with PWSCF's runtime summary. Gamma-only k-point counts are directly
checked. Automatic and explicit grids retain the requested and runtime counts as
separate evidence because symmetry and time reversal may reduce or expand the
set.

### Orbitron contract development

Orbitron is an optional subprocess integration. The current adapter allows
only `--version`, `info <source> --json`, and
`inspect <source> --json`, plus the geometry, orbital, population, and vibration
forms of `analyze <kind> <source> --json`. The `inspect_with_orbitron`,
`analyze_geometry_with_orbitron`, `analyze_orbitals_with_orbitron`,
`analyze_populations_with_orbitron`, `analyze_vibrations_with_orbitron`, and
`render_with_orbitron` MCP tools each accept one local path. None accepts a
command, remote target, output path, or arbitrary Orbitron arguments.

Set `CHEMTOOLS_ORBITRON_CLI` before starting the MCP server. Successful calls
return Orbitron evidence with the schema, producer version, commit, and
structured warnings kept separate from Chemtools scientific judgments. The
tool reports unavailable, incompatible, and refused outcomes without making
Orbitron a required dependency. Chemtools caps the source at 2 GiB and the
JSON response at 2 MiB. Rendered PNG data is capped at 8 MiB.

For programs already registered in Chemtools, the response also includes a
canonical external-tool producer identity and artifact classification. Output,
NWChem `.movecs` and `.hess`, and DIRAC `.h5` subjects use the existing backend
artifact declarations. Unsupported programs and companion formats remain
explicitly unresolved. Orbitron `inspect/2` supplies atom and bond counts, but
not coordinates or a complete periodic-system specification, so Chemtools
reports `insufficient_evidence` instead of constructing a scientific-system
model from partial data.

`analyze_geometry_with_orbitron` returns the versioned
`orbitron.analyze.geometry/3` evidence: atom and bond counts, element and
coordination summaries, bond-length statistics, bounds, and unit-cell vectors.
Chemtools checks that counts agree, numeric vectors are finite, bond statistics
match the reported bond count, unit-cell periodic flags are well formed, and
all distance-valued fields are labelled in angstroms. It also validates the
geometry role as `input`, `single_point`, `converged_final`, or
`last_attempted`, and requires a non-empty source description. A
`last_attempted` geometry remains available for diagnosis but carries explicit
uncertainty that it is not a converged structure.

`analyze_orbitals_with_orbitron` fixes the frontier window at three orbitals on
each side. Chemtools validates total, occupied, and virtual counts; finite
energies and occupancies; Hartree-to-eV conversions; HOMO-LUMO gap arithmetic;
frontier membership; the reported occupancy threshold; and restricted or
alpha/beta spin-channel partitions. Orbitron `orbitron.analyze.orbitals/2`
returns separate frontier summaries for unrestricted channels rather than one
cross-channel gap. Periodic band data is outside this molecular-orbital
operation and remains a tool refusal.

`analyze_populations_with_orbitron` fixes the largest-charge window at eight
atoms. Chemtools validates atom counts, finite charges, descending absolute-
charge order, derived totals and extrema, mean absolute charge, per-atom maps,
and top-charge membership. It also promotes each method warning into structured
uncertainty. Orbitron `orbitron.analyze.populations/2` reports the expected
system charge, its source, and the residual when the input establishes that
charge. Chemtools validates their nullability and arithmetic. When the source
does not establish the expected charge, Chemtools reports that limitation
without assuming neutrality. Population evidence remains external evidence
until Chemtools has a canonical atomic-population summary model.

`analyze_vibrations_with_orbitron` calls the fixed raw-mode operation with
`--top 10`. Chemtools validates counts, frequency statistics, sample ordering,
magnitudes, mode indices, `cm^-1` units, the unscaled-frequency policy,
total and per-mode displacement availability, and unit-labelled
thermochemistry. The `/4` response also identifies the geometry's role and
source. Chemtools treats a `last_attempted` geometry as structured uncertainty,
along with a missing thermochemistry pressure or imaginary-mode displacement
vector. Orbitron 0.4.0 at commit `20e81d225b4c` selects the last complete
Molcas MCLR frequency block, and the pinned HCN reference agrees at 9 modes
with one imaginary mode.

`render_with_orbitron` uses the fixed 1024 by 768 headless PNG operation. It
writes only in an ephemeral sibling directory, never accepts a destination or
camera setting, validates PNG signature and dimensions, and returns the image
as a separate MCP image-content item after JSON provenance. A render does not
change the source artifact.

Set the local CLI and external reference corpus, then run the pinned eight-case
contract:

```bash
export CHEMTOOLS_ORBITRON_CLI=/path/to/orbitron
export CHEMTOOLS_REFERENCE_CORPUS=/path/to/input_examples
.venv/bin/python scripts/check_orbitron_contract.py
```

### Companion scientific runtime

The optional companion scientific runtime keeps native scientific Python
packages outside the MCP server environment. Set
`CHEMTOOLS_SCIENCE_PYTHON` to the explicit path of its interpreter, then call
`inspect_science_runtime` to obtain fixed, read-only import and version
evidence for PySCF, RDKit, Open Babel, h5py, and Orbitron's Python API. The probe
does not install packages, run calculations, accept a caller-selected module,
or fall back to the MCP server's Python interpreter.

`inspect_qmcpack_hdf5` is a fixed companion operation for one absolute local
QMCPACK HDF5 artifact. It recognizes the QE/pw2qmcpack electronic-structure
wavefunction layout plus QMCPACK variational-parameter, walker-configuration,
and statistics sidecars. It reports only named small metadata fields such as
species, electron-spin populations, k-point count, parameter count, walker
shape, and estimator names. It does not read coefficients, density grids,
walker coordinates, estimator values, or arbitrary datasets.

`preflight_molecule_with_rdkit` accepts only SMILES or MDL mol blocks. It
returns RDKit's canonical molecular evidence, while retaining the submitted
source and reporting disconnected fragments and explicit radical electrons as
warnings. It does not rewrite a calculation input or claim complete
coordination-chemistry perception.

`convert_molecule_with_openbabel` converts only declared SMILES and MDL mol
blocks through the same companion runtime. It returns the converted text and
hashes, then independent RDKit evidence on both forms. A mismatch in canonical
SMILES, formula, connectivity, charge, aromaticity, or stereochemistry is
reported rather than silently repaired. It refuses output that RDKit cannot
inspect. SMILES-to-mol conversion never generates coordinates, and any
zero-coordinate MOL block is marked as connectivity-only evidence. See
[`docs/openbabel-conversion.md`](docs/openbabel-conversion.md).
The reviewed seven-case [Open Babel fixture corpus](tests/fixtures/openbabel/)
pins neutral, charged, aromatic, disconnected, radical, chiral, and reverse
MOL-to-SMILES behavior; run its opt-in checker with
`scripts/check_openbabel_fixture_corpus.py` after configuring the companion
interpreter.

`inspect_periodic_electronic_structure_with_orbitron` is a fixed read-only
operation through Orbitron's Python API. It returns bounded band-structure and
DOS summary evidence from one absolute local path, including source hash and
package provenance, while omitting raw curves and projections. It reports
parsed facts only and does not assess the calculation method, k-point path, or
scientific validity. See
[`docs/orbitron-periodic-python-api.md`](docs/orbitron-periodic-python-api.md).

`inspect_structure_identity_with_orbitron` is a fixed read-only companion
operation for molecular and coordination-structure files. It returns source
and environment provenance, atom and bond counts, bond-order counts including
`Dative` when assigned by Orbitron, and formula, InChI, InChIKey, and SMILES
evidence when each is available. It does not change the input or judge whether
the constructed coordination model is chemically correct. The reviewed zinc
chloride fixture pins this boundary and can be checked with
`scripts/check_orbitron_structure_identity_python_api.py`; see
[`docs/orbitron-structure-identity-python-api.md`](docs/orbitron-structure-identity-python-api.md).

`inspect_nbo_with_orbitron` returns bounded Natural Bond Orbital evidence from
one supported output: orbital-type counts, occupancy range, per-atom entry
counts, and at most twelve `BD`, `BD*`, `LP`, or `LP*` samples. It does not
return raw NBO tables or turn NBO labels into a bonding or oxidation-state
verdict. The live UO₂ regression source remains in Orbitron's own fixture
corpus and is hash-pinned by
[`scripts/check_orbitron_nbo_python_api.py`](scripts/check_orbitron_nbo_python_api.py).

`run_pyscf_single_point` is available in local execution mode for typed
molecular RHF, UHF, RKS, and UKS single points. It passes a fixed JSON request
to the companion interpreter, records the normal execution-service launch
evidence, and reports SCF convergence separately from process completion. It
has a small reviewed [fixture corpus](tests/fixtures/pyscf/) covering each
supported SCF entry point, an intentionally unconverged SCF, and an
electron-spin-inconsistent runtime refusal. The fixtures are version-scoped
regression evidence, not method-validation references for unrelated programs.

`compare_cube_densities` compares two caller-declared density CUBE files from
PySCF or another program only on the same nuclear geometry and real-space
grid. It requires the scalar unit for each field, normalizes to electrons per
cubic angstrom, and reports electron-count, L1, L2, RMS, and maximum-difference
evidence. It does not interpolate, resample, or decide whether the two
calculations are scientifically comparable. See
[`docs/cube-density-comparison.md`](docs/cube-density-comparison.md).
`compare_cube_orbitals` compares one caller-matched non-degenerate orbital
CUBE pair on that same strict grid and geometry contract. It reports signed
overlap, the phase flip when necessary, phase-aligned overlap, and aligned L2
distance. It does not choose orbital correspondence or compare degenerate
subspaces. See [`docs/cube-orbital-comparison.md`](docs/cube-orbital-comparison.md).
`compare_cube_orbital_subspaces` compares two caller-declared orbital sets
through phase- and rotation-invariant principal overlaps and angles. It
requires a shared CUBE grid and rejects rank-deficient sets. See
[`docs/cube-orbital-subspace-comparison.md`](docs/cube-orbital-subspace-comparison.md).
`compare_pyscf_reference_calculation` combines explicit PySCF/reference
geometry, settings, electron-count, convergence, energy, and optional CUBE
evidence into one report. It records differences without selecting a correct
calculation. See
[`docs/pyscf-reference-comparison.md`](docs/pyscf-reference-comparison.md).
`draft_nwchem_pyscf_reference` prepares that reference from one NWChem input
and optional output, while leaving the PySCF method, density-fitting setting,
and effective electron count as explicit caller declarations. It reports every
unresolved required field rather than guessing an equivalence. See
[`docs/nwchem-pyscf-reference-draft.md`](docs/nwchem-pyscf-reference-draft.md).
`run_nwchem_pyscf_matched_reference` composes that draft with the bounded
PySCF runner and returns the evidence-only report in one local-execution call.
It refuses to start PySCF unless the NWChem output is converged and all
comparison settings have been declared. For DFT, `pyscf_xc` is an explicit
PySCF declaration, while the NWChem `xc` line remains source evidence. See
[`docs/nwchem-pyscf-matched-run.md`](docs/nwchem-pyscf-matched-run.md).
Provide a pre-written NWChem total-density CUBE as `reference_density_cube`
with `density_cube_grid_points` to include density evidence in that same run.
The report refuses to resample mismatched CUBEs and returns `not_comparable`
unless their grid and geometry match exactly.
Use the same point count as `pyscf_compatible_grid_points` in
`draft_nwchem_cube_input` to derive the required NWChem `limitxyz` box from
the explicit-unit input geometry, including PySCF's 3 bohr margin and NWChem's
one-fewer-spacings convention.
The reviewed NWChem inputs, outputs, hashes, and matched-reference assertions
live in [`tests/fixtures/nwchem_pyscf`](tests/fixtures/nwchem_pyscf).
Set `density_cube_grid_points` on `run_pyscf_single_point` to write a bounded
total-density CUBE after converged SCF; its derived filename, SHA-256, grid,
and `electron_per_bohr3` value unit are returned as an artifact record.
Set `orbital_cube_grid_points` plus up to eight zero-based
`orbital_cube_requests` to write selected PySCF MO CUBEs. Restricted methods
use `restricted`; unrestricted methods use `alpha` or `beta`. Each artifact
returns its spin, index, MO energy, occupation, derived label, and hash. See
[`docs/pyscf-orbital-cubes.md`](docs/pyscf-orbital-cubes.md).
does not accept custom Python, periodic cells, geometry optimization, MP2,
multireference methods, or QMCPACK export. Set `CHEMTOOLS_REGISTRY_DB` to a
writable SQLite path when the server's default `$HOME/.chemtools` location is
not writable.

The Python boundary is intentionally limited to periodic electronic structure
and structure identity. Other Orbitron Python API methods need a stable,
bounded response contract and a passing owned fixture before MCP exposure.
An absent companion runtime leaves Chemtools' existing analysis and execution
tools unchanged.

[`environments/chemtools-science.yml`](environments/chemtools-science.yml)
defines the optional Conda or micromamba environment for PySCF, RDKit, and
Open Babel. Install Orbitron's Python bridge explicitly from its intended
wheel or local checkout. The linux-64 resolved lock is bundled with Chemtools
so every fixed science-runner result includes a `runtime_provenance` record:
the fixed runner operation and request hash, lock SHA-256, configured
interpreter, and PySCF, RDKit, Open Babel, and Orbitron version evidence.
Editable Orbitron installations also record the native-extension SHA-256. See
[environments/README.md](environments/README.md).

The checker reads only paths listed in
`references/orbitron_contract_cases.json`. It verifies each pinned size before
hashing, then verifies each SHA-256 hash before parsing and prints a JSON
report with separate `agree`, `disagree`,
`tool_refused`, and `no_reference` outcomes. A changed or missing reference is
never treated as a parser disagreement. The current QE geometry cases compare
Si SCF, Fe SCF, and converged FeO `vc-relax` atom counts, elements, Cartesian
bounds, periodic flags, cell vectors, and geometry provenance against Orbitron.
The failed benzene relaxation separately pins `last_attempted` and its source
description without presenting that structure as converged.

Exit codes are 0 for full agreement, 1 for a field disagreement, 2 when
Orbitron refuses or fails a case, and 3 when a pinned reference is
unavailable. Use `--output report.json` to keep the report. See
[`ADR 005`](docs/adr/005-reference-corpus-boundaries.md) for the committed
fixture, scientific dataset, and external corpus boundaries. The broader
integration work remains in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

### Restricting to one program

Loading all four programs means 274 tool definitions in your agent's
context. For a session focused on one program, filter:

```json
"chemtools-molcas": {
  "command": "chemtools",
  "env": { "CHEMTOOLS_PROGRAMS": "molcas" }
}
```

In analysis mode, `CHEMTOOLS_PROGRAMS=molcas` exposes 91 tools: 40 Molcas
analysis tools and 51 generic analysis tools. The corresponding local and HPC
counts are 100 and 101. Other choices are `nwchem`, `dirac`, and `grasp`.
Comma-separate multiple programs (`nwchem,molcas`).

### Server modes

| Mode | Tools visible | Use when |
|---|---|---|
| `analysis` (default if no `CHEMTOOLS_RUNNER_PROFILES`) | 272 | Post-hoc parsing, drafting, and planning; no chemistry executable needed |
| `local` | 323 | Programs run as subprocesses on this machine (`launcher.kind: "direct"`) |
| `hpc` | 326 | Submit to SLURM/PBS/LSF on an HPC cluster (`launcher.kind: "scheduler"`) |

Mode is auto-detected from your runner profile (see below). Override with
`CHEMTOOLS_MODE=analysis` or the `--mode` flag.

---

## Runner profiles

To actually **launch** jobs (not just parse pre-existing output), point
`CHEMTOOLS_RUNNER_PROFILES` at a YAML or JSON file describing your environment.
The repo includes ready-to-copy examples:

| Example | What it shows |
|---|---|
| `chemtools/runner_profiles.local.example.json` | Minimal local-workstation profile (single direct subprocess) |
| `chemtools/runner_profiles.example.yaml` | Canonical reference covering local + SLURM/PBS HPC profiles |
| `examples/tacc_stampede3/runner_profiles.yaml` | Real TACC Stampede3 SLURM config (SKX / ICX / SPR partitions) |
| `examples/local_workstation/` | Direct-launch workstation profile |

Program installations use the same shape for local and scheduler targets:

```yaml
launcher:
  kind: "scheduler"
programs:
  nwchem:
    launcher_argv: ["ibrun"]
    executable_argv: ["/path/to/nwchem"]
```

Containerized programs put the container prefix in `launcher_argv` and the
program command in `executable_argv`. Molcas keeps
`parallel_caspt2_supported` in `programs.molcas`; DIRAC keeps
`default_mpi`, `default_mw`, and `default_nw` in `programs.dirac`. Modules,
hooks, environment, resources, and scheduler commands remain target-wide.
Version 1 profiles using the previous program-specific field locations are
still accepted, but `programs.<name>` takes precedence.

Copy one, edit the paths to point at your NWChem / OpenMolcas binary, then:

```json
{
  "mcpServers": {
    "chemtools": {
      "command": "chemtools",
      "env": {
        "CHEMTOOLS_RUNNER_PROFILES": "/path/to/runner_profiles.yaml"
      }
    }
  }
}
```

The server auto-detects the right mode (`local` for direct profiles, `hpc` for
scheduler profiles), filters the tool surface accordingly, and exposes
`launch_nwchem_run`, `watch_nwchem_run`, `terminate_nwchem_run`, etc.

NWChem MCP launch and termination use a process-owned execution service.
Dry runs remain read-only. Live direct and SLURM launches record the effective
argument array, paths, resources, PID or job ID, and state in the registry.
Termination only accepts a PID or job ID launched by the same running MCP
server. Restart the server and its previous local process handles are no
longer cancelable through this tool.

The typed path currently supports direct and SLURM version 1 profiles whose
working directory is the input directory. PBS, LSF, alternate working
directories, and `write_script=false` scheduler submission are not supported
yet. The tool returns a clear error instead of falling back to the older
shell-based execution path.

QE `pw.x` uses the same tracked launch boundary through `render_qe_launch` and
`launch_qe_run`. Its typed plan passes `-in <input-file>` and requires the
profile working directory to be the input directory. The process result is not
an SCF verdict. Follow a completed launch with `inspect_run` to check for
convergence and `JOB DONE.` evidence. `ph.x` and `pw2qmcpack.x` execution are
not part of this slice.

The reproducible local and Slurm execution check is documented in
[`docs/execution-smoke-tests.md`](docs/execution-smoke-tests.md). It runs a
small H₂ SCF input through launch ownership, monitoring, run registration,
artifact hashing, and parsed-energy verification.

Molcas uses the same execution boundary. Live launches preserve the
`parallel_caspt2_supported=false` safeguard by changing both `pymolcas -np`
and the requested Slurm ranks to one. `MOLCAS_PROJECT` and `MOLCAS_NPROCS`
come from the reviewed plan and cannot be replaced through `env_overrides`.
Slurm profiles put module loads in `modules` and scratch initialization in
`hooks.pre_run`; see
[`examples/tacc_stampede3/runner_profiles.yaml`](examples/tacc_stampede3/runner_profiles.yaml).
The existing `prepare_molcas_launch` tool remains a read-only preview. Explicit
status and watch requests for owned local or Slurm launches now use the typed
execution record. Unowned identifiers and auto-detected `.jobid` files retain
the legacy path. Molcas scientific-run registration and artifact observations
are still pending.

DIRAC live launches now use the same tracked direct and Slurm path. The
program plan records `pam-dirac --mpi`, the paired `.inp` and `.mol`
filenames, and `programs.dirac.default_mw` and `default_nw` settings. Slurm
does not prefix DIRAC with `ibrun` because `pam-dirac` manages its own MPI
launch. The two input files must be in the same working directory. Advanced
checkpoint flags (`--copy`, `--put`, and `--get`) remain read-only through
`prepare_dirac_launch` until live staging has explicit destination and
overwrite rules.

GRASP workflow scripts also use the tracked direct and Slurm path. A target
runs the complete ordered recipe as `apptainer exec <sif> bash <workflow>`;
this fixes the older Stampede3 template, which invoked `bash` on the host
despite describing an in-container run. The workflow is caller-supplied
executable content and should be reviewed before launch. Interactive
`run_grasp_<exe>` tools now use the same permission and launch-record service
through a synchronous local contract. The target owns an explicit executable
allowlist. Each run records its effective command, stdin SHA-256 and byte
count, return code, elapsed time, and `completed`, `failed`, or `timed_out`
state without storing stdin content in SQLite. Captured stdout and stderr
remain in the MCP response, and the compatibility adapter still writes
optional capture files and `grasp_session.md`. Structured workflows route
each executable through this service and permit typed `cp` file actions in
place of arbitrary pre-step or post-step shell commands.

For HPC profiles, `suggest_nwchem_resources` analyzes your input against
the profile's hardware specs and recommends optimal nodes / ranks / walltime
/ memory directives — no manual guessing.

---

## What you get

The 326 tools cover these areas. Generic tools auto-detect the program where
the underlying operation supports it.

| Area | NWChem | Molcas | Generic | Notes |
|---|---:|---:|---:|---|
| Guided input review | Parse + lint | Lint | `review_input` | DIRAC parses without lint; GRASP single-file review is unsupported |
| Guided run inspection | ✓ | ✓ | `inspect_run` | Normalizes evidence, verdict, uncertainty, and next actions across all four current programs |
| Independent file inspection | | | `inspect_with_orbitron` | Optional fixed-command Orbitron evidence with pinned JSON schema and build provenance |
| Geometry summary | | | `analyze_geometry_with_orbitron` | Validated Orbitron counts, bond statistics, bounds, and unit-cell evidence |
| Molecular-orbital summary | | | `analyze_orbitals_with_orbitron` | Validated restricted or alpha/beta frontier orbitals, occupancy policy, and channel-local gaps |
| Atomic-population summary | | | `analyze_populations_with_orbitron` | Validated per-atom charges, expected system charge provenance, and charge residuals |
| Vibrational summary | | | `analyze_vibrations_with_orbitron` | Validated raw frequencies, units, scaling policy, displacements, and thermochemistry metadata |
| Image render | | | `render_with_orbitron` | Fixed 1024 by 768 PNG returned as MCP image content, with no caller-selected destination |
| Curated knowledge | ✓ | ✓ | `search_knowledge_cards` | Accepted cards are the default; other curation states require an explicit status filter |
| F-block atomic references | | | | `lookup_grasp_fblock_state` retrieves reviewed states; `plan_fblock_atomic_state` emits recorded ATSP2K and GRASP inputs, validates all 132 external donor aliases against a consumer-scoped review ledger, and preserves unresolved donors as manual requirements |
| Parse output (basic) | ✓ | ✓ | `parse_output`, `summarize_output` | Auto-detects program |
| Parse output (deep) | `parse_nwchem_output` | `parse_molcas_output` | | Per-module rich data |
| Geometry / freq / thermo / trajectory | `*_nwchem_*` | `*_molcas_*` | `extract_geometry`, `parse_frequencies`, `parse_thermochem`, `parse_trajectory`, `inspect_geometry` | Generic versions auto-dispatch |
| Input drafting | 17 tools | `draft_molcas_input`, 6 orchestrators (CASSCF, CASPT2 chain, opt+freq, excited states, IRC, scans, atomization) | | |
| Lint | `lint_nwchem_input` | `lint_molcas_input` | | |
| Case analysis | `analyze_nwchem_case` | `analyze_molcas_case` | `analyze_case`, `summarize_output` | Auto-dispatch |
| Recovery suggestion | `suggest_nwchem_recovery` | `suggest_molcas_recovery`, `apply_molcas_recovery`, `try_molcas_run_with_recovery` | `suggest_recovery`, `apply_recovery` | Auto-dispatch |
| Active-space tools | — | `analyze_molcas_active_space`, `validate_molcas_caspt2_setup`, `refine_molcas_active_space`, `suggest_molcas_orbital_swaps` | | Multireference |
| Basis / ECP | 4 tools | `list_molcas_basis_sets` | `suggest_basis_set` | Bundled libraries |
| Documentation | 7 tools (29 docs bundled) | 7 tools (133 docs bundled) | | Plus runtime forum search for NWChem |
| HPC / resources | 6 tools | `prepare_molcas_launch` | `suggest_resources`, `render_job_script` | Scheduler-aware |
| Registry + campaigns | 9 tools (program='nwchem' default) | — | 8 tools | Cross-program SQLite registry |
| Workflow protocols | `list_nwchem_protocols`, `plan_nwchem_calculation`, `create_nwchem_workflow`, `advance_nwchem_workflow` | — | | DAG engine in core/ |

**Bundled data** (no separate downloads):
- 608 NWChem basis-set files
- 91 OpenMolcas basis-set files
- 29 NWChem documentation pages
- 133 OpenMolcas documentation pages
- 180 DIRAC documentation pages

**Optional dependencies**:
- `pip install chemtools[dirac]` adds `h5py` for reading DIRAC HDF5 checkpoints.
  The optional `chemtools-science` environment declares `h5py` for its fixed
  QMCPACK HDF5 metadata inspection operation.

---

## Three-line agent workflows

The tools are designed to chain. A few worked examples the agent can drive:

**Review an input, then inspect its output**:
```
review_input(input_file)                → checks, uncertainty, edit actions
# run with an approved local or scheduler target
inspect_run(output_file, artifact_files=[input_file, stderr, checkpoint])
                                        → evidence, verdict, next actions
```

`checks_passed` means the configured parser and linter found no known problem.
It does not mean Chemtools validated every rule accepted by the chemistry
program. If an explicit `review_input` program conflicts with positive input
content evidence, the tool returns `program_content_mismatch` and the matching
program names instead of reviewing the file with the wrong grammar. Explicit
selection remains available for sparse inputs with no recognized markers.
Automatic input detection may also use a unique filename extension after
content detection finds no match.

`inspect_run` parses the primary output and classifies only the related paths
supplied by the caller. It does not scan a working directory or read related
binary files. Whole-file parsers accept primary outputs up to 128 MiB; larger
files return `primary_output_too_large` before parsing. Before parsing, it
identifies a standalone NBO analysis and
returns `unsupported_output_format` with `detected_format: nbo`; the caller
must supply the parent quantum-chemistry output. If an explicit program
override conflicts with a positive match from another registered backend,
`inspect_run` and generic output tools return `program_content_mismatch` and
the matching program names. An explicit override remains valid when no
detector recognizes the content, which preserves sparse and truncated output
support. When automatic detection finds multiple positive backend matches,
`inspect_run` and the generic dispatch tools return
`program_detection_ambiguous` with the candidate names instead of choosing by
registration order. Pass `program` explicitly to resolve that case. Detector
crashes return
`program_detector_error` with the failing backend and any successful
candidates. Unreadable source files return `program_source_error` with the
underlying error type and `errno`. The low-level `detect_from_*` helpers keep
their earlier lossy behavior for compatibility. A truncated NWChem fragment
can retain a supported printed total-energy record without its preceding
`NWChem Input Module` header, but the task remains `incomplete` unless
completion evidence is present. An SCF or DFT module banner also retains an
incomplete energy task when the output ends before the first total-energy
record. A fatal NWChem input-error message produces a failed unknown task when
execution stops before a scientific module starts; method and operation
comparisons then remain `not_checked`. Each related artifact declared as
`text` contributes
at most 16 KiB, with a 64 KiB total excerpt budget per inspection. Stderr uses
its tail; other large text files use separate head and tail segments. Chemtools
reports truncation, invalid UTF-8, and exhausted excerpt budgets as uncertainty.
Binary and `unknown` artifacts remain metadata only. Molcas `INPORB` is
classified as formatted text; binary `JOBIPH` has a separate checkpoint and
wavefunction artifact kind.

When the related paths contain exactly one primary input, `inspect_run` asks
the backend for supported input-output checks. QE compares calculation mode,
system counts, UPF-derived electron count, and runtime cutoffs while preserving
the requested and symmetry-processed k-point counts separately. NWChem compares the
normalized echoed-deck hash, task methods and operations, explicit charge,
spin multiplicity, atom count, electron count, electron/spin parity,
alpha/beta occupations, wavefunction class, AO basis representation, input
geometry, per-task state, and external restart references. For multi-task
decks, it tracks changes to charge, multiplicity, explicit
RHF/ROHF/UHF/ODFT/RODFT reference selection, spherical or Cartesian basis
selection, explicit named `xc` aliases, ECP core-electron replacement, named AO
basis indirection, named ECP indirection through `ecp basis`, and `set geometry`
selections as NWChem reads the input. Named ECP blocks remain stored but
inactive until selected; redefining the unnamed default restores `ecp basis`.
Chemtools pairs the selected state with evidence from the corresponding
top-level output task. DFT and TDDFT tasks compare B3LYP, PBE0, SCAN, BHLYP,
and M06-2X with NWChem's runtime XC Method label. For TDDFT, the parser keeps
the internal DFT ground-state calculation and excited-state calculation in
one task and reports the printed target-state energy. Each completed DPLOT
section and each explicit NWChem Property Module section is a property task.
The property handler's printed `energy failure` marker makes an unfinished
property task failed rather than merely incomplete. Weighted and
component-level `xc` expressions remain `not_checked` until Chemtools can
compare their resolved components and coefficients.
Raman tasks keep their Raman label and raw NWChem operation while using the
shared `frequency` task kind. Input-output checks treat `task ... raman` as
that same operation. Omitted task operations use NWChem's parsed default, such
as `energy` for `task scf`, instead of forcing the comparison to abstain.
Explicit TCE model keywords are kept separately from the `tce` execution
module, so a `task tce energy` block can pair with CCSD, CCSD(T), or MP2
evidence in the output. Unrecognized TCE models remain unresolved. An explicit
`NWChem TDDFT Gradient Module` section is reported as a gradient task while
retaining TDDFT as its method and the target excited-state energy.
An open-shell DFT reference activated by a completed task remains active for
later tasks in the same NWChem database unless the input explicitly changes
the reference. This matters when a later fragment uses `mult 1` but retains
the earlier spin-polarized determinant.
NWChem total-energy parsing accepts the printed `=`, `:`, and delimiter-free
forms used by SCF, DFT, MP2, CCSD, and CCSD(T). When an optimization prints
both reference and correlated energies, its final energy and trajectory use
the highest-level recognized method rather than the SCF reference.
The compact task `basis` field reports a family only when echoed library
assignments resolve to one name. Manual and mixed-family bases remain unset
there; per-element runtime basis tables carry their descriptions and counts.
Parsed task summaries use 1-based inclusive output line ranges. NWChem keeps
byte offsets only in its raw parser evidence for internal section matching;
it does not expose them as line numbers.
The basis checks verify that each element in the selected geometry appears in
NWChem's populated runtime AO basis table. They preserve each tag's basis
description, shell count, function count, and function types as evidence, but
do not recalculate the final AO dimension from library input. Electron counts
use the active geometry, molecular charge, explicit ECP `nelec` values, and
standard ECP assignments resolved from the bundled NWChem library. The
evidence records the library family and source file used for each resolved
element. Printed ECP replacement counts provide a separate check when NWChem
emits them.
The parity check verifies separately that the input-derived and
output-reported electron count minus `(multiplicity - 1)` is even.
Alpha/beta populations must sum to the expected electron count and differ by
`multiplicity - 1`. Reported `RHF` and `closed shell` references normalize to
closed shell; `ROHF`, `UHF`, `ODFT`, `RODFT`, `open shell`, and
`spin polarized` normalize to open shell.
The check abstains when an ECP family is absent from the bundled library, an
ECP directive names an external `file`, a selected ECP name has no definition,
or restart state or explicit geometry-center charges make the effective
nuclear charge unresolved. Geometry comparison uses element order and pair
distances, so translation and rotation do not create false mismatches. Input
geometry units
must be explicit for coordinate comparison, but are not required for electron
counting. A field remains `not_checked` when the output is too sparse or task
boundaries cannot be paired safely. Restart databases and orbital files count
only when their paths are supplied explicitly in `artifact_files`.

**Parse a run you don't recognize** (any program):
```
parse_output(output_file)              → tasks, energies, diagnosis
summarize_output(output_file)          → high-signal narrative
analyze_case(output_file)              → verdict + issues + next_actions
```

**Recover a failed CASPT2 run** (Molcas):
```
analyze_molcas_case(output_file)       → verdict=problematic, issues list
suggest_molcas_recovery(output_file)   → failure_class + fix_recipe
apply_molcas_recovery(input_file, output_file)  → writes patched input
```

Recovery patchers check that the input is recognizable as the selected
program before writing a new file. GRASP automatic parsing also recognizes
RMCDHF `.sum`, RCI `.csum`, hyperfine `.(c)h(lsj)`, isotope-shift `.(c)i`, and
LSJ radiative-transition `.(c)t.lsj` outputs as distinct file families.

**Submit and monitor an NWChem job on HPC**:
```
suggest_nwchem_resources(input_file, profile)   → optimal nodes/ranks
launch_nwchem_run(input_file, profile, auto_watch=true)  → block until done
analyze_nwchem_case(output_file)        → quality verdict
```

**Set up a CASSCF / CASPT2 calculation from scratch**:
```
prepare_molcas_casscf_setup(molecule, cas=(M,N), method="CASPT2")  → input
prepare_molcas_launch(input_file)       → safe pymolcas command
# run the command
analyze_molcas_case(output_file)        → check verdict before trusting energy
```

---

## CLI debugging

```bash
chemtools --show-mode                          # mode + reason + program filter (JSON)
chemtools --list-tools                         # tool names visible under current filters
chemtools --mode analysis                      # force analysis mode (no executable needed)
chemtools --mode analysis --toolset guided     # review + inspect + knowledge search
chemtools --programs molcas                    # only Molcas + generic tools
chemtools --mode local --programs nwchem,molcas
```

Inside an agent session, call the `get_server_mode` tool to introspect at
runtime — useful when a tool fails with "not available in mode."

---

## Architecture

Proposed architecture decisions are recorded in [`docs/adr`](docs/adr).
ADR 001 defines optional program capabilities and the built-in backend
catalog. ADR 002 separates execution permission from named local or Slurm
targets. ADR 003 defines runs as typed artifact collections with provenance.
ADR 004 defines canonical public tool names and compatibility-alias removal
rules. ADR 005 separates committed fixtures, scientific datasets, and
manifest-selected external references. These ADRs define the migration
target; [`PROJECT_PLAN.md`](PROJECT_PLAN.md) records what is implemented and
which compatibility paths remain.

The Phase 3 execution boundary now supports copy and symlink staging for
auxiliary inputs. Both paths must resolve under the selected target's allowed
roots. The executor validates the full staging list before writing, refuses
existing destinations and collisions with launch outputs, and records the
resolved staging manifest with the launch. Successful NWChem auto-registration
now creates the scientific run and its one-to-one execution-launch link in one
SQLite transaction. The NWChem adapter checks service ownership, program
identity, and the recorded input before registration.

Execution code is split by responsibility. `_common.py` owns command
rendering, allowed-root checks, and staging. `local.py` owns live process
handles, synchronous execution, status, and signals. `slurm.py` owns batch
scripts, submission, status, and cancellation. The old `executors.py` import
path remains as a compatibility facade.

The version 1 profile runner now defines program-neutral
`run_calculation`, `render_calculation_run`, `inspect_run_status`, and
`watch_run` entry points. Molcas, DIRAC, and GRASP scheduler wrappers use
those names. The previous NWChem-named functions remain direct aliases for
existing Python callers. Profile loading and default merging live in
`execution/legacy_profiles.py`. Unowned PID and scheduler inspection, file
status, tailing, cancellation, and optional NWChem progress parsing live in
`execution/legacy_status.py`. `core/runner.py` remains the compatibility
import path and contains the legacy render and launch implementation.

Local NWChem status checks and explicit watches now poll only the live process
handle owned by the execution service. The retained handle remains
authoritative while a partial output file is present, so an incomplete parse
cannot end a live watch. A terminal check records the return code and elapsed
time, updates the linked scientific run, and stores point-in-time SHA-256
observations for stdout and stderr. Repeated checks reuse that terminal state
without duplicating observations.

Owned NWChem Slurm status checks use the target's `squeue` command for active
jobs and a target-owned `sacct` fallback for terminal state, exit code, signal,
and elapsed time. This follows Slurm's documented split: default `squeue`
reports pending, running, and completing jobs, while `sacct` exposes accounting
state and exit codes. If neither command returns the job, Chemtools reports
`not_found` and leaves the launch submitted. It does not infer success from an
empty queue result. See the official [squeue](https://slurm.schedmd.com/squeue.html),
[sacct](https://slurm.schedmd.com/sacct.html), and
[job-state](https://slurm.schedmd.com/job_state_codes.html) documentation.

NWChem scheduler auto-watch and the explicit `watch_nwchem_run` tool use the
same owned status path for local and Slurm launches. Their polling loop
retains adaptive intervals and compact history while terminal execution state
updates the launch, linked run, and output observations once. Typed
`not_found` results continue polling instead of treating an output file as
proof of completion. The explicit tool checks launch ownership before polling,
then sends unowned PIDs or job IDs to the legacy watcher without an extra
process or scheduler query. Watch responses expose their final overall status
so the MCP can recommend analysis after incomplete or failed output.

Molcas, DIRAC, and asynchronous GRASP workflow status and watch tools share
the same owned execution projection. They use retained local process handles
or target-owned Slurm queue and accounting queries, persist terminal launch
state, and keep the legacy watcher for unowned identifiers. These programs do
not create scientific-run links yet, so the status tools do not claim artifact
observations or chemistry success from an execution exit code.

GRASP per-executable tools remain synchronous and return terminal execution
results directly. Staged-input hashes and copy provenance still await their
planned integration. The same boundary supports captured local runs with
closed stdin when no payload is supplied, explicit timeouts, and persistent
terminal states.

The [current-to-target module map](docs/current-to-target-module-map.md)
records which existing modules stay in place, which responsibilities move in
each phase, and the dependency crossings the refactor must remove.

```
chemtools/
  core/                          program-agnostic shared infrastructure
    program.py                   backend providers, capabilities, and validation
    registry.py                  plugin registry + program auto-detection
    runner.py                    legacy profile, submit, render, and status facade
    monitoring.py                polling, terminal checks, and watch history
    workflow.py                  DAG engine for multi-step protocols
    basis_advisor.py             basis-set + ECP recommendation
    units.py, thermochem.py,
    geometry.py, issues.py,      shared math + helpers
    recovery.py, case_analysis.py, session.py
    run_records.py               SQLite run records + execution links
    run_registry.py              Compatibility facade + campaigns/workflows
  programs/
    nwchem/                      NWChem plugin
      parse/                     output / input / freq / mos / tasks / tce parsers
      input/                     input file drafting + lint
      strategy/                  diagnose, recovery, case_review, progress, resources
      binary/                    movecs / hessian / fdrst readers
      data/                      bundled basis library + docs
      _plugin_*.py               sub-protocol implementations
    molcas/                      OpenMolcas plugin (mirrors nwchem/)
    dirac/                       DIRAC plugin (4c / X2C / AOC / KPSELE)
    grasp/                       GRASP2018 plugin (multi-exe atomic workflows)
  mcp/
    decorator.py                 @_tool registration with program / needs tags
    modes.py                     mode + program filtering
    server.py                    JSON-RPC entry point
    cli.py                       `chemtools` CLI entry point
    tools/
      nwchem.py                  NWChem tool definitions + handlers (101 tools)
      molcas.py                  Molcas tool definitions + handlers (45 tools)
      dirac.py                   DIRAC tool definitions + handlers (39 tools)
      grasp.py                   GRASP tool definitions + handlers (51 tools)
      qe.py                      Quantum ESPRESSO definitions + handlers (20 tools)
      qmcpack.py                 QMCPACK definitions + handlers (14 tools)
      generic.py                 Cross-program definitions + handlers (56 tools)
      guided.py                  Guided cross-program workflow tools
```

Tools are tagged with `program=nwchem|molcas|dirac|grasp|qe|qmcpack|generic` and
`needs=none|registry|runner_profile|executable_or_scheduler|executable|scheduler`.
The active mode + program filter decides which subset is exposed at
`tools/list` time. Generic tools auto-detect the program at call time via
`registry.resolve(program=None, path=output_file)`.

---

## Adding a new program

Each program is a plugin under `chemtools/programs/<name>/`:

1. Implement the providers needed by the backend's declared operations from
   `chemtools/core/program.py`.
2. Export one validated `ProgramBackend` from
   `chemtools/programs/<name>/__init__.py`. Program modules do not register
   themselves.
3. Add one `BuiltinBackendSpec` to `chemtools/mcp/catalog.py`. This is the
   only built-in membership and registration point.
4. Add MCP tool definitions in `chemtools/mcp/tools/<name>.py`, with one
   `@_tool("name", program="<name>")` handler per tool.
5. Bundle docs / basis libraries under `chemtools/programs/<name>/data/`
   (or pull from `chemtools/data/<name>/` if shared).

The backend contract is documented in `chemtools/core/program.py`. NWChem is
the complete reference declaration; Molcas, DIRAC, and GRASP show partial
capability sets.

---

## Troubleshooting

- **"No program registered for path"** — the auto-detection didn't recognize
  the output. Check that the file is a real NWChem `.out` or OpenMolcas `.log`.
  Or call the per-program tool (`parse_nwchem_output` / `parse_molcas_output`)
  directly.
- **Tool fails with "not available in mode"** — call `get_server_mode` to
  see which mode you're in. Either switch mode (`CHEMTOOLS_MODE=local`)
  or use a different tool that's available in your mode.
- **Runner profile not loading** — verify `CHEMTOOLS_RUNNER_PROFILES` points
  at a readable file; check `chemtools --show-mode` for the resolution
  result. Profile YAML / JSON syntax errors are logged on stderr.
- **NWChem job stuck** — `watch_nwchem_run` detects known output-silent
  phases (SAD X2C guess, DFT grid generation, frequency Hessian
  differentiation, TCE AO→MO transform) and reports "expected slow"
  rather than treating them as hung.

For help or to file an issue:
[github.com/charliecpeterson/chemtoolsmcp/issues](https://github.com/charliecpeterson/chemtoolsmcp/issues)

---

## License

See LICENSE file.

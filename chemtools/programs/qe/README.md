# Quantum ESPRESSO support

This package reviews `pw.x` inputs and outputs for `scf`, `relax`, and
`vc-relax` calculations. Input parsing covers scalar namelist assignments and
the `ATOMIC_SPECIES`, `ATOMIC_POSITIONS`, `CELL_PARAMETERS`, and `K_POINTS`
cards. Other documented `pw.x` calculation modes remain parseable, but the
review reports that their semantics have not been checked. `bands.x`, `dos.x`,
`pp.x`, and `projwfc.x` inputs receive one explicit unsupported-program finding
instead of a cascade of missing-`pw.x` diagnostics.

Run inspection is limited to `pw.x` (PWSCF) output. Banners for `bands.x`,
`dos.x`, `pp.x`, and `projwfc.x` are identified before parsing and return
`unsupported_output_format`, rather than being mislabelled as PWSCF SCF runs.
Bannerless launcher-abort fragments remain unassigned because they do not
establish which QE program produced them.

Input geometry review covers explicit `ibrav=0` cells in alat, angstrom, or bohr and
every nonzero Bravais form documented for QE 7.5. It accepts documented
`celldm` or conventional `A/B/C` and cosine parameters. Atom positions may use
alat, angstrom, bohr, or crystal coordinates. It uses periodic minimum-image
distances and applies molecular bond-network checks only when the cell has
enough vacuum. A sub-0.6-angstrom contact is an error; main-group
overcoordination is a warning. Periodic solids receive metrics without a
molecular verdict. Missing, mixed, or invalid lattice parameters return
`not_assessed` for this check. Numeric coordinate fields accept QE's documented
restricted arithmetic syntax. The parser evaluates only numeric constants and
the `+`, `-`, `*`, `/`, and `^` operations, with finite results and bounded
expression size; names, function calls, malformed expressions, and excessive
exponents remain invalid coordinates. A unitless `CELL_PARAMETERS` card uses
QE's deprecated `alat` or bohr default and records that as `implicit_alat` or
`implicit_bohr` in the coordinate contract. Unitless `ATOMIC_POSITIONS` uses
the documented deprecated `alat` default and is recorded as `implicit_alat`.
`crystal_sg` syntax requires a positive `space_group` number. Structural
analysis and atom-count checks remain unavailable until Chemtools expands it.

Path-aware review also reads the bounded preamble of referenced UPF 2 files.
It reports element, pseudopotential type, relativistic treatment, functional,
valence charge, selected flags, and positive cutoff suggestions from
`PP_HEADER`. Relative `pseudo_dir` values use the input directory as an
explicit review assumption. Missing files, element mismatches, and inputs below
the hardest positive cutoff suggestion remain separate findings.

Charge and spin review covers relationships that can be decided from the
input itself: `nspin`, `noncolin`, and `lspinorb` mode consistency; species
index bounds; fixed versus starting magnetization; missing magnetic seeds;
and whether inspected UPFs advertise spin-orbit data. If every atomic species
has a parsed `z_valence`, Chemtools sums valence electrons over
`ATOMIC_POSITIONS` and applies the documented `tot_charge` sign. This is
accounting evidence only. It does not decide whether the requested charge or
spin state is physically appropriate, and it does not impose integer-electron
or spin-parity rules on calculations that may use fractional occupations.

K-point review parses all eight documented `K_POINTS` forms. It validates
automatic grids, explicit point counts and rows, and the use of `tpiba_b` or
`crystal_b` with a bands calculation. The response keeps requested full-grid
counts separate from irreducible counts because only `pw.x` output establishes
the latter after symmetry reduction. It also records `nosym`, `nosym_evc`,
`noinv`, `no_t_rev`, and `force_symmorphic` effects.

Automatic meshes receive a three-stage candidate series. Chemtools preserves
the shift and mesh parity, targets 25 and 50 percent refinement, and leaves
axes currently at one unchanged. Small meshes can require larger steps to keep
the parity and stages distinct. This is a labelled heuristic, not a QE default.
The caller must confirm dimensionality, select tolerances, keep the other
numerical controls fixed, and extend the series when the requested energy,
force, or stress tolerance has not been met.

Output parsing keeps converged `! total energy` records separate from ordinary
SCF iteration energies. It reports SCF cycle outcomes, the last estimated SCF
accuracy, system and cutoff summaries, total force, stress, BFGS convergence,
final enthalpy, and final coordinates in their native QE units. A `vc-relax`
result retains both the converged relaxation enthalpy and the final SCF energy.
Geometry extraction normalizes the runtime cell and Cartesian site table from
`alat` to angstroms for SCF-style output. For a converged relaxation, it reads
the final-coordinate block and handles `angstrom`, `bohr`, `alat`, and
`crystal` positions plus `angstrom`, `bohr`, and `alat` cells. A failed or
incomplete relaxation has no output geometry because its last printed
coordinates are an attempted structure, not a converged result.

The separate trajectory operation keeps the initial runtime geometry and each
distinct relaxation update. Every frame includes its cell, atom coordinates,
source lines, and role. The last frame is `converged_final` only when PWSCF
prints BFGS convergence and a final-coordinate block; otherwise it is
`last_attempted`. Completed non-convergence and truncated output remain
separate statuses. SCF energies are returned by output parsing but are not
assigned to trajectory frames by position because a relaxation may print an
extra final SCF calculation at the relaxed geometry.

Coordinate handling has one low-level owner. `_coordinates.py` parses PWSCF
cards, runtime site tables, and units. `geometry.py` decides whether one
snapshot satisfies the converged-output contract, while `trajectory.py`
preserves optimization history. New coordinate forms belong in the shared
normalizer rather than either consumer.

Each available trajectory also includes a bounded structural analysis.
Distances use the periodic minimum image, including skewed cells. Molecular
bond-network checks run only when every frame has at least 5 angstrom of empty
periodic space along each lattice direction and the covalent-radius table
covers every element. These checks flag sub-0.6-angstrom contacts,
main-group overcoordination, new fragments, and new atoms without a bonded
neighbor. Extended periodic structures receive geometric metrics without a
molecular connectivity verdict. A cell-volume change of at least 20 percent
is reported as an observation, not as proof that the structure is wrong.
Each concern records whether it was present in the input geometry or developed
along the trajectory. A trajectory with both receives the summary origin
`mixed` without losing the origin of each finding.

`inspect_run` includes a compact copy of this evidence and adds structural
concerns or large cell changes to its assessment reasons. It returns a
`parse_trajectory` next action when the full frames need inspection. Automatic
inspection reads trajectory detail only for outputs no larger than 16 MiB.
Structural analysis also stops at 512 frames or 250,000 pair evaluations. A
case beyond any limit returns `not_assessed` with the applicable count.

The generic `inspect_run` tool diagnoses converged SCF and BFGS runs, SCF
nonconvergence, truncated output, and runtime errors. Repeated MPI copies of the
same error collapse into one line-anchored diagnostic. `readpp` file-not-found
errors are identified specifically and point to `review_input` when the related
input file is available. A `JOB DONE` marker establishes clean process shutdown;
it does not override a printed SCF failure.

Bands and NSCF output receive a limited inspection. Their shared PWSCF output
markers establish that the band calculation ended, but do not identify which
input mode produced it. Chemtools therefore reports `bands_or_nscf` unless the
input is inspected separately.

When `inspect_run` receives one explicit `.in` artifact, it compares the input
with PWSCF's runtime summary. The current checks cover calculation mode, atom
and atomic-type counts, UPF-derived electron count, `ecutwfc`, and the explicit
or defaulted `ecutrho`. Bounded UPF-header evidence also preserves the declared
local channel and total nonlocal-projector count. When every declared
`PP_BETA` opening tag is visible in that bounded preamble, it reports counts by
angular channel. It does not parse radial projector data or infer DMC
suitability. Gamma-only sampling must produce one runtime k-point.
Other k-point counts remain `not_checked` because a requested mesh and a
symmetry-reduced runtime count are different quantities. Output that stops
before the runtime summary also abstains instead of turning missing evidence
into mismatches.

The contract follows the Quantum ESPRESSO 7.5
[`INPUT_PW`](https://www.quantum-espresso.org/Doc/INPUT_PW.html) reference.
The initial regression cases come from `/home/charlie/input_examples/qe`.

The guided `launch_run` tool uses the shared execution service for a reviewed
`pw.x` input through a schema-2 target or version 1 migration profile. The
typed plan passes the input as `-in <filename>`, runs in the input directory,
and records the configured command, resources, output paths, and local process
or scheduler identity. The launch result establishes process state only;
inspect the produced output before treating an SCF or relaxation as complete.

This slice does not execute `ph.x` or `pw2qmcpack.x`, establish cutoff or
k-point convergence, parse radial UPF data, choose a physical magnetic state,
recover geometries from incomplete runtime tables or unsupported cards, compare
general k-point sets after symmetry reduction, construct nonzero-`ibrav` cells
for structural review, or interpret band eigenvalues. Those need separate
evidence and tests.

`check_qe_qmcpack_conversion_ready` is a separate pre-conversion check. It
requires an SCF input, checks that `disk_io` preserves the wavefunctions,
requires an explicit `K_POINTS crystal` gamma point, and rejects
`assume_isolated='m-t'`. Unknown variants stay `review_required` rather than
being treated as compatible. It does not inspect a QE output or invoke
`pw2qmcpack`.

`inspect_qe_qmcpack_conversion` combines every supported conversion check in
one response: artifact lineage, deck references, pseudopotentials,
pseudopotential and ion species, valence, DMC projector evidence, electron and
atom counts, geometry, fixed collinear spin, and charge accounting. The
granular conversion tools remain available for diagnosis. The aggregate check
has the same evidence boundaries as those tools.

`inspect_qe_qmcpack_conversion_artifacts` takes explicit QE input, QE output,
and `.pwscf.h5` paths. It repeats the input preconditions, requires a clean
completed converged SCF output, checks the HDF5 signature at supported
superblock offsets, and requires the artifact to be no older than the input or
output. It does not decode HDF5, invoke `pw2qmcpack`, inspect a QMCPACK input,
or compare energies.

`inspect_qe_qmcpack_conversion_deck` also takes a QMCPACK XML input. It follows
the existing bounded include graph and requires a resolved HDF5 reference to
the exact declared `.pwscf.h5` artifact. It does not merge XML, decode HDF5,
compare particle or pseudopotential semantics, invoke a converter, or compare
energies.

`inspect_qe_qmcpack_conversion_pseudopotentials` also inspects every QMCPACK
pseudopotential referenced by that bounded graph. It requires the supported
semilocal XML evidence: Hartree units, `r*V` encoding, linear channel grids,
matching declared grid counts, and an existing local channel. It does not
establish transferability or equivalence to the QE UPF.

`inspect_qe_qmcpack_conversion_projectors` checks bounded QE `PP_BETA`
projector counts for nonlocal `NC` UPFs when the primary or bounded included
QMCPACK XML declares DMC. Native `SL` UPFs do not require KB-projector
evidence. It keeps each observed DMC block tied to its source file and local
block index. A
repeated angular channel or a source UPF type other than `NC` or `SL` returns
review and asks whether the QMCPACK card came from a separately generated
semilocal potential. It does not establish pseudopotential family equivalence,
card provenance, or DMC compatibility.

`inspect_qe_qmcpack_conversion_electrons` compares QE's printed runtime
electron count, and complete UPF-based valence accounting when available, with
the particle-group sizes of the QMCPACK Hamiltonian target. Missing count
evidence remains `review_required`. This does not establish a physical charge
or spin state.

`inspect_qe_qmcpack_conversion_atoms` compares QE's declared `nat` with the
explicit sizes of QMCPACK particle sets other than the Hamiltonian target.
Missing particle-set sizes remain `review_required`. This does not compare
element identities, cells, or coordinates.

`inspect_qe_qmcpack_conversion_ion_species` compares QE atomic element counts
with explicitly sized QMCPACK ion-group labels, including bounded XML includes.
It works without ion positions but does not compare pseudopotential identity,
charge, spin, or coordinates.

`inspect_qe_qmcpack_conversion_geometry` compares an explicit QE periodic cell
and atom positions with one QMCPACK `bohr` simulation cell and ion particle
set, modulo periodic translations. It requires `p p p` boundaries and returns
`not_ready` with observed cell volumes when either periodic cell is non-finite
or singular. It does not compare pseudopotentials, spin, or energy conventions.

`inspect_qe_qmcpack_conversion_spin` compares an explicit QE `nspin=2`
`tot_magnetization` with the `u` minus `d` population in the QMCPACK electron
particle set. Other QE spin modes and electron-group layouts remain
`review_required`. This is fixed-moment compatibility evidence, not a physical
spin-state or spin-density comparison.

`inspect_qe_qmcpack_conversion_charge` sums QE UPF `z_valence` over the input
atoms, then compares that total and QE `tot_charge` with QMCPACK ion-group
`valence` parameters and selected electron-particle counts. It requires every
ion group to have an explicit size and numeric `valence`; other layouts remain
`review_required`. This is accounting evidence, not a pseudopotential-family
or physical charge-state verdict.

# Quantum ESPRESSO, QMCPACK, and pseudopotential generation — f-block notes

Neither QE nor QMCPACK is installed on the dev box yet and neither has MCP
tooling. These notes exist so that when they do, the agent starts with the
tricks already paid for rather than rediscovering them. Everything below was
learned the hard way in a single 2026-07-27/28 session building a plane-wave
pseudopotential family for Ce (project `ECPgen`, family name "sECP"), running
QE 7.5 + ONCVPSP 4.0.1 in Docker and QMCPACK 4.3.0 on Slurm.

Written as input for chemtools, not as a spec.

---

## 0. The programs and how they chain

```
ONCVPSP (oncvpsp.x / oncvpspr.x)   ->  UPF or psp8 pseudopotential
   or ld1.x (QE's atomic code)     ->  UPF / old-NC semilocal
        |
        v
QE pw.x  (SCF with that potential)
        |
        +--> pw2qmcpack.x --> .pwscf.h5 spline orbitals --> QMCPACK trial WF
        |
        +--> EOS / defect / property runs (the DFT deliverable)
```

Two generators, and **the choice between them is not cosmetic**:

- `oncvpsp.x` writes **multi-projector Kleinman-Bylander** potentials (2
  projectors per channel). Required whenever two shells of the same l are in
  valence — e.g. 5s and 6s both explicit for a lanthanide with semicore.
- `ld1.x` (with `pseudotype = 1`, `tm = .true.`) writes **semilocal** V_l(r)
  channels, one projector per l.

**QMCPACK's DMC evaluates semilocal potentials only.** A 2-projector KB
potential has no exact semilocal equivalent, and the DMC locality
approximation / T-moves are formulated around semilocal forms. So a project
that wants both plane-wave DFT *and* DMC needs **two sibling potentials**, not
one file converted twice. Budget for that from the start.

---

## 1. ONCVPSP: the traps

### 1.1 stdin must be a seekable file

`oncvpsp.x` **seeks on unit 5**. Piping input into it through a Docker attach
pipe dies with:

```
Fortran runtime error: Illegal seek
#3 __input_text_m_MOD_cmtskp
```

Wrong:

```bash
docker run --rm -i image oncvpsp.x < input.dat        # pipe -> Illegal seek
```

Right — put the file inside the container and redirect there:

```bash
docker run --rm -v "$PWD:/work" -w /work image bash -c "oncvpsp.x < input.dat"
```

This matters enormously for optimizer loops: the naive `subprocess.run(...,
input=text)` pattern fails 100% of the time and the error message names
neither stdin nor the pipe.

### 1.2 Structural constraints (each found by a loud failure)

For a 4f-in-valence lanthanide with 5s5p semicore:

- **rc(s) has a floor** — set it too tight and you get
  `ERROR pspot: first pseudo wave function has node, program will stop`.
  For Ce with a 46-electron core the floor bisected to between 2.25 and 2.30
  bohr (the 5s outermost node). The error names the symptom, never the channel;
  find it by bisection.
- **rc(local) must be ≤ every channel's rc**, else
  `test_data: rc < rc(lloc) for l = 2. Not allowed.` This kills "use s as the
  local channel" for any element whose f channel needs a tight rc.
- **The l=0 positive-energy GHOST warning is endemic**, not a defect, when two
  s shells (5s + 6s) are in valence. Proof: the ONCVPSP authors' own bundled
  `57_La.dat` test input emits the same flag (+0.73 Ha). Roughly ten Ce
  candidates were discarded chasing it before checking the stock inputs.
  **Calibrate any generator warning against the code's own test suite before
  treating it as a candidate defect.** Adjudicate ghosts by behavior (does the
  ion ladder converge in pw.x with correct occupations?), not by the flag.

### 1.3 The ionized-reference trick

Generating from the neutral atom puts *two* occupied s shells (5s, 6s) into the
reference, which pins the s projectors and is the source of the endemic ghost.
Generating from an **ionized reference** (Ce²⁺ 5s²5p⁶4f¹5d¹ — no 6s) removes
6s from the projector structure entirely; the 6s window is then covered by
`debl` (the second-projector energy offset) and the neutral atom becomes a
*test* configuration, which it passes at ~1.7e-4 Ha.

This trick is what makes the **semilocal `ld1.x` route viable at all** — one
projector per l cannot carry 5s and 6s, but from an ionized reference there is
only one s shell to carry.

### 1.4 Fully-relativistic / SO variant is free

`oncvpspr.x` consumes the **identical input file** and produces j-resolved
channels. On Ce²⁺ this reproduced a 4f j-splitting of 0.34 eV. If the project
needs spin-orbit (SOREP for QMCPACK spinor DMC, or 2c DFT), it costs one extra
run, not a new input.

### 1.5 Bundled test inputs are the best templates

`/opt/oncvpsp/share/tests/data/` ships `57_La.dat`, `60_Nd_GHOST.dat` (a
deliberate negative control!), `73_Ta.dat`, etc. For any new element, start
from the nearest bundled input rather than from scratch — and run the stock
input first to calibrate what "normal" warnings look like for that structure.

---

## 2. ld1.x (semilocal generation)

Input quirks that cost iterations:

- `rcloc` **must be set and positive** whenever `lloc = -1`, else
  `Error in routine ld1_readin (1): rcloc must be positive`.
- **One wavefunction per l, full stop.** Listing both `5S` and `6S` gives
  `Error in routine ld1_readin (1): Two wavefunctions for the same l`.
  (Again: the ionized reference is what makes this survivable.)
- Writing the pseudopotential in **native old-NC semilocal format** rather than
  UPF is done purely by the output filename in `file_pseudopw` — a name not
  ending in `.UPF` gets the old format, which contains the true `Pseudopot.
  l=0..3` + `Local PP` channel blocks on a log grid. **This is the format to
  parse when you need real semilocal channels** (see §4.2).
- `ld1`'s **test mode (`iswitch = 2`) is not a usable comparator**: test
  configurations get silently frozen, charge-matching is quirky, and an f²
  configuration collapsed outright. Use the generator's own AE-vs-PSP
  excitation table (ONCVPSP style) or real pw.x calculations instead.

---

## 3. Quantum ESPRESSO: pw.x for atoms-in-boxes and f-block solids

### 3.1 Building QE with `pw2qmcpack`

The plugin is not in the default build. Three things bite, in order:

1. `-D QE_ENABLE_PLUGINS=pw2qmcpack` requires **HDF5 dev headers**
   (`libhdf5-dev`); without them cmake fails with
   `Could NOT find HDF5 (missing: HDF5_LIBRARIES ...)` deep in
   `external/pw2qmcpack/CMakeLists.txt`.
2. Even after it compiles, the **install rule is broken upstream**: it looks
   for `build/external/pw2qmcpack/mod/pw2qmcpack_esh5` which the build never
   creates. Workaround — `mkdir -p` that path between `cmake --build` and
   `cmake --install`.
3. On a multiarch image, QE's own cmake needs the right `SCALAPACK_LIBDIR`
   (`/usr/lib/x86_64-linux-gnu` vs `aarch64-...`); parameterize it rather than
   hardcoding, or the same Dockerfile can't serve both an Apple-silicon laptop
   and an x86 box.

### 3.2 Atoms in boxes (for generating molecular/atomic trial WFs)

- **Open-shell f atoms are SCF-multistable in boxes.** A neutral Ce
  (4f¹5d¹6s²) in a 24-bohr cell landed in different f-orientation minima on
  repeated runs — total magnetization flopping across 4/−2/0/2 with the same
  input, and one point in an ecut scan came back *non-variationally* 13 mRy
  above its neighbour. Use `tot_magnetization` (fixed-moment) for these, and
  treat a non-monotonic ecut curve as a basin flag, not noise.
- Ion states are much better behaved than neutrals (Ce⁴⁺ converged in 11
  iterations, Ce³⁺ in 46, the neutral not at all reliably).
- `assume_isolated = 'm-t'` (Martyna-Tuckerman) is right for a genuinely
  isolated system **but see §4.3** — it silently breaks consistency with
  QMCPACK.

### 3.3 EOS / solid-state workflow that worked

For CeO₂ (fluorite, the canonical 4f solid test): 3-atom primitive cell,
`ibrav = 2`, 85 Ry, 6×6×6 shifted k-mesh, 9-point `celldm(1)` scan, then a
Birch-Murnaghan fit. Fit residuals of ~2 µHa are achievable and give a₀ to
better than a milli-ångström. Reference values worth having in the tool:
CeO₂ experiment a₀ = 5.411 Å, B₀ ≈ 204–220 GPa; PBE-PAW literature ≈ 5.47 Å,
172–190 GPa. (Our untuned NC potential: 5.4545 Å, 179.6 GPa — i.e. PAW-grade,
with the residual vs experiment being the *functional*, not the potential.)

Useful negative control: swapping a hand-made oxygen for a PseudoDojo-grade
oxygen moved a₀ by 0.0002 Å and B₀ by 0.6 GPa. **The metal card dominates;
don't chase the light-element partner.**

### 3.4 Defect supercells

O-vacancy formation energy `E_f = E(Ce₄O₇) + ½E(O₂) − E(Ce₄O₈)` came out at
4.08 eV unrelaxed — inside the plain-PBE window (~3.6–4.2). Two warnings the
tool should surface:

- **Plain PBE delocalizes the leftover electrons.** The vacancy cell carried
  |m| = 2 µB (correct — two Ce³⁺-like f electrons) but smeared equally over all
  four Ce (−0.61 µB each). Localization needs PBE+U on the 4f (or hybrid), a
  cell big enough that the vacancy's neighbours aren't *all* the metal atoms,
  and ionic relaxation. Anything smaller is a pilot, not a result.
- The triplet O₂ reference needs `nspin=2, tot_magnetization=2.0` or the
  formation energy is silently wrong.

---

## 4. QMCPACK

### 4.1 Getting orbitals out of QE — three hard requirements

1. **`disk_io = 'medium'` (or higher) in the SCF.** With `'low'` or `'none'`
   QE never writes the wavefunctions and `pw2qmcpack.x` dies with
   `read_file_new: Wavefunctions not in collected format?!?`.
2. **No gamma trick.** `K_POINTS gamma` produces a reduced G-space QMCPACK
   cannot read; the converter says so explicitly:
   *"Using gamma trick results a reduced G space that is not supported by
   QMCPACK ... Please run pw.x with k point 0.0 0.0 0.0 instead of gamma."*
   Use an explicit `K_POINTS crystal` block with `0.0 0.0 0.0 1.0`.
3. The resulting `.pwscf.h5` is **large** (146 MB for a single atom in a
   24-bohr box at 85 Ry). Plan transport: these must not go into git (see §6).

### 4.2 Converting a pseudopotential to QMCPACK XML

`ppconvert` (shipped with QMCPACK) **only reads UPF v1**. Handed a v2 file (what
modern `ld1.x` writes) it aborts on an assertion:

```
ppconvert: NLPPClass.cc:1197: bool PseudoClass::ReadUPF_PP(std::string):
Assertion `parser.FindToken("<PP_HEADER>")' failed.
```

Two routes out, and **only one of them is correct**:

- ✗ `upfconv -c` (UPF → CASINO) then convert. This *reconstructs* semilocal
  channels from the KB beta projectors + pseudo-wavefunctions. It reproduces
  the reference eigenstates exactly and is **multi-hartree wrong off-reference**
  — verified: it gave s(0) = −11.98 Ha where the true channel is −8.40, and
  p(0) = −20.31 vs −9.01. Symptom in production: VMC energies *below* the DFT
  reference by state-dependent amounts (−1.8 / −2.4 / −8.9 Ha), i.e. spurious
  attraction.
- ✓ Generate the semilocal potential natively (§2), parse the true
  `Pseudopot. l=…` blocks, and write QMCPACK's grid XML directly. The target
  schema is simple: `<pseudo version="0.5">` → `<header>` → `<grid>` →
  `<semilocal units="hartree" format="r*V" npots-down="N" l-local="L">` with
  one `<vps l="s|p|d|f|g">` per channel containing a linear-grid `<radfunc>`.
  Sanity check every channel: `r*V → −Z_eff` at large r.

### 4.3 The trial wavefunction and the QMC Hamiltonian must share conventions

QMCPACK evaluates periodic Ewald + neutralizing background. If the DFT that
made the orbitals used `assume_isolated='m-t'`, the two are **different
Hamiltonians** and every energy comparison is meaningless. Symptom: determinant-
only VMC sitting hartrees below the DFT reference with a state-dependent offset.

**The diagnostic that settles it** (cheap, and worth building into any
QMC-from-DFT workflow): run *determinant-only* VMC (no Jastrow, no
optimization) on 2–3 states and compare to the matched DFT energies. Expect a
**small, positive, state-ordered** offset (missing correlation + exchange
treatment) — for Ce ions this was +1.04 / +0.44 / +0.25 Ha across neutral /
3+ / 4+. Anything negative or state-scrambled means a broken card or a
Hamiltonian mismatch, and no amount of Jastrow optimization will fix it.

Three-way discriminator worth remembering:

| symptom | culprit |
|---|---|
| all states off by the same small positive amount | fine — that's physics |
| neutral clean, charged states off | charged-cell / background handling |
| state-*dependent*, especially negative | the pseudopotential card or orbital mapping |

### 4.4 Jastrow optimization on hostile landscapes

For heavy open-f ions, optimizing a Jastrow from a zeros start is a **coin
flip** — measured ~25% success over four attempts on Yb ions with byte-identical
inputs. The gate: after optimization, VMC energy must sit **below** the trial
SCF energy (`VMC ≤ E_SCF`); above it means the Jastrow is worse than useless
and the DMC that follows carries T-move bias.

What works, in order of preference:

1. **More optimization**, not more attempts: 6 loops × 60k samples was
   marginal; 8 × 240k passed a state that had failed four times.
2. **Seed from a converged Jastrow** of a neighbouring charge state — *but see
   the trap below*.
3. Two-stage cost function (variance-first, energy-weighted final loops) if
   both of the above stall.

### 4.5 The `vp.h5` trap (this one is vicious)

When a QMCPACK optimization writes
`<override_variational_parameters href="...vp.h5"/>` into its `.opt.xml`, the
**coefficient arrays printed in that XML are stale display values** — the
actual optimized parameters live in the HDF5 sidecar. Harvesting the XML arrays
to seed another run transplants a *never-optimized* Jastrow. Symptom: the new
run's very first optimization block sits 100+ Ha above the SCF reference,
before the optimizer has done anything.

**Rule: if a `vp.h5` override tag is present, carry the sidecar or re-optimize
— never copy the coefficient arrays.**

### 4.6 Production DMC protocol that has held up

Per state, one input file, sections in order:

1. `<loop max="6..8">` of `method="linear"` Jastrow optimization
   (`MinMethod=OneShiftOnly`, cost 0.1 energy / 0.9 unreweighted variance).
2. VMC with the optimized Jastrow — **this is the gate** (§4.4).
3. A DMC τ ladder, each τ with more blocks than the last:
   0.005 → 0.0025 → 0.00125 → 0.000625 (→ 0.0003125), `nonlocalmoves=yes`.
4. One repeat at a middle τ with `nonlocalmoves=no` — the T-move-off control.

Analysis: discard the first ~25% of blocks, reblock the rest (≈32 blocks),
then a weighted linear fit in τ for the τ→0 extrapolation. The difference
between the T-move-on and T-move-off runs is the **locality shift** — for f
elements this is the number to watch (tens of mHa; the deep f channel is the
stressor).

Walker-count caveat: mixing runs with different walker counts in one τ
extrapolation is legitimate only if you know which points came from which; a
fine-τ high-walker run grafted onto a coarse-τ low-walker ladder from a
*different potential* produced a −1198 eV "gap" in one analysis pass.

---

## 5. Cross-checking two pseudopotential families

If a project has two potentials for the same element tuned to the same
reference data (e.g. a Gaussian-basis semilocal ECP and a plane-wave NC one),
DMC on the same atomic states is a **two-family consistency check** that
neither family can give alone. Worked example: Ce³⁺ → Ce⁴⁺ (the 4th ionization
potential, experimental 36.76 eV) came out at 36.98 eV from the Gaussian-side
open-boundary DMC — a genuine validation datum. The plane-wave side needs
finite-size corrections before it is comparable (see §6).

---

## 6. Practical HPC / repo hygiene

- **Spline `.h5` files are ~140 MB each and must never enter git.** GitHub's
  100 MB hard limit rejects the push *after* the objects transfer, and the
  files land in the repo twice if the job script copies inputs into a run
  directory that is itself inside the repo. Gitignore the pattern
  (`**/*.pwscf.h5`) at the same time you write the job script, not after the
  rejected push.
- Charged cells in periodic QMC carry Makov-Payne-scale artifacts: for a +4 ion
  in a 24-bohr box the leading correction is many eV. Absolute energies are not
  comparable to open-boundary results; differences partially cancel. A box
  ladder (24/32/40 bohr) is the only honest resolution.
- Analysis scripts that pick "the newest run" per system need to know about
  *potential generation*: a stats tool with no notion of which card produced a
  run will silently blend two eras of data. Either version-stamp runs or
  quarantine superseded ones into a sibling directory.

---

## 7. What chemtools could offer here

Ranked by how much pain each removes:

1. **A `qmc_from_dft` workflow validator** — check `disk_io`, k-points,
   isolation scheme, and spin settings *before* the user burns an SCF, and run
   the determinant-only VMC discriminator (§4.3) before any production DMC.
2. **A UPF/psp8/QMCPACK-XML inspector** that reports z_valence, l_local,
   projector count per channel, and the `r*V → −Z_eff` tail check — plus a
   loud warning if a multi-projector potential is about to be used for DMC.
3. **A DMC series analyzer**: reblocking, τ→0 fit, locality shift, and the
   Jastrow gate verdict, from a directory of `.scalar.dat` files.
4. **ONCVPSP input drafting** from an element + core choice, seeded from the
   bundled test inputs, with the rc floor / rc(local) constraints enforced and
   the endemic-ghost calibration built in.
5. A `vp.h5`-aware Jastrow extractor, so seeding a new run from an old one is
   a supported operation rather than a footgun.

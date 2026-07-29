# f-block worked examples

Real input files from the ECPgen reference campaign (2026-06/07), not
synthesized illustrations. Every one of these ran and produced usable output;
where a file encodes a hard-won trick, the trick is called out below and
explained in the sibling notes.

## The baseline table (start here)

- **`fblock-reference-configs.json`** / **`.md`** — 31 elements (Y, La–Lu,
  Ac–Lr), 633 states, all converged in GRASP2018 at DC+Breit level. For each
  state: the occupation line in GRASP/ATSP syntax, the J blocks it actually
  produces (with CSF counts), the converged DC+B energy, and — most valuable —
  **which states cannot be converged cold**. 534 of 633 need a donor start
  guess; 51 need staged orbital birth. That distribution is the single best
  argument for why f-block atomic work needs tooling.

This is the "do not fight the atom again" artifact: given an element and a
target configuration, the table says what worked, what J blocks to expect, and
what seeding the state needs.

## Program examples

### `oncvpsp/`
- `Ce-4f-in-valence.oncvpsp.dat` — 4f-in-valence NC pseudopotential, 46-e core,
  2 projectors per channel. **Encodes the ionized-reference trick** (generation
  config is Ce²⁺ 5s²5p⁶4f¹5d¹, with the neutral demoted to a test
  configuration) and an ox-state test battery (neutral / 3+ / 4+ / f²).
- `Ce-semilocal-for-QMC.ld1.in` — the semilocal sibling for QMC, one projector
  per l. Only viable *because* of the ionized reference (`ld1` rejects two
  wavefunctions with the same l). Note `rcloc` is set explicitly — omitting it
  is an immediate input error.

### `qe/`
- `CeO2-fluorite-eos-point.in` — one point of a 9-point E(V) scan; primitive
  3-atom fluorite cell, 85 Ry, 6×6×6 shifted mesh. Birch-Murnaghan over the
  scan gave a₀ = 5.4545 Å, B₀ = 179.6 GPa (PAW-grade).
- `CeO2-O-vacancy-supercell.in` — the vacancy cell, spin-polarized. Read with
  the caveats in the notes: plain PBE delocalizes the two leftover electrons.
- `Ce-ion-in-box-for-qmcpack.in` — atom-in-box SCF whose orbitals feed
  `pw2qmcpack`. **Three non-obvious requirements are baked in**:
  `disk_io = 'medium'`, an explicit `K_POINTS crystal` gamma point (never
  `K_POINTS gamma`), and no Martyna-Tuckerman isolation (so the DFT and QMC
  Hamiltonians match).

### `qmcpack/`
- `Ce-ion-dmc-production.xml` — the full production chain in one file: 6–8
  linear-method Jastrow optimization loops, VMC (the gate), a four-point DMC τ
  ladder with T-moves, and a T-move-off control at a middle τ.
- `Ce-ion.ptcl.xml` — particle/cell definition for a single ion in a periodic
  box; note the up/down electron counts must match the DFT occupation exactly.

### `dirac/`
- `U-5d-hole-4c.mol` — all-electron 4c uranium with an even-tempered f block
  (24 exponents from 3e7 down). The huge exponent ceiling is what a core-hole
  state needs.
- `U-5d-hole-cosci.inp` — 4c-DC SCF + COSCI for the U 5d⁹5f¹ core-hole
  multiplet. **The `.KPSELE` block is the whole point**: supersymmetry
  selection by κ, telling DIRAC how many spinors of each symmetry are closed,
  open, and in which open shell. Getting a core hole to converge without it is
  not realistic. Even with it, this state fights DIIS — see the DIRAC notes.

### `grasp/`
- A complete production run's stdin for every executable in the chain (Ce3+
  4f1, DC+Breit configuration average), with the four most error-prone answers
  called out. For GRASP the *answer sequence* is the artifact, not an input
  file.

## What is deliberately not here

ATSP2K examples live inside `atsp2k.md` for the same reason — its input is a
13-line stdin sequence whose meaning is the documentation.

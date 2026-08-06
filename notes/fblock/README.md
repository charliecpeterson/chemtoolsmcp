# f-block computational chemistry — field notes for chemtools

Knowledge captured from a 2026-06/07 campaign that built energy-consistent
pseudopotential families for the entire f block: 31 elements (Y, La–Lu,
Ac–Lr), 633 all-electron reference states, and downstream families for
Gaussian-basis QC, plane-wave DFT, and quantum Monte Carlo.

These began as input for chemtools. The table records what has since been
implemented. The organizing claim is simple: f-block atoms break tooling that
works fine for main-group and transition-metal chemistry, often silently, and
the failure modes are specific enough to encode.

## The files

| File | Program | Status in chemtools |
|---|---|---|
| `grasp-fblock.md` | GRASP2018 | 53 tools exist; these are the f-block specifics beyond them |
| `atsp2k.md` | ATSP2K (`hf`/`mchf`) | **no tooling** — this is the spec-in-prose |
| `dirac-fblock.md` | DIRAC | 38 tools exist; these are the actinide lessons on top |
| `qe-qmcpack-oncvpsp.md` | Quantum ESPRESSO, QMCPACK, ONCVPSP/`ld1` | **no tooling, not installed** — read before installing |
| `catalog-state-semantics.md` | the bundled f-block catalog itself | dataset is **current and correct**; these are semantics the records do not carry |
| `grasp-atomic-semantics-audit.md` | GRASP2018 | multireference, excitation, angular-census, ASF-block, and orbital-role contracts are implemented and live-tested |
| `examples/` | all of the above | real inputs that ran, plus the 633-state reference table |
| [`chemtools/data/fblock`](../../chemtools/data/fblock) | GRASP · ATSP2K · DIRAC | **per-element seed library**: 31 elements × every state, including configuration, J structure, converged DC+B energy, and the seeding recipe each state needs. Start here to run a new f-block calculation. |

Start with [`chemtools/data/fblock`](../../chemtools/data/fblock) if you want the payload rather than the
methodology: every validated configuration, its J blocks, its converged
DC+Breit energy, and the seeding recipe it needs. Read the program notes when
something breaks — they are organized by failure mode.

## The five things that generalize

Everything in these notes is an instance of one of these. If chemtools learns
only five lessons from the f block, make it these.

**1. Silent success is the dominant failure mode.** A GRASP SCF that exhausted
its cycle limit exits 0 with a normal-looking output file. A pseudopotential
generator warns about a "ghost" that its own test suite also triggers. A
configuration average silently taken over a truncated J manifold passes every
internal consistency check the pipeline has. In each case the calculation
*reports success*. Exit codes and the absence of error strings are not
evidence; verification has to be positive and independent.

**2. Consistency between two artifacts from the same upstream step validates
nothing about that step.** The J-truncation defect (`grasp-fblock.md` §7)
survived a full production campaign because the level-count check compared two
numbers that both descended from the same truncated input. This generalizes
directly to how a validation tool should be designed: the check must come from
a different derivation than the thing being checked.

**3. Agreement between two runs is one measurement unless their starting
points come from different classes.** Two independently written GRASP inputs
for the same heavy atom agreed to the last digit — both wrong, because both
started from Thomas-Fermi estimates that fall into the same spurious basin.
Cross-validation must vary the *class* of initial guess, not just the author.

**4. Cheap, low-theory cross-checks catch basin errors that expensive ones
miss.** A 9 eV error in a relativistic multiconfiguration calculation was found
by a non-relativistic Hartree-Fock run taking seconds — not because it was more
accurate, but because the *sign* of the implied relativistic shift was
physically impossible. Sign arguments and monotonicity arguments survive
uncertainty about magnitudes; a tool should reach for them first.

**5. Optimizer sentinels must dominate the objective, in both directions.**
A failure return value smaller than a physical objective value lets an
optimizer tunnel into the failure region and converge there. This project hit
both polarities, in two different codes, a month apart. Any tool that wraps a
scientific code in an optimizer needs this rule stated once and enforced.

## The f-block-specific pattern underneath

Open f shells are near-degenerate with the d shell next to them, and both are
near-degenerate with each other across a row. Practically:

- Orbitals must often be **born one at a time**, in a frozen potential, from a
  donor calculation — and two near-degenerate orbitals cannot be born beside
  each other at all. 597 of 633 reference states need seeded orbitals (427 a
  single donor, 110 a two-donor merge, 60 a converted non-relativistic
  calculation); 51 additionally need staged birth. Only 36 start cold.
- SCF **bistability is frequently physical**, not numerical. An oscillating
  open-shell actinide calculation may be reporting a real near-degeneracy —
  the same state converges instantly once the deep core is removed, and then
  gives the wrong coupling pattern.
- Anything that changes the **f occupation** is where transferability dies:
  it is the wall for pseudopotential accuracy, the axis along which fixed-
  valence potentials give up redox chemistry, and the reason plain DFT
  delocalizes defect states in f-element oxides.

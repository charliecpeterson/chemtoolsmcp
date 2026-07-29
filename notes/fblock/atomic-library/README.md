# f-block atomic library — seeds, not documentation

Per-element starting material for GRASP2018, ATSP2K and DIRAC across the whole
f block: 31 elements (Y, La–Lu, Ac–Lr), 633 atomic states spanning neutral
through the closed-shell anchor ion of each element.

The point of this directory is that **f-block atoms are expensive to converge
the first time and nearly free the second time**, provided you know the
configuration, the J structure, and where the starting orbitals come from.
That knowledge is what is captured here. **597 of the 633 states cannot be
converged from a cold start** — only 36 work from the default estimate.

Everything comes from a production campaign, not from worked examples, and the
GRASP numbers are the **v2 references** (rebuilt 2026-07-28 with a
per-configuration J ceiling — the earlier vintage silently truncated the J
manifold of high-f configurations, worth up to 250 meV; see
`../grasp-fblock.md` §7).

---

## Layout

```
grasp/
  fblock-all.json        every element, every state, machine-readable
  <El>.md                per-element recipe table (31 files)
  stdin-examples/        complete answer sequences, one per seeding class
atsp/
  <El>.md                per-element hf configuration inputs (31 files)
dirac/                   the hard actinide jobs
```

### `grasp/<El>.md`

One table per element. For every state: the slug, the occupation line, the
core menu selection, the `rcsfgenerate` active set and 2J range, the J blocks
it actually produces with CSF counts, **how it must be seeded**, and its
converged DC+Breit configuration average (absolute and relative to the
element's anchor ion).

Seeding column vocabulary, in increasing order of difficulty:

| entry | count | meaning |
|---|---|---|
| donor: `<slug>` | 427 | needs a converged donor's orbitals |
| multi-donor merge | 110 | needs **two** donors merged — the orbitals it needs exist in no single converged state |
| ATSP-hf seed | 60 | relativistic estimates land in a spurious basin; seed from a converged non-relativistic calculation |
| cold (Thomas-Fermi) | 36 | works from the default estimate |
| + staged birth `4f-,4f` | 51 | additionally needs the named orbital converged alone in the frozen potential first — **both j components must be listed** |

**29 of the 31 elements set the ATSP-hf seed as their element-level default**
(all but Y and Ce). A state showing a donor in the table still inherits that
default when the donor is unavailable — the per-element file states this in a
banner. Treating a heavy element as cold-startable is how the false vacuum
gets you.

### `atsp/<El>.md`

The `hf` stdin per state: closed-shell string (fixed 4-character fields —
`'  5s  5p'`, not `' 5s 5p'`), open configuration, plus each element's Z,
Z_eff, core size, and its seed state. Where an element has states whose
warm restart reproducibly crashes, the donor pinning is listed.

---

## Reading the stdin examples

`grasp/stdin-examples/` holds five complete runs chosen to cover every seeding
class in the family:

| example | why it is here |
|---|---|
| `Ce-ion3_4f1` | the ordinary case — 2 blocks, 1 CSF each |
| `Th-ion0_6d27s2` | the **false-vacuum** element: Thomas-Fermi converges to a stationary spurious solution ~9 eV high that passes every internal check |
| `U-ion4_5f16d1` | **multi-donor merge**: the 5f/6d near-degeneracy means neither orbital can be born in the other's presence |
| `Tb-ion0_4f85d16s2` | the widest J manifold in the lanthanide row (15 blocks, 2725 levels) |
| `Am-ion0_5f77s2` | actinide mid-row, half-filled f |

**A caution that these files cannot express on their own:** the
`rwfnestimate` stdin for the ATSP-seeded case and the multi-donor case is
byte-identical (`y / 1 / prev.w / * / 3 / *`). The difference is entirely in
how `prev.w` was *built* — converted from a non-relativistic calculation in
one case, merged from two converged donors in the other. The answer sequence
tells you nothing about the strategy; the per-element table does.

---

## `dirac/` — the hard actinide jobs

These are the ones that fought back, and they are worth keeping precisely for
that reason.

- `U-5d-hole-ECP.mol` / `-cosci.inp` — U⁵⁺ 5d⁹5f¹ core-hole multiplet with a
  60-electron ECP. **Converges in 11 iterations** and produces the full
  140-state manifold via `.RESOLVE` (COSCI).
- `U-5d-hole-AE-4c.mol` / `-cosci.inp` — the same state all-electron, 4c.
  **Does not converge, and that is the finding.** Six independent approaches
  fail (restart from parent, two level-shift settings, DIIS-off with damping,
  clean KPSELE short and run to 198 iterations). The open-shell density is
  bistable between two near-degenerate configurations — which *is* the
  jj-coupled doublet the calculation exists to measure. The ECP version
  converges easily because it lacks the degeneracy, and then gives the wrong
  coupling pattern. Keep both files together: the contrast is the result.
- `U-krci-*.mol` / `.inp` — the frozen-orbital CI route: converge the
  closed-shell parent, then CI in a windowed active space. Note
  `**MOLTRA / .ACTIVE energy -10.0 -2.05 1.0`, which windows the integral
  transform **by orbital energy rather than index** — necessary because 6s and
  6p sit energetically between 5d and 5f. Also note the `.mol` header
  `C   1    3  Z  Y  X`, forcing D2h: root selection in linear symmetry needs
  a kernel whose keywords are undocumented and whose selection is silently
  overridden.

Further DIRAC specifics — the fixed-format `.mol` traps, KPSELE partition
tables, the aufbau-collapse diagnostic, and the spin-orbit 2/(2l+1)
convention — are in `../dirac-fblock.md`.

---

## Using this as a seed

For a state already in the library: take its row, generate the inputs, and
seed as the table says. For a state that is *not* in it, the nearest neighbour
in charge and configuration is almost always a viable donor — cross-element
donors work too, because GRASP's orbital records carry their own radial grid,
so a Z mismatch interpolates rather than failing.

Two rules that survive every element:

1. **Birth an orbital only when it is the first of its kind above a closed
   anchor.** Everything else should descend by occupation change from a
   converged neighbour.
2. **Never birth two near-degenerate orbitals beside each other.** Merge two
   donors instead.

# F-block catalog: state semantics the records do not carry

Raised 2026-08-04 from a week of ECP-fitting work against the same GRASP dataset
this catalog bundles (`chemtools/data/fblock/grasp/fblock-all.json`), where
each item below cost real time before it was understood.

Implemented 2026-08-05 in dataset contract v3. Exact lookup now returns
occupancy, transfer risk, and f/d separation; GRASP artifact validation binds
generated CSFs and ASFs to the catalog; generic NWChem and PySCF atomic paths
disclose that occupation control is unavailable. The same audit also found
and corrected 15 charge/slug mismatches in Gd, Lu, Pa, Cm, and Lr. The encoded
GRASP configurations, J/CSF blocks, and energies were already correct.

A second GRASP execution audit now covers multireference and mixed excitation
lists, conditional ASF prompts, labeled block selection, and correlation-
orbital node policy. The results and retained live cases are in
[`grasp-atomic-semantics-audit.md`](grasp-atomic-semantics-audit.md).

**First, the good news, verified rather than assumed.** All 633 states in
the bundled catalog agree with the corrected upstream references to better
than 1 meV, so the dataset carries the per-configuration J-ceiling fix and
not the truncated earlier values. There is no stale science here. Checked
by comparing every `E_rel_au` against the live `reference_energies.json`
for all 31 elements; zero states differ by more than 1 meV.

Everything below is about semantics the records do not express, all of
which are silent-wrong-answer shaped rather than crash shaped.

---

## 1. The `closed` anchors are not ground states at high Z

`Tm ion15_closed` carries `confline = 4d(10,i)5s(2,i)5p(6,i)`, `note =
"anchor, +15"`. That is a well-defined reference configuration and the
energies attached to it are correct. It is **not** the ground state of
Tm XVI: NIST ASD gives `4f⁸ ⁷F₆`. The closed-shell configuration is an
excited state by roughly 14.8 Ha there.

The `(n,i)` inactive markers express the constraint, but only GRASP reads
them. A consumer that takes `config: "closed"` and runs an unconstrained
SCF in any other program converges a different state and gets a plausible
number with no warning. This is the trap that cost the source project
several days; it surfaced only from an external NIST cross-check, not from
anything internal.

Where it applies: the 4f row from Sm (Z=62) onward, the 5f row from Es
(Z=99) onward, with a band of *bistable* elements just below each crossing
(Pm in the 4f row; Np through Cf in the 5f row) whose SCF may land either
way depending on the initial guess.

**Suggested fields**, both derivable from data already in the record:

- `is_aufbau_ground: bool` — whether an unconstrained SCF for this ion and
  charge should be expected to reach this configuration.
- `implied_occupancy: {"s": .., "p": .., "d": .., "f": ..}` — the per-l
  electron count the confline implies, so a consumer can *verify* what it
  converged instead of trusting it.

A lookup that returns "this state requires constraint, here is the
occupancy to check" turns a silent wrong answer into a visible one.

## 2. D2h cannot express these configurations

For any state where an f electron competes with p, D2h `irrep_nelec`
pinning is satisfiable by the wrong configuration: f spans
Au + 2B1u + 2B2u + 2B3u and shares the u irreps with p. A state pinned as
`Au=[1,0]`, `B1u/B2u/B3u=[2,2]` is satisfied equally well by a collapsed f
filling the u irreps while p empties.

Measured in the source project: 15 elements converged their `ion(N)_f1`
state at n_f = 7 instead of 1, and their `ion(N)_d1` state at n_f = 6
instead of 0, while obeying every pinned D2h number. Tm's f1 state sat
11.9 Ha below its own design state that way. The fix is per-(l,m) pinning
under SO3 atomic symmetry, where each component is its own irrep.

This matters here because the MCP drives PySCF, OpenMolcas and NWChem,
none of which read the confline. **Suggested tool**: given a catalog slug,
return the constraint in the target program's own idiom — SO3
`irrep_nelec` for PySCF, the equivalent for Molcas — plus the population
to verify afterwards.

The generalizable rule, worth stating wherever the MCP advises on
constrained SCF: *symmetry-group pinning that looks sufficient can be
satisfied by the wrong configuration. Verify by population, not by
convergence.*

## 3. The f/d crossing is derivable and would make a useful lookup

The catalog already contains both one-electron-outside-noble-core
isoelectronic series. Taking the f¹→d¹ configuration-center separation:

| series | ions | range | slope | crossing |
|---|---|---|---|---|
| Cs-like (55 e⁻) | La III … Lu XVII | −1.86 → +170.83 eV | 12.34 eV/element | La III → Ce IV |
| Fr-like (87 e⁻) | Ac III … Lr XVII | −3.67 → +99.45 eV | 7.37 eV/element | Ac III → Th IV |

The 5f crossing is ~40% shallower (slope ratio 0.597), which is why the
bistable band is one element wide in the 4f row and six in the 5f row.

A lookup answering "for this ion, is f above or below d, and by how much"
lets an agent predict *before* running whether a state needs constraining,
rather than diagnosing it from a failed or wrong SCF afterwards.

## 4. Smaller items

**Slug grammar is ambiguous under the obvious parser.** `5f16d17s2` is
5f¹ 6d¹ 7s², but a greedy `(\d)([spdf])(\d+)` reads it as 5f¹⁶. This broke
an audit script in the source project and produced a nonsense comparison
table before it was noticed. Either publish a parser alongside the
catalog, or add an explicit `occupancy` field and let consumers stop
parsing the slug at all. The non-greedy form that works:
`(\d)([spdf])(\d+?)(?=(?:\d[spdf])|$)`.

**No external cross-check field.** NIST ASD has ground configurations for
many of these ions and is the only external check available on the
catalog's physics. A `nist_ground_config` field (with the access date)
would let a consumer see at a glance where a catalog state diverges from
the measured ground state — which is finding 1, made self-evident.

**Reproducibility expectations are unstated.** Downstream artifacts built
from these references reproduce to ~5–6 significant figures, not bit
identically, because SCF `conv_tol` leaves contraction coefficients moving
in the 7th–8th digit. Worth saying wherever the MCP hands back derived
numbers, so a user diffing two runs does not report it as a defect.

---

## Priority, if it helps

1 and 2 are the same failure mode at two layers and are the ones that
produce wrong numbers rather than errors. 3 is genuinely useful and
cheap since the data is already present. 4 is hygiene.

Source material: `ECPgen/notes/26-closed-anchor-4f-collapse.md` (the
decision record for the closed anchors) and `ECPgen/notes/27-paper-values.md`
(the measured values, including the isoelectronic table and the NIST
comparison).

# GRASP2018 for f-block atoms — reference-quality DC+Breit energies

chemtools already ships 37 GRASP tools covering the multi-executable workflow
and an hf-bootstrap for high Z. These notes record what a 2026-06/07 campaign
learned producing **633 converged states across 31 f-block elements** (Y,
La–Lu, Ac–Lr) — the exact answer sequences, how ASFs and configuration averages
are really specified, and the failure modes that make lanthanides and actinides
different from everything else.

Validated per-element configurations, J blocks and seeding requirements:
`examples/fblock-reference-configs.{json,md}`. **534 of those 633 states cannot
be converged from a cold start**, which is the whole argument for tooling here.

Physics invariants used throughout: static Fermi nucleus, configuration average
over all levels of every J block with (2J+1) weights, Dirac-Coulomb + Breit via
the `rci` transverse photon in the low-frequency limit, no QED, no mass shifts.

---

## 1. The chain

```
rnucleus      -> isodata
rcsfgenerate  -> rcsf.out (copy to rcsf.inp)      + the block table on stdout
rangular      -> mcp.30 … mcp.39
rwfnestimate  -> rwfn.inp
rmcdhf        -> rwfn.out, rmcdhf.sum             (DC)
   copy rcsf.inp -> ref.c, rwfn.out -> ref.w
rci           -> ref.csum                          (DC + Breit)
```

`jj2lsj`, `rlevels`, `rhfs`, `rtransition` are not needed for reference
energies. Save each program's stdin next to its output — a run directory that
cannot tell you what was answered is unreproducible six weeks later.

---

## 2. Answer sequences, decoded prompt by prompt

### rnucleus

```
90        atomic number
232       mass number (0 = point nucleus)
n         revise default RMS radius / skin thickness?
0         mass of neutral atom in amu — 0 means STATIC nucleus
0.5       nuclear spin
1         nuclear dipole moment
1         nuclear quadrupole moment
```

The `0` for atomic mass is load-bearing: it makes the nucleus static, which is
what matches a fixed-nucleus non-relativistic Hamiltonian on the other side of
a comparison. The last three answers do not affect energies but must be given.

### rcsfgenerate

```
*                                        ordering (*/r/s/u)
5                                        core: 0 none 1 He 2 Ne 3 Ar 4 Kr 5 Xe 6 Rn
4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)    configuration
                                         blank line ends the list
6s,6p,5d,5f                              active orbitals
1,21                                     2*J range: lower, higher
0                                        number of excitations
n                                        generate more lists?
```

Four traps:

1. **`,i` markers are mandatory** — `i` = inactive (fixed occupation). Without
   them the shell opens to excitations and the CSF count explodes.
2. **The active-orbital list must be the highest n per l, in s,p,d,f,g order.**
   Giving it in configuration order (`3d,4s,4p` instead of `4s,4p,3d`) silently
   drops shells — this cost 150 Ha once, in a run that reported success.
   ```python
   per_l = {}
   for sh in re.findall(r"(\d[spdfg])", confline):
       per_l[sh[1]] = max(per_l.get(sh[1], 0), int(sh[0]))
   orbs = ",".join(f"{per_l[l]}{l}" for l in "spdfg" if l in per_l)
   ```
3. **The 2*J range must have the right parity** (odd totals for odd electron
   counts). GRASP re-prompts on a mismatch, which in piped stdin shifts every
   later answer by one line.
4. **The upper limit must actually cover the manifold.** See §7 — getting this
   wrong is silent, self-consistent, and was live in production data for weeks.

Everything below the ECP valence is carried **inactive in the peel**, never as
`c`losed, so those orbitals remain available for variation: Ln use core Kr with
prefix `4d(10,i)5s(2,i)5p(6,i)`; An use core Xe with
`4f(14,i)5d(10,i)6s(2,i)6p(6,i)`.

Parse the block table off stdout — it is the contract for every later step:

```python
re.finditer(r"^\s*\d+\s+(\S+)[+-]\s+(\d+)\s*$", stdout, re.M)   # (J, ncsf)
```

### rangular

```
y
```

### rwfnestimate

Cold start (Thomas-Fermi):
```
y
2
*
```
Seeded from converged orbitals:
```
y
1
prev.w      file to read
*           which orbitals to take from it
3           screened-hydrogenic for whatever is left
*
```
That trailing `3 / *` **seed-fill is essential** — without it, orbitals the
donor lacks are left unestimated. A known historical bug was answering `2` for
`*` after building a seed, silently discarding it.

Because `rwfn` records carry their own radial grid, **cross-element donors work**
(the Z mismatch interpolates). That is what makes neighbour-chaining viable.

### rmcdhf

```
y            default settings
1-5          ASF serial numbers, block 1
1-3          ASF serial numbers, block 2      (one line per block)
5            level weights: 1 equal, 5 standard (2J+1), 9 user
*            orbitals to be varied
*            which are spectroscopic
100          maximum SCF cycles
```

Three things worth knowing:

- **The weights prompt only appears when more than one level is selected.**
  With a single level GRASP skips it. A driver that always sends the weights
  line shifts every subsequent answer. Mirror the condition:
  `weights = "5\n" if nlevels > 1 else ""`.
- **rmcdhf is always EOL in GRASP2018** — the "(E)OL type calculation?" prompt
  is hard-wired. "Configuration average" here means *selecting all levels of
  every block with (2J+1) weights*, not a separate code path.
- **Spectroscopic orbitals: `*` for single-configuration reference runs.**
  Leaving it blank marks them as correlation orbitals, which turns off node
  checking — worth 0.17 Ha of silent error on one 4d state. Blank is correct
  only for genuine correlation orbitals.

### rci

```
y        default settings   (also suppresses the per-block iccut prompt)
ref      state name -> ref.c, ref.w, ref.csum
y        include H (Transverse)?          <- Breit
y        modify all transverse photon frequencies?
1.d-6    scale factor                     <- low-frequency (Breit) limit
n        vacuum polarisation
n        normal mass shift
n        specific mass shift
n        estimate self-energy
1-5      ASF serial numbers per block (same selection as rmcdhf)
```

Breit is worth having: at Z = 90 it is ~35 Ha, and it removes a systematic from
cross-element comparisons for almost no cost at single-configuration sizes.

**A stale `rci.res` changes the prompt sequence** (rci asks about restarting),
so wipe a state directory before re-running rather than reusing it.

---

## 3. ASF selection

The block table from `rcsfgenerate` defines everything. After it, GRASP asks
once for ASF serial numbers and then reads **one line per block**.

Accepted syntax: singles and ranges, `1`, `2, 5, 7-10`. Rejections re-prompt
(fatal for piped stdin): undecodable input, out-of-range serial numbers, and
descending ranges like `8-3`. **An empty line selects nothing in that block**;
if every block is empty GRASP restarts the whole block loop.

Three selection idioms:

| Purpose | Selection | Notes |
|---|---|---|
| Configuration average | `1-N` per block, weights `5` | all levels, (2J+1) weights |
| Term-specific | narrow the `rcsfgenerate` 2*J range to the term's J span, then `1` per block | assert the block J set is exactly the expected span |
| Single term, state-specific | 2*J range = only the ground-term J → one block → optimize its lowest level | the cleanest way to get a term energy |

**"Lowest level per block" is not a state selector.** Two costly instances:
adding an extra orbital to an active set let `rcsfgenerate` build
singly-excited CSFs of a *lower-lying* configuration, and "lowest level" then
silently returned a different physical state (−18.7 mHa error). The actinide
analogue flipped a J=4 root to 74% of the wrong configuration while neighbouring
J stayed clean — **J-resolved deltas are the intruder detector**. If the space
can contain intruders, prune the CSF list explicitly.

---

## 4. Configuration average vs term-specific — pick deliberately

Configuration averages are the right quantity when the *other* side of a
comparison is also a configuration average (e.g. fitting to a non-relativistic
config-average code). They are the **wrong** quantity for anything about term
structure: they wash out term-specific exchange stabilization and produce
sign-flipped or compressed gaps versus experiment.

Three ways to get a term energy, only one valid:

1. Fixed-spin CAS in a molecular code — **wrong**: spin pinning prevents the
   competing configuration from even being represented.
2. Lowest level of a config-average run — **invalid**: the orbitals are
   optimized for the average, not the term. This misassigns ground
   configurations across half a row.
3. State-specific MCDHF with a narrowed J range — **valid**, and still not
   sufficient: reference-level MCDHF gave a lanthanide f–d gap with the right
   magnitude and the **wrong sign** versus experiment, because it misses
   differential f–f vs f–d correlation. Trust MCDHF's sign only for large gaps.

Also: **`rci` across unequal CSF spaces is apples-to-oranges.** Comparing two
configurations whose reference spaces differ by an order of magnitude (38 vs
326 CSFs) produced a 6 eV disagreement with the MCDHF gap — intra-configuration
mixing lowers the larger space far more.

---

## 5. Convergence: the trap that must be tooled

> **`rmcdhf` exits 0 whether its SCF criterion was met or it simply ran out of
> cycles.**

No error, no non-zero status, a normal-looking output file. Count iteration
blocks in the stdout (the `Subshell / Energy / Method` header repeats once per
iteration) and raise if the count reached the maximum.

Do **not** guard on the per-orbital consistency column instead: a correct
diffuse outer orbital pins consistency near 1 through the same numerical
artifact as a pathological one, so that guard produces only false positives.

---

## 6. Failure modes specific to the f block

### 6.1 The false vacuum (the worst one)

At Z ≈ 90 and above, Thomas-Fermi and screened-hydrogenic estimates converge to
a **stationary spurious solution ~9 eV above the true ground state** that passes
every internal check: converged in 5–8 iterations, exit 0, stationary to eight
digits, plausible intra-configuration splittings.

Five independent checks all failed to catch it:

- No solver failure — in fact the *crashing* states were the healthy ones,
  refusing bad estimates. The clean run was the wrong one.
- Per-orbital consistency looked fine (see §5).
- Path independence: two seed chains agreed to 1e-7 Ha — vacuously, since both
  descended from the same Thomas-Fermi start.
- Cross-validation against an independently written input agreed to the last
  digit. **At heavy Z, everyone's TF-started run agrees on the same wrong
  number.**
- Physical plausibility of one run: the whole configuration is displaced
  together, so internal splittings look right.

**What actually found it:** a *cluster* of same-direction cross-configuration
anomalies, then the decisive cheap test — **compare configuration-average
differences against a non-relativistic calculation and check the sign of the
implied relativistic shift.** Relativity stabilizes s at high Z, so removing an
outer s electron must cost *more* relativistically. The false basin implied the
wrong sign, which no uncertainty about magnitudes can excuse.

**Fingerprint (usable post hoc):** the outer s orbital pinned at consistency
≈ 1 with its sign-flip flag set every iteration while every other orbital sits
at ~1e-5, plus an outer-s eigenvalue about a factor of two too shallow.

**Fix:** converge the state first in a non-relativistic atomic code, convert
those orbitals, and seed. One byte-level detail: the converter wants the
orbital-label digit **right-justified** in a 3-character field where the
non-relativistic code left-justifies — patch it or shells get mis-assigned.

**Rule worth burning in:** *never accept agreement between two runs as
validation unless their starting estimates come from different classes.*
Same-class agreement is one measurement, not two.

Scope: clean at Z=39 and Z=58 (<0.1 meV). Treat everything from the third
transition row upward as check-first.

### 6.2 The converter Z-ceiling

Beyond Z ≈ 94 the non-relativistic→relativistic orbital converter corrupts
records — the underlying calculation is fine, the converted 1s arrives broken.
It works at Z=62 for the same charge, so it is a Z ceiling, not a charge
ceiling. Above it, the only route is neighbour-donor chains. (One element below
the ceiling, Z=91, is also converter-hostile.)

### 6.3 `IMPROV` "Convergence not obtained"

This is an error stop **before the first iteration**: a bad starting shape, not
too few cycles. More iterations never helps; a better donor does.

The structural instance: both heavy-row chains failed at the *same* state — the
charge-2 f^(n−1)s¹ configuration — because a graded-charge battery has no nearby
s donor (its own s state sits at high charge with a far-too-tight orbital) and a
screened-hydrogenic 7s at Z=97 never brackets. Staged variation alone did not
save it; the fix was donating the **previous element's** converged charge-2 s
orbital and staging that.

### 6.4 Node hunts and staged birth

Warnings to recognize: `difficulty with node-counting procedure`, `lower bound
on energy exceeds upper bound`, `Method N unable to solve for <nl> orbital`,
followed by node counts and oscillation signs.

Rules distilled from ~120 element-batteries:

- **Birth an orbital only when it is the first of its kind above a closed
  anchor**, and then stage it: vary only that orbital in the frozen seeded
  potential first (`5f-,5f` — **both j components must be listed**), then
  release everything with a warm start from that pass.
- **Never birth f and d beside each other.** At uranium the 5f/6d near
  degeneracy forbids it in both directions. The fix is a **multi-donor seed**:
  merge the d orbital from one converged state and the f from another.
- **A p birth needs a relativistic donor.** Thomas-Fermi gives one shape for
  both j components and the j-resolved solve dies even when the radial size is
  right. Birth it at a higher charge state where binding is forgiving, then
  donate the converged j pair down.
- **Staged variation of a single orbital in a single-CSF state crashes rmcdhf
  at input.** Use a plain donor seed for those.
- Barely-bound Rydberg states may simply be unreachable; replace the probe
  rather than fighting it.

### 6.5 Multi-donor orbital merging

`rwfn` is a sequence of Fortran-framed records: a header, then orbital triples.
Merging is ~30 lines — take the header and all triples from the first donor,
then add triples from later donors whose (n, κ) is not already present, first
file winning on duplicates. This single capability is what makes mixed-occupancy
f/d states reachable.

### 6.6 Use the serial build

The MPI build of `rmcdhf` crashes on diffuse outer s orbitals in the
neutral-heavy, small-reference regime — i.e. every neutral actinide.

---

## 7. Case study: a silent truncation that survived a full campaign

Worth reading as a design lesson for validation tooling, because it was found
*after* 633 states were in production use.

The driver hard-coded the `rcsfgenerate` 2*J range as `1,21` / `0,20`.
Configurations whose J manifold extends past 21/2 are **silently truncated** —
the high-J blocks are simply never built, so the configuration average is taken
over a partial manifold.

Verified by re-running `rcsfgenerate` on a shipped configuration with a wide
range, everything else identical:

| state | as shipped | full manifold | missing |
|---|---|---|---|
| Tb 4f⁸5d¹6s² | 11 blocks, top 21/2, 2666 levels | 15 blocks, top 29/2, 2725 | 59 levels = **4.9% of (2J+1) weight** |
| Am 5f⁷7s² | 11 blocks, top 21/2, 323 levels | 13 blocks, 327 | **2.9% of weight** |

210 of 633 states carry the signature (exactly 11 blocks with the top block
sitting exactly at the cap), concentrated in mid-to-late f occupations with a d
companion.

**Why no check caught it:** the driver validated that the number of levels
found equalled the number of levels *predicted from the same truncated
`rcsfgenerate` output*. The pipeline was perfectly self-consistent and wrong.

**Lessons for a validation tool:**

1. A consistency check between two artifacts derived from the same upstream
   step validates nothing about that step.
2. Any hard-coded numeric limit in generated input is a silent-truncation
   risk. The J range should be **derived from the configuration** (or set
   unreachably wide), and a tool should flag "top block sits exactly at the
   requested limit" as suspicious — that one heuristic finds this class of bug
   instantly.
3. Truncation of this kind does **not** cancel in energy differences when the
   reference state is unaffected, which is exactly the case for an f⁰ anchor
   versus f-rich states.

---

## 8. Energy extraction

Levels appear under `Eigenenergies:` with the Hartree value in column 4 in
Fortran `D` notation:

```
Level  J Parity       Hartrees              Kaysers                eV
  1   5/2 -   -2.64745696394063D+04 -5.81049641229232D+09 ...
```

```python
sections = text.split("Eigenenergies:")[1:]
if from_rci:
    sections = sections[0::2]      # see below
for s in sections:
    s = re.split(r"Energy of each level|Weights of major|Self Energy", s)[0]
    for m in re.finditer(r"^\s*\d+\s+(\d+)(/2)?\s+[+-]\s+(-?[\d.]+D[+-]\d+)", s, re.M):
        twoj = int(m.group(1)) if m.group(2) else 2 * int(m.group(1))
        w = twoj + 1
```

**`rci` prints two tables per block** — the second includes an estimated
self-energy *even when QED was declined*. At Z=90 they differ by **31 Ha**.
Keep the even-indexed sections. Truncating each section before the
`Energy of each level` / `Weights of major contributors` tables also prevents
level-difference lines being read as energies.

Store the Hamiltonian description alongside the numbers so provenance travels
with the data, and record both the DC and DC+Breit values.

---

## 9. What chemtools could add

Ranked by damage prevented:

1. **`rmcdhf` convergence guard** reading the stdout termination reason, not
   the exit code (§5).
2. **A seeding manager**: donor selection, `rwfn` merging, staged-birth
   orchestration, and the seed-fill answer pattern. Without it two thirds of
   f-block states are unreachable.
3. **A false-vacuum detector**: the pinned-outer-s fingerprint, the
   non-relativistic sign check, and the different-start-class validation rule
   (§6.1).
4. **An input-limit auditor**: flag any generated input whose result sits
   exactly at a requested limit (§7). Cheap, general, and it catches a class of
   bug that survives every other check.
5. **ASF/block consistency checking** across `rcsfgenerate` → `rmcdhf` → `rci`,
   including the conditional weights prompt.
6. **A DC+Breit preset** emitting the low-frequency transverse-photon chain plus
   its Hamiltonian description string.

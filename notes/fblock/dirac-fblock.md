# DIRAC for f-block atoms — lessons beyond the current tooling

chemtools already ships 38 DIRAC tools including AOC and KPSELE support. These
notes cover what a 2026-06/07 actinide/lanthanide campaign learned *on top of*
that: the fixed-format traps that produce unhelpful errors, what KPSELE does and
does not fix, the open-shell core-hole wall, and the diagnostics that
distinguish "converged to the wrong state" from "converged."

Working examples: `examples/dirac/U-5d-hole-4c.mol` and
`U-5d-hole-cosci.inp`.

---

## 1. Invocation and resource traps

```bash
pam --inp=cosci --mol=u --mpi=1 --noarch --mw=1500
```

Output lands in **`<inp>_<mol>.out`** (`cosci_u.out`), checkpoint in
`<inp>_<mol>.h5`. DIRAC 26 uses `CHECKPOINT.h5`, not the older `DFCOEF`;
restart by copying the parent's `.h5` to `CHECKPOINT.h5` and passing
`--put=CHECKPOINT.h5`.

**`--mw` is a hard allocation, not a ceiling.** An int64 build reserves the
full workspace up front at 8 bytes/word: `--mw=8000` is 64 GB and will exceed a
40 GB container limit before any physics happens. Working values for atoms:
1200–1800. This is worth a preflight check in any submission tool — the failure
mode is an allocation abort that looks like a machine problem.

Convergence test: the string `Convergence after` in the output.

---

## 2. `.mol` fixed-format traps

The `.mol` parser is 1990s fixed-format Fortran and its errors do not name the
column that broke.

| Trap | Symptom | Fix |
|---|---|---|
| Header keyword | parse failure | First line must be `DIRAC` (not `INTGRL`) |
| Angstrom flag | silently wrong units | The `A` must sit at **column 20** of the symmetry card |
| **Charge field** | `Error reading line 5` | Line 5 is `<charge>. <n_atoms>` in a **fixed-width field**. A three-digit Z (Lr, Z=103) overflows it. Right-align: `f"{f'{Z}.':>10}    1"` |
| Comments in basis blocks | breaks `convert4qmc -dirac` | That parser finds blocks by searching for `"f "`, so a `# f functions` comment is caught. Keep basis blocks comment-free. |

Symmetry card options that matter:
- `C   1              A` — auto-detect (a lone atom becomes linear D∞h).
- `C   1    3  Z  Y  X` — **force D2h.** This is the escape hatch for several
  CI blockers (see §5).

### Even-tempered pools vs library bases

Pools that worked for SCF-level gates (ratio 1.8–1.9, per-l `(top, n)`):

| Case | s | p | d | f |
|---|---|---|---|---|
| U all-electron 4c | 3e7 / 24 | 5e5 / 20 | 1e4 / 18 | 2e2 / 14 |
| Ce all-electron 4c | 5e7 / 28 | 1e6 / 22 | 2e4 / 18 | 1e3 / 18 |
| U with 60-e ECP | 5e5 / 26 | 1e5 / 24 | 2e4 / 20 | 8e3 / 16 |

All-electron tops reach ~5e7 to cover the 1s **small-component** tights;
ECP pools start ~5e5 because the deep core is gone.

**But a hand-rolled ET pool is not good enough for a CI reference.** One left
U 5s/6s *unbound* (+19 and +178 Ha) — usable for an eigenvalue gate, useless
underneath a CI. Switching to `LARGE BASIS dyall.cv3z` bound everything (1s
−4284 … 6s −4.60 Ha) and moved the total by ~470 Ha. **Rule: verify every
orbital energy is negative before building CI on a parent SCF.**

---

## 3. `.KPSELE` — atomic supersymmetry

The block that makes open-shell f-element SCF tractable:

```
.KPSELE
7
 -1 1 -2 2 -3 3 -4          <- kappa list: s½ p½ p3/2 d3/2 d5/2 f5/2 f7/2
 12 10 20 8 12 6 8          <- closed-shell spinors per kappa
 0 0 0 4 6 0 0              <- open shell 1 (here: 5d, 10 spinors)
 0 0 0 0 0 6 8              <- open shell 2 (here: 5f, 14 spinors)
```

One row per open shell, in the same order as `.OPEN SHELL`. Consistency check a
tool should enforce: the closed row must sum to the total closed electron
count, and `.CLOSED SHELL` (which is per-fermion-irrep, gerade then ungerade)
must match the same partition — for the row above, g = 12+8+12 = 32 and
u = 10+20+6+8 = 44.

Worked partitions:

| System | closed row | `.CLOSED SHELL` |
|---|---|---|
| U⁶⁺ [Rn] parent (AE) | `12 10 20 12 18 6 8` | `42 44` |
| U⁵⁺ 5d⁹5f¹ (AE) | `12 10 20 8 12 6 8` | `32 44` |
| Ce⁴⁺ [Xe] parent (AE) | `10 8 16 8 12 0 0` | `30 24` |
| Ce 4d⁹4f¹ (AE) | `10 8 16 4 6 0 0` | `20 24` |
| Either, with a 28/60-e ECP | `4 4 8 0 0 0 0` | `4 12` |

`.OPEN SHELL` syntax is `<n_electrons>/<n_gerade>,<n_ungerade>` — so `9/10,0`
is nine electrons in a ten-spinor gerade shell.

### What KPSELE fixes, and what it does not

**Fixes:** occupation rearrangement between kappas. Concretely, a D2h SCF
*without* KPSELE relaxed a [Rn] closed shell into a lower f-occupied
configuration (−28057.9 vs −28052.6 Ha) — a legitimate lower state, and the
wrong one. KPSELE pins the intended configuration.

**Does not fix:** near-degenerate final states. Occupation-pinning per kappa
cannot choose between two near-degenerate radial/angular solutions of the same
occupation. See §5.

---

## 4. The aufbau-collapse trap (and the multiplicity diagnostic)

At anchor charges ≳ +7, f-collapse puts 4f/5f **at or below** the valence p, so
a `.CLOSED SHELL` specification intended as an f⁰ anchor converges instead to a
legitimate, jj-closed, *f-occupied* state. Nothing errors. Downstream property
gates then read as catastrophic nonsense (one sweep produced a "0.000 eV"
semicore splitting and a "+223%" error) and the natural conclusion — "the
potential is broken" — is wrong.

**The diagnostic is degeneracy multiplicity, never positional indexing:**

```python
def groups(evs, tol=1e-6):
    out = []
    for e in sorted(evs):
        if out and abs(e - out[-1][0]) < tol: out[-1][1] += 1
        else: out.append([e, 1])
    return out
# gerade multiplicities [1,2,3,1] + ungerade [1,2] == the intended aufbau
```

A triple-degenerate ungerade group (to 11 digits) is an f₅/₂⁶ shell, not a p
shell. **No exact degeneracies at all means a symmetry-broken p/f hybrid** —
which is what elements sitting near the crossing (Pm, Np, Pu, Am) produce.

Workaround for high anchors: gate on the **valence-p-stripped closed ion**
(anchor+6, e.g. `[5s²5p⁶5d¹⁰6s²]`, `.CLOSED SHELL 14 6`), where aufbau is
unambiguous because f sits ~6 Ha above the deepest hole.

A second fingerprint, for AOC probe pairs: `E_avg` must lie *between* the
J-resolved hole energies, with `E_{1/2} = 3·E_avg − 2·E_{3/2}` holding exactly.
If it doesn't, the SCF landed on an f-occupied solution.

---

## 5. The open-shell core-hole wall

Getting an all-electron 4c SCF for a core-hole state like U⁵⁺ 5d⁹5f¹ is, as of
this campaign, **unsolved with the standard controls**. Five approaches, all
failing the same way:

| Approach | Outcome |
|---|---|
| Restart from converged closed-shell parent | 5f overlap oscillates 0.06 ↔ 0.99 |
| `.LSHIFT 0.3` / `.OLEVEL 0.2` | damped to a 0.15–0.5 limit cycle |
| `.LSHIFT 0.6` / `.OLEVEL 0.5` | no better |
| `.NODIIS` + `.DAMPFC 0.85` | stalls in a limit cycle around iteration 50 |
| Clean KPSELE (atomic start, no restart, no shifts) | still oscillates |

DIRAC's own message during this is unusually diagnostic and worth surfacing
verbatim in any recovery tool:

```
DIIS aborted because of the last two iterations
the lowest energy has the largest gradient and DIIS minimizes gradient !!!
```

**The interpretation is physics, not numerics.** The open-shell density is
*bistable* between two near-degenerate configurations, and that near-degeneracy
is exactly the jj-coupled doublet the calculation exists to measure. Supporting
evidence: the same state with an ECP core converges in **11 iterations**,
precisely because removing the deep core removes the near-degeneracy — and the
ECP result then shows the *wrong* (Landé-spaced) coupling pattern.

So: an f-element open-shell SCF that oscillates forever may be reporting a real
degeneracy. Useful diagnostic when there are no eigenvalues to print — track
the open-shell overlap across iterations (`DVOVLP( 2) = ...`) and report the
trajectory rather than just "not converged."

Escape routes that remain: frozen-orbital CI in the converged closed-shell
parent (initial-state approximation), or a genuinely j-resolved potential.

---

## 6. CI (COSCI / KRCI) setup

**COSCI** is nearly free once an AOC SCF converges: add `.RESOLVE` under
`**WAVE FUNCTION` and DIRAC diagonalizes the complete open-shell CI, printing
one `Eigenvalues` block per boson irrep.

```python
evs = []
for b in re.findall(r"Eigenvalues((?:\s+-\d+\.\d+)+)", txt):
    evs += [float(x) for x in re.findall(r"-\d+\.\d+", b)]
evs.sort()
rel = [(e - evs[0]) * 27.211386 for e in evs]
# then group by degeneracy at ~2 meV to recover term multiplicities
```

**KRCI** needs more care:

- Root selection in *linear* symmetry requires a kernel whose keywords are
  undocumented, and requests for it are silently overridden. **Force D2h**
  (`C   1    3  Z  Y  X`) and use ordinary `.CIROOTS` on point-group irreps.
  Multiplet energies are symmetry-independent; only the labels change.
- Window the integral transform **by orbital energy, not index**:
  `**MOLTRA / .ACTIVE / energy -10.0 -2.05 1.0`. For uranium this cleanly
  separates 5d/6s/6p/5f without needing explicit MO reordering — which matters
  because 6s and 6p sit energetically *between* 5d and 5f.
- `.GAS SHELLS` lines are `<min_accum> <max_accum> / <n_gerade> <n_ungerade>`;
  setting min = max at every space reproduces a COSCI-sized space exactly.
- In D2h, DIRAC's boson irrep order is Ag1 B1u2 B2u3 B3g4 B3u5 B2g6 B1g7 Au8;
  ungerade states of a d⁹f¹ configuration live in irreps 2, 3, 5, 8.

Known unresolved blocker: the CI module can allocate, print its banner, and
then exit **without producing root energies and without an error** — setup is
accepted but the diagonalization output never appears. Suspects are the
MOLTRA-window→active-space handoff, rigid min=max GAS spaces, and input block
ordering.

---

## 7. Using an ECP inside DIRAC

Only in `.ECP` mode (non-relativistic kinetic + averaged relativistic potential
+ optional spin-orbit blocks). **Putting an energy-consistent relativistic ECP
into a full Dirac solver double-counts relativity.** The same rule forbids
DKH/X2C in other codes and rules out ANO-RCC-type bases.

The ECP block goes in the `.mol` file before `FINISH`:

```
ECP 60 5 3          <- n_core, lmax+1, n_SO_blocks
# ul
4
  1  8.00000000  32.00000000
  ...
# p-so
2
  2  1.72854738  3.77279498
```

**Spin-orbit convention conversion:** DIRAC (and NWChem) want the coefficients
premultiplied by **2/(2l+1)** — p × 2/3, d × 2/5, f × 2/7 — where Molpro-style
cards carry that factor in the operator. Getting this wrong scales every
splitting and is invisible in a single-code test.

Validation the tooling should know about: two independent 2c codes agreed to
0.5 mHa on a 4p₁/₂–4p₃/₂ splitting (60.1 vs 59.6 mHa), which makes this a good
cross-code regression target.

---

## 8. Reference results worth keeping as regression targets

- **Ce³⁺ 4d core-hole multiplet:** an energy-consistent 28-core ECP with SO
  reproduced all-electron 4c literature values to **<0.06 eV**
  (0.000/0.591/1.767 vs 0.00/0.57/1.71 eV, degeneracies 1/3/5 = ³P₀,₁,₂),
  across 140 states spanning 42 eV.
- **U⁵⁺ 5d core-hole:** the same approach gives 0/1.00/2.83 eV obeying the
  Landé interval rule, where all-electron reference is jj-compressed at
  0/0.1 eV. **Doubling the spin-orbit strength widens rather than compresses**
  (0/1.32/3.50) — proof that the discrepancy is a per-j *radial* representation
  limit, not a spin-orbit magnitude error.
- Corollary worth stating in any tool that offers "j-resolved" potentials:
  algebraically splitting an averaged potential plus SO term into per-j
  channels is **information-null** — the solver already forms exactly that
  internally. A real j-dependent potential must be fitted independently to
  j-resolved reference spinors, energies *and* radii.

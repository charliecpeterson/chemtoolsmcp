# ATSP2K (`hf` / `mchf`) — a program chemtools does not support yet

ATSP2K is Froese Fischer's non-relativistic atomic structure package. It has no
MCP tooling today; these notes are the spec-in-prose for adding it. Everything
here comes from a 2026-06/07 campaign that used a **modified** ATSP2K (a ported
`libecp`, so `hf`/`mchf` can run with an effective core potential) as the
fitting engine for f-block pseudopotentials.

Why it is worth supporting: `hf` in config-average mode is a **grid-based**
atomic solver, so it has no basis-set incompleteness error, and a single SCF
takes ~0.1 s. That makes it the only practical inner loop for an optimizer that
needs thousands of atomic energies. The cost is that it is a 1980s interactive
Fortran program with a stdin protocol full of traps.

---

## 1. Invocation

`hf` is driven **entirely by stdin, one line per prompt**, and reads/writes
files relative to the working directory. Every state therefore needs its own
directory.

```python
(d / "ecp.inp").write_text(card)          # optional; presence = ECP mode
stdin = (f"{el},AV,{Z}.\n{closed}\n{cfg}\nall\ny\n"
         f"n\ny\ny\nn\n99 2\ny\nn\nn\n")
subprocess.run([HF], input=stdin, text=True,
               capture_output=True, cwd=d, timeout=300)
```

Files in the run directory:

| File | Role |
|---|---|
| `ecp.inp` | ECP card. **If present, ECP mode activates automatically** — no flag, no prompt. |
| `wfn.inp` | Starting orbitals (unformatted). Absent → screened-hydrogenic estimates. |
| `wfn.out` | Converged orbitals, always written. |
| `hf.log` | The parseable output. |

Prompts go to stderr, results to `hf.log`. Capturing stdout tells you nothing.

---

## 2. The stdin answer sequence, line by line

Thirteen lines for a standard config-average call. Each line answers one
prompt, and **the order is not negotiable**:

| # | Line | Meaning |
|---|---|---|
| 1 | `Ce,AV,58.` | ATOM label, TERM, Z. `AV` = **configuration average** — this is what pairs with GRASP EAL config averages. The trailing `.` matters (read as `F3.0`). |
| 2 | `  5s  5p` | Closed shells, **fixed 4-character fields**: one blank + 3-char label. `' 5s 5p'` (single spaces) silently mis-parses. Blank line if none. |
| 3 | `4f(1)5d(1)` | Open configuration, free-format `nl(occ)` groups. Blank if none. |
| 4 | `all` | Vary all orbitals. |
| 5 | `y` | Accept default per-orbital parameters. |
| 6 | `n` | **Do not accept default remaining parameters** — this is what opens the block that lets you raise the iteration count. |
| 7 | `y` | Keep grid size and strong orthogonalization. |
| 8 | `y` | Keep `PRINT=.FALSE.`, `SCFTOL = 1e-8`. |
| 9 | `n` | Open the NSCF/IC entry. |
| 10 | `99 2` | **NSCF=99** (default 12), IC=2. Format is `I2,1X,I1`. The default 12 cycles cannot kill 4s oscillations in bare-ion states. |
| 11 | `y` | TRACE off (note the inverted logic — only `n` turns tracing *on*). |
| 12 | `n` | Skip the post-run integral menu. |
| 13 | `n` | No isoelectronic-sequence continuation. |

### The hidden 14th prompt — the single worst trap

If the SCF exhausts NSCF without converging, `hf` emits an **extra
mid-stream prompt**: `Do you wish to continue ? (Y/N/H)`. It eats line 12,
sets an internal `FAIL` flag, and jumps to end-of-case **without printing the
energy summary**.

So a non-converged run does not announce itself with an error — it produces a
log with no total-energy block, and the remaining stdin lines land in the wrong
prompts. **Detect non-convergence by the absence of the energy block, not by a
return code:**

```python
if "Non-Relativistic" not in txt:
    return None          # this IS the convergence check
```

---

## 3. Configuration syntax

`(closed_shells, open_config)` pairs. Real examples:

```python
# Ce, 28-electron core ([Ar]3d10)
"ion4_closed":    ("  5s  5p",     "")
"ion3_4f1":       ("  5s  5p",     "4f(1)")
"ion0_4f15d16s2": ("  5s  5p  6s", "4f(1)5d(1)")
"ion5_5p5":       ("  5s",         "5p(5)")     # semicore p-hole

# U, 60-electron core
CORE6 = "  5s  5p  5d  6s  6p"
"ion6_closed":    (CORE6,          "")
"ion0_5f36d17s2": (CORE6 + "  7s", "5f(3)6d(1)")
```

Rules:
- Shells inside the ECP core are **dropped entirely**. A real bug came from
  leaking GRASP's `4f(14,i)` bookkeeping into a 60-core ATSP closed list —
  GRASP counts the frozen shell, ATSP must not.
- Full shell → closed string (4-char fields); partial → open string as `nl(occ)`.
- GRASP's `(n,i)` / `(n,*)` inactive markers are **GRASP syntax only**. ATSP
  wants bare `4f(1)5d(1)`.

---

## 4. The `ecp.inp` card format

Molpro-style, commas or spaces. Header, then the local channel, then l = 0…lmax−1:

```
ECP, 58, 28, 4, 0          ! Z, n_core, l_max, n_SO_blocks
4                          ! local channel: 4 terms
1 8.0000000000 30.0000000000
3 8.0000000000 240.0000000000
2 8.2000000000 -157.0786308192
2 3.5000000000 -4.9729938390
2                          ! l=0: 2 terms
2 20.1378290000 429.4308209535
...
```

Term convention: `coefficient * r**(power-2) * exp(-exponent * r^2)`, so input
power **2 → r⁰, 1 → r⁻¹, 3 → r¹**. The semilocal blocks are *difference*
potentials (V_l − V_local); the local block is added to every l.

A detail that matters for QMC-safe potentials: the pair
```
1  a   Z_eff        !  +Z_eff/r
3  a   Z_eff*a      !  Z_eff*a*r
```
cancels the −Z_eff/r divergence at the origin to O(r³). Its absence is what
makes a "free-form" or Stuttgart-style card divergent and DMC-hostile.

Sanity warnings the loader emits (worth mirroring in any tool): non-closed-shell
`n_core`, and a Z mismatch between the card and the run.

---

## 5. Parsing `hf.log`

**Total energy** — take the *non-relativistic* value:

```python
float(txt.split("Non-Relativistic")[-1].split()[0])
```

`hf` also prints a "Relativistic Shift". Deliberately ignore it when the
reference data is relativistic (GRASP DC+B): using both double-counts
relativity.

**Convergence:**

```python
conv = re.findall(r"MAXIMUM WEIGHTED CHANGE IN FUNCTIONS\s*=\s*([\d.]+D[+-]\d+)", txt)
ok = conv and float(conv[-1].replace("D", "E")) <= 1.0e-4
```

Internal tolerance is 1e-8, but it is *doubled every time the loop stalls*, so
the printed final change can legitimately exceed it. 1e-4 is the pragmatic cut.

**Orbital ⟨r⟩** (useful for shape constraints — config-average energies leave
the f-shell *shape* unconstrained, and one measurement showed a fitted card
reproducing energies well while getting the f-f interaction 2.4× worse than a
reference potential):

```python
m = re.search(rf"^\s+{orb}\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s*$",
              txt.split("1/R**3")[-1], re.M)   # 5th numeric column
```

---

## 6. Warm starts: the part that makes or breaks an optimizer

`hf` reads `wfn.inp` if present and always writes `wfn.out`. Everything else is
driver policy, and the policy is where the difficulty lives. Three layers, all
of them necessary:

**Layer 1 — self-restart.** Copy the previous evaluation's `wfn.out` to
`wfn.inp`. This is what makes an 18-state objective evaluation cost ~0.1 s.

**Layer 2 — last-known-good backup.** A failed run must not poison the web:

```python
if not converged:
    (d / "wfn.out").unlink(missing_ok=True)
    if (d / "wfn.good").exists():
        shutil.copy(d / "wfn.good", d / "wfn.out")   # restore
    return None
shutil.copy(d / "wfn.out", d / "wfn.good")           # promote on success
```

**Layer 3 — donor seeding.** A designated well-behaved state (usually the
neutral) is run first every evaluation and its orbitals seed any state with no
orbitals of its own. Heavy elements often **cannot cold-start at all** and must
bootstrap from a neighbouring element's converged pseudo-orbitals — the reference
campaign ran U's states off Th's, and each element seeded the next.

### Pinning: the fix for a non-deterministic objective

Some states' *self*-restart reproducibly crashes or flips between two SCF
solutions. Warm-starting them makes the objective non-deterministic, which
silently corrupts an optimizer path. The fix is to freeze a snapshot and seed
from it every single evaluation:

```python
pins = set(os.environ.get("PIN_STATES", "").split(","))
# once: wfn.good -> wfn.pin
# every eval: wfn.pin -> wfn.inp, and delete wfn.out first
```

This costs a few SCF iterations and buys determinism. It has been rediscovered
independently at least three times in this project (a uranium ladder, a
gadolinium stall, and a promethium fit that plateaued 4× above its convergence
gate for 3000 evaluations before one pinned state fixed it in a single run).

**Diagnostic signature to teach a tool:** an objective that plateaus far above
its gate while individual residuals look reasonable, and that returns *different
values for the same parameters*, is a warm-start bistability, not a form limit.
The forensic is a per-residual dump at the stalled parameter vector — the
offending state usually shows up as the one that fails to converge outright.

---

## 7. Failure modes

**The whipsaw (load sensitivity).** Each evaluation seeds from the previous
one, and there is a per-call timeout. Under machine contention, calls run slow,
effectively time out, and the optimizer walks a corrupted path — the same
parameters then give different energies (a measured 2.0 ↔ 4.6 eV swing).
**Never run fits on a saturated machine.** Light single-threaded Fortran
coexists fine with heavy jobs; it is the timeout, not the CPU, that kills.

**Hydrogenic bounds vs pseudo-orbitals.** A pseudo-orbital is nodeless but its
eigenvalue tracks the all-electron orbital with *physical* n. A solver that
computes its energy bounds from the node-reduced effective n will clamp the
update and limit-cycle — for *every* candidate potential, which reads exactly
like "the potential is bad." Two days of pointless tuning are avoidable by
checking the solver's bounds first. (The modified code patches this; stock
ATSP2K in ECP mode does not.)

**Node-hunting failures.** Watch for `WARNING: DIFFICULTY WITH NODE COUNTING
PROCEDURE`, `LOWER BOUND ON ED GREATER THAN UPPER BOUND`, and silent fallback
to a hydrogenic function. Correlation orbitals bypass node checking entirely,
which is how diffuse garbage (⟨r⟩ = 122 bohr where 4.5 was expected) gets
accepted without complaint. **Always check converged ⟨r⟩ against expectation.**

**Never seed correlation orbitals from all-electron shapes.** AE-shaped starts
carry inner core-node oscillations that are pure kinetic energy in a pseudo
context: the CSF diagonal inflates, the CI coefficient collapses to ~1e-5, and
the channel dies silently. Use polynomial seeds (r^m × an existing
pseudo-orbital), Gram-Schmidt orthonormalized per l — unorthonormalized seeds
make the Davidson CI return garbage without an error.

**Form-family incompatibility.** Optimized parameter vectors do not transfer
between potential *forms*. A bounded-form winner used as a divergent-form start
had its f-channel coefficient sign flip, the SCF oscillated, and the whole
battery died with the objective pinned at the penalty value.

**Stale orbitals in downstream consumers.** Any tool that reads orbitals from a
fit working directory must **re-converge them under the card it is currently
solving against**. A spin-orbit refit that skipped this sat on months-old
orbitals and regressed its holdouts by 10–100× — silently, because every
individual step "worked."

**Sentinels must dominate the objective.** The failure return value has to be
larger than any physical objective value, or an optimizer will tunnel into the
failure region and "converge" there. This project hit both polarities: a
too-*attractive* sentinel in one code, and (a month later, in a different code)
an out-of-bounds penalty of `1.0 + pen` sitting *below* a physical objective of
~40, so Nelder-Mead walked into the boundary and declared victory.

**`gencl`'s "Parity is wrong!"** is a genuine physics check, not a nuisance —
it catches configuration/term combinations that cannot couple.

---

## 8. What chemtools could offer

1. **A stdin-sequence builder** — the single highest-value item. Given element,
   Z, core size, closed shells, and open configuration, emit the exact 13-line
   answer sequence with the fixed-width field formatting handled.
2. **A convergence classifier** that knows about the hidden 14th prompt: absent
   energy block = non-convergence, not a crash.
3. **A run-directory manager** implementing the three warm-start layers plus
   pinning, since every serious use of `hf` reinvents them.
4. **A `hf.log` parser**: total energy, final weighted change, per-orbital
   ⟨r⟩/⟨r²⟩/⟨1/r³⟩ moments, node warnings.
5. **An `ecp.inp` reader/writer** with the power-2 convention documented and
   the origin-cancellation pair recognized (so a tool can say "this card is
   QMC-safe" or "this card diverges at the origin").

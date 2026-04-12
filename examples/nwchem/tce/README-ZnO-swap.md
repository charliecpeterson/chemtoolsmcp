# ZnO TCE MP2: Freezing Zn 3s/3p While Correlating All O Orbitals

## The Problem

For ZnO with a 10-electron ECP on Zn (`nelec 10`, replacing the 1s²2s²2p⁶ core), the
canonical SCF orbital ordering is:

| MO | Character | Energy (h) |
|----|-----------|------------|
| 1  | O  1s     | −20.49     |
| 2  | Zn 3s     |  −5.92     |
| 3  | Zn 3p     |  −4.00     |
| 4  | Zn 3p     |  −3.99     |
| 5  | Zn 3p     |  −3.99     |
| 6–14 | valence | ... |

O 1s sits lowest because its orbital energy (−20.49 h) is more negative than Zn 3s (−5.92 h),
even though chemically Zn 3s/3p are the "core" orbitals we want to freeze.

The goal is:
- **Freeze** Zn 3s and all three Zn 3p (MOs 2–5 in canonical ordering)
- **Correlate** O 1s and all valence orbitals

A naive `freeze 4` freezes MOs 1–4 = **O 1s + Zn 3s + only 2 of 3 Zn 3p**. Wrong on two counts.

## Why the NWChem `vectors swap` Approach Fails

The obvious fix is to swap O 1s and the last Zn 3p before running TCE:

```
scf
  vectors input zno.movecs swap 1 5 output zno-swap.movecs
end
task scf energy
```

This does not work. NWChem's SCF always re-converges to **canonical (energy-ordered)** MOs.
The swap modifies the initial guess, but since all 14 occupied orbitals are unchanged the
Fock matrix built from the swapped density is identical to the original. Diagonalization
immediately restores O 1s to position 1.

Using `maxiter 0` writes the swapped vectors to disk (with the correct ordering), but leaves
the RTDB (runtime database) flagged as "SCF not converged." When `task tce energy` runs, TCE
internally re-triggers the SCF to get a converged reference — which re-orders the orbitals
back to canonical before TCE even starts.

## The Solution

The key insight is that NWChem checks its RTDB to decide whether to re-run the SCF. If the
RTDB says "SCF converged" and the geometry/basis have not changed, it reads the movecs file
directly and skips the re-run. We exploit this by:

1. **Running SCF to convergence** — writes `zno.movecs` (canonical ordering) and marks the
   RTDB "SCF converged."
2. **Swapping orbitals 1 and 5 directly in the binary movecs file** using `swap_movecs.py`.
   The RTDB is untouched; it still sees "SCF converged" pointing to `zno.movecs`.
3. **Restarting NWChem for TCE** — NWChem reports "The SCF is already converged," reads the
   modified `zno.movecs`, and never re-runs SCF. With `freeze 4`, it now freezes MOs 1–4
   (all Zn 3s/3p) and O 1s at position 5 enters the correlation space.

## Files

| File | Purpose |
|------|---------|
| `ZnO-scf.nw` | NWChem input: RHF SCF only, writes `zno.movecs` |
| `swap_movecs.py` | Python script: swaps two orbitals in a binary NWChem movecs file |
| `ZnO-tce-freeze-Zn.nw` | NWChem input: `restart zno` + TCE MP2 with `freeze 4` |
| `run_ZnO_tce_freeze_Zn.sh` | Shell script: runs all three steps in order |

## Running It

```bash
bash run_ZnO_tce_freeze_Zn.sh
```

Or step by step:

```bash
# 1. Converged SCF
rm -f zno.db zno.movecs
mpirun -np 12 apptainer exec /home/charlie/mycontainers/nwchem_7.2.2.sif nwchem ZnO-scf.nw > ZnO-scf.out

# 2. Swap MOs 1 and 5 in the binary movecs file (RTDB unchanged)
python3 swap_movecs.py --inplace zno.movecs 1 5

# 3. TCE restart — sees "SCF already converged", uses swapped vectors
mpirun -np 12 apptainer exec /home/charlie/mycontainers/nwchem_7.2.2.sif nwchem ZnO-tce-freeze-Zn.nw > ZnO-tce-freeze-Zn.out
```

You can inspect the movecs eigenvalue ordering at any point:

```bash
python3 swap_movecs.py --parse zno.movecs
```

## Results

| Calculation | Frozen orbitals | MBPT(2) (h) |
|-------------|-----------------|-------------|
| Wrong: `freeze 4` (canonical order) | O 1s + Zn 3s + 2×Zn 3p | −0.720165 |
| Correct: swap 1↔5 then `freeze 4` | Zn 3s + 3×Zn 3p | −0.661187 |

The correct result is less negative. This is expected: O 1s sits at −20.49 h, so
excitations from it have very large MP2 denominators (ε_i + ε_j − ε_a − ε_b ≫ 0)
and contribute little to the correlation energy. The Zn 3p orbital that was incorrectly
correlated in the wrong run sits at −3.99 h, much closer to the virtual space, and
contributed more.

## How `swap_movecs.py` Works

NWChem movecs files are Fortran unformatted binary. For a closed-shell RHF calculation
the structure is:

```
Records 0–8:   header (title, basis name, nsets, nbf, nmo, ...)
Record  9:     occupation numbers  [nmo × float64]
Record 10:     eigenvalues         [nmo × float64]
Records 11…:   MO coefficients, one record per orbital [nbf × float64 each]
Last record:   SCF total energy
```

The script reads all Fortran records, swaps:
- eigenvalues[i−1] ↔ eigenvalues[j−1] in Record 10
- occupation[i−1] ↔ occupation[j−1] in Record 9
- the entire coefficient records for MOs i and j

and writes the result back. No NWChem process is involved — it is a pure binary file edit.

## Generalizing to Other Systems

Any time you need to freeze non-lowest occupied MOs in NWChem TCE:

1. Run SCF and note the orbital energies and characters from the vector analysis.
2. Decide which swap(s) bring the orbitals you want frozen into positions 1…N.
3. Apply the swap(s) with `swap_movecs.py` (chain multiple calls if needed).
4. Restart for TCE with `freeze N`.

For example, if two swaps are needed:

```bash
python3 swap_movecs.py zno.movecs 1 5 zno-tmp.movecs
python3 swap_movecs.py zno-tmp.movecs 2 6 zno.movecs
```

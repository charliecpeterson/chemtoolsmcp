#!/usr/bin/env bash
# ============================================================================
# ZnO MP2 with frozen Zn 3s/3p and ALL O orbitals correlated
#
# Problem: canonical SCF ordering for ZnO (ECP10 on Zn) is:
#   MO 1: O  1s  (E=-20.49 h)  <- lowest energy → freeze 4 would freeze this
#   MO 2: Zn 3s  (E=-5.92  h)  |
#   MO 3: Zn 3p  (E=-4.00  h)  | <- we want to freeze only these 4
#   MO 4: Zn 3p  (E=-3.99  h)  |
#   MO 5: Zn 3p  (E=-3.99  h)  /
#   MO 6-14: valence
#
# A naive "freeze 4" freezes O 1s + Zn 3s + only 2 of 3 Zn 3p.
#
# Solution:
#   1. Run SCF to convergence (writes zno.movecs, RTDB marked "converged")
#   2. Swap MOs 1 and 5 in zno.movecs using swap_movecs.py (pure binary edit,
#      RTDB stays "converged" since it only checks the energy hash)
#   3. Restart NWChem for TCE — it sees "SCF already converged", reads the
#      modified zno.movecs directly without re-running SCF, and freeze 4 now
#      correctly freezes MOs 1-4 = Zn 3p + Zn 3s + Zn 3p + Zn 3p.
#      O 1s is now at MO 5 and enters the correlation space.
#
# Note: do NOT try to do the swap via NWChem's "vectors swap" + task scf energy,
# because SCF always re-converges to canonical (energy-ordered) MOs.
# ============================================================================

set -euo pipefail
NWCHEM="mpirun -np 12 apptainer exec /home/charlie/mycontainers/nwchem_7.2.2.sif nwchem"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Step 1: Converged SCF ==="
rm -f zno.db zno.movecs
$NWCHEM ZnO-scf.nw > ZnO-scf.out
echo "SCF done. Energy: $(grep 'Total SCF energy' ZnO-scf.out | tail -1)"

echo ""
echo "=== Step 2: Swap MOs 1 and 5 in zno.movecs ==="
python3 swap_movecs.py --inplace zno.movecs 1 5
echo "After swap:"
echo "  MO 1 = Zn 3p  (was O  1s)"
echo "  MO 5 = O  1s  (was Zn 3p)"

echo ""
echo "=== Step 3: TCE MP2 with freeze 4 (freezes only Zn 3s/3p) ==="
$NWCHEM ZnO-tce-freeze-Zn.nw > ZnO-tce-freeze-Zn.out
echo "TCE done."
grep "Alpha frozen cores"   ZnO-tce-freeze-Zn.out
grep "MBPT(2) correlation"  ZnO-tce-freeze-Zn.out
grep "MBPT(2) total"        ZnO-tce-freeze-Zn.out

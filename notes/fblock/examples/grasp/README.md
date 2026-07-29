# A complete GRASP2018 run: Ce3+ 4f1, DC+Breit configuration average

Verbatim stdin from a production run (`~/scratch/Ce-refdata/ion3_4f1/`).
Run in this order, in one directory; copy `rcsf.out` -> `rcsf.inp` after
rcsfgenerate, and `rcsf.inp` -> `ref.c` plus `rwfn.out` -> `ref.w` before rci.

Result: two J blocks (5/2, 7/2), one CSF each, E(DC+Breit) = -8852.4565 Ha,
-1.2849 Ha relative to the Ce4+ closed-shell anchor.

Line-by-line decoding is in `../../grasp-fblock.md` §2. The four answers most
likely to be wrong in a generated input:

- `rcsfgenerate` line 5 (`5s,5p,4d,4f`) — highest n per l, in s,p,d,f order.
  Configuration order silently drops shells.
- `rcsfgenerate` line 6 (`1,21`) — 2*J range. Parity must match the electron
  count, and the upper limit must cover the manifold (see grasp-fblock.md §7:
  a hard-coded cap silently truncated 210 production states).
- `rmcdhf` line 4 (`5`) — (2J+1) level weights. **Only prompted when more than
  one level is selected**; sending it unconditionally shifts every later
  answer.
- `rci` line 5 (`1.d-6`) — transverse-photon scale factor, i.e. the
  low-frequency Breit limit.

# Ac (Z=89, A=227) — GRASP2018 reference recipe

Xe-menu core; hand ladder (nf=0)

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=227, static

Energies are (2J+1)-weighted configuration averages in Hartree, from
the 2026-07-28 v2 rebuild (per-configuration J ceiling; the earlier
vintage truncated high-f manifolds by up to 250 meV).

> **This element defaults to the ATSP-hf seed.** Relativistic
> starting estimates (Thomas-Fermi / screened hydrogenic) are not
> trusted here: at high Z they can converge to a stationary
> spurious solution several eV above the true ground state that
> passes every internal check. Converge non-relativistically
> first, convert, then seed.

## Run recipe

```
rnucleus     : 89 / 227 / n / 0 / 0.5 / 1 / 1
rangular     : y
rmcdhf       : y / <ASF lines, one per block> / 5 (only if >1 level) /
               <vary> / * / 100
rci          : y / ref / y / y / 1.d-6 / n / n / n / n / <ASF lines>
```

`5` in rmcdhf selects (2J+1) level weights — that answer is what makes
these configuration averages, and it is only prompted when more than
one level is selected.

## States

| slug | ion | confline | core | active set / 2J | J blocks (ncsf) | seeding | E(DC+B) Ha | rel. anchor |
|---|---|---|---|---|---|---|---|---|
| `ion3_closed` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -25727.795155 | +0.000000 |
| `ion2_5f1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Thf1  +  staged birth `5f-,5f` | -25728.254952 | -0.459797 |
| `ion2_6d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -25728.389936 | -0.594780 |
| `ion2_7s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_ths | -25728.405728 | -0.610573 |
| `ion1_6d2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -25728.717885 | -0.922730 |
| `ion1_5f16d1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion1_6d2 + ion2_5f1 | -25728.560366 | -0.765210 |
| `ion1_5f2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion2_5f1 | -25728.380350 | -0.585195 |
| `ion1_6d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `0,8` | 1(1) 2(2) 3(1) | multi-donor merge: ion1_6d2 + ion2_7s1 | -25728.768687 | -0.973532 |
| `ion1_5f17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,10` | 2(1) 3(2) 4(1) | multi-donor merge: ion2_5f1 + ion2_7s1 | -25728.615950 | -0.820794 |
| `ion0_6d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: ion1_6d17s1 | -25728.950315 | -1.155160 |
| `ion0_5f17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: ion1_5f17s1  +  staged birth `5f-,5f` | -25728.799981 | -1.004826 |
| `ion0_6d27s1` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `1,13` | 1/2(3) 3/2(4) 5/2(4) 7/2(3) 9/2(2) | multi-donor merge: ion1_6d2 + ion1_6d17s1 | -25728.896370 | -1.101215 |
| `ion4_6p5` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion3_closed | -25726.091992 | +1.703163 |
| `ion3_6p55f1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion2_5f1 | -25726.964097 | +0.831058 |

# Th (Z=90, A=232) — GRASP2018 reference recipe

n<=4 shells common core (60e, small-core An standard per the Ce 28e decision); Z_eff = 30; valence 5s 5p 5d 6s 6p + 5f 6d 7s

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=232, static

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
rnucleus     : 90 / 232 / n / 0 / 0.5 / 1 / 1
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
| `ion0_6d27s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.800397 | -2.215843 |
| `ion0_5f16d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.743513 | -2.158959 |
| `ion0_5f16d27s1` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(2,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,20` | 0(7) 1(20) 2(29) 3(32) 4(29) 5(22) 6(14) 7(7) 8(2) | donor: ion0_5f16d17s2 | -26475.689355 | -2.104801 |
| `ion1_6d3` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(3,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,17` | 1/2(2) 3/2(5) 5/2(5) 7/2(3) 9/2(3) 11/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.562944 | -1.978391 |
| `ion1_6d27s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `1,13` | 1/2(3) 3/2(4) 5/2(4) 7/2(3) 9/2(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.601168 | -2.016614 |
| `ion1_6d17s2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.610388 | -2.025835 |
| `ion1_5f16d2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,19` | 1/2(7) 3/2(13) 5/2(16) 7/2(16) 9/2(13) 11/2(9) 13/2(5) 15/2(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.507095 | -1.922541 |
| `ion1_5f26d1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion1_5f16d2 | -26475.380302 | -1.795748 |
| `ion1_5f3` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion1_5f26d1 | -26475.208467 | -1.623913 |
| `ion1_5f16d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,15` | 1/2(4) 3/2(7) 5/2(8) 7/2(8) 9/2(7) 11/2(4) 13/2(1) | donor: ion1_6d27s1  +  staged birth `5f-,5f` | -26475.557585 | -1.973031 |
| `ion1_5f27s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,17` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | donor: ion1_5f16d17s1 | -26475.438538 | -1.853984 |
| `ion2_5f16d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | donor: ion1_5f16d2 | -26475.170044 | -1.585490 |
| `ion2_5f2` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion1_5f26d1 | -26475.070088 | -1.485534 |
| `ion2_6d2` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -26475.194637 | -1.610084 |
| `ion2_6d17s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,4f` / `0,8` | 1(1) 2(2) 3(1) | donor: ion1_6d27s1  +  staged birth `7s` | -26475.193735 | -1.609181 |
| `ion2_5f17s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,10` | 2(1) 3(2) 4(1) | donor: ion1_5f27s1  +  staged birth `7s` | -26475.184310 | -1.599756 |
| `ion3_5f1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: ion2_5f2 | -26474.559880 | -0.975326 |
| `ion3_6d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -26474.545071 | -0.960518 |
| `ion3_7s1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: ion2_6d17s1  +  staged birth `7s` | -26474.498878 | -0.914325 |
| `ion4_closed` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -26473.584554 | +0.000000 |
| `ion5_6p5` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -26471.357660 | +2.226894 |
| `ion4_6p55f1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion3_5f1 | -26472.794903 | +0.789651 |

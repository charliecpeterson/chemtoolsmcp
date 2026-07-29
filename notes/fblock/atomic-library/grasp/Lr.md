# Lr (Z=103, A=262) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=262, static

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
rnucleus     : 103 / 262 / n / 0 / 0.5 / 1 / 1
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
| `ion17_closed` | 17+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -37552.529476 | +0.000000 |
| `ion16_5f1` | 16+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Nof1  +  staged birth `5f-,5f` | -37565.418750 | -12.889274 |
| `ion16_6d1` | 16+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -37561.764098 | -9.234622 |
| `ion16_7s1` | 16+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -37560.283343 | -7.753867 |
| `ion15_6d2` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -37570.349971 | -17.820495 |
| `ion15_5f16d1` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion15_6d2 + ion16_5f1 | -37573.896755 | -21.367279 |
| `ion5_5f116d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: donor_chemfd | -37643.801576 | -91.272100 |
| `ion5_5f12` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: donor_chemf | -37644.713750 | -92.184274 |
| `ion4_5f126d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion5_5f116d1 | -37646.267083 | -93.737607 |
| `ion4_5f13` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,93` | 5/2(1) 7/2(1) | donor: ion5_5f12 | -37646.961212 | -94.431736 |
| `ion3_5f136d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,98` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | donor: ion4_5f126d1 | -37648.048154 | -95.518678 |
| `ion3_5f14` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,4` | 0(1) | donor: ion4_5f13 | -37648.530005 | -96.000529 |
| `ion2_5f137s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,94` | 2(1) 3(2) 4(1) | multi-donor merge: ion4_5f13 + donor_chems  +  staged birth `7s` | -37648.070698 | -95.541222 |
| `ion1_5f147s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,5` | 1/2(1) | multi-donor merge: ion3_5f14 + ion2_5f137s1 | -37649.287325 | -96.757849 |
| `ion1_5f136d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,99` | 1/2(4) 3/2(7) 5/2(8) 7/2(8) 9/2(7) 11/2(4) 13/2(1) | multi-donor merge: ion2_5f137s1 + ion3_5f136d1 | -37648.838815 | -96.309339 |
| `ion0_5f146d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,7` | 3/2(1) 5/2(1) | donor: ion1_5f136d17s1 | -37649.901096 | -97.371620 |
| `ion1_5f147s2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,4` | 0(1) | donor: ion1_5f147s1 | -37649.769022 | -97.239546 |
| `ion0_5f147s27p1` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(2,i)7p(1,i)` | Xe (menu 5) | `7s,7p,5d,5f` / `1,5` | 1/2(1) 3/2(1) | multi-donor merge: ion1_5f147s2 + donor_7p  +  staged birth `7p-,7p` | -37649.906394 | -97.376918 |
| `ion18_6p5` | 18+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion17_closed | -37540.065553 | +12.463924 |
| `ion17_6p55f1` | 17+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion16_5f1 | -37553.803055 | -1.273579 |

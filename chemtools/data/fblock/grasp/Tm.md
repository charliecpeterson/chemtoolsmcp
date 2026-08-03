# Tm (Z=69, A=169) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=169, static

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
rnucleus     : 69 / 169 / n / 0 / 0.5 / 1 / 1
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
| `ion15_closed` | 15+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -13472.102602 | +0.000000 |
| `ion14_4f1` | 14+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Erf1  +  staged birth `4f-,4f` | -13486.153076 | -14.050474 |
| `ion14_5d1` | 14+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -13481.075080 | -8.972477 |
| `ion14_6s1` | 14+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -13478.899809 | -6.797207 |
| `ion13_5d2` | 13+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -13489.318439 | -17.215837 |
| `ion13_4f15d1` | 13+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion13_5d2 + ion14_4f1 | -13494.251390 | -22.148787 |
| `ion5_4f95d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: donor_chemfd | -13554.553947 | -82.451344 |
| `ion5_4f10` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: donor_chemf | -13555.578282 | -83.475679 |
| `ion4_4f105d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion5_4f95d1 | -13557.261231 | -85.158629 |
| `ion4_4f11` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion5_4f10 | -13557.940889 | -85.838286 |
| `ion3_4f115d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion4_4f105d1 | -13559.104821 | -87.002219 |
| `ion3_4f12` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion4_4f11 | -13559.466327 | -87.363725 |
| `ion2_4f125d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion3_4f115d1 | -13560.166027 | -88.063424 |
| `ion2_4f13` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,93` | 5/2(1) 7/2(1) | donor: ion3_4f12 | -13560.237376 | -88.134773 |
| `ion2_4f126s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,87` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | multi-donor merge: ion3_4f12 + donor_chems  +  staged birth `6s` | -13560.183454 | -88.080851 |
| `ion1_4f136s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,94` | 2(1) 3(2) 4(1) | multi-donor merge: ion2_4f13 + ion2_4f126s1 | -13560.647204 | -88.544601 |
| `ion1_4f125d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,92` | 0(8) 1(23) 2(34) 3(38) 4(36) 5(30) 6(22) 7(14) 8(7) 9(2) | multi-donor merge: ion2_4f126s1 + ion2_4f125d1 | -13560.600790 | -88.498188 |
| `ion0_4f136s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,93` | 5/2(1) 7/2(1) | donor: ion1_4f136s1 | -13560.834234 | -88.731632 |
| `ion0_4f125d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion1_4f125d16s1 | -13560.806054 | -88.703452 |
| `ion16_5p5` | 16+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion15_closed | -13460.164986 | +11.937617 |
| `ion15_5p54f1` | 15+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion14_4f1 | -13475.176988 | -3.074386 |

# Fm (Z=100, A=257) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=257, static

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
rnucleus     : 100 / 257 / n / 0 / 0.5 / 1 / 1
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
| `ion14_closed` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -34753.652007 | +0.000000 |
| `ion13_5f1` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Esf1  +  staged birth `5f-,5f` | -34763.068343 | -9.416336 |
| `ion13_6d1` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -34760.464130 | -6.812123 |
| `ion13_7s1` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -34759.385399 | -5.733392 |
| `ion12_6d2` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -34766.693576 | -13.041570 |
| `ion12_5f16d1` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion12_6d2 + ion13_5f1 | -34769.202990 | -15.550983 |
| `ion5_5f86d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: donor_chemfd | -34806.596644 | -52.944637 |
| `ion5_5f9` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: donor_chemf | -34807.376547 | -53.724540 |
| `ion4_5f96d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion5_5f86d1 | -34808.898867 | -55.246860 |
| `ion4_5f10` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion5_5f9 | -34809.477271 | -55.825264 |
| `ion3_5f106d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion4_5f96d1 | -34810.548827 | -56.896821 |
| `ion3_5f11` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion4_5f10 | -34810.932787 | -57.280780 |
| `ion2_5f116d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion3_5f106d1 | -34811.594355 | -57.942348 |
| `ion2_5f12` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion3_5f11 | -34811.790898 | -58.138891 |
| `ion2_5f117s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,80` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | multi-donor merge: ion3_5f11 + donor_chems  +  staged birth `7s` | -34811.660783 | -58.008776 |
| `ion1_5f127s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,87` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | multi-donor merge: ion2_5f12 + ion2_5f117s1 | -34812.215360 | -58.563353 |
| `ion1_5f116d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,85` | 1/2(48) 3/2(86) 5/2(112) 7/2(122) 9/2(115) 11/2(98) 13/2(75) 15/2(50) 17/2(30) 19/2(16) 21/2(6) 23/2(1) | multi-donor merge: ion2_5f117s1 + ion2_5f116d1 | -34812.048388 | -58.396381 |
| `ion0_5f127s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion1_5f127s1 | -34812.411540 | -58.759533 |
| `ion0_5f116d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion1_5f116d17s1 | -34812.268831 | -58.616825 |
| `ion15_6p5` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion14_closed | -34744.086192 | +9.565815 |
| `ion14_6p55f1` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion13_5f1 | -34754.272176 | -0.620169 |

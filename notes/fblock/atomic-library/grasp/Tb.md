# Tb (Z=65, A=159) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=159, static

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
rnucleus     : 65 / 159 / n / 0 / 0.5 / 1 / 1
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
| `ion11_closed` | 11+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -11662.399599 | +0.000000 |
| `ion10_4f1` | 10+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Gdf1  +  staged birth `4f-,4f` | -11670.812631 | -8.413032 |
| `ion10_5d1` | 10+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -11667.857802 | -5.458203 |
| `ion10_6s1` | 10+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -11666.580651 | -4.181052 |
| `ion9_5d2` | 9+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -11672.708092 | -10.308493 |
| `ion9_4f15d1` | 9+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion9_5d2 + ion10_4f1 | -11675.548137 | -13.148538 |
| `ion5_4f55d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: donor_chemfd | -11695.565613 | -33.166015 |
| `ion5_4f6` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: donor_chemf | -11696.530409 | -34.130810 |
| `ion4_4f65d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion5_4f55d1 | -11698.172246 | -35.772647 |
| `ion4_4f7` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion5_4f6 | -11698.806806 | -36.407207 |
| `ion3_4f75d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: ion4_4f65d1 | -11699.950422 | -37.550823 |
| `ion3_4f8` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion4_4f7 | -11700.282148 | -37.882549 |
| `ion2_4f85d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion3_4f75d1 | -11700.979005 | -38.579406 |
| `ion2_4f9` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion3_4f8 | -11701.038204 | -38.638605 |
| `ion2_4f86s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,59` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion3_4f8 + donor_chems  +  staged birth `6s` | -11700.966361 | -38.566762 |
| `ion1_4f96s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,66` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion2_4f9 + ion2_4f86s1 | -11701.429429 | -39.029830 |
| `ion1_4f85d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,64` | 0(130) 1(376) 2(575) 3(700) 4(748) 5(724) 6(642) 7(526) 8(397) 9(274) 10(174) 11(101) 12(51) 13(22) 14(8) 15(2) | multi-donor merge: ion2_4f86s1 + ion2_4f85d1 | -11701.392185 | -38.992587 |
| `ion0_4f96s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion1_4f96s1 | -11701.608035 | -39.208436 |
| `ion0_4f85d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion1_4f85d16s1 | -11701.586059 | -39.186460 |
| `ion12_5p5` | 12+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion11_closed | -11654.569357 | +7.830241 |
| `ion11_5p54f1` | 11+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion10_4f1 | -11663.800118 | -1.400519 |

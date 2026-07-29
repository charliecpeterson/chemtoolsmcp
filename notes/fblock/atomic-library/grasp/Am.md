# Am (Z=95, A=243) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=243, static

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
rnucleus     : 95 / 243 / n / 0 / 0.5 / 1 / 1
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
| `ion9_closed` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -30421.078494 | +0.000000 |
| `ion8_5f1` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Puf1  +  staged birth `5f-,5f` | -30425.628476 | -4.549982 |
| `ion8_6d1` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -30424.511879 | -3.433385 |
| `ion8_7s1` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -30424.022415 | -2.943922 |
| `ion7_6d2` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -30427.482312 | -6.403818 |
| `ion7_5f16d1` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion7_6d2 + ion8_5f1 | -30428.527900 | -7.449406 |
| `ion7_5f2` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion8_5f1 | -30429.448032 | -8.369538 |
| `ion6_5f26d1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion7_5f16d1 | -30431.838138 | -10.759644 |
| `ion6_5f3` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion7_5f2 | -30432.568405 | -11.489911 |
| `ion5_5f36d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion6_5f26d1 | -30434.475722 | -13.397228 |
| `ion5_5f4` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion6_5f3 | -30435.023505 | -13.945012 |
| `ion4_5f46d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion5_5f36d1 | -30436.476925 | -15.398431 |
| `ion4_5f5` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion5_5f4 | -30436.850540 | -15.772046 |
| `ion3_5f56d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion4_5f46d1 | -30437.882473 | -16.803979 |
| `ion3_5f6` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion4_5f5 | -30438.091639 | -17.013145 |
| `ion2_5f66d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion3_5f56d1 | -30438.740109 | -17.661615 |
| `ion2_5f7` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion3_5f6 | -30438.796634 | -17.718140 |
| `ion2_5f67s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,45` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion3_5f6 + donor_Pu7s | -30438.770851 | -17.692357 |
| `ion1_5f77s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,52` | 0(17) 1(48) 2(73) 3(92) 4(96) 5(88) 6(77) 7(61) 8(44) 9(29) 10(16) 11(8) 12(4) 13(1) | multi-donor merge: ion2_5f7 + ion2_5f67s1 | -30439.194130 | -18.115636 |
| `ion1_5f66d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,50` | 0(130) 1(376) 2(575) 3(700) 4(748) 5(724) 6(642) 7(526) 8(397) 9(274) 10(174) 11(101) 12(51) 13(22) 14(8) 15(2) | multi-donor merge: ion2_5f67s1 + ion2_5f66d1 | -30439.161679 | -18.083185 |
| `ion0_5f77s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion1_5f77s1 | -30439.378078 | -18.299585 |
| `ion0_5f66d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion1_5f66d17s1 | -30439.364364 | -18.285871 |
| `ion10_6p5` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion9_closed | -30415.639788 | +5.438705 |
| `ion9_6p55f1` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion8_5f1 | -30420.820115 | +0.258379 |

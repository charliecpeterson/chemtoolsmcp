# Cm (Z=96, A=247) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=247, static

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
rnucleus     : 96 / 247 / n / 0 / 0.5 / 1 / 1
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
| `ion10_closed` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -31256.312659 | +0.000000 |
| `ion9_5f1` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Amf1  +  staged birth `5f-,5f` | -31261.739772 | -5.427113 |
| `ion9_6d1` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -31260.353464 | -4.040805 |
| `ion9_7s1` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -31259.755909 | -3.443250 |
| `ion8_6d2` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -31263.905978 | -7.593319 |
| `ion8_5f16d1` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion8_6d2 + ion9_5f1 | -31265.216221 | -8.903562 |
| `ion8_5f2` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion9_5f1 | -31266.395309 | -10.082650 |
| `ion7_5f26d1` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion8_5f16d1 | -31269.330757 | -13.018098 |
| `ion7_5f3` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion8_5f2 | -31270.308614 | -13.995955 |
| `ion6_5f36d1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion7_5f26d1 | -31272.727900 | -16.415241 |
| `ion6_5f4` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion7_5f3 | -31273.510955 | -17.198296 |
| `ion5_5f46d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion6_5f36d1 | -31275.440695 | -19.128036 |
| `ion5_5f5` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion6_5f4 | -31276.036157 | -19.723498 |
| `ion4_5f56d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion5_5f46d1 | -31277.505514 | -21.192855 |
| `ion4_5f6` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion5_5f5 | -31277.921342 | -21.608683 |
| `ion3_5f66d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion4_5f56d1 | -31278.962991 | -22.650333 |
| `ion3_5f7` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion4_5f6 | -31279.208359 | -22.895700 |
| `ion3_5f67s1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,45` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion4_5f6 + donor_Am7s | -31278.930330 | -22.617671 |
| `ion2_5f77s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,52` | 0(17) 1(48) 2(73) 3(92) 4(96) 5(88) 6(77) 7(61) 8(44) 9(29) 10(16) 11(8) 12(4) 13(1) | multi-donor merge: ion3_5f7 + ion3_5f67s1 | -31279.897506 | -23.584847 |
| `ion2_5f66d17s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,50` | 0(130) 1(376) 2(575) 3(700) 4(748) 5(724) 6(642) 7(526) 8(397) 9(274) 10(174) 11(101) 12(51) 13(22) 14(8) 15(2) | multi-donor merge: ion3_5f67s1 + ion3_5f66d1 | -31279.679364 | -23.366705 |
| `ion0_5f76d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: ion2_5f66d17s1 | -31280.494786 | -24.182127 |
| `ion0_5f87s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion2_5f77s1 | -31280.535617 | -24.222958 |
| `ion11_6p5` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion10_closed | -31250.119801 | +6.192857 |
| `ion10_6p55f1` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion9_5f1 | -31256.206335 | +0.106324 |

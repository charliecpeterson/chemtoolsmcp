# Eu (Z=63, A=153) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=153, static

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
rnucleus     : 63 / 153 / n / 0 / 0.5 / 1 / 1
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
| `ion9_closed` | 9+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | donor: donor_closed | -10813.666576 | +0.000000 |
| `ion8_4f1` | 8+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Smf1  +  staged birth `4f-,4f` | -10819.672153 | -6.005577 |
| `ion8_5d1` | 8+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -10817.635629 | -3.969053 |
| `ion8_6s1` | 8+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | donor: donor_s1 | -10816.741192 | -3.074616 |
| `ion7_5d2` | 7+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -10821.060913 | -7.394337 |
| `ion7_4f15d1` | 7+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion7_5d2 + ion8_4f1 | -10822.998567 | -9.331991 |
| `ion7_4f2` | 7+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion8_4f1 | -10824.642775 | -10.976199 |
| `ion6_4f25d1` | 6+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion7_4f15d1 | -10827.362282 | -13.695706 |
| `ion6_4f3` | 6+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion7_4f2 | -10828.635481 | -14.968904 |
| `ion5_4f35d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion6_4f25d1 | -10830.783675 | -17.117099 |
| `ion5_4f4` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion6_4f3 | -10831.708736 | -18.042160 |
| `ion4_4f45d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion5_4f35d1 | -10833.326072 | -19.659496 |
| `ion4_4f5` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion5_4f4 | -10833.927964 | -20.261388 |
| `ion3_4f55d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion4_4f45d1 | -10835.058148 | -21.391572 |
| `ion3_4f6` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion4_4f5 | -10835.364828 | -21.698252 |
| `ion2_4f65d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion3_4f55d1 | -10836.057928 | -22.391352 |
| `ion2_4f7` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion3_4f6 | -10836.101146 | -22.434570 |
| `ion2_4f66s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,45` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion3_4f6 + ion8_6s1 | -10836.032286 | -22.365710 |
| `ion1_4f76s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,52` | 0(17) 1(48) 2(73) 3(92) 4(96) 5(88) 6(77) 7(61) 8(44) 9(29) 10(16) 11(8) 12(4) 13(1) | multi-donor merge: ion2_4f7 + ion2_4f66s1 | -10836.482937 | -22.816361 |
| `ion1_4f65d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,50` | 0(130) 1(376) 2(575) 3(700) 4(748) 5(724) 6(642) 7(526) 8(397) 9(274) 10(174) 11(101) 12(51) 13(22) 14(8) 15(2) | multi-donor merge: ion2_4f66s1 + ion2_4f65d1 | -10836.460438 | -22.793862 |
| `ion0_4f76s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion1_4f76s1 | -10836.657288 | -22.990712 |
| `ion0_4f65d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion1_4f65d16s1 | -10836.648883 | -22.982307 |
| `ion10_5p5` | 10+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion9_closed | -10807.618080 | +6.048496 |
| `ion9_5p54f1` | 9+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion8_4f1 | -10814.366623 | -0.700047 |

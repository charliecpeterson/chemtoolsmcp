# Dy (Z=66, A=164) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=164, static

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
rnucleus     : 66 / 164 / n / 0 / 0.5 / 1 / 1
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
| `ion12_closed` | 12+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -12100.627724 | +0.000000 |
| `ion11_4f1` | 11+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Tbf1  +  staged birth `4f-,4f` | -12110.348619 | -9.720895 |
| `ion11_5d1` | 11+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -12106.898262 | -6.270538 |
| `ion11_6s1` | 11+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -12105.412752 | -4.785028 |
| `ion10_5d2` | 10+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -12112.529863 | -11.902139 |
| `ion10_4f15d1` | 10+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion10_5d2 + ion11_4f1 | -12115.857716 | -15.229992 |
| `ion5_4f65d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: donor_chemfd | -12143.855098 | -43.227374 |
| `ion5_4f7` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: donor_chemf | -12144.837125 | -44.209401 |
| `ion4_4f75d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: ion5_4f65d1 | -12146.490184 | -45.862460 |
| `ion4_4f8` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion5_4f7 | -12147.138377 | -46.510653 |
| `ion3_4f85d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion4_4f75d1 | -12148.287838 | -47.660114 |
| `ion3_4f9` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion4_4f8 | -12148.629363 | -48.001639 |
| `ion2_4f95d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion3_4f85d1 | -12149.327458 | -48.699734 |
| `ion2_4f10` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion3_4f9 | -12149.391983 | -48.764259 |
| `ion2_4f96s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,66` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion3_4f9 + donor_chems  +  staged birth `6s` | -12149.321857 | -48.694134 |
| `ion1_4f106s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,73` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion2_4f10 + ion2_4f96s1 | -12149.787869 | -49.160146 |
| `ion1_4f95d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,71` | 1/2(187) 3/2(345) 5/2(459) 7/2(519) 9/2(521) 11/2(475) 13/2(397) 15/2(305) 17/2(216) 19/2(139) 21/2(79) 23/2(40) 25/2(18) 27/2(6) 29/2(1) | multi-donor merge: ion2_4f96s1 + ion2_4f95d1 | -12149.745987 | -49.118263 |
| `ion0_4f106s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion1_4f106s1 | -12149.968583 | -49.340860 |
| `ion0_4f95d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion1_4f95d16s1 | -12149.942629 | -49.314905 |
| `ion13_5p5` | 13+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion12_closed | -12091.838284 | +8.789440 |
| `ion12_5p54f1` | 12+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion11_4f1 | -12102.413461 | -1.785738 |

# Ho (Z=67, A=165) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=165, static

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
rnucleus     : 67 / 165 / n / 0 / 0.5 / 1 / 1
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
| `ion13_closed` | 13+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -12548.301137 | +0.000000 |
| `ion12_4f1` | 12+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Dyf1  +  staged birth `4f-,4f` | -12559.398048 | -11.096911 |
| `ion12_5d1` | 12+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -12555.428439 | -7.127302 |
| `ion12_6s1` | 12+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -12553.723638 | -5.422501 |
| `ion11_5d2` | 11+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -12561.886318 | -13.585181 |
| `ion11_4f15d1` | 11+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion11_5d2 + ion12_4f1 | -12565.725812 | -17.424675 |
| `ion5_4f75d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: donor_chemfd | -12603.047130 | -54.745993 |
| `ion5_4f8` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: donor_chemf | -12604.044796 | -55.743659 |
| `ion4_4f85d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion5_4f75d1 | -12605.708409 | -57.407272 |
| `ion4_4f9` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion5_4f8 | -12606.368610 | -58.067473 |
| `ion3_4f95d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion4_4f85d1 | -12607.523359 | -59.222222 |
| `ion3_4f10` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion4_4f9 | -12607.873068 | -59.571931 |
| `ion2_4f105d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion3_4f95d1 | -12608.572018 | -60.270881 |
| `ion2_4f11` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion3_4f10 | -12608.640265 | -60.339128 |
| `ion2_4f106s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,73` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion3_4f10 + donor_chems  +  staged birth `6s` | -12608.573794 | -60.272657 |
| `ion1_4f116s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,80` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | multi-donor merge: ion2_4f11 + ion2_4f106s1 | -12609.040809 | -60.739672 |
| `ion1_4f105d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,78` | 0(54) 1(153) 2(231) 3(278) 4(288) 5(268) 6(229) 7(177) 8(123) 9(78) 10(44) 11(21) 12(8) 13(2) | multi-donor merge: ion2_4f106s1 + ion2_4f105d1 | -12608.995924 | -60.694787 |
| `ion0_4f116s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion1_4f116s1 | -12609.223626 | -60.922488 |
| `ion0_4f105d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion1_4f105d16s1 | -12609.195379 | -60.894241 |
| `ion14_5p5` | 14+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion13_closed | -12538.507293 | +9.793844 |
| `ion13_5p54f1` | 13+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion12_4f1 | -12550.494584 | -2.193447 |

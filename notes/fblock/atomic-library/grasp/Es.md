# Es (Z=99, A=252) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=252, static

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
rnucleus     : 99 / 252 / n / 0 / 0.5 / 1 / 1
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
| `ion13_closed` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -33855.032996 | +0.000000 |
| `ion12_5f1` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Cff1  +  staged birth `5f-,5f` | -33863.381889 | -8.348893 |
| `ion12_6d1` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -33861.102267 | -6.069271 |
| `ion12_7s1` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -33860.150405 | -5.117408 |
| `ion11_6d2` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -33866.611643 | -11.578647 |
| `ion11_5f16d1` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion11_6d2 + ion12_5f1 | -33868.800973 | -13.767977 |
| `ion5_5f76d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: donor_chemfd | -33897.778981 | -42.745985 |
| `ion5_5f8` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: donor_chemf | -33898.513682 | -43.480686 |
| `ion4_5f86d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion5_5f76d1 | -33900.024197 | -44.991200 |
| `ion4_5f9` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion5_5f8 | -33900.562861 | -45.529865 |
| `ion3_5f96d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion4_5f86d1 | -33901.628077 | -46.595081 |
| `ion3_5f10` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion4_5f9 | -33901.978276 | -46.945280 |
| `ion2_5f106d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion3_5f96d1 | -33902.638343 | -47.605347 |
| `ion2_5f11` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion3_5f10 | -33902.807971 | -47.774975 |
| `ion2_5f107s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,73` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion3_5f10 + donor_chems  +  staged birth `7s` | -33902.696597 | -47.663601 |
| `ion1_5f117s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,80` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | multi-donor merge: ion2_5f11 + ion2_5f107s1 | -33903.226997 | -48.194001 |
| `ion1_5f106d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,78` | 0(54) 1(153) 2(231) 3(278) 4(288) 5(268) 6(229) 7(177) 8(123) 9(78) 10(44) 11(21) 12(8) 13(2) | multi-donor merge: ion2_5f107s1 + ion2_5f106d1 | -33903.085748 | -48.052752 |
| `ion0_5f117s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion1_5f117s1 | -33903.420676 | -48.387680 |
| `ion0_5f106d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion1_5f106d17s1 | -33903.302420 | -48.269424 |
| `ion14_6p5` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion13_closed | -33846.363344 | +8.669653 |
| `ion13_6p55f1` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion12_5f1 | -33855.455042 | -0.422046 |

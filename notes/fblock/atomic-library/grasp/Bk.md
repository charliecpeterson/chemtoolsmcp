# Bk (Z=97, A=247) — GRASP2018 reference recipe

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
rnucleus     : 97 / 247 / n / 0 / 0.5 / 1 / 1
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
| `ion11_closed` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -32106.643419 | +0.000000 |
| `ion10_5f1` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Cmf1  +  staged birth `5f-,5f` | -32112.996803 | -6.353384 |
| `ion10_6d1` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -32111.326479 | -4.683060 |
| `ion10_7s1` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -32110.615550 | -3.972131 |
| `ion9_6d2` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -32115.496737 | -8.853317 |
| `ion9_5f16d1` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion9_6d2 + ion10_5f1 | -32117.086120 | -10.442700 |
| `ion5_5f56d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: donor_chemfd | -32132.535983 | -25.892563 |
| `ion5_5f6` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: donor_chemf | -32133.178476 | -26.535057 |
| `ion4_5f66d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion5_5f56d1 | -32134.662594 | -28.019174 |
| `ion4_5f7` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: ion5_5f6 | -32135.119989 | -28.476569 |
| `ion3_5f76d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: ion4_5f66d1 | -32136.170340 | -29.526920 |
| `ion3_5f8` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion4_5f7 | -32136.451266 | -29.807847 |
| `ion2_5f86d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion3_5f76d1 | -32137.106816 | -30.463397 |
| `ion2_5f9` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion3_5f8 | -32137.221005 | -30.577586 |
| `ion2_5f87s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,59` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion3_5f8 + donor_chems  +  staged birth `7s` | -32137.150194 | -30.506774 |
| `ion1_5f97s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,66` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion2_5f9 + ion2_5f87s1 | -32137.629256 | -30.985836 |
| `ion1_5f86d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,64` | 0(130) 1(376) 2(575) 3(700) 4(748) 5(724) 6(642) 7(526) 8(397) 9(274) 10(174) 11(101) 12(51) 13(22) 14(8) 15(2) | multi-donor merge: ion2_5f87s1 + ion2_5f86d1 | -32137.541214 | -30.897795 |
| `ion0_5f97s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion1_5f97s1 | -32137.818024 | -31.174605 |
| `ion0_5f86d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion1_5f86d17s1 | -32137.750708 | -31.107289 |
| `ion12_6p5` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion11_closed | -32099.660472 | +6.982947 |
| `ion11_6p55f1` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion10_5f1 | -32106.701638 | -0.058219 |

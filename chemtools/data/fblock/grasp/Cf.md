# Cf (Z=98, A=251) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=251, static

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
rnucleus     : 98 / 251 / n / 0 / 0.5 / 1 / 1
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
| `ion12_closed` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -32972.663846 | +0.000000 |
| `ion11_5f1` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Bkf1  +  staged birth `5f-,5f` | -32979.991488 | -7.327642 |
| `ion11_6d1` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -32978.023264 | -5.359418 |
| `ion11_7s1` | 11+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -32977.194095 | -4.530249 |
| `ion10_6d2` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -32982.846049 | -10.182203 |
| `ion10_5f16d1` | 10+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion10_6d2 + ion11_5f1 | -32984.728597 | -12.064751 |
| `ion5_5f66d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,49` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: donor_chemfd | -33006.410709 | -33.746863 |
| `ion5_5f7` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,51` | 1/2(17) 3/2(31) 5/2(42) 7/2(50) 9/2(46) 11/2(42) 13/2(35) 15/2(26) 17/2(18) 19/2(11) 21/2(5) 23/2(3) 25/2(1) | donor: donor_chemf | -33007.099592 | -34.435746 |
| `ion4_5f76d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,56` | 0(73) 1(213) 2(326) 3(397) 4(426) 5(414) 6(366) 7(299) 8(227) 9(158) 10(101) 11(58) 12(29) 13(13) 14(5) 15(1) | donor: ion5_5f66d1 | -33008.597388 | -35.933542 |
| `ion4_5f8` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,58` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion5_5f7 | -33009.095725 | -36.431880 |
| `ion3_5f86d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: ion4_5f76d1 | -33010.153902 | -37.490057 |
| `ion3_5f9` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion4_5f8 | -33010.469763 | -37.805917 |
| `ion2_5f96d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion3_5f86d1 | -33011.127847 | -38.464001 |
| `ion2_5f10` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion3_5f9 | -33011.270027 | -38.606182 |
| `ion2_5f97s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,66` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion3_5f9 + donor_chems  +  staged birth `7s` | -33011.178402 | -38.514557 |
| `ion1_5f107s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,73` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion2_5f10 + ion2_5f97s1 | -33011.683663 | -39.019817 |
| `ion1_5f96d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,71` | 1/2(187) 3/2(345) 5/2(459) 7/2(519) 9/2(521) 11/2(475) 13/2(397) 15/2(305) 17/2(216) 19/2(139) 21/2(79) 23/2(40) 25/2(18) 27/2(6) 29/2(1) | multi-donor merge: ion2_5f97s1 + ion2_5f96d1 | -33011.568712 | -38.904866 |
| `ion0_5f107s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion1_5f107s1 | -33011.874860 | -39.211014 |
| `ion0_5f96d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion1_5f96d17s1 | -33011.781737 | -39.117891 |
| `ion13_6p5` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion12_closed | -32964.855212 | +7.808634 |
| `ion12_6p55f1` | 12+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion11_5f1 | -32972.898398 | -0.234553 |

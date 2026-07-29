# Pu (Z=94, A=244) — GRASP2018 reference recipe

n<=4 shells common core (60e); Z_eff = 34

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=244, static

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
rnucleus     : 94 / 244 / n / 0 / 0.5 / 1 / 1
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
| `ion8_closed` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -29601.829305 | +0.000000 |
| `ion7_5f1` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Np5f1  +  staged birth `5f-,5f` | -29605.552753 | -3.723448 |
| `ion7_6d1` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_6d1 | -29604.690976 | -2.861671 |
| `ion7_7s1` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_7s1 | -29604.304013 | -2.474709 |
| `ion6_5f2` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion7_5f1 | -29608.588892 | -6.759588 |
| `ion6_6d2` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_6d2 | -29607.116069 | -5.286764 |
| `ion6_5f16d1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion6_6d2 + ion7_5f1 | -29607.912094 | -6.082789 |
| `ion5_5f3` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion6_5f2 | -29610.971662 | -9.142357 |
| `ion5_5f26d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion6_5f16d1 | -29610.472266 | -8.642961 |
| `ion4_5f4` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion5_5f3 | -29612.738459 | -10.909154 |
| `ion4_5f36d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion5_5f26d1 | -29612.407793 | -10.578488 |
| `ion3_5f5` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion4_5f4 | -29613.931743 | -12.102439 |
| `ion3_5f46d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion4_5f36d1 | -29613.759520 | -11.930215 |
| `ion3_5f47s1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,31` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion4_5f4 + ion7_7s1 | -29613.717771 | -11.888466 |
| `ion2_5f6` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion3_5f5 | -29614.602220 | -12.772915 |
| `ion2_5f57s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,38` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion3_5f5 + ion3_5f47s1 | -29614.600872 | -12.771568 |
| `ion1_5f67s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,45` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion2_5f6 + ion2_5f57s1 | -29614.994277 | -13.164973 |
| `ion1_5f56d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,43` | 1/2(187) 3/2(345) 5/2(459) 7/2(519) 9/2(521) 11/2(475) 13/2(397) 15/2(305) 17/2(216) 19/2(139) 21/2(79) 23/2(40) 25/2(18) 27/2(6) 29/2(1) | multi-donor merge: ion2_5f57s1 + ion3_5f46d1 | -29614.990572 | -13.161267 |
| `ion0_5f67s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion1_5f67s1 | -29615.175856 | -13.346552 |
| `ion0_5f56d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion1_5f56d17s1 | -29615.189961 | -13.360656 |
| `ion9_6p5` | 9+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion8_closed | -29597.108402 | +4.720903 |
| `ion8_6p55f1` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion7_5f1 | -29601.432186 | +0.397119 |

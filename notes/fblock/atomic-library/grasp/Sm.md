# Sm (Z=62, A=152) — GRASP2018 reference recipe

[Ar]3d10 common core (28e); Z_eff = 34

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=152, static

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
rnucleus     : 62 / 152 / n / 0 / 0.5 / 1 / 1
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
| `ion8_closed` | 8+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | donor: donor_closed | -10403.097889 | +0.000000 |
| `ion7_4f1` | 7+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Nd4f1  +  staged birth `4f-,4f` | -10408.006397 | -4.908508 |
| `ion7_5d1` | 7+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | donor: donor_5d1 | -10406.391565 | -3.293675 |
| `ion7_6s1` | 7+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | donor: donor_6s1 | -10405.670855 | -2.572965 |
| `ion6_4f2` | 6+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion7_4f1 | -10411.946324 | -8.848435 |
| `ion6_5d2` | 6+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_5d2 | -10409.174957 | -6.077068 |
| `ion6_4f15d1` | 6+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion6_5d2 + ion7_4f1 | -10410.699172 | -7.601283 |
| `ion5_4f3` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion6_4f2 | -10414.977311 | -11.879421 |
| `ion5_4f25d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion6_4f15d1 | -10414.074954 | -10.977065 |
| `ion4_4f4` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion5_4f3 | -10417.163829 | -14.065940 |
| `ion4_4f35d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion5_4f25d1 | -10416.581212 | -13.483322 |
| `ion3_4f5` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion4_4f4 | -10418.577408 | -15.479519 |
| `ion3_4f45d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion4_4f35d1 | -10418.286294 | -15.188404 |
| `ion3_4f46s1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,31` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion4_4f4 + ion7_6s1 | -10418.143629 | -15.045740 |
| `ion2_4f6` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion3_4f5 | -10419.300070 | -16.202181 |
| `ion2_4f56s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,38` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion3_4f5 + ion3_4f46s1 | -10419.236348 | -16.138458 |
| `ion1_4f66s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,45` | 1/2(33) 3/2(56) 5/2(74) 7/2(83) 9/2(83) 11/2(75) 13/2(62) 15/2(44) 17/2(31) 19/2(19) 21/2(10) 23/2(4) 25/2(2) | multi-donor merge: ion2_4f6 + ion2_4f56s1 | -10419.677059 | -16.579170 |
| `ion1_4f55d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,43` | 1/2(187) 3/2(345) 5/2(459) 7/2(519) 9/2(521) 11/2(475) 13/2(397) 15/2(305) 17/2(216) 19/2(139) 21/2(79) 23/2(40) 25/2(18) 27/2(6) 29/2(1) | multi-donor merge: ion2_4f56s1 + ion3_4f45d1 | -10419.665034 | -16.567145 |
| `ion0_4f66s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,44` | 0(14) 1(19) 2(37) 3(37) 4(46) 5(37) 6(38) 7(24) 8(20) 9(11) 10(8) 11(2) 12(2) | donor: ion1_4f66s1 | -10419.849255 | -16.751366 |
| `ion0_4f55d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,42` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion1_4f55d16s1 | -10419.850805 | -16.752916 |
| `ion9_5p5` | 9+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion8_closed | -10397.871249 | +5.226640 |
| `ion8_5p54f1` | 8+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion7_4f1 | -10403.484209 | -0.386320 |

# Nd (Z=60, A=144) — GRASP2018 reference recipe

[Ar]3d10 common core (28e); Z_eff = 32; valence 4s 4p 4d 5s 5p + 4f 5d 6s

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=144, static

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
rnucleus     : 60 / 144 / n / 0 / 0.5 / 1 / 1
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
| `ion6_closed` | 6+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -9609.198134 | +0.000000 |
| `ion5_4f1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: ion6_closed  +  staged birth `4f-,4f` | -9612.135202 | -2.937069 |
| `ion5_5d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -9611.284683 | -2.086549 |
| `ion5_6s1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -9610.873986 | -1.675852 |
| `ion4_4f2` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion5_4f1 | -9614.246758 | -5.048625 |
| `ion4_5d2` | 4+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -9612.932359 | -3.734226 |
| `ion4_4f15d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion4_5d2 + ion5_4f1 | -9613.709537 | -4.511403 |
| `ion3_4f3` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion4_4f2 | -9615.604101 | -6.405968 |
| `ion3_4f25d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion4_4f15d1 | -9615.351378 | -6.153244 |
| `ion3_4f26s1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,17` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | multi-donor merge: ion4_4f2 + ion5_6s1 | -9615.200838 | -6.002705 |
| `ion2_4f4` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion3_4f3 | -9616.290084 | -7.091950 |
| `ion2_4f35d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion3_4f25d1 | -9616.287356 | -7.089223 |
| `ion2_4f36s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,24` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | donor: ion3_4f26s1 | -9616.245571 | -7.047437 |
| `ion1_4f46s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,31` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | donor: ion2_4f36s1 | -9616.657222 | -7.459088 |
| `ion1_4f35d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,29` | 1/2(48) 3/2(86) 5/2(112) 7/2(122) 9/2(115) 11/2(98) 13/2(75) 15/2(50) 17/2(30) 19/2(16) 21/2(6) 23/2(1) | multi-donor merge: ion2_4f36s1 + ion2_4f35d1 | -9616.673710 | -7.475576 |
| `ion0_4f46s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion1_4f46s1 | -9616.824989 | -7.626855 |
| `ion0_4f35d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion1_4f35d16s1 | -9616.854170 | -7.656036 |
| `ion7_5p5` | 7+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion6_closed | -9605.474577 | +3.723557 |
| `ion6_5p54f1` | 6+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion5_4f1 | -9609.035699 | +0.162435 |
| `ion1_4f45d1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion2_4f35d1 | -9616.612409 | -7.414275 |
| `ion4_4f16s1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,10` | 2(1) 3(2) 4(1) | multi-donor merge: ion5_4f1 + ion5_6s1 | -9613.435137 | -4.237003 |
| `ion2_4f25d2` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,26` | 0(21) 1(40) 2(70) 3(71) 4(78) 5(61) 6(52) 7(31) 8(21) 9(8) 10(4) | donor: ion3_4f25d1 | -9616.089127 | -6.890994 |

# Pm (Z=61, A=147) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=147, static

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
rnucleus     : 61 / 147 / n / 0 / 0.5 / 1 / 1
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
| `ion7_closed` | 7+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | donor: donor_closed | -10001.639245 | +0.000000 |
| `ion6_4f1` | 6+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Ndf1  +  staged birth `4f-,4f` | -10005.524113 | -3.884868 |
| `ion6_5d1` | 6+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -10004.304971 | -2.665726 |
| `ion6_6s1` | 6+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | donor: donor_s1 | -10003.745689 | -2.106444 |
| `ion5_5d2` | 5+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -10006.495218 | -4.855973 |
| `ion5_4f15d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion5_5d2 + ion6_4f1 | -10007.632223 | -5.992977 |
| `ion5_4f2` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion6_4f1 | -10008.509789 | -6.870544 |
| `ion4_4f25d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion5_4f15d1 | -10010.099390 | -8.460145 |
| `ion4_4f3` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion5_4f2 | -10010.660527 | -9.021282 |
| `ion3_4f35d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion4_4f25d1 | -10011.774503 | -10.135258 |
| `ion3_4f4` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion4_4f3 | -10012.047730 | -10.408485 |
| `ion2_4f45d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion3_4f35d1 | -10012.734946 | -11.095701 |
| `ion2_4f5` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion3_4f4 | -10012.753767 | -11.114522 |
| `ion2_4f46s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,31` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion3_4f4 + ion6_6s1 | -10012.698021 | -11.058776 |
| `ion1_4f56s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,38` | 0(10) 1(31) 2(49) 3(58) 4(59) 5(55) 6(46) 7(36) 8(25) 9(14) 10(8) 11(4) 12(1) | multi-donor merge: ion2_4f5 + ion2_4f46s1 | -10013.125883 | -11.486638 |
| `ion1_4f45d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,36` | 0(54) 1(153) 2(231) 3(278) 4(288) 5(268) 6(229) 7(177) 8(123) 9(78) 10(44) 11(21) 12(8) 13(2) | multi-donor merge: ion2_4f46s1 + ion2_4f45d1 | -10013.126731 | -11.487486 |
| `ion0_4f56s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion1_4f56s1 | -10013.295891 | -11.656646 |
| `ion0_4f45d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion1_4f45d16s1 | -10013.309845 | -11.670600 |
| `ion8_5p5` | 8+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion7_closed | -9997.187799 | +4.451446 |
| `ion7_5p54f1` | 7+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion6_4f1 | -10001.737588 | -0.098343 |

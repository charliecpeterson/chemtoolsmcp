# Er (Z=68, A=166) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=166, static

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
rnucleus     : 68 / 166 / n / 0 / 0.5 / 1 / 1
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
| `ion14_closed` | 14+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -13005.411304 | +0.000000 |
| `ion13_4f1` | 13+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Hof1  +  staged birth `4f-,4f` | -13017.951624 | -12.540320 |
| `ion13_5d1` | 13+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -13013.439373 | -8.028069 |
| `ion13_6s1` | 13+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -13011.504569 | -6.093265 |
| `ion12_5d2` | 12+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -13020.767982 | -15.356677 |
| `ion12_4f15d1` | 12+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion12_5d2 + ion13_4f1 | -13025.142608 | -19.731304 |
| `ion5_4f85d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,63` | 1/2(130) 3/2(246) 5/2(329) 7/2(371) 9/2(377) 11/2(347) 13/2(295) 15/2(231) 17/2(166) 19/2(108) 21/2(66) 23/2(35) 25/2(16) 27/2(6) 29/2(2) | donor: donor_chemfd | -13073.207378 | -67.796074 |
| `ion5_4f9` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,65` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: donor_chemf | -13074.219150 | -68.807845 |
| `ion4_4f95d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: ion5_4f85d1 | -13075.892715 | -70.481411 |
| `ion4_4f10` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion5_4f9 | -13076.563379 | -71.152075 |
| `ion3_4f105d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion4_4f95d1 | -13077.722947 | -72.311643 |
| `ion3_4f11` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion4_4f10 | -13078.079269 | -72.667965 |
| `ion2_4f115d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion3_4f105d1 | -13078.778748 | -73.367444 |
| `ion2_4f12` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion3_4f11 | -13078.849250 | -73.437946 |
| `ion2_4f116s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,80` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | multi-donor merge: ion3_4f11 + donor_chems  +  staged birth `6s` | -13078.788203 | -73.376899 |
| `ion1_4f126s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,87` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | multi-donor merge: ion2_4f12 + ion2_4f116s1 | -13079.254438 | -73.843134 |
| `ion1_4f115d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,85` | 1/2(48) 3/2(86) 5/2(112) 7/2(122) 9/2(115) 11/2(98) 13/2(75) 15/2(50) 17/2(30) 19/2(16) 21/2(6) 23/2(1) | multi-donor merge: ion2_4f116s1 + ion2_4f115d1 | -13079.208061 | -73.796757 |
| `ion0_4f126s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion1_4f126s1 | -13079.439362 | -74.028058 |
| `ion0_4f115d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion1_4f115d16s1 | -13079.410387 | -73.999083 |
| `ion15_5p5` | 15+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion14_closed | -12994.568023 | +10.843281 |
| `ion14_5p54f1` | 14+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion13_4f1 | -13008.034453 | -2.623149 |

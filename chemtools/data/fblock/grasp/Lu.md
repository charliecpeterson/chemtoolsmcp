# Lu (Z=71, A=175) — GRASP2018 reference recipe

Kr-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=175, static

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
rnucleus     : 71 / 175 / n / 0 / 0.5 / 1 / 1
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
| `ion17_closed` | 17+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -14434.376317 | +0.000000 |
| `ion16_4f1` | 16+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Ybf1  +  staged birth `4f-,4f` | -14451.645268 | -17.268951 |
| `ion16_5d1` | 16+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -14445.367377 | -10.991060 |
| `ion16_6s1` | 16+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -14442.680588 | -8.304270 |
| `ion15_5d2` | 15+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_ybd2 | -14455.570946 | -21.194628 |
| `ion15_4f15d1` | 15+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion15_5d2 + ion16_4f1 | -14461.689166 | -27.312849 |
| `ion5_4f115d1` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: donor_chemfd | -14551.246818 | -116.870500 |
| `ion5_4f12` | 5+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: donor_chemf | -14552.292346 | -117.916028 |
| `ion4_4f125d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion5_4f115d1 | -14553.992400 | -119.616083 |
| `ion4_4f13` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,93` | 5/2(1) 7/2(1) | donor: ion5_4f12 | -14554.685940 | -120.309622 |
| `ion3_4f135d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,98` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | donor: ion4_4f125d1 | -14555.857357 | -121.481039 |
| `ion3_4f14` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,4` | 0(1) | donor: ion4_4f13 | -14556.225200 | -121.848883 |
| `ion3_4f136s1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,94` | 2(1) 3(2) 4(1) | multi-donor merge: ion4_4f13 + donor_chems  +  staged birth `6s` | -14555.776191 | -121.399873 |
| `ion2_4f146s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,5` | 1/2(1) | multi-donor merge: ion3_4f14 + ion3_4f136s1 | -14556.958690 | -122.582373 |
| `ion2_4f135d16s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,99` | 1/2(4) 3/2(7) 5/2(8) 7/2(8) 9/2(7) 11/2(4) 13/2(1) | multi-donor merge: ion3_4f136s1 + ion3_4f135d1 | -14556.618534 | -122.242217 |
| `ion0_4f145d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: ion2_4f135d16s1 | -14557.581703 | -123.205386 |
| `ion18_5p5` | 18+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion17_closed | -14420.115653 | +14.260665 |
| `ion17_5p54f1` | 17+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion16_4f1 | -14438.416217 | -4.039900 |

# Pr (Z=59, A=141) — GRASP2018 reference recipe

[Ar]3d10 common core (28e, small-core Ln standard per the Ce decision); Z_eff = 31; valence 4s 4p 4d 5s 5p + 4f 5d 6s

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=141, static

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
rnucleus     : 59 / 141 / n / 0 / 0.5 / 1 / 1
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
| `ion5_closed` | 5+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -9225.730331 | +0.000000 |
| `ion4_4f1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: ion5_closed  +  staged birth `4f-,4f` | -9227.798896 | -2.068565 |
| `ion4_5d1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -9227.288320 | -1.557989 |
| `ion4_6s1` | 4+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -9227.012625 | -1.282294 |
| `ion3_4f2` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion4_4f1 | -9229.122364 | -3.392033 |
| `ion3_5d2` | 3+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -9228.446664 | -2.716333 |
| `ion3_4f15d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion3_5d2 + ion4_4f1 | -9228.893159 | -3.162828 |
| `ion2_4f3` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion3_4f2 | -9229.784207 | -4.053876 |
| `ion2_4f25d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion3_4f15d1 | -9229.800798 | -4.070467 |
| `ion2_4f26s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,17` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | multi-donor merge: ion3_4f2 + ion4_6s1 | -9229.754789 | -4.024457 |
| `ion1_4f36s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,24` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | donor: ion2_4f26s1 | -9230.146208 | -4.415877 |
| `ion1_4f35d1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion2_4f25d1 | -9230.106510 | -4.376179 |
| `ion1_4f25d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,22` | 0(8) 1(23) 2(34) 3(38) 4(36) 5(30) 6(22) 7(14) 8(7) 9(2) | multi-donor merge: ion2_4f26s1 + ion2_4f25d1 | -9230.181636 | -4.451305 |
| `ion0_4f36s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion1_4f36s1 | -9230.311670 | -4.581338 |
| `ion0_4f25d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion1_4f25d16s1 | -9230.359432 | -4.629101 |
| `ion6_5p5` | 6+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion5_closed | -9222.686498 | +3.043833 |
| `ion5_5p54f1` | 5+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion4_4f1 | -9225.336379 | +0.393952 |

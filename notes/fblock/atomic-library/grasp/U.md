# U (Z=92, A=238) — GRASP2018 reference recipe

n<=4 shells common core (60e, small-core An standard); Z_eff = 32; valence 5s 5p 5d 6s 6p + 5f 6d 7s

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=238, static

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
rnucleus     : 92 / 238 / n / 0 / 0.5 / 1 / 1
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
| `ion6_closed` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -28008.422091 | +0.000000 |
| `ion5_5f1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: ion6_closed  +  staged birth `5f-,5f` | -28010.652549 | -2.230459 |
| `ion5_6d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -28010.252278 | -1.830187 |
| `ion5_7s1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -28010.051172 | -1.629082 |
| `ion4_5f2` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion5_5f1 | -28012.291444 | -3.869353 |
| `ion4_6d2` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -28011.703121 | -3.281030 |
| `ion4_5f16d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion4_6d2 + ion5_5f1 | -28012.049188 | -3.627098 |
| `ion3_5f3` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion4_5f2 | -28013.382242 | -4.960151 |
| `ion3_5f26d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion4_5f16d1 | -28013.286395 | -4.864305 |
| `ion2_5f4` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion3_5f3 | -28013.978117 | -5.556026 |
| `ion2_5f36d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion3_5f26d1 | -28014.012835 | -5.590744 |
| `ion2_5f37s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,24` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | donor: ion3_5f3  +  staged birth `7s` | -28014.030257 | -5.608166 |
| `ion2_5f26d2` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,26` | 0(21) 1(40) 2(70) 3(71) 4(78) 5(61) 6(52) 7(31) 8(21) 9(8) 10(4) | donor: ion3_5f26d1 | -28013.958326 | -5.536235 |
| `ion1_5f37s2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion2_5f37s1 | -28014.442342 | -6.020251 |
| `ion1_5f36d2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,33` | 1/2(93) 3/2(172) 5/2(225) 7/2(248) 9/2(240) 11/2(211) 13/2(166) 15/2(120) 17/2(77) 19/2(44) 21/2(21) 23/2(9) 25/2(2) | donor: ion2_5f36d1 | -28014.359057 | -5.936966 |
| `ion1_5f36d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,29` | 1/2(48) 3/2(86) 5/2(112) 7/2(122) 9/2(115) 11/2(98) 13/2(75) 15/2(50) 17/2(30) 19/2(16) 21/2(6) 23/2(1) | donor: ion2_5f36d1  +  staged birth `7s` | -28014.414843 | -5.992753 |
| `ion0_5f36d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion1_5f36d17s1 | -28014.607697 | -6.185606 |
| `ion0_5f47s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion1_5f37s2 | -28014.535789 | -6.113699 |
| `ion7_6p5` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion6_closed | -28005.025372 | +3.396719 |
| `ion6_6p55f1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion5_5f1 | -28007.792182 | +0.629909 |

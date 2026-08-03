# Np (Z=93, A=237) — GRASP2018 reference recipe

n<=4 shells common core (60e); Z_eff = 33; valence 5s 5p 5d 6s 6p + 5f 6d 7s

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=237, static

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
rnucleus     : 93 / 237 / n / 0 / 0.5 / 1 / 1
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
| `ion7_closed` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -28798.353958 | +0.000000 |
| `ion6_5f1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_U5f1  +  staged birth `5f-,5f` | -28801.303340 | -2.949382 |
| `ion6_6d1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -28800.680733 | -2.326775 |
| `ion6_7s1` | 6+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -28800.390157 | -2.036199 |
| `ion5_5f2` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion6_5f1 | -28803.611319 | -5.257361 |
| `ion5_6d2` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -28802.598650 | -4.244692 |
| `ion5_5f16d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion5_6d2 + ion6_5f1 | -28803.161075 | -4.807117 |
| `ion4_5f3` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion5_5f2 | -28805.315501 | -6.961542 |
| `ion4_5f26d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion5_5f16d1 | -28805.028600 | -6.674642 |
| `ion3_5f4` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion4_5f3 | -28806.458804 | -8.104846 |
| `ion3_5f36d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,28` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion4_5f26d1 | -28806.324319 | -7.970361 |
| `ion3_5f37s1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,24` | 0(2) 1(8) 2(13) 3(14) 4(14) 5(12) 6(8) 7(6) 8(4) 9(1) | multi-donor merge: ion4_5f3 + ion6_7s1 | -28806.279557 | -7.925599 |
| `ion2_5f5` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion3_5f4 | -28807.093023 | -8.739065 |
| `ion2_5f47s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,31` | 1/2(13) 3/2(24) 5/2(30) 7/2(32) 9/2(33) 11/2(27) 13/2(20) 15/2(14) 17/2(9) 19/2(4) 21/2(2) | multi-donor merge: ion3_5f4 + ion3_5f37s1 | -28807.117596 | -8.763638 |
| `ion2_5f46d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion3_5f36d1 | -28807.096612 | -8.742654 |
| `ion1_5f47s2` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,30` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: ion2_5f47s1 | -28807.536514 | -9.182556 |
| `ion1_5f46d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,36` | 0(54) 1(153) 2(231) 3(278) 4(288) 5(268) 6(229) 7(177) 8(123) 9(78) 10(44) 11(21) 12(8) 13(2) | multi-donor merge: ion2_5f46d1 + ion2_5f47s1 | -28807.505291 | -9.151333 |
| `ion0_5f46d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,35` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion1_5f46d17s1 | -28807.701425 | -9.347467 |
| `ion0_5f57s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,37` | 1/2(10) 3/2(21) 5/2(28) 7/2(30) 9/2(29) 11/2(26) 13/2(20) 15/2(16) 17/2(9) 19/2(5) 21/2(3) 23/2(1) | donor: ion1_5f47s2 | -28807.658773 | -9.304815 |
| `ion8_6p5` | 8+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion7_closed | -28794.313964 | +4.039994 |
| `ion7_6p55f1` | 7+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion6_5f1 | -28797.832492 | +0.521466 |

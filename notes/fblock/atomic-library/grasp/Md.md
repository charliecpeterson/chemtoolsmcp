# Md (Z=101, A=258) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=258, static

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
rnucleus     : 101 / 258 / n / 0 / 0.5 / 1 / 1
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
| `ion15_closed` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -35669.353356 | +0.000000 |
| `ion14_5f1` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Fmf1  +  staged birth `5f-,5f` | -35679.882612 | -10.529256 |
| `ion14_6d1` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -35676.940888 | -7.587532 |
| `ion14_7s1` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -35675.731527 | -6.378170 |
| `ion13_6d2` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -35683.923378 | -14.570022 |
| `ion13_5f16d1` | 13+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion13_6d2 + ion14_5f1 | -35686.765873 | -17.412517 |
| `ion5_5f96d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,70` | 0(49) 1(138) 2(207) 3(252) 4(267) 5(254) 6(221) 7(176) 8(129) 9(87) 10(52) 11(27) 12(13) 13(5) 14(1) | donor: donor_chemfd | -35733.749705 | -64.396349 |
| `ion5_5f10` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,72` | 0(6) 1(7) 2(17) 3(13) 4(19) 5(14) 6(13) 7(7) 8(7) 9(2) 10(2) | donor: donor_chemf | -35734.574259 | -65.220903 |
| `ion4_5f106d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: ion5_5f96d1 | -35736.107667 | -66.754311 |
| `ion4_5f11` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: ion5_5f10 | -35736.725186 | -67.371830 |
| `ion3_5f116d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion4_5f106d1 | -35737.802435 | -68.449079 |
| `ion3_5f12` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion4_5f11 | -35738.219590 | -68.866234 |
| `ion2_5f126d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion3_5f116d1 | -35738.882224 | -69.528868 |
| `ion2_5f13` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,93` | 5/2(1) 7/2(1) | donor: ion3_5f12 | -35739.105140 | -69.751784 |
| `ion2_5f127s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,87` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | multi-donor merge: ion3_5f12 + donor_chems  +  staged birth `7s` | -35738.957292 | -69.603936 |
| `ion1_5f137s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,94` | 2(1) 3(2) 4(1) | multi-donor merge: ion2_5f13 + ion2_5f127s1 | -35739.535106 | -70.181750 |
| `ion1_5f126d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,92` | 0(8) 1(23) 2(34) 3(38) 4(36) 5(30) 6(22) 7(14) 8(7) 9(2) | multi-donor merge: ion2_5f127s1 + ion2_5f126d1 | -35739.343004 | -69.989648 |
| `ion0_5f137s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,93` | 5/2(1) 7/2(1) | donor: ion1_5f137s1 | -35739.733840 | -70.380484 |
| `ion0_5f126d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion1_5f126d17s1 | -35739.567369 | -70.214013 |
| `ion16_6p5` | 16+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion15_closed | -35658.856388 | +10.496969 |
| `ion15_6p55f1` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion14_5f1 | -35670.181771 | -0.828415 |

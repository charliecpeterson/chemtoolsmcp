# No (Z=102, A=259) — GRASP2018 reference recipe

Xe-menu core; generated ladder

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=259, static

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
rnucleus     : 102 / 259 / n / 0 / 0.5 / 1 / 1
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
| `ion16_closed` | 16+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `0,4` | 0(1) | donor: donor_closed | -36602.247128 | +0.000000 |
| `ion15_5f1` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Mdf1  +  staged birth `5f-,5f` | -36613.934206 | -11.687078 |
| `ion15_6d1` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `1,7` | 3/2(1) 5/2(1) | donor: donor_d1 | -36610.642266 | -8.395137 |
| `ion15_7s1` | 15+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,4f` / `1,5` | 1/2(1) | donor: donor_s1 | -36609.298799 | -7.051671 |
| `ion14_6d2` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | Xe (menu 5) | `6s,6p,6d,4f` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | donor: donor_d2 | -36618.410350 | -16.163222 |
| `ion14_5f16d1` | 14+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion14_6d2 + ion15_5f1 | -36621.598708 | -19.351580 |
| `ion5_5f106d1` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,77` | 1/2(54) 3/2(99) 5/2(132) 7/2(146) 9/2(142) 11/2(126) 13/2(103) 15/2(74) 17/2(49) 19/2(29) 21/2(15) 23/2(6) 25/2(2) | donor: donor_chemfd | -36679.401673 | -77.154545 |
| `ion5_5f11` | 5+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,79` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | donor: donor_chemf | -36680.270314 | -78.023186 |
| `ion4_5f116d1` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,84` | 0(13) 1(35) 2(51) 3(61) 4(61) 5(54) 6(44) 7(31) 8(19) 9(11) 10(5) 11(1) | donor: ion5_5f106d1 | -36681.814028 | -79.566900 |
| `ion4_5f12` | 4+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,86` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion5_5f11 | -36682.470134 | -80.223006 |
| `ion3_5f126d1` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `1,91` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | donor: ion4_5f116d1 | -36683.552494 | -81.305365 |
| `ion3_5f13` | 3+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `1,93` | 5/2(1) 7/2(1) | donor: ion4_5f12 | -36684.002280 | -81.755152 |
| `ion2_5f136d1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)` | Xe (menu 5) | `6s,6p,6d,5f` / `0,98` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | donor: ion3_5f126d1 | -36684.665601 | -82.418473 |
| `ion2_5f14` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,4` | 0(1) | donor: ion3_5f13 | -36684.914355 | -82.667227 |
| `ion2_5f137s1` | 2+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,94` | 2(1) 3(2) 4(1) | multi-donor merge: ion3_5f13 + donor_chems  +  staged birth `7s` | -36684.749749 | -82.502621 |
| `ion1_5f147s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(1,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `1,5` | 1/2(1) | multi-donor merge: ion2_5f14 + ion2_5f137s1 | -36685.349892 | -83.102764 |
| `ion1_5f136d17s1` | 1+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)7s(1,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `1,99` | 1/2(4) 3/2(7) 5/2(8) 7/2(8) 9/2(7) 11/2(4) 13/2(1) | multi-donor merge: ion2_5f137s1 + ion2_5f136d1 | -36685.133264 | -82.886135 |
| `ion0_5f147s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(2,i)` | Xe (menu 5) | `7s,6p,5d,5f` / `0,4` | 0(1) | donor: ion1_5f147s1 | -36685.551229 | -83.304101 |
| `ion0_5f136d17s2` | 0+ | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)7s(2,i)` | Xe (menu 5) | `7s,6p,6d,5f` / `0,98` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | donor: ion1_5f136d17s1 | -36685.361711 | -83.114583 |
| `ion17_6p5` | 17+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | Xe (menu 5) | `6s,6p,5d,4f` / `1,17` | 1/2(1) 3/2(1) | donor: ion16_closed | -36590.784107 | +11.463021 |
| `ion16_6p55f1` | 16+ | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | Xe (menu 5) | `6s,6p,5d,5f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion15_5f1 | -36603.293481 | -1.046353 |

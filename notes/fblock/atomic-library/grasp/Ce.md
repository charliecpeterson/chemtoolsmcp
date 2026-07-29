# Ce (Z=58, A=140) — GRASP2018 reference recipe

[Kr]4d10 common core (46e, ccECP core choice); Z_eff = 12; valence 4f 5s 5p 5d 6s

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=140, static

Energies are (2J+1)-weighted configuration averages in Hartree, from
the 2026-07-28 v2 rebuild (per-configuration J ceiling; the earlier
vintage truncated high-f manifolds by up to 250 meV).

## Run recipe

```
rnucleus     : 58 / 140 / n / 0 / 0.5 / 1 / 1
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
| `ion4_closed` | 4+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | cold (Thomas-Fermi) | -8851.171673 | +0.000000 |
| `ion3_4f1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | cold (Thomas-Fermi) | -8852.456545 | -1.284873 |
| `ion3_5d1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | cold (Thomas-Fermi) | -8852.254412 | -1.082740 |
| `ion3_6s1` | 3+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | cold (Thomas-Fermi) | -8852.099005 | -0.927333 |
| `ion2_4f2` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | cold (Thomas-Fermi) | -8853.089198 | -1.917525 |
| `ion2_5d16s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d` / `0,8` | 1(1) 2(2) 3(1) | cold (Thomas-Fermi) | -8852.900042 | -1.728369 |
| `ion2_4f15d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | cold (Thomas-Fermi) | -8853.129077 | -1.957405 |
| `ion2_5d2` | 2+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | cold (Thomas-Fermi) | -8852.980980 | -1.809308 |
| `ion2_4f16s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,10` | 2(1) 3(2) 4(1) | cold (Thomas-Fermi) | -8853.079627 | -1.907954 |
| `ion1_4f15d2` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,19` | 1/2(7) 3/2(13) 5/2(16) 7/2(16) 9/2(13) 11/2(9) 13/2(5) 15/2(2) | cold (Thomas-Fermi) | -8853.492153 | -2.320480 |
| `ion1_4f15d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `1,15` | 1/2(4) 3/2(7) 5/2(8) 7/2(8) 9/2(7) 11/2(4) 13/2(1) | cold (Thomas-Fermi) | -8853.504269 | -2.332596 |
| `ion1_4f26s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,17` | 1/2(3) 3/2(4) 5/2(4) 7/2(4) 9/2(4) 11/2(3) 13/2(2) | cold (Thomas-Fermi) | -8853.445822 | -2.274149 |
| `ion0_4f15d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | cold (Thomas-Fermi) | -8853.679367 | -2.507695 |
| `ion0_4f26s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | cold (Thomas-Fermi) | -8853.608869 | -2.437196 |
| `ion5_5p5` | 5+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion4_closed | -8848.758258 | +2.413414 |
| `ion4_5p54f1` | 4+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion3_4f1 | -8850.578789 | +0.592884 |
| `ion1_4f3` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,23` | 1/2(2) 3/2(6) 5/2(7) 7/2(7) 9/2(7) 11/2(5) 13/2(3) 15/2(3) 17/2(1) | cold (Thomas-Fermi) | -8853.189025 | -2.017353 |
| `ion1_4f25d1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `1,21` | 1/2(8) 3/2(15) 5/2(19) 7/2(19) 9/2(17) 11/2(13) 13/2(9) 15/2(5) 17/2(2) | cold (Thomas-Fermi) | -8853.410688 | -2.239016 |
| `ion1_5d3` | 1+ | `4d(10,i)5s(2,i)5p(6,i)5d(3,i)` | Kr (menu 4) | `5s,5p,5d` / `1,17` | 1/2(2) 3/2(5) 5/2(5) 7/2(3) 9/2(3) 11/2(1) | cold (Thomas-Fermi) | -8853.386180 | -2.214508 |
| `ion1_5d26s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d` / `1,13` | 1/2(3) 3/2(4) 5/2(4) 7/2(3) 9/2(2) | cold (Thomas-Fermi) | -8853.372050 | -2.200377 |
| `ion0_4f15d26s1` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(2,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d,4f` / `0,20` | 0(7) 1(20) 2(29) 3(32) 4(29) 5(22) 6(14) 7(7) 8(2) | cold (Thomas-Fermi) | -8853.655429 | -2.483756 |

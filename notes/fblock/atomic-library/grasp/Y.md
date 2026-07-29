# Y (Z=39, A=89) — GRASP2018 reference recipe

[Kr] common to all states; ECP core will be [Ar]3d10 (28e)

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=89, static

Energies are (2J+1)-weighted configuration averages in Hartree, from
the 2026-07-28 v2 rebuild (per-configuration J ceiling; the earlier
vintage truncated high-f manifolds by up to 250 meV).

## Run recipe

```
rnucleus     : 39 / 89 / n / 0 / 0.5 / 1 / 1
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
| `ion0_4d15s2` | 0+ | `` | None (menu None) | `` / `` | 3/2(1) 5/2(1) | cold (Thomas-Fermi) | -3381.883583 | -1.325483 |
| `ion0_4d25s1` | 0+ | `` | None (menu None) | `` / `` | 1/2(3) 3/2(4) 5/2(4) 7/2(3) 9/2(2) | cold (Thomas-Fermi) | -3381.822022 | -1.263923 |
| `ion0_5s25p1` | 0+ | `` | None (menu None) | `` / `` | 1/2(1) 3/2(1) | cold (Thomas-Fermi) | -3381.839692 | -1.281593 |
| `ion0_4d15s15p1` | 0+ | `` | None (menu None) | `` / `` | 1/2(4) 3/2(7) 5/2(7) 7/2(4) 9/2(1) | cold (Thomas-Fermi) | -3381.817049 | -1.258949 |
| `ion0_4d3` | 0+ | `` | None (menu None) | `` / `` | 1/2(2) 3/2(5) 5/2(5) 7/2(3) 9/2(3) 11/2(1) | cold (Thomas-Fermi) | -3381.728327 | -1.170228 |
| `ion1_5s2` | 1+ | `` | None (menu None) | `` / `` | 0(1) | cold (Thomas-Fermi) | -3381.687768 | -1.129669 |
| `ion1_4d15s1` | 1+ | `` | None (menu None) | `` / `` | 1(1) 2(2) 3(1) | cold (Thomas-Fermi) | -3381.690982 | -1.132882 |
| `ion1_4d2` | 1+ | `` | None (menu None) | `` / `` | 0(2) 1(1) 2(3) 3(1) 4(2) | cold (Thomas-Fermi) | -3381.640599 | -1.082499 |
| `ion1_5s15p1` | 1+ | `` | None (menu None) | `` / `` | 0(1) 1(2) 2(1) | cold (Thomas-Fermi) | -3381.586179 | -1.028080 |
| `ion2_4d1` | 2+ | `` | None (menu None) | `` / `` | 3/2(1) 5/2(1) | cold (Thomas-Fermi) | -3381.277412 | -0.719313 |
| `ion2_5s1` | 2+ | `` | None (menu None) | `` / `` | 1/2(1) | cold (Thomas-Fermi) | -3381.251026 | -0.692927 |
| `ion2_5p1` | 2+ | `` | None (menu None) | `` / `` | 1/2(1) 3/2(1) | cold (Thomas-Fermi) | -3381.100909 | -0.542810 |
| `ion3_closed` | 3+ | `` | None (menu None) | `` / `` | 0(1) | cold (Thomas-Fermi) | -3380.558099 | +0.000000 |
| `ion2_4f1` | 2+ | `` | None (menu None) | `` / `` | 5/2(1) 7/2(1) | cold (Thomas-Fermi) | -3380.843874 | -0.285774 |
| `ion2_6s1` | 2+ | `` | None (menu None) | `` / `` | 1/2(1) | cold (Thomas-Fermi) | -3380.908618 | -0.350519 |
| `ion1_4d15p1` | 1+ | `` | None (menu None) | `` / `` | 0(1) 1(3) 2(4) 3(3) 4(1) | cold (Thomas-Fermi) | -3381.577547 | -1.019447 |
| `ion1_5p2` | 1+ | `` | None (menu None) | `` / `` | 0(2) 1(1) 2(2) | cold (Thomas-Fermi) | -3381.437247 | -0.879148 |
| `ion4_4p5` | 4+ | `3d(10,i)4s(2,i)4p(5,i)` | Ar (menu 3) | `4s,4p,3d` / `1,17` | 1/2(1) 3/2(1) | donor: ion3_closed | -3378.352986 | +2.205114 |

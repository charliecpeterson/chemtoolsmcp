# La (Z=57, A=139) — GRASP2018 reference recipe

Kr-menu core; hand ladder (nf=0)

Hamiltonian: Dirac-Coulomb + Breit (rci transverse photon, low-frequency limit; no QED, no mass shifts); rmcdhf EAL, (2J+1) weights, all levels
Nucleus: Fermi, A=139, static

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
rnucleus     : 57 / 139 / n / 0 / 0.5 / 1 / 1
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
| `ion3_closed` | 3+ | `4d(10,i)5s(2,i)5p(6,i)` | Kr (menu 4) | `5s,5p,4d` / `0,4` | 0(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -8485.477240 | +0.000000 |
| `ion2_4f1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: donor_Cef1  +  staged birth `4f-,4f` | -8486.074004 | -0.596764 |
| `ion2_5d1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -8486.142439 | -0.665199 |
| `ion2_6s1` | 2+ | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d` / `1,5` | 1/2(1) | ATSP-hf seed (non-relativistic orbitals, converted) | -8486.090552 | -0.613312 |
| `ion1_5d2` | 1+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | Kr (menu 4) | `5s,5p,5d` / `0,12` | 0(2) 1(1) 2(3) 3(1) 4(2) | ATSP-hf seed (non-relativistic orbitals, converted) | -8486.503226 | -1.025986 |
| `ion1_4f15d1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | Kr (menu 4) | `5s,5p,5d,4f` / `0,14` | 0(1) 1(3) 2(4) 3(4) 4(4) 5(3) 6(1) | multi-donor merge: ion1_5d2 + ion2_4f1 | -8486.393388 | -0.916148 |
| `ion1_4f2` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,16` | 0(2) 1(1) 2(3) 3(1) 4(3) 5(1) 6(2) | donor: ion2_4f1 | -8486.159791 | -0.682550 |
| `ion1_5d16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d` / `0,8` | 1(1) 2(2) 3(1) | multi-donor merge: ion1_5d2 + ion2_6s1 | -8486.511771 | -1.034531 |
| `ion1_4f16s1` | 1+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(1,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `0,10` | 2(1) 3(2) 4(1) | multi-donor merge: ion2_4f1 + ion2_6s1 | -8486.424840 | -0.947599 |
| `ion0_5d16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,5d` / `1,7` | 3/2(1) 5/2(1) | donor: ion1_5d16s1 | -8486.684105 | -1.206865 |
| `ion0_4f16s2` | 0+ | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(2,i)` | Kr (menu 4) | `6s,5p,4d,4f` / `1,9` | 5/2(1) 7/2(1) | donor: ion1_4f16s1 | -8486.585271 | -1.108031 |
| `ion0_5d26s1` | 0+ | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)6s(1,i)` | Kr (menu 4) | `6s,5p,5d` / `1,13` | 1/2(3) 3/2(4) 5/2(4) 7/2(3) 9/2(2) | multi-donor merge: ion1_5d2 + ion1_5d16s1 | -8486.663782 | -1.186542 |
| `ion4_5p5` | 4+ | `4d(10,i)5s(2,i)5p(5,i)` | Kr (menu 4) | `5s,5p,4d` / `1,17` | 1/2(1) 3/2(1) | donor: ion3_closed | -8483.643386 | +1.833854 |
| `ion3_5p54f1` | 3+ | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | Kr (menu 4) | `5s,5p,4d,4f` / `0,24` | 1(1) 2(3) 3(4) 4(3) 5(1) | donor: ion2_4f1 | -8484.724219 | +0.753021 |

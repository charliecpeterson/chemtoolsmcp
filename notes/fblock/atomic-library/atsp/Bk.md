# Bk — ATSP2K `hf` configuration inputs

`Z = 97`, `Z_eff = 37.0`, `n_core = 60`, `l_max = 4`  (module `fit_ecp_bk`)

Seed state (run first every evaluation, seeds all others): `ion0_5f97s2`

## stdin per state

```
Bk,AV,97.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion11_closed` | `  5s  5p  5d  6s  6p` | `(none)` |
| `ion10_5f1` | `  5s  5p  5d  6s  6p` | `5f(1)` |
| `ion10_6d1` | `  5s  5p  5d  6s  6p` | `6d(1)` |
| `ion10_7s1` | `  5s  5p  5d  6s  6p` | `7s(1)` |
| `ion9_6d2` | `  5s  5p  5d  6s  6p` | `6d(2)` |
| `ion9_5f16d1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)` |
| `ion5_5f56d1` | `  5s  5p  5d  6s  6p` | `5f(5)6d(1)` |
| `ion5_5f6` | `  5s  5p  5d  6s  6p` | `5f(6)` |
| `ion4_5f66d1` | `  5s  5p  5d  6s  6p` | `5f(6)6d(1)` |
| `ion4_5f7` | `  5s  5p  5d  6s  6p` | `5f(7)` |
| `ion3_5f76d1` | `  5s  5p  5d  6s  6p` | `5f(7)6d(1)` |
| `ion3_5f8` | `  5s  5p  5d  6s  6p` | `5f(8)` |
| `ion2_5f86d1` | `  5s  5p  5d  6s  6p` | `5f(8)6d(1)` |
| `ion2_5f9` | `  5s  5p  5d  6s  6p` | `5f(9)` |
| `ion2_5f87s1` | `  5s  5p  5d  6s  6p` | `5f(8)7s(1)` |
| `ion1_5f97s1` | `  5s  5p  5d  6s  6p` | `5f(9)7s(1)` |
| `ion1_5f86d17s1` | `  5s  5p  5d  6s  6p` | `5f(8)6d(1)7s(1)` |
| `ion0_5f97s2` | `  5s  5p  5d  6s  6p  7s` | `5f(9)` |
| `ion0_5f86d17s2` | `  5s  5p  5d  6s  6p  7s` | `5f(8)6d(1)` |
| `ion12_6p5` | `  5s  5p  5d  6s` | `6p(5)` |
| `ion11_6p55f1` | `  5s  5p  5d  6s` | `6p(5)5f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

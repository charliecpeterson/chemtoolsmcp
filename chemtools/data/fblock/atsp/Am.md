# Am — ATSP2K `hf` configuration inputs

`Z = 95`, `Z_eff = 35.0`, `n_core = 60`, `l_max = 4`  (module `fit_ecp_am`)

Seed state (run first every evaluation, seeds all others): `ion0_5f77s2`

## stdin per state

```
Am,AV,95.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion9_closed` | `  5s  5p  5d  6s  6p` | `(none)` |
| `ion8_5f1` | `  5s  5p  5d  6s  6p` | `5f(1)` |
| `ion8_6d1` | `  5s  5p  5d  6s  6p` | `6d(1)` |
| `ion8_7s1` | `  5s  5p  5d  6s  6p` | `7s(1)` |
| `ion7_6d2` | `  5s  5p  5d  6s  6p` | `6d(2)` |
| `ion7_5f16d1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)` |
| `ion7_5f2` | `  5s  5p  5d  6s  6p` | `5f(2)` |
| `ion6_5f26d1` | `  5s  5p  5d  6s  6p` | `5f(2)6d(1)` |
| `ion6_5f3` | `  5s  5p  5d  6s  6p` | `5f(3)` |
| `ion5_5f36d1` | `  5s  5p  5d  6s  6p` | `5f(3)6d(1)` |
| `ion5_5f4` | `  5s  5p  5d  6s  6p` | `5f(4)` |
| `ion4_5f46d1` | `  5s  5p  5d  6s  6p` | `5f(4)6d(1)` |
| `ion4_5f5` | `  5s  5p  5d  6s  6p` | `5f(5)` |
| `ion3_5f56d1` | `  5s  5p  5d  6s  6p` | `5f(5)6d(1)` |
| `ion3_5f6` | `  5s  5p  5d  6s  6p` | `5f(6)` |
| `ion2_5f66d1` | `  5s  5p  5d  6s  6p` | `5f(6)6d(1)` |
| `ion2_5f7` | `  5s  5p  5d  6s  6p` | `5f(7)` |
| `ion2_5f67s1` | `  5s  5p  5d  6s  6p` | `5f(6)7s(1)` |
| `ion1_5f77s1` | `  5s  5p  5d  6s  6p` | `5f(7)7s(1)` |
| `ion1_5f66d17s1` | `  5s  5p  5d  6s  6p` | `5f(6)6d(1)7s(1)` |
| `ion0_5f77s2` | `  5s  5p  5d  6s  6p  7s` | `5f(7)` |
| `ion0_5f66d17s2` | `  5s  5p  5d  6s  6p  7s` | `5f(6)6d(1)` |
| `ion10_6p5` | `  5s  5p  5d  6s` | `6p(5)` |
| `ion9_6p55f1` | `  5s  5p  5d  6s` | `6p(5)5f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

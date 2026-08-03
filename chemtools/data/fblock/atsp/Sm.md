# Sm — ATSP2K `hf` configuration inputs

`Z = 62`, `Z_eff = 34.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_sm`)

Seed state (run first every evaluation, seeds all others): `ion0_4f66s2`

## stdin per state

```
Sm,AV,62.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion8_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion7_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion7_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion7_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion6_4f2` | `  4s  4p  4d  5s  5p` | `4f(2)` |
| `ion6_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion6_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion5_4f3` | `  4s  4p  4d  5s  5p` | `4f(3)` |
| `ion5_4f25d1` | `  4s  4p  4d  5s  5p` | `4f(2)5d(1)` |
| `ion4_4f4` | `  4s  4p  4d  5s  5p` | `4f(4)` |
| `ion4_4f35d1` | `  4s  4p  4d  5s  5p` | `4f(3)5d(1)` |
| `ion3_4f5` | `  4s  4p  4d  5s  5p` | `4f(5)` |
| `ion3_4f45d1` | `  4s  4p  4d  5s  5p` | `4f(4)5d(1)` |
| `ion3_4f46s1` | `  4s  4p  4d  5s  5p` | `4f(4)6s(1)` |
| `ion2_4f6` | `  4s  4p  4d  5s  5p` | `4f(6)` |
| `ion2_4f56s1` | `  4s  4p  4d  5s  5p` | `4f(5)6s(1)` |
| `ion1_4f66s1` | `  4s  4p  4d  5s  5p` | `4f(6)6s(1)` |
| `ion1_4f55d16s1` | `  4s  4p  4d  5s  5p` | `4f(5)5d(1)6s(1)` |
| `ion0_4f66s2` | `  4s  4p  4d  5s  5p  6s` | `4f(6)` |
| `ion0_4f55d16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(5)5d(1)` |
| `ion9_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion8_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

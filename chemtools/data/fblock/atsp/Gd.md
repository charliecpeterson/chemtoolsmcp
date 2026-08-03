# Gd — ATSP2K `hf` configuration inputs

`Z = 64`, `Z_eff = 36.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_gd`)

Seed state (run first every evaluation, seeds all others): `ion0_4f75d16s2`

## stdin per state

```
Gd,AV,64.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion10_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion9_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion9_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion9_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion8_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion8_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion8_4f2` | `  4s  4p  4d  5s  5p` | `4f(2)` |
| `ion7_4f25d1` | `  4s  4p  4d  5s  5p` | `4f(2)5d(1)` |
| `ion7_4f3` | `  4s  4p  4d  5s  5p` | `4f(3)` |
| `ion6_4f35d1` | `  4s  4p  4d  5s  5p` | `4f(3)5d(1)` |
| `ion6_4f4` | `  4s  4p  4d  5s  5p` | `4f(4)` |
| `ion5_4f45d1` | `  4s  4p  4d  5s  5p` | `4f(4)5d(1)` |
| `ion5_4f5` | `  4s  4p  4d  5s  5p` | `4f(5)` |
| `ion4_4f55d1` | `  4s  4p  4d  5s  5p` | `4f(5)5d(1)` |
| `ion4_4f6` | `  4s  4p  4d  5s  5p` | `4f(6)` |
| `ion3_4f65d1` | `  4s  4p  4d  5s  5p` | `4f(6)5d(1)` |
| `ion3_4f7` | `  4s  4p  4d  5s  5p` | `4f(7)` |
| `ion2_4f66s1` | `  4s  4p  4d  5s  5p` | `4f(6)6s(1)` |
| `ion1_4f76s1` | `  4s  4p  4d  5s  5p` | `4f(7)6s(1)` |
| `ion1_4f65d16s1` | `  4s  4p  4d  5s  5p` | `4f(6)5d(1)6s(1)` |
| `ion0_4f75d16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(7)5d(1)` |
| `ion0_4f86s2` | `  4s  4p  4d  5s  5p  6s` | `4f(8)` |
| `ion11_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion10_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

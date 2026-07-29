# Nd — ATSP2K `hf` configuration inputs

`Z = 60`, `Z_eff = 32.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_nd`)

Seed state (run first every evaluation, seeds all others): `ion0_4f46s2`

## stdin per state

```
Nd,AV,60.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion6_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion5_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion5_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion5_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion4_4f2` | `  4s  4p  4d  5s  5p` | `4f(2)` |
| `ion4_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion4_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion3_4f3` | `  4s  4p  4d  5s  5p` | `4f(3)` |
| `ion3_4f25d1` | `  4s  4p  4d  5s  5p` | `4f(2)5d(1)` |
| `ion3_4f26s1` | `  4s  4p  4d  5s  5p` | `4f(2)6s(1)` |
| `ion2_4f4` | `  4s  4p  4d  5s  5p` | `4f(4)` |
| `ion2_4f35d1` | `  4s  4p  4d  5s  5p` | `4f(3)5d(1)` |
| `ion2_4f36s1` | `  4s  4p  4d  5s  5p` | `4f(3)6s(1)` |
| `ion1_4f46s1` | `  4s  4p  4d  5s  5p` | `4f(4)6s(1)` |
| `ion1_4f35d16s1` | `  4s  4p  4d  5s  5p` | `4f(3)5d(1)6s(1)` |
| `ion0_4f46s2` | `  4s  4p  4d  5s  5p  6s` | `4f(4)` |
| `ion0_4f35d16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(3)5d(1)` |
| `ion1_4f45d1` | `  4s  4p  4d  5s  5p` | `4f(4)5d(1)` |
| `ion4_4f16s1` | `  4s  4p  4d  5s  5p` | `4f(1)6s(1)` |
| `ion2_4f25d2` | `  4s  4p  4d  5s  5p` | `4f(2)5d(2)` |
| `ion7_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion6_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

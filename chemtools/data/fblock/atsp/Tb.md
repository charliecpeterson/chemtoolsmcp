# Tb — ATSP2K `hf` configuration inputs

`Z = 65`, `Z_eff = 37.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_tb`)

Seed state (run first every evaluation, seeds all others): `ion0_4f96s2`

## stdin per state

```
Tb,AV,65.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion11_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion10_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion10_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion10_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion9_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion9_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion5_4f55d1` | `  4s  4p  4d  5s  5p` | `4f(5)5d(1)` |
| `ion5_4f6` | `  4s  4p  4d  5s  5p` | `4f(6)` |
| `ion4_4f65d1` | `  4s  4p  4d  5s  5p` | `4f(6)5d(1)` |
| `ion4_4f7` | `  4s  4p  4d  5s  5p` | `4f(7)` |
| `ion3_4f75d1` | `  4s  4p  4d  5s  5p` | `4f(7)5d(1)` |
| `ion3_4f8` | `  4s  4p  4d  5s  5p` | `4f(8)` |
| `ion2_4f85d1` | `  4s  4p  4d  5s  5p` | `4f(8)5d(1)` |
| `ion2_4f9` | `  4s  4p  4d  5s  5p` | `4f(9)` |
| `ion2_4f86s1` | `  4s  4p  4d  5s  5p` | `4f(8)6s(1)` |
| `ion1_4f96s1` | `  4s  4p  4d  5s  5p` | `4f(9)6s(1)` |
| `ion1_4f85d16s1` | `  4s  4p  4d  5s  5p` | `4f(8)5d(1)6s(1)` |
| `ion0_4f96s2` | `  4s  4p  4d  5s  5p  6s` | `4f(9)` |
| `ion0_4f85d16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(8)5d(1)` |
| `ion12_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion11_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

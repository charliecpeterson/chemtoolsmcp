# Tm — ATSP2K `hf` configuration inputs

`Z = 69`, `Z_eff = 41.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_tm`)

Seed state (run first every evaluation, seeds all others): `ion0_4f136s2`

## stdin per state

```
Tm,AV,69.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion15_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion14_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion14_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion14_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion13_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion13_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion5_4f95d1` | `  4s  4p  4d  5s  5p` | `4f(9)5d(1)` |
| `ion5_4f10` | `  4s  4p  4d  5s  5p` | `4f(10)` |
| `ion4_4f105d1` | `  4s  4p  4d  5s  5p` | `4f(10)5d(1)` |
| `ion4_4f11` | `  4s  4p  4d  5s  5p` | `4f(11)` |
| `ion3_4f115d1` | `  4s  4p  4d  5s  5p` | `4f(11)5d(1)` |
| `ion3_4f12` | `  4s  4p  4d  5s  5p` | `4f(12)` |
| `ion2_4f125d1` | `  4s  4p  4d  5s  5p` | `4f(12)5d(1)` |
| `ion2_4f13` | `  4s  4p  4d  5s  5p` | `4f(13)` |
| `ion2_4f126s1` | `  4s  4p  4d  5s  5p` | `4f(12)6s(1)` |
| `ion1_4f136s1` | `  4s  4p  4d  5s  5p` | `4f(13)6s(1)` |
| `ion1_4f125d16s1` | `  4s  4p  4d  5s  5p` | `4f(12)5d(1)6s(1)` |
| `ion0_4f136s2` | `  4s  4p  4d  5s  5p  6s` | `4f(13)` |
| `ion0_4f125d16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(12)5d(1)` |
| `ion16_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion15_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

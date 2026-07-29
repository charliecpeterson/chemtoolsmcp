# Er — ATSP2K `hf` configuration inputs

`Z = 68`, `Z_eff = 40.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_er`)

Seed state (run first every evaluation, seeds all others): `ion0_4f126s2`

## stdin per state

```
Er,AV,68.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion14_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion13_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion13_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion13_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion12_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion12_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion5_4f85d1` | `  4s  4p  4d  5s  5p` | `4f(8)5d(1)` |
| `ion5_4f9` | `  4s  4p  4d  5s  5p` | `4f(9)` |
| `ion4_4f95d1` | `  4s  4p  4d  5s  5p` | `4f(9)5d(1)` |
| `ion4_4f10` | `  4s  4p  4d  5s  5p` | `4f(10)` |
| `ion3_4f105d1` | `  4s  4p  4d  5s  5p` | `4f(10)5d(1)` |
| `ion3_4f11` | `  4s  4p  4d  5s  5p` | `4f(11)` |
| `ion2_4f115d1` | `  4s  4p  4d  5s  5p` | `4f(11)5d(1)` |
| `ion2_4f12` | `  4s  4p  4d  5s  5p` | `4f(12)` |
| `ion2_4f116s1` | `  4s  4p  4d  5s  5p` | `4f(11)6s(1)` |
| `ion1_4f126s1` | `  4s  4p  4d  5s  5p` | `4f(12)6s(1)` |
| `ion1_4f115d16s1` | `  4s  4p  4d  5s  5p` | `4f(11)5d(1)6s(1)` |
| `ion0_4f126s2` | `  4s  4p  4d  5s  5p  6s` | `4f(12)` |
| `ion0_4f115d16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(11)5d(1)` |
| `ion15_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion14_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

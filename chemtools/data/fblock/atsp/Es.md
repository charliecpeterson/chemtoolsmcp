# Es — ATSP2K `hf` configuration inputs

`Z = 99`, `Z_eff = 39.0`, `n_core = 60`, `l_max = 4`  (module `fit_ecp_es`)

Seed state (run first every evaluation, seeds all others): `ion0_5f117s2`

## stdin per state

```
Es,AV,99.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion13_closed` | `  5s  5p  5d  6s  6p` | `(none)` |
| `ion12_5f1` | `  5s  5p  5d  6s  6p` | `5f(1)` |
| `ion12_6d1` | `  5s  5p  5d  6s  6p` | `6d(1)` |
| `ion12_7s1` | `  5s  5p  5d  6s  6p` | `7s(1)` |
| `ion11_6d2` | `  5s  5p  5d  6s  6p` | `6d(2)` |
| `ion11_5f16d1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)` |
| `ion5_5f76d1` | `  5s  5p  5d  6s  6p` | `5f(7)6d(1)` |
| `ion5_5f8` | `  5s  5p  5d  6s  6p` | `5f(8)` |
| `ion4_5f86d1` | `  5s  5p  5d  6s  6p` | `5f(8)6d(1)` |
| `ion4_5f9` | `  5s  5p  5d  6s  6p` | `5f(9)` |
| `ion3_5f96d1` | `  5s  5p  5d  6s  6p` | `5f(9)6d(1)` |
| `ion3_5f10` | `  5s  5p  5d  6s  6p` | `5f(10)` |
| `ion2_5f106d1` | `  5s  5p  5d  6s  6p` | `5f(10)6d(1)` |
| `ion2_5f11` | `  5s  5p  5d  6s  6p` | `5f(11)` |
| `ion2_5f107s1` | `  5s  5p  5d  6s  6p` | `5f(10)7s(1)` |
| `ion1_5f117s1` | `  5s  5p  5d  6s  6p` | `5f(11)7s(1)` |
| `ion1_5f106d17s1` | `  5s  5p  5d  6s  6p` | `5f(10)6d(1)7s(1)` |
| `ion0_5f117s2` | `  5s  5p  5d  6s  6p  7s` | `5f(11)` |
| `ion0_5f106d17s2` | `  5s  5p  5d  6s  6p  7s` | `5f(10)6d(1)` |
| `ion14_6p5` | `  5s  5p  5d  6s` | `6p(5)` |
| `ion13_6p55f1` | `  5s  5p  5d  6s` | `6p(5)5f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

# Pa — ATSP2K `hf` configuration inputs

`Z = 91`, `Z_eff = 31.0`, `n_core = 60`, `l_max = 4`  (module `fit_ecp_pa`)

Seed state (run first every evaluation, seeds all others): `ion0_5f26d17s2`

## stdin per state

```
Pa,AV,91.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion5_closed` | `  5s  5p  5d  6s  6p` | `(none)` |
| `ion4_5f1` | `  5s  5p  5d  6s  6p` | `5f(1)` |
| `ion4_6d1` | `  5s  5p  5d  6s  6p` | `6d(1)` |
| `ion4_7s1` | `  5s  5p  5d  6s  6p` | `7s(1)` |
| `ion3_6d2` | `  5s  5p  5d  6s  6p` | `6d(2)` |
| `ion3_5f16d1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)` |
| `ion3_5f2` | `  5s  5p  5d  6s  6p` | `5f(2)` |
| `ion3_5f17s1` | `  5s  5p  5d  6s  6p` | `5f(1)7s(1)` |
| `ion2_5f27s1` | `  5s  5p  5d  6s  6p` | `5f(2)7s(1)` |
| `ion2_5f16d17s1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)7s(1)` |
| `ion0_5f26d17s2` | `  5s  5p  5d  6s  6p  7s` | `5f(2)6d(1)` |
| `ion0_5f37s2` | `  5s  5p  5d  6s  6p  7s` | `5f(3)` |
| `ion6_6p5` | `  5s  5p  5d  6s` | `6p(5)` |
| `ion5_6p55f1` | `  5s  5p  5d  6s` | `6p(5)5f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

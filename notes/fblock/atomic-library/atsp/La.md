# La — ATSP2K `hf` configuration inputs

`Z = 57`, `Z_eff = 29.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_la`)

Seed state (run first every evaluation, seeds all others): `ion0_5d16s2`

## stdin per state

```
La,AV,57.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion3_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion2_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion2_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion2_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion1_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion1_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion1_4f2` | `  4s  4p  4d  5s  5p` | `4f(2)` |
| `ion1_5d16s1` | `  4s  4p  4d  5s  5p` | `5d(1)6s(1)` |
| `ion1_4f16s1` | `  4s  4p  4d  5s  5p` | `4f(1)6s(1)` |
| `ion0_5d16s2` | `  4s  4p  4d  5s  5p  6s` | `5d(1)` |
| `ion0_4f16s2` | `  4s  4p  4d  5s  5p  6s` | `4f(1)` |
| `ion0_5d26s1` | `  4s  4p  4d  5s  5p` | `5d(2)6s(1)` |
| `ion4_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion3_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

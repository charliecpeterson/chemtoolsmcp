# Y — ATSP2K `hf` configuration inputs

`Z = 39`, `Z_eff = 11.0`, `n_core = 28`, `l_max = 3`  (module `fit_ecp`)

Seed state (run first every evaluation, seeds all others): `ion0_4d15s2`

## stdin per state

```
Y,AV,39.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion0_4d15s2` | `  4s  4p  5s` | `4d(1)` |
| `ion0_4d25s1` | `  4s  4p` | `4d(2)5s(1)` |
| `ion0_5s25p1` | `  4s  4p  5s` | `5p(1)` |
| `ion0_4d15s15p1` | `  4s  4p` | `4d(1)5s(1)5p(1)` |
| `ion0_4d3` | `  4s  4p` | `4d(3)` |
| `ion1_5s2` | `  4s  4p  5s` | `(none)` |
| `ion1_4d15s1` | `  4s  4p` | `4d(1)5s(1)` |
| `ion1_4d2` | `  4s  4p` | `4d(2)` |
| `ion1_5s15p1` | `  4s  4p` | `5s(1)5p(1)` |
| `ion2_4d1` | `  4s  4p` | `4d(1)` |
| `ion2_5s1` | `  4s  4p` | `5s(1)` |
| `ion2_5p1` | `  4s  4p` | `5p(1)` |
| `ion3_closed` | `  4s  4p` | `(none)` |
| `ion2_4f1` | `  4s  4p` | `4f(1)` |
| `ion2_6s1` | `  4s  4p` | `6s(1)` |
| `ion1_4d15p1` | `  4s  4p` | `4d(1)5p(1)` |
| `ion1_5p2` | `  4s  4p` | `5p(2)` |
| `ion4_4p5` | `  4s` | `4p(5)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

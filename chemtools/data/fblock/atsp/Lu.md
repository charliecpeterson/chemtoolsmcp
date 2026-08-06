# Lu — ATSP2K `hf` configuration inputs

`Z = 71`, `Z_eff = 43.0`, `n_core = 28`, `l_max = 4`  (module `fit_ecp_lu`)

Seed state (run first every evaluation, seeds all others): `ion0_4f145d16s2`

## stdin per state

```
Lu,AV,71.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion17_closed` | `  4s  4p  4d  5s  5p` | `(none)` |
| `ion16_4f1` | `  4s  4p  4d  5s  5p` | `4f(1)` |
| `ion16_5d1` | `  4s  4p  4d  5s  5p` | `5d(1)` |
| `ion16_6s1` | `  4s  4p  4d  5s  5p` | `6s(1)` |
| `ion15_5d2` | `  4s  4p  4d  5s  5p` | `5d(2)` |
| `ion15_4f15d1` | `  4s  4p  4d  5s  5p` | `4f(1)5d(1)` |
| `ion5_4f115d1` | `  4s  4p  4d  5s  5p` | `4f(11)5d(1)` |
| `ion5_4f12` | `  4s  4p  4d  5s  5p` | `4f(12)` |
| `ion4_4f125d1` | `  4s  4p  4d  5s  5p` | `4f(12)5d(1)` |
| `ion4_4f13` | `  4s  4p  4d  5s  5p` | `4f(13)` |
| `ion3_4f135d1` | `  4s  4p  4d  5s  5p` | `4f(13)5d(1)` |
| `ion3_4f14` | `  4s  4p  4d  5s  5p  4f` | `(none)` |
| `ion3_4f136s1` | `  4s  4p  4d  5s  5p` | `4f(13)6s(1)` |
| `ion2_4f146s1` | `  4s  4p  4d  5s  5p  4f` | `6s(1)` |
| `ion2_4f135d16s1` | `  4s  4p  4d  5s  5p` | `4f(13)5d(1)6s(1)` |
| `ion0_4f145d16s2` | `  4s  4p  4d  5s  5p  4f  6s` | `5d(1)` |
| `ion18_5p5` | `  4s  4p  4d  5s` | `5p(5)` |
| `ion17_5p54f1` | `  4s  4p  4d  5s` | `5p(5)4f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

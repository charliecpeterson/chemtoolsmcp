# Lr — ATSP2K `hf` configuration inputs

`Z = 103`, `Z_eff = 43.0`, `n_core = 60`, `l_max = 4`  (module `fit_ecp_lr`)

Seed state (run first every evaluation, seeds all others): `ion0_5f146d17s2`

## stdin per state

```
Lr,AV,103.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion17_closed` | `  5s  5p  5d  6s  6p` | `(none)` |
| `ion16_5f1` | `  5s  5p  5d  6s  6p` | `5f(1)` |
| `ion16_6d1` | `  5s  5p  5d  6s  6p` | `6d(1)` |
| `ion16_7s1` | `  5s  5p  5d  6s  6p` | `7s(1)` |
| `ion15_6d2` | `  5s  5p  5d  6s  6p` | `6d(2)` |
| `ion15_5f16d1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)` |
| `ion5_5f116d1` | `  5s  5p  5d  6s  6p` | `5f(11)6d(1)` |
| `ion5_5f12` | `  5s  5p  5d  6s  6p` | `5f(12)` |
| `ion4_5f126d1` | `  5s  5p  5d  6s  6p` | `5f(12)6d(1)` |
| `ion4_5f13` | `  5s  5p  5d  6s  6p` | `5f(13)` |
| `ion3_5f136d1` | `  5s  5p  5d  6s  6p` | `5f(13)6d(1)` |
| `ion3_5f14` | `  5s  5p  5d  6s  6p  5f` | `(none)` |
| `ion3_5f137s1` | `  5s  5p  5d  6s  6p` | `5f(13)7s(1)` |
| `ion2_5f147s1` | `  5s  5p  5d  6s  6p  5f` | `7s(1)` |
| `ion2_5f136d17s1` | `  5s  5p  5d  6s  6p` | `5f(13)6d(1)7s(1)` |
| `ion0_5f146d17s2` | `  5s  5p  5d  6s  6p  5f  7s` | `6d(1)` |
| `ion1_5f147s2` | `  5s  5p  5d  6s  6p  5f  7s` | `(none)` |
| `ion0_5f147s27p1` | `  5s  5p  5d  6s  6p  5f  7s` | `7p(1)` |
| `ion18_6p5` | `  5s  5p  5d  6s` | `6p(5)` |
| `ion17_6p55f1` | `  5s  5p  5d  6s` | `6p(5)5f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

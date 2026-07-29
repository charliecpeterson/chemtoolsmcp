# U — ATSP2K `hf` configuration inputs

`Z = 92`, `Z_eff = 32.0`, `n_core = 60`, `l_max = 4`  (module `fit_ecp_u`)

Seed state (run first every evaluation, seeds all others): `ion0_5f36d17s2`

Donor-pinned states (self-restart reproducibly crashes; re-seeded
from a proven donor every evaluation):

- `ion5_7s1` ← `ion3_7s1`
- `ion7_6p5` ← `ion5_6p5`
- `ion6_6p55f1` ← `ion4_6p55f1`
- `ion3_5f3` ← `ion4_5f2`
- `ion2_5f4` ← `ion3_5f3`
- `ion1_5f36d2` ← `ion2_5f36d1`

## stdin per state

```
U,AV,92.        <- TERM=AV = configuration average
<closed shells>     <- FIXED 4-char fields: two spaces + 2-char label
<open config>       <- free-format nl(occ), blank if none
all / y / n / y / y / n / 99 2 / y / n / n
```

| slug | closed shells | open configuration |
|---|---|---|
| `ion6_closed` | `  5s  5p  5d  6s  6p` | `(none)` |
| `ion5_5f1` | `  5s  5p  5d  6s  6p` | `5f(1)` |
| `ion5_6d1` | `  5s  5p  5d  6s  6p` | `6d(1)` |
| `ion5_7s1` | `  5s  5p  5d  6s  6p` | `7s(1)` |
| `ion4_5f2` | `  5s  5p  5d  6s  6p` | `5f(2)` |
| `ion4_5f16d1` | `  5s  5p  5d  6s  6p` | `5f(1)6d(1)` |
| `ion4_6d2` | `  5s  5p  5d  6s  6p` | `6d(2)` |
| `ion3_5f3` | `  5s  5p  5d  6s  6p` | `5f(3)` |
| `ion3_5f26d1` | `  5s  5p  5d  6s  6p` | `5f(2)6d(1)` |
| `ion2_5f4` | `  5s  5p  5d  6s  6p` | `5f(4)` |
| `ion2_5f36d1` | `  5s  5p  5d  6s  6p` | `5f(3)6d(1)` |
| `ion2_5f37s1` | `  5s  5p  5d  6s  6p` | `5f(3)7s(1)` |
| `ion2_5f26d2` | `  5s  5p  5d  6s  6p` | `5f(2)6d(2)` |
| `ion1_5f37s2` | `  5s  5p  5d  6s  6p  7s` | `5f(3)` |
| `ion1_5f36d2` | `  5s  5p  5d  6s  6p` | `5f(3)6d(2)` |
| `ion1_5f36d17s1` | `  5s  5p  5d  6s  6p` | `5f(3)6d(1)7s(1)` |
| `ion0_5f36d17s2` | `  5s  5p  5d  6s  6p  7s` | `5f(3)6d(1)` |
| `ion0_5f47s2` | `  5s  5p  5d  6s  6p  7s` | `5f(4)` |
| `ion7_6p5` | `  5s  5p  5d  6s` | `6p(5)` |
| `ion6_6p55f1` | `  5s  5p  5d  6s` | `6p(5)5f(1)` |

Shells inside the ECP core are **dropped entirely** — do not carry
GRASP's `4f(14,i)`-style bookkeeping into these strings.

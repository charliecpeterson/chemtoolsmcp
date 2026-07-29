# f-block reference configurations — validated baseline

Every state below was converged in GRASP2018 (DC+Breit) during the ECPgen
reference campaign, 2026-06/07. 31 elements, 633 states. This is the
'do not fight the atom again' table: the configuration lines, the J blocks
each state actually produces, and which states need seeding tricks.

Machine-readable: `fblock-reference-configs.json` (same data, all fields).

Columns: slug · confline (GRASP/ATSP occupation syntax, `i` = inactive) ·
J blocks found · seeding note.

`SEED` = cannot be converged cold, needs a donor start guess (`estimate_from`).
`STAGE` = needs staged orbital birth (`vary_first`) or the orbital never
forms — the single most common f-block failure.

## Y (Z=39, A=89)

Core: [Kr] common to all states; ECP core will be [Ar]3d10 (28e)

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion0_4d15s2` | `` | 3/2 5/2 | neutral ground 2D |
| `ion0_4d25s1` | `` | 1/2 3/2 5/2 7/2 9/2 |  |
| `ion0_5s25p1` | `` | 1/2 3/2 |  |
| `ion0_4d15s15p1` | `` | 1/2 3/2 5/2 7/2 9/2 |  |
| `ion0_4d3` | `` | 1/2 3/2 5/2 7/2 9/2 11/2 |  |
| `ion1_5s2` | `` | 0 |  |
| `ion1_4d15s1` | `` | 1 2 3 | Y+ ground config |
| `ion1_4d2` | `` | 0 1 2 3 4 |  |
| `ion1_5s15p1` | `` | 0 1 2 |  |
| `ion2_4d1` | `` | 3/2 5/2 | Y2+ ground |
| `ion2_5s1` | `` | 1/2 |  |
| `ion2_5p1` | `` | 1/2 3/2 |  |
| `ion3_closed` | `` | 0 | Y3+ [Kr] closed shell; anchor for ionization ladder |
| `ion2_4f1` | `` | 5/2 7/2 | v2: probes the local/f channel |
| `ion2_6s1` | `` | 1/2 | v2: second s constraint |
| `ion1_4d15p1` | `` | 0 1 2 3 4 | v2: mixed d-p |
| `ion1_5p2` | `` | 0 1 2 | v2: p-channel pair |
| `ion4_4p5` | `3d(10,i)4s(2,i)4p(5,i)` | 1/2 3/2 | from ion3_closed |

## La (Z=57, A=139)

Core: Kr-menu core; hand ladder (nf=0)

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion3_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +3 (chemical ion) |
| `ion2_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Cef1 |
| `ion2_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion2_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion1_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion1_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_5d2,ion2_4f1 |
| `ion1_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion2_4f1 |
| `ion1_5d16s1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)6s(1,i)` | 1 2 3 | SEED; from ion1_5d2,ion2_6s1 |
| `ion1_4f16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(1,i)` | 2 3 4 | SEED; from ion2_4f1,ion2_6s1 |
| `ion0_5d16s2` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)6s(2,i)` | 3/2 5/2 | SEED; from ion1_5d16s1 |
| `ion0_4f16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(2,i)` | 5/2 7/2 | SEED; from ion1_4f16s1 |
| `ion0_5d26s1` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 | SEED; from ion1_5d2,ion1_5d16s1 |
| `ion4_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion3_closed |
| `ion3_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion2_4f1 |

## Ce (Z=58, A=140)

Core: [Kr]4d10 common core (46e, ccECP core choice); Z_eff = 12; valence 4f 5s 5p 5d 6s

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion4_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor |
| `ion3_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | Ce3+ ground, f channel |
| `ion3_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 | d channel |
| `ion3_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 | s channel |
| `ion2_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | f pair |
| `ion2_5d16s1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)6s(1,i)` | 1 2 3 | d-s cross holdout (replaces Ce3+ 6p1: Rydberg 6p not convergeable in GRASP from TF/hydrogenic estimates) |
| `ion2_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 |  |
| `ion2_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion2_4f16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(1,i)` | 2 3 4 |  |
| `ion1_4f15d2` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 | Ce+ ground region |
| `ion1_4f15d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 |  |
| `ion1_4f26s1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 |  |
| `ion0_4f15d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 | neutral ground config |
| `ion0_4f26s2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(2,i)` | 0 1 2 3 4 5 6 |  |
| `ion5_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | from ion4_closed |
| `ion4_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | from ion3_4f1 |
| `ion1_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | v2: 4f3, deep f probe |
| `ion1_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | v2: f2-d |
| `ion1_5d3` | `4d(10,i)5s(2,i)5p(6,i)5d(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 | v2: pure d3 |
| `ion1_5d26s1` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 | v2: d2-s |
| `ion0_4f15d26s1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(2,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 | v2: neutral fds2 variant |

## Pr (Z=59, A=141)

Core: [Ar]3d10 common core (28e, small-core Ln standard per the Ce decision); Z_eff = 31; valence 4s 4p 4d 5s 5p + 4f 5d 6s

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion5_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, Pr5+ 4f0 |
| `ion4_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from ion5_closed |
| `ion4_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 | d channel at high charge |
| `ion4_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 | s channel at high charge |
| `ion3_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_4f1 |
| `ion3_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion3_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion3_5d2,ion4_4f1 |
| `ion2_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion3_4f2 |
| `ion2_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion3_4f15d1 |
| `ion2_4f26s1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion3_4f2,ion4_6s1 |
| `ion1_4f36s1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion2_4f26s1 |
| `ion1_4f35d1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f25d1 |
| `ion1_4f25d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion2_4f26s1,ion2_4f25d1 |
| `ion0_4f36s2` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_4f36s1 |
| `ion0_4f25d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_4f25d16s1 |
| `ion6_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion5_closed |
| `ion5_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion4_4f1 |

## Nd (Z=60, A=144)

Core: [Ar]3d10 common core (28e); Z_eff = 32; valence 4s 4p 4d 5s 5p + 4f 5d 6s

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion6_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor |
| `ion5_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from ion6_closed |
| `ion5_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion5_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion4_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion5_4f1 |
| `ion4_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion4_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_5d2,ion5_4f1 |
| `ion3_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_4f2 |
| `ion3_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_4f15d1 |
| `ion3_4f26s1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion4_4f2,ion5_6s1 |
| `ion2_4f4` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f3 |
| `ion2_4f35d1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f25d1 |
| `ion2_4f36s1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion3_4f26s1 |
| `ion1_4f46s1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f36s1 |
| `ion1_4f35d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f36s1,ion2_4f35d1 |
| `ion0_4f46s2` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f46s1 |
| `ion0_4f35d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f35d16s1 |
| `ion7_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion6_closed |
| `ion6_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion5_4f1 |
| `ion1_4f45d1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f35d1 |
| `ion4_4f16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)6s(1,i)` | 2 3 4 | SEED; from ion5_4f1,ion5_6s1 |
| `ion2_4f25d2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f25d1 |

## Pm (Z=61, A=147)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion7_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | SEED; from donor_closed |
| `ion6_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Ndf1 |
| `ion6_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion6_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion5_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion5_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion5_5d2,ion6_4f1 |
| `ion5_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion6_4f1 |
| `ion4_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_4f15d1 |
| `ion4_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_4f2 |
| `ion3_4f35d1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f25d1 |
| `ion3_4f4` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f3 |
| `ion2_4f45d1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f35d1 |
| `ion2_4f5` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f4 |
| `ion2_4f46s1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f4,ion6_6s1 |
| `ion1_4f56s1` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f5,ion2_4f46s1 |
| `ion1_4f45d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f46s1,ion2_4f45d1 |
| `ion0_4f56s2` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f56s1 |
| `ion0_4f45d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f45d16s1 |
| `ion8_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion7_closed |
| `ion7_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion6_4f1 |

## Sm (Z=62, A=152)

Core: [Ar]3d10 common core (28e); Z_eff = 34

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion8_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | SEED; from donor_closed |
| `ion7_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Nd4f1 |
| `ion7_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 | SEED; from donor_5d1 |
| `ion7_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 | SEED; from donor_6s1 |
| `ion6_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion7_4f1 |
| `ion6_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 | SEED; from donor_5d2 |
| `ion6_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion6_5d2,ion7_4f1 |
| `ion5_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion6_4f2 |
| `ion5_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion6_4f15d1 |
| `ion4_4f4` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f3 |
| `ion4_4f35d1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f25d1 |
| `ion3_4f5` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f4 |
| `ion3_4f45d1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f35d1 |
| `ion3_4f46s1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f4,ion7_6s1 |
| `ion2_4f6` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f5 |
| `ion2_4f56s1` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f5,ion3_4f46s1 |
| `ion1_4f66s1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f6,ion2_4f56s1 |
| `ion1_4f55d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f56s1,ion3_4f45d1 |
| `ion0_4f66s2` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f66s1 |
| `ion0_4f55d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f55d16s1 |
| `ion9_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion8_closed |
| `ion8_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion7_4f1 |

## Eu (Z=63, A=153)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion9_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | SEED; from donor_closed |
| `ion8_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Smf1 |
| `ion8_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion8_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion7_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion7_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion7_5d2,ion8_4f1 |
| `ion7_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion8_4f1 |
| `ion6_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion7_4f15d1 |
| `ion6_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion7_4f2 |
| `ion5_4f35d1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion6_4f25d1 |
| `ion5_4f4` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion6_4f3 |
| `ion4_4f45d1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f35d1 |
| `ion4_4f5` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f4 |
| `ion3_4f55d1` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f45d1 |
| `ion3_4f6` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f5 |
| `ion2_4f65d1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f55d1 |
| `ion2_4f7` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f6 |
| `ion2_4f66s1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f6,ion8_6s1 |
| `ion1_4f76s1` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f7,ion2_4f66s1 |
| `ion1_4f65d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f66s1,ion2_4f65d1 |
| `ion0_4f76s2` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f76s1 |
| `ion0_4f65d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f65d16s1 |
| `ion10_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion9_closed |
| `ion9_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion8_4f1 |

## Gd (Z=64, A=158)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion10_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | SEED; from donor_closed |
| `ion9_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Smf1 |
| `ion9_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion9_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion8_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion8_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion8_5d2,ion9_4f1 |
| `ion8_4f2` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion9_4f1 |
| `ion7_4f25d1` | `4d(10,i)5s(2,i)5p(6,i)4f(2,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion8_4f15d1 |
| `ion7_4f3` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion8_4f2 |
| `ion6_4f35d1` | `4d(10,i)5s(2,i)5p(6,i)4f(3,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion7_4f25d1 |
| `ion6_4f4` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion7_4f3 |
| `ion5_4f45d1` | `4d(10,i)5s(2,i)5p(6,i)4f(4,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion6_4f35d1 |
| `ion5_4f5` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion6_4f4 |
| `ion4_4f55d1` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f45d1 |
| `ion4_4f6` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f5 |
| `ion3_4f65d1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f55d1 |
| `ion3_4f7` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f6 |
| `ion2_4f66s1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f6,ion9_6s1 |
| `ion1_4f76s1` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f7,ion2_4f66s1 |
| `ion1_4f65d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f66s1,ion3_4f65d1 |
| `ion0_4f75d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f65d16s1 |
| `ion0_4f86s2` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f76s1 |
| `ion11_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion10_closed |
| `ion10_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion9_4f1 |

## Tb (Z=65, A=159)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion11_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +11 |
| `ion10_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Gdf1 |
| `ion10_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion10_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion9_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion9_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion9_5d2,ion10_4f1 |
| `ion5_4f55d1` | `4d(10,i)5s(2,i)5p(6,i)4f(5,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_4f6` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemf |
| `ion4_4f65d1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f55d1 |
| `ion4_4f7` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f6 |
| `ion3_4f75d1` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f65d1 |
| `ion3_4f8` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f7 |
| `ion2_4f85d1` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f75d1 |
| `ion2_4f9` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f8 |
| `ion2_4f86s1` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; STAGE(6s); from ion3_4f8,donor_chems |
| `ion1_4f96s1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f9,ion2_4f86s1 |
| `ion1_4f85d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f86s1,ion2_4f85d1 |
| `ion0_4f96s2` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f96s1 |
| `ion0_4f85d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f85d16s1 |
| `ion12_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion11_closed |
| `ion11_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion10_4f1 |

## Dy (Z=66, A=164)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion12_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +12 |
| `ion11_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Tbf1 |
| `ion11_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion11_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion10_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion10_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion10_5d2,ion11_4f1 |
| `ion5_4f65d1` | `4d(10,i)5s(2,i)5p(6,i)4f(6,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemfd |
| `ion5_4f7` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemf |
| `ion4_4f75d1` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f65d1 |
| `ion4_4f8` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f7 |
| `ion3_4f85d1` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f75d1 |
| `ion3_4f9` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f8 |
| `ion2_4f95d1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f85d1 |
| `ion2_4f10` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f9 |
| `ion2_4f96s1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; STAGE(6s); from ion3_4f9,donor_chems |
| `ion1_4f106s1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f10,ion2_4f96s1 |
| `ion1_4f95d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f96s1,ion2_4f95d1 |
| `ion0_4f106s2` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f106s1 |
| `ion0_4f95d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f95d16s1 |
| `ion13_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion12_closed |
| `ion12_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion11_4f1 |

## Ho (Z=67, A=165)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion13_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +13 |
| `ion12_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Dyf1 |
| `ion12_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion12_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion11_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion11_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion11_5d2,ion12_4f1 |
| `ion5_4f75d1` | `4d(10,i)5s(2,i)5p(6,i)4f(7,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_4f8` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemf |
| `ion4_4f85d1` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f75d1 |
| `ion4_4f9` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f8 |
| `ion3_4f95d1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f85d1 |
| `ion3_4f10` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f9 |
| `ion2_4f105d1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_4f95d1 |
| `ion2_4f11` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion3_4f10 |
| `ion2_4f106s1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; STAGE(6s); from ion3_4f10,donor_chems |
| `ion1_4f116s1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion2_4f11,ion2_4f106s1 |
| `ion1_4f105d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_4f106s1,ion2_4f105d1 |
| `ion0_4f116s2` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_4f116s1 |
| `ion0_4f105d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_4f105d16s1 |
| `ion14_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion13_closed |
| `ion13_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion12_4f1 |

## Er (Z=68, A=166)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion14_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +14 |
| `ion13_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Hof1 |
| `ion13_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion13_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion12_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion12_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion12_5d2,ion13_4f1 |
| `ion5_4f85d1` | `4d(10,i)5s(2,i)5p(6,i)4f(8,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemfd |
| `ion5_4f9` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemf |
| `ion4_4f95d1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f85d1 |
| `ion4_4f10` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f9 |
| `ion3_4f105d1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_4f95d1 |
| `ion3_4f11` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_4f10 |
| `ion2_4f115d1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_4f105d1 |
| `ion2_4f12` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | 0 1 2 3 4 5 6 | SEED; from ion3_4f11 |
| `ion2_4f116s1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; STAGE(6s); from ion3_4f11,donor_chems |
| `ion1_4f126s1` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_4f12,ion2_4f116s1 |
| `ion1_4f115d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_4f116s1,ion2_4f115d1 |
| `ion0_4f126s2` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)6s(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_4f126s1 |
| `ion0_4f115d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_4f115d16s1 |
| `ion15_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion14_closed |
| `ion14_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion13_4f1 |

## Tm (Z=69, A=169)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion15_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +15 |
| `ion14_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Erf1 |
| `ion14_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion14_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion13_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion13_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion13_5d2,ion14_4f1 |
| `ion5_4f95d1` | `4d(10,i)5s(2,i)5p(6,i)4f(9,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_4f10` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemf |
| `ion4_4f105d1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_4f95d1 |
| `ion4_4f11` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_4f10 |
| `ion3_4f115d1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_4f105d1 |
| `ion3_4f12` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_4f11 |
| `ion2_4f125d1` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion3_4f115d1 |
| `ion2_4f13` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)` | 5/2 7/2 | SEED; from ion3_4f12 |
| `ion2_4f126s1` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; STAGE(6s); from ion3_4f12,donor_chems |
| `ion1_4f136s1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(1,i)` | 2 3 4 | SEED; from ion2_4f13,ion2_4f126s1 |
| `ion1_4f125d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)6s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion2_4f126s1,ion2_4f125d1 |
| `ion0_4f136s2` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(2,i)` | 5/2 7/2 | SEED; from ion1_4f136s1 |
| `ion0_4f125d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)6s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_4f125d16s1 |
| `ion16_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion15_closed |
| `ion15_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion14_4f1 |

## Yb (Z=70, A=174)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion16_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +16 |
| `ion15_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Tmf1 |
| `ion15_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion15_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion14_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 |  |
| `ion14_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion14_5d2,ion15_4f1 |
| `ion5_4f105d1` | `4d(10,i)5s(2,i)5p(6,i)4f(10,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemfd |
| `ion5_4f11` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from donor_chemf |
| `ion4_4f115d1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_4f105d1 |
| `ion4_4f12` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | 0 1 2 3 4 5 6 | SEED; from ion5_4f11 |
| `ion3_4f125d1` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_4f115d1 |
| `ion3_4f13` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)` | 5/2 7/2 | SEED; from ion4_4f12 |
| `ion2_4f135d1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion3_4f125d1 |
| `ion2_4f14` | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)` | 0 | SEED; from ion3_4f13 |
| `ion2_4f136s1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(1,i)` | 2 3 4 | SEED; STAGE(6s); from ion3_4f13,donor_chems |
| `ion1_4f146s1` | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)6s(1,i)` | 1/2 | SEED; from ion2_4f14,ion2_4f136s1 |
| `ion1_4f135d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_4f136s1,ion2_4f135d1 |
| `ion0_4f146s2` | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)6s(2,i)` | 0 | SEED; from ion1_4f146s1 |
| `ion0_4f135d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)6s(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_4f135d16s1 |
| `ion17_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion16_closed |
| `ion16_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion15_4f1 |

## Lu (Z=71, A=175)

Core: Kr-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion17_closed` | `4d(10,i)5s(2,i)5p(6,i)` | 0 | anchor, +17 |
| `ion16_4f1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)` | 5/2 7/2 | SEED; STAGE(4f-,4f); from donor_Ybf1 |
| `ion16_5d1` | `4d(10,i)5s(2,i)5p(6,i)5d(1,i)` | 3/2 5/2 |  |
| `ion16_6s1` | `4d(10,i)5s(2,i)5p(6,i)6s(1,i)` | 1/2 |  |
| `ion15_5d2` | `4d(10,i)5s(2,i)5p(6,i)5d(2,i)` | 0 1 2 3 4 | SEED; from donor_ybd2 |
| `ion15_4f15d1` | `4d(10,i)5s(2,i)5p(6,i)4f(1,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion15_5d2,ion16_4f1 |
| `ion5_4f115d1` | `4d(10,i)5s(2,i)5p(6,i)4f(11,i)5d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_4f12` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)` | 0 1 2 3 4 5 6 | SEED; from donor_chemf |
| `ion4_4f125d1` | `4d(10,i)5s(2,i)5p(6,i)4f(12,i)5d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_4f115d1 |
| `ion4_4f13` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)` | 5/2 7/2 | SEED; from ion5_4f12 |
| `ion3_4f135d1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_4f125d1 |
| `ion3_4f14` | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)` | 0 | SEED; from ion4_4f13 |
| `ion2_4f136s1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)6s(1,i)` | 2 3 4 | SEED; STAGE(6s); from ion4_4f13,donor_chems |
| `ion1_4f146s1` | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)6s(1,i)` | 1/2 | SEED; from ion3_4f14,ion2_4f136s1 |
| `ion1_4f135d16s1` | `4d(10,i)5s(2,i)5p(6,i)4f(13,i)5d(1,i)6s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_4f136s1,ion3_4f135d1 |
| `ion0_4f145d16s2` | `4d(10,i)5s(2,i)5p(6,i)4f(14,i)5d(1,i)6s(2,i)` | 3/2 5/2 | SEED; from ion1_4f135d16s1 |
| `ion18_5p5` | `4d(10,i)5s(2,i)5p(5,i)` | 1/2 3/2 | SEED; from ion17_closed |
| `ion17_5p54f1` | `4d(10,i)5s(2,i)5p(5,i)4f(1,i)` | 1 2 3 4 5 | SEED; from ion16_4f1 |

## Ac (Z=89, A=227)

Core: Xe-menu core; hand ladder (nf=0)

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion3_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | anchor, +3 (chemical ion) |
| `ion2_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Thf1 |
| `ion2_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 |  |
| `ion2_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_ths |
| `ion1_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 |  |
| `ion1_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_6d2,ion2_5f1 |
| `ion1_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion2_5f1 |
| `ion1_6d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(1,i)` | 1 2 3 | SEED; from ion1_6d2,ion2_7s1 |
| `ion1_5f17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(1,i)` | 2 3 4 | SEED; from ion2_5f1,ion2_7s1 |
| `ion0_6d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(2,i)` | 3/2 5/2 | SEED; from ion1_6d17s1 |
| `ion0_5f17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(2,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from ion1_5f17s1 |
| `ion0_6d27s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 | SEED; from ion1_6d2,ion1_6d17s1 |
| `ion4_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion3_closed |
| `ion3_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion2_5f1 |

## Th (Z=90, A=232)

Core: n<=4 shells common core (60e, small-core An standard per the Ce 28e decision); Z_eff = 30; valence 5s 5p 5d 6s 6p + 5f 6d 7s

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion0_6d27s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(2,i)` | 0 1 2 3 4 | neutral ground config (Th grounds d2s2, unlike Ce) |
| `ion0_5f16d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 | the f-d competition, key Th observable; births 5f (staged: 7s node-hunts if everything moves at once) |
| `ion0_5f16d27s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(2,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 | SEED; from ion0_5f16d17s2 |
| `ion1_6d3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 | pure d3; run before the ion1 7s states so 7s can be born at ion1 |
| `ion1_6d27s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 | Th+ ground region |
| `ion1_6d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(2,i)` | 3/2 5/2 |  |
| `ion1_5f16d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 |  |
| `ion1_5f26d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_5f16d2 |
| `ion1_5f3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_5f26d1 |
| `ion1_5f16d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; STAGE(5f-,5f); from ion1_6d27s1 |
| `ion1_5f27s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion1_5f16d17s1 |
| `ion2_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_5f16d2 |
| `ion2_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_5f26d1 |
| `ion2_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 |  |
| `ion2_6d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)7s(1,i)` | 1 2 3 | SEED; STAGE(7s); from ion1_6d27s1 |
| `ion2_5f17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(1,i)` | 2 3 4 | SEED; STAGE(7s); from ion1_5f27s1 |
| `ion3_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; from ion2_5f2 |
| `ion3_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | d channel |
| `ion3_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; STAGE(7s); from ion2_6d17s1 |
| `ion4_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | anchor, Th4+ [Rn] |
| `ion5_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | 6p hole, hard p probe |
| `ion4_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion3_5f1 |

## Pa (Z=91, A=231)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion5_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion4_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Thf1 |
| `ion4_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion4_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion3_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion3_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion3_6d2,ion4_5f1 |
| `ion3_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_5f1 |
| `ion2_5f17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)7s(1,i)` | 2 3 4 | SEED; STAGE(7s); from ion4_5f1 |
| `ion1_5f27s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion3_5f2,ion2_5f17s1 |
| `ion1_5f16d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_5f17s1,ion3_5f16d1 |
| `ion0_5f26d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_5f16d17s1 |
| `ion0_5f37s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_5f27s1 |
| `ion6_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion5_closed |
| `ion5_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion4_5f1 |

## U (Z=92, A=238)

Core: n<=4 shells common core (60e, small-core An standard); Z_eff = 32; valence 5s 5p 5d 6s 6p + 5f 6d 7s

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion6_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | anchor, U6+ [Rn] |
| `ion5_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from ion6_closed |
| `ion5_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | d channel at high charge |
| `ion5_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | s channel at high charge; hf_seed attempt, fallback = file-seed from a 7s carrier |
| `ion4_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion5_5f1 |
| `ion4_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 |  |
| `ion4_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_6d2,ion5_5f1 |
| `ion3_5f3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_5f2 |
| `ion3_5f26d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_5f16d1 |
| `ion2_5f4` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f3 |
| `ion2_5f36d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f26d1 |
| `ion2_5f37s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; STAGE(7s); from ion3_5f3 |
| `ion2_5f26d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f26d1 |
| `ion1_5f37s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion2_5f37s1 |
| `ion1_5f36d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_5f36d1 |
| `ion1_5f36d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; STAGE(7s); from ion2_5f36d1 |
| `ion0_5f36d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f36d17s1 |
| `ion0_5f47s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f37s2 |
| `ion7_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion6_closed |
| `ion6_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion5_5f1 |

## Np (Z=93, A=237)

Core: n<=4 shells common core (60e); Z_eff = 33; valence 5s 5p 5d 6s 6p + 5f 6d 7s

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion7_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | anchor |
| `ion6_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_U5f1 |
| `ion6_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 |  |
| `ion6_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 |  |
| `ion5_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion6_5f1 |
| `ion5_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 |  |
| `ion5_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion5_6d2,ion6_5f1 |
| `ion4_5f3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_5f2 |
| `ion4_5f26d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_5f16d1 |
| `ion3_5f4` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f3 |
| `ion3_5f36d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f26d1 |
| `ion3_5f37s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion4_5f3,ion6_7s1 |
| `ion2_5f5` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f4 |
| `ion2_5f47s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f4,ion3_5f37s1 |
| `ion2_5f46d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f36d1 |
| `ion1_5f47s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f47s1 |
| `ion1_5f46d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f46d1,ion2_5f47s1 |
| `ion0_5f46d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f46d17s1 |
| `ion0_5f57s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f47s2 |
| `ion8_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion7_closed |
| `ion7_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion6_5f1 |

## Pu (Z=94, A=244)

Core: n<=4 shells common core (60e); Z_eff = 34

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion8_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion7_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Np5f1 |
| `ion7_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_6d1 |
| `ion7_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_7s1 |
| `ion6_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion7_5f1 |
| `ion6_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_6d2 |
| `ion6_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion6_6d2,ion7_5f1 |
| `ion5_5f3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion6_5f2 |
| `ion5_5f26d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion6_5f16d1 |
| `ion4_5f4` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f3 |
| `ion4_5f36d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f26d1 |
| `ion3_5f5` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f4 |
| `ion3_5f46d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f36d1 |
| `ion3_5f47s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f4,ion7_7s1 |
| `ion2_5f6` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f5 |
| `ion2_5f57s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f5,ion3_5f47s1 |
| `ion1_5f67s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_5f6,ion2_5f57s1 |
| `ion1_5f56d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_5f57s1,ion3_5f46d1 |
| `ion0_5f67s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f67s1 |
| `ion0_5f56d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f56d17s1 |
| `ion9_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion8_closed |
| `ion8_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion7_5f1 |

## Am (Z=95, A=243)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion9_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion8_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Puf1 |
| `ion8_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion8_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion7_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion7_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion7_6d2,ion8_5f1 |
| `ion7_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion8_5f1 |
| `ion6_5f26d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion7_5f16d1 |
| `ion6_5f3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion7_5f2 |
| `ion5_5f36d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion6_5f26d1 |
| `ion5_5f4` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion6_5f3 |
| `ion4_5f46d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f36d1 |
| `ion4_5f5` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f4 |
| `ion3_5f56d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f46d1 |
| `ion3_5f6` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f5 |
| `ion2_5f66d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f56d1 |
| `ion2_5f7` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f6 |
| `ion2_5f67s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f6,donor_Pu7s |
| `ion1_5f77s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f7,ion2_5f67s1 |
| `ion1_5f66d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f67s1,ion2_5f66d1 |
| `ion0_5f77s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f77s1 |
| `ion0_5f66d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f66d17s1 |
| `ion10_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion9_closed |
| `ion9_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion8_5f1 |

## Cm (Z=96, A=247)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion10_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion9_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Amf1 |
| `ion9_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion9_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion8_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion8_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion8_6d2,ion9_5f1 |
| `ion8_5f2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion9_5f1 |
| `ion7_5f26d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(2,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion8_5f16d1 |
| `ion7_5f3` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion8_5f2 |
| `ion6_5f36d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(3,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion7_5f26d1 |
| `ion6_5f4` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion7_5f3 |
| `ion5_5f46d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(4,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion6_5f36d1 |
| `ion5_5f5` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion6_5f4 |
| `ion4_5f56d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f46d1 |
| `ion4_5f6` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f5 |
| `ion3_5f66d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f56d1 |
| `ion3_5f7` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f6 |
| `ion2_5f67s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f6,donor_Am7s |
| `ion1_5f77s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f7,ion2_5f67s1 |
| `ion1_5f66d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f67s1,ion3_5f66d1 |
| `ion0_5f76d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f66d17s1 |
| `ion0_5f87s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f77s1 |
| `ion11_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion10_closed |
| `ion10_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion9_5f1 |

## Bk (Z=97, A=247)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion11_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion10_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Cmf1 |
| `ion10_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion10_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion9_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion9_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion9_6d2,ion10_5f1 |
| `ion5_5f56d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(5,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_5f6` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemf |
| `ion4_5f66d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f56d1 |
| `ion4_5f7` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f6 |
| `ion3_5f76d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f66d1 |
| `ion3_5f8` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f7 |
| `ion2_5f86d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f76d1 |
| `ion2_5f9` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f8 |
| `ion2_5f87s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; STAGE(7s); from ion3_5f8,donor_chems |
| `ion1_5f97s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f9,ion2_5f87s1 |
| `ion1_5f86d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f87s1,ion2_5f86d1 |
| `ion0_5f97s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f97s1 |
| `ion0_5f86d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f86d17s1 |
| `ion12_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion11_closed |
| `ion11_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion10_5f1 |

## Cf (Z=98, A=251)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion12_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion11_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Bkf1 |
| `ion11_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion11_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion10_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion10_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion10_6d2,ion11_5f1 |
| `ion5_5f66d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(6,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemfd |
| `ion5_5f7` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemf |
| `ion4_5f76d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f66d1 |
| `ion4_5f8` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f7 |
| `ion3_5f86d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f76d1 |
| `ion3_5f9` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f8 |
| `ion2_5f96d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f86d1 |
| `ion2_5f10` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f9 |
| `ion2_5f97s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; STAGE(7s); from ion3_5f9,donor_chems |
| `ion1_5f107s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_5f10,ion2_5f97s1 |
| `ion1_5f96d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_5f97s1,ion2_5f96d1 |
| `ion0_5f107s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f107s1 |
| `ion0_5f96d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f96d17s1 |
| `ion13_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion12_closed |
| `ion12_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion11_5f1 |

## Es (Z=99, A=252)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion13_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion12_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Cff1 |
| `ion12_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion12_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion11_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion11_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion11_6d2,ion12_5f1 |
| `ion5_5f76d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(7,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_5f8` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemf |
| `ion4_5f86d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f76d1 |
| `ion4_5f9` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f8 |
| `ion3_5f96d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f86d1 |
| `ion3_5f10` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f9 |
| `ion2_5f106d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion3_5f96d1 |
| `ion2_5f11` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion3_5f10 |
| `ion2_5f107s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; STAGE(7s); from ion3_5f10,donor_chems |
| `ion1_5f117s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion2_5f11,ion2_5f107s1 |
| `ion1_5f106d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion2_5f107s1,ion2_5f106d1 |
| `ion0_5f117s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_5f117s1 |
| `ion0_5f106d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion1_5f106d17s1 |
| `ion14_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion13_closed |
| `ion13_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion12_5f1 |

## Fm (Z=100, A=257)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion14_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion13_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Esf1 |
| `ion13_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion13_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion12_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion12_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion12_6d2,ion13_5f1 |
| `ion5_5f86d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(8,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemfd |
| `ion5_5f9` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemf |
| `ion4_5f96d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f86d1 |
| `ion4_5f10` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f9 |
| `ion3_5f106d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion4_5f96d1 |
| `ion3_5f11` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_5f10 |
| `ion2_5f116d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion3_5f106d1 |
| `ion2_5f12` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | 0 1 2 3 4 5 6 | SEED; from ion3_5f11 |
| `ion2_5f117s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; STAGE(7s); from ion3_5f11,donor_chems |
| `ion1_5f127s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_5f12,ion2_5f117s1 |
| `ion1_5f116d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion2_5f117s1,ion2_5f116d1 |
| `ion0_5f127s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)7s(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_5f127s1 |
| `ion0_5f116d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion1_5f116d17s1 |
| `ion15_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion14_closed |
| `ion14_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion13_5f1 |

## Md (Z=101, A=258)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion15_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion14_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Fmf1 |
| `ion14_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion14_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion13_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion13_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion13_6d2,ion14_5f1 |
| `ion5_5f96d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(9,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_5f10` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemf |
| `ion4_5f106d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from ion5_5f96d1 |
| `ion4_5f11` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_5f10 |
| `ion3_5f116d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion4_5f106d1 |
| `ion3_5f12` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_5f11 |
| `ion2_5f126d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion3_5f116d1 |
| `ion2_5f13` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)` | 5/2 7/2 | SEED; from ion3_5f12 |
| `ion2_5f127s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; STAGE(7s); from ion3_5f12,donor_chems |
| `ion1_5f137s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(1,i)` | 2 3 4 | SEED; from ion2_5f13,ion2_5f127s1 |
| `ion1_5f126d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)7s(1,i)` | 0 1 2 3 4 5 6 7 8 9 | SEED; from ion2_5f127s1,ion2_5f126d1 |
| `ion0_5f137s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(2,i)` | 5/2 7/2 | SEED; from ion1_5f137s1 |
| `ion0_5f126d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)7s(2,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion1_5f126d17s1 |
| `ion16_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion15_closed |
| `ion15_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion14_5f1 |

## No (Z=102, A=259)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion16_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion15_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Mdf1 |
| `ion15_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion15_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion14_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion14_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion14_6d2,ion15_5f1 |
| `ion5_5f106d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(10,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 19/2 21/2 | SEED; from donor_chemfd |
| `ion5_5f11` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from donor_chemf |
| `ion4_5f116d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from ion5_5f106d1 |
| `ion4_5f12` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | 0 1 2 3 4 5 6 | SEED; from ion5_5f11 |
| `ion3_5f126d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion4_5f116d1 |
| `ion3_5f13` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)` | 5/2 7/2 | SEED; from ion4_5f12 |
| `ion2_5f136d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion3_5f126d1 |
| `ion2_5f14` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)` | 0 | SEED; from ion3_5f13 |
| `ion2_5f137s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(1,i)` | 2 3 4 | SEED; STAGE(7s); from ion3_5f13,donor_chems |
| `ion1_5f147s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(1,i)` | 1/2 | SEED; from ion2_5f14,ion2_5f137s1 |
| `ion1_5f136d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_5f137s1,ion2_5f136d1 |
| `ion0_5f147s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(2,i)` | 0 | SEED; from ion1_5f147s1 |
| `ion0_5f136d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)7s(2,i)` | 0 1 2 3 4 5 6 | SEED; from ion1_5f136d17s1 |
| `ion17_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion16_closed |
| `ion16_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion15_5f1 |

## Lr (Z=103, A=262)

Core: Xe-menu core; generated ladder

| slug | confline | J blocks | notes |
|---|---|---|---|
| `ion17_closed` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)` | 0 | SEED; from donor_closed |
| `ion16_5f1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)` | 5/2 7/2 | SEED; STAGE(5f-,5f); from donor_Nof1 |
| `ion16_6d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(1,i)` | 3/2 5/2 | SEED; from donor_d1 |
| `ion16_7s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)7s(1,i)` | 1/2 | SEED; from donor_s1 |
| `ion15_6d2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)6d(2,i)` | 0 1 2 3 4 | SEED; from donor_d2 |
| `ion15_5f16d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(1,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion15_6d2,ion16_5f1 |
| `ion5_5f116d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(11,i)6d(1,i)` | 0 1 2 3 4 5 6 7 8 9 10 | SEED; from donor_chemfd |
| `ion5_5f12` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)` | 0 1 2 3 4 5 6 | SEED; from donor_chemf |
| `ion4_5f126d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(12,i)6d(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 15/2 17/2 | SEED; from ion5_5f116d1 |
| `ion4_5f13` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)` | 5/2 7/2 | SEED; from ion5_5f12 |
| `ion3_5f136d1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)` | 0 1 2 3 4 5 6 | SEED; from ion4_5f126d1 |
| `ion3_5f14` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)` | 0 | SEED; from ion4_5f13 |
| `ion2_5f137s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)7s(1,i)` | 2 3 4 | SEED; STAGE(7s); from ion4_5f13,donor_chems |
| `ion1_5f147s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(1,i)` | 1/2 | SEED; from ion3_5f14,ion2_5f137s1 |
| `ion1_5f136d17s1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(13,i)6d(1,i)7s(1,i)` | 1/2 3/2 5/2 7/2 9/2 11/2 13/2 | SEED; from ion2_5f137s1,ion3_5f136d1 |
| `ion0_5f146d17s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)6d(1,i)7s(2,i)` | 3/2 5/2 | SEED; from ion1_5f136d17s1 |
| `ion1_5f147s2` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(2,i)` | 0 | SEED; from ion1_5f147s1 |
| `ion0_5f147s27p1` | `4f(14,i)5d(10,i)6s(2,i)6p(6,i)5f(14,i)7s(2,i)7p(1,i)` | 1/2 3/2 | SEED; STAGE(7p-,7p); from ion1_5f147s2,donor_7p |
| `ion18_6p5` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)` | 1/2 3/2 | SEED; from ion17_closed |
| `ion17_6p55f1` | `4f(14,i)5d(10,i)6s(2,i)6p(5,i)5f(1,i)` | 1 2 3 4 5 | SEED; from ion16_5f1 |

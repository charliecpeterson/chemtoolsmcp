orphan  

# Known problems

## 14.0

- LAO shieldings are wrong in combination with the .SELECT keyword.
- Analytic molecular gradient is wrong for an open-shell system.
- LAO shieldings are wrong in combination with nonzero .HFXMU (e.g.
  LCPBE0).
- CAMB3LYP analytic molecular gradient is wrong.
- Several tests show failures on Intel 15 with --int64.
- XC grid generation does not correctly handle ghost atoms.
- .CNVINT keyword is ignored.
- .PROPERTIES can reset OPTIMIZE parameters (e.g. cancel .NUMGRA).

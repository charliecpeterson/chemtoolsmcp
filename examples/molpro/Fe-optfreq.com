***, 3d-metal example: geometry optimization then frequency (start from off geometry)
memory,400,m
symmetry,nosym
geometry={
Fe    0.000000    0.000000    0.000000
O     2.100000    0.000000    0.000000
H     2.100000    0.960000    0.000000
H     2.100000   -0.960000    0.000000
O    -2.100000    0.000000    0.000000
H    -2.100000    0.960000    0.000000
H    -2.100000   -0.960000    0.000000
O     0.000000    2.100000    0.000000
H     0.000000    2.100000    0.960000
H     0.000000    2.100000   -0.960000
O     0.000000   -2.100000    0.000000
H     0.000000   -2.100000    0.960000
H     0.000000   -2.100000   -0.960000
O     0.000000    0.000000    2.100000
H     0.960000    0.000000    2.100000
H    -0.960000    0.000000    2.100000
O     0.000000    0.000000   -2.100000
H     0.960000    0.000000   -2.100000
H    -0.960000    0.000000   -2.100000
}
set,charge=2
symmetry,nosym   ! disable point-group symmetry to avoid symmetry-convergence problems

basis={
default,cc-pVDZ         ! basis for main-group atoms (fast for initial optim)
}

{uhf;wf,,,4}         ! start an unrestricted HF (good for open-shell 3d metals)

optg,maxit=200

frequencies       ! calculate vibrational frequencies
print,low         ! print frequencies+modes of zero frequencies
thermo
temp,200,400,50 


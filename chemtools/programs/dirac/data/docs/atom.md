orphan  

Now we turn to the cadmium atom in $`D_{2h}`$ with the
`HAMILTONIAN_.SPINFREE` Hamiltonian. The $`D_{2h}`$ double group has two
fermionic IRREPS (g and u) leading to XREPS=1,2. By using the
`HAMILTONIAN_.SPINFREE` option a real calculation in the ordinary
$`D_{2h}`$ point group is performed in DIRAC leading to eight symmetries
as known from the character table. Therefore we set
XREPS=1,2,3,4,5,6,7,8 if we need all symmetries. The relation of the
irrep name to the number can be obtained from the Dirac output and is
reproduced in the ADC code as well. After the calculation the $`A_{1g}`$
final states are the in SSPEC.01, the $`B_{2g}`$ in SSPEC.02 asf. The
complete spectrum is then the merge of SSPEC.01 ... SSPEC.08. Remember
that for the plot we only need the IP and the corresponding pole
strength. We therefore do a grep '@' on the merged SSPEC.X files (cat
SSPEC.\* \> SPEC.all) and obtain all lines in each symmetry. If we need
only a specific range we do "sort -n SPEC.all \> SPEC.range" and edit
according to our needs. Then we can use gnuplot with "plot "SPEC.range"
u 1:2 w i". For ADC(3) with constant diagrams and 600 Lanczos iterations
we would have an input like this:

    **DIRAC
    .TITLE
     input for Cd atom
    .WAVE F
    .4INDEX
    **GENERAL
    .DIRECT
     1 1 1
    **INTEGRALS
    *READIN
    *TWOINT
    .SOFOCK
    .SCREEN
    1.E-16
    **HAMILTONIAN
    .SPINFREE
    **WAVE FUNCTIONS
    .SCF
    .RELADC
    *SCF
    .CLOSED
     30 18
    .FCKCNV
    5.0E-09
    .INTFLG
    1 1 1
    **MOLTRA
    .INTFLG
    1 1 1
    .CORE
    1..8
    1..6
    .ACTIVE
    9..25
    7..22
    **RELADC
    .DOSIPS
    .ADCLEVEL
     3
    .SIPREPS
     8
     1,2,3,4,5,6,7,8
    **LACNZOS
    .SIPITER
     600
    *END OF

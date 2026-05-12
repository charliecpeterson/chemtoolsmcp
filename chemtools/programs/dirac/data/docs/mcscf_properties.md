orphan  

# Properties at the MCSCF level of theory

In this tutorial we will look at a simple example of computing
one-electron properties using a Kramers-restricted MCSCF (KRMC) and
Hartree-Fock (HF) wave function in DIRAC. For more details on the
implementation and the method, see Jensen1996, Thyssen2004 , Thyssen2008
and Knecht2009, respectively.

We consider the Be atom using $`\mbox{C}_{2v}`$ symmetry:

    INTGRL
    Be atom in uncontracted Pierloot basis set, MOLCAS 5, ANO-S
    Generated small component via RKB
    C   1    2XY Y         .10D-15
            4.    1
    Be 1           .00000000           .00000000           .00000000
    LARGE    3    2    1    1
    H   7    0    3
          2732.3281
           410.31981
            93.672648
            26.587957
             8.6295600
             3.0562640
             1.1324240
    H   3    0    3
              .18173200
              .05917000
              .02071000
    H   4    0    3
             1.1677000
              .36500000
              .11410000
              .03570000
    H   3    0    3
              .54680000
              .14650000
              .03930000
    FINISH

and generate the following menu file:

    **DIRAC
    .TITLE
     Testing KRMC and KR-CI for Be in Pierloot basis
    .ANALYZE
    .WAVE FUNCTION
    .PROPERTIES
    **PROPERTIES
    .RHONUC
    .EFFDEN
    .WAVE F
    2
     DHF
     KRMC
    **HAMILTONIAN
    .X2C
    **WAVE FUNCTION
    .SCF
    .KRMCSCF
    *SCF
    .CLOSED SHELL
     4
    *KRMCSCF
    .CI PROGRAM
    LUCIAREL
    .INACTIVE
     1
    .GAS SHELL
     2
     0 2 / 1
     2 2 / 3
    .SYMMETRY
     1
    .PRINT
     5
    *OPTIMI
    .ANALYZ
    **END OF

Here we ask for the contact density $`\rho(0)`$ (.RHONUC) as well as
effective density $`\bar{\rho}`$ (.EFFDEN) at the Be nucleus calculated
at the HF and MCSCF level of theory using the exact two-component
Hamiltonian. In the MCSCF step we define a CAS(2,4) space correlating 2
electrons in 4 Kramers pairs (2s2p shell of Be).

The results read as follows:

    HF:
    Rho at nuc Be 01 :     33.94937443146 a.u.
    EFFD:Be 01       :     33.94934902177 a.u.

    KRMC:
    Rho at nuc Be 01 :     33.89686536890 a.u.
    EFFD:Be 01       :     33.89683999776 a.u.

From this we can conclude that non-dynamical (and partially dynamical)
correlation included in our CAS wave function seems to reduce both,
$`\rho(0)`$ and $`\bar{\rho}`$ at the Be nucleus compared to the
uncorrelated Hartree-Fock values.

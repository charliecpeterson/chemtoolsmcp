orphan  

# CFREAD

CFREAD is a utility program for getting information from the file DFCOEF
containig MO-coefficients, orbital energies and, if relevant,
information about boson or linear symmetries. Usage is straightforward,
for instance:

    dirac 26>cfread.x DFCOEF

     CFREAD; Information read from file : DFCOEF

     - Heading :DIRAC: No title specified !!!                     Tue Sep 24 23:05:49 2013
     - Total energy: -75.1621606438559979
     - Fermion ircops          :      1
    -----------------------------------
     - Negative-energy orbitals:      0
     - Positive-energy orbitals:     21
     - AO-basis functions      :     21

Note that the electronic energy is calculated before diagonalization, so
that coefficients from iteration *n* refer to electronic energy from
iteration *n-1*. At convergence this should not matter.

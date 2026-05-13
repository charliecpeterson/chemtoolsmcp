orphan  

# Generate a FCIDUMP file for external DMRG calculations

This tutorial briefly demonstrates how to generate a FCIDUMP integral
file which can be used to perform relativistic DMRG calculations with an
external DMRG code. The interface and the first relativistic DMRG
implementation and calculation are described in Knecht2014a.

We will explain the basic steps to generate a FCIDUMP file by looking at
the HF molecule using a small cc-pVDZ basis set for each atom. The
molecular input file **hf.xyz** is:

    2

    H      0.0000000000   0.0000000000   0.8547056701
    F      0.0000000000   0.0000000000  -0.0453403237

and the corresponding wave function input file **fci.inp** is:

    **DIRAC
    .TITLE
     FCIDUMP file tutorial
    .WAVE FUNCTION
    **HAMILTONIAN
    .DOSSSS
    **WAVE FUNCTION
    .SCF
    .KR CI
    *SCF
    .CLOSED SHELL
     10
    *KRCI
    .CI PROGRAM
    LUCIAREL
    .CIROOTS
     0  1
    .MAX CI
     100
    .INACTIVE
     1
    .GAS SHELLS
     1
     8 8 / 9
    **MOLECULE
    *BASIS
    .DEFAULT
    cc-pVDZ
    *END OF INPUT

So let's have a closer look at the above wave function input. The
default Hamiltonian (to be specified under \*\*Hamiltonian) in Dirac is
the Dirac-Coulomb Hamiltonian and we explicitly ask to also include the
class of (SS\|SS) integrals. The wave function input asks for a
self-consistent-field Hartree-Fock calculation for the closed-shell HF
molecule with 10 electrons followed by a FCI/CASCI calculation for the
$`\Omega=0`$ state using an active space of **8** electrons in **9**
Kramer's pairs (== 18 spinors). The syntax (min max \# of electrons in
the CAS space / \# of Kramer's pairs) reads in detail as:

    .GAS SHELLS
     1
     8 8 / 9

Note that for this FCI/CASCI calculation we have frozen the 1s shell of
F by setting:

    .INACTIVE
     1

To run the above example we use DIRACs pyhton script *pam* with the
following command line:

    $ $path-to-dirac-build-directory/pam --inp=fci.inp --mol=hf.xyz  

A closer look at the output of the CI program at the bottom of the Dirac
calculation will tell us the final FCI/CASCI energy for the $`\Omega=0`$
state to be **-100.1956847...** Hartree. This is the reference energy we
should expect to get from the DMRG calculation on the same active space.

To obtain now the integral file **FCIDUMP** for the above FCI/CASCI
example we first need to modify the above input file by introducing the
keyword .FCIDUMP and change the CI program from *LUCIAREL* to *GASCIP*
for technical reasons. Save the file as **fcidump.inp** and run Dirac
with the following command line:

    $ $path-to-dirac-build-directory/pam --inp=fcidump.inp --mol=hf.xyz --get=FCIDUMP 

The **fcidump.inp** reads as:

    **DIRAC
    .TITLE
     FCIDUMP file tutorial
    .WAVE FUNCTION
    **HAMILTONIAN
    .DOSSSS
    **WAVE FUNCTION
    .SCF
    .KR CI
    *SCF
    .CLOSED SHELL
     10
    *KRCI
    .CI PROGRAM
    GASCIP
    .FCIDUMP
    .CIROOTS
     0  1
    .MAX CI
     100
    .INACTIVE
     1
    .GAS SHELLS
     1
     8 8 / 9
    **MOLECULE
    *BASIS
    .DEFAULT
    cc-pVDZ
    *END OF INPUT

Happy DMRG computing!

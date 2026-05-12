orphan  

Here we consider a bent triatomic molecule in $`C_{s}`$ symmetry. We
have only one fermionic IRREP and complex MOs. This is indicated in the
ADC output reporting complex ADC matrices. We want to do single
ionizations (DOSIPS=T) at the ADC(3) level (ADCLEVEL=3) including
constant diagrams (DOCONST=default) and compute all final states in
symmetry 1, the only available one (XREPS=1). If the basis set is large,
we restrict the iterations for the constant diagrams by releasing
convergence a bit (VCONV=1.0E-05) because this step takes longest. We
request 1000 Lanczos iterations and consider ionization energies up to
100 eV on screen. The complete spectrum is written to SSPEC.01. You get
as many entries as you have Lanczos iterations. A characteristic of the
iterative diagonalizer is that the eigenvalues at the edge of the
spectrum converge very fast and are reproduced for high iteration
numbers. In order to get higher eigenvalues equally tightly converged
the number of iterations has to be increased accordingly. In this case
the spurious (already converged) eigenvalues are projected out. For the
$`C_{s}`$ molecule the ADC input would then look like this:

    **RELADC
    .DOSIPS
    .SIPREPS
     1 # How many requiered IRREPS are going to follow in the next line
     1
    .VCONV
     1.0E-05
    **LANCZOS
    .SIPITER
     1000
    .SIPPRNT
     100.0

This part has to be included into an input file, which contains the
information about the Hamiltonian, the appropriate number of electrons,
... .

Since we do not activate DIPs nothing else needs to be specified in the
Lanczos section.

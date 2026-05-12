orphan  

# DIRAC Workshop on Relativistic Quantum Chemistry

## Specifying electronic states with single determinant

# Hartree-Fock SCF in DIRAC : Hamiltonians

- specify one-component, two-component (X2C) or four-component
  Hamiltonian
- decide whether neglect spin-orbital effects
- if X2C, you can neglect the AMFI 2-electron correction
- in the case of 4-comp. Hamiltonian specify integral classes

# Hartree-Fock SCF in DIRAC : Electronic occupation

- either use the automatic occupation of set the electronic occupation
  manually
- recommended (but not perfect) approach: start with automatic
  occupation
- after switch to manuall electronic occupation
- always add orbital analysis to check resulting molecular spinors

# Getting SCF convergence in DIRAC

- first empty valence shell(s) and see
- atomic start SCF as the best method so far
- for some systems force keeping order of MOs

orphan  

<div class="index">

\*STEX

</div>

# \*STEX

## Core excitation spectra in the Static Exchange approximation

A well known limitation of the time dependent Hartree-Fock approximation
is that electronic relaxation is not accounted for in the calculation of
excitation energies. This relaxation can be understood as a correlation
effect, since it is due to the simultaneous excitation of an electron
and the relaxation of the remaining N-1 electrons (or rather orbitals)
due to the "hole" left behind by the excited electron. On the other hand
the relaxation is well described already at the Hartree-Fock level when
the excited states are optimized separately, in the so-called ΔSCF
approach.

The Static Exchange (STEX) approximation is a cheap way to incorporate
relaxation effects into the calculation of core excitation spectra,
where it is particularly important for excitation energies and
transition probabilities. The method works by making a configuration
interaction singles expansion of the excited states with a reference
determinant optimized for core holes in a particular set of orbitals.
The excited states are then not orthogonal to the Hartree-Fock ground
state, and wavefunction overlaps have to be explicitly taken into
account.

A STEX calculation typically proceeds in the following way:

1.  The Hartree-Fock groundstate is calculated with Dirac, and the
    wavefunction saved in the `DFCOEF` file.
2.  The groundstate output is examined and the indices of the core
    orbital of interest are noted. If the core orbitals are not well
    localized to individual atoms it may be necessary to add a small
    fractional charge to break a symmetry of the molecule. This change
    can then be removed in the final calculation.
3.  An average-of-configuration open shell Hartree-Fock calculation is
    performed to get an approximate wavefunction of the core ionized
    molecule. This wavefunction should then be saved to a file
    `DFCOEF.ION`. In order for the core ionized state to converge it is
    necessary to use overlap selection and possibly orbital freezing.
4.  The STEX calculation is run with the specified hole orbitals,
    starting from the previously calculated `DFCOEF` and `DFCOEF.ION`
    files.

please provide an  
example !

## Directives

**Basic**

<div class="index">

.HOLES

</div>

### .HOLES

Indicate (singly occupied) hole orbitals. Example:

    .HOLES
    ! Expects format like (NFSYM=2)
    2    ! nr of holes in fermion symmetry 1.
    5 9  ! holes in fs 1
    1    ! nr of holes in fermion symmetry 2.
    1    ! holes in fs 2

<div class="index">

.PRINT

</div>

### .PRINT

Print level. *Default:* 0

**Advanced keywords**

<div class="index">

.BATCH

</div>

### .BATCH

Number of simultaneous Fock matrices to build. *Default:* 4

<div class="index">

.CUBE

</div>

### .CUBE

Deactivated ?

<div class="index">

.CUTOFF

</div>

### .CUTOFF

Set default cutoff in eV. *Default:* 100 eV

<div class="index">

.SCREEN

</div>

### .SCREEN

Screening threshold.

orphan  

# One-electron property operators

4-component one-electron operators are discussed in
`one_electron_operators`. Here we look at the handling of matrix
representation of such operators inside DIRAC. The general form of
4-component one-electron operators is

``` math
{\,}^{4c}\Omega = \sum_{i=1}^{n_{comp}}{\,}^{4c}M_i f_i P_i
```

where $`{\,}^{4c}M_i`$ is a $`4\times 4`$ matrix, $`f_i`$ a real factor,
$`P_i`$ a real scalar operator and the sum running over the numbers of
components of the operator (not to be confused with the number of
components of a spinor).

The matrices $`{\,}^{4c}M`$ are either *even*, sampling LL- and SS-
blocks of the matrix of the scalar operator $`P`$, or *odd*, sampling
the LS- and SL-blocks of the scalar operator.

The general input routine for 4-component one-electron operators is
XPRINP (src/prp/paminp.F).

## List of 4-component operators

Each unique operator gets an index and is added to the list of
4-component one-electron operators stored in the common block
*dcbxpr.h*. For each 4-component operator the following information is
stored:

*PRPNAM*  
(user-defined) operator name

*IPRTYP*  
operator type (see `one_electron_operators`), corresponding to a
particular linear combination of $`4\times4`$ matrices . This
automatically defines $`n_{comp}`$.

*IPRPSYM*  
spatial symmetry (boson irrep)

*IPRPTIM*  
time-reversal symmetry (-1,0,+1)

For each component:

*FACPRP*  
real factor $`f_i`$

*IPRPLBL*  
pointer to scalar operator $`P_i`$

## List of scalar operators

A list of scalar operators is kept in common block *dcbprl.h*. For each
scalar operator the following information is recorded:

*PRPLBL*  
name of property label identifying the corresponding integrals on file
AOPROPER

*IPRLREP*  
spatial symmetry (boson irrep) of scalar operator

*IPRLTYP*  
symmetry of scalar operator under matrix transpose (-1,0,1)

*PDOINT*  
a CHARACTER\*4 array where the four slots correspond to the LL, SL, LS
and SS block of the scalar property matrix and have three possible
values:

> - '0': the block is not used (and therefore not generated)
> - '+': the block is used
> - '-': the block is used and scaled with minus one

## Generating the matrix of a 4-component property operator

In the present setup DIRAC/HERMIT generates the integrals corresponding
to the list of scalar operators in *dcbprl.h* and writes them to disk on
the file AOPROPER. For property matrices that are symmetric or
anti-symmetric only the lower triangle is stored.

DIRAC comes with a utility program *labread.x* which can be used to
print the list of labels (and more) corresponding to the integrals
stored in AOPROPER.

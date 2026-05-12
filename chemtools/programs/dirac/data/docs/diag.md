orphan  

# Problems with the diagonalization

Diagonalizations are important part of (relativistic) quantum chemistry
methods.

In the DIRAC program suite, there is the diagonalization of the Lowdin
matrix, then of the Jz operator matrix in the case of linear symmetry
and, finally, of the Fock MO matrix.

In DIRAC, there are few routines to accomplish these tasks:

- non-deafult lapack' routine DSYEVR (for real symmetric matrixes only)
- eispack's routine RS (for real symmetric matrixes only)
- own DIRAC routine JACOBI (for real symmetric matrices only)
- own DIRAC routine QJACOBI (for quaternion Hermitian matrices)
- and own DIRAC routine based on Householder diagonalization (for
  quaternion Hermitian matrices)

For certain cases one has to carefully choose the diagonalization
method, depending either on the quality (accuracy) of obtained results -
eigenvalues and eigenvectors - or on the time performance.

Note that DSYEVR is not the default diagonalization routine in DIRAC,
but can be activated via *./setup -D LAPACK_QDIAG=1 ...*.

## Problem case - DSYEVR

The lapack's routine for diagonalization, DSYEVR, which is fast enough,
is giving - for certain input matrices - numerically unstable
eigenvectors. We discovered it in the linear symmetry.

### The $`O_2`$ test system

The real test system, where numerical discrepancies with active DSYEVR
are observed, is the $`O_2`$ molecule of the *fscc_highspin* test: :

    pam --noarch --inp=hf.inp --mol=O2.mol

(Input files for download are from the DIRAC test directory, namely
`hf.inp <../../test/fscc_highspin/hf.inp>` and
`O2.mol <../../test/fscc_highspin/O2.mol>` )

The proper Total SCF energy of this system is, :

    -149.68666145184687

while the DSYEVR routine employed in the LINSYM procedure gives
different value by about $`10^{-7}`$, like (obtained by Miro) :

    -149.68666198782267

together with some positron states intruding.

Molecular symmetries lower than linear symmetry (D2h, C2v ...) give
proper and identical SCF energies.

### The smallest test system

By tracing the error to the smallest system where error is clearly
showing up we came to the artificial *X2* one-electron linear system in
the smallest 1s1p1d basis: :

    pam --noarch --inp=dc.1el.2fs.inp  --mol=X2.1s1p1d.asd.mol --get "Jz_SS_matrix.fermirp2-2"
    pam --noarch --inp=dc.1el.2fs.inp  --mol=X2.1s1p1d.D2h.mol

(Input files for download are
`dc.1el.2fs.inp <../../test/fscc_highspin/dc.1el.2fs.inp>`,
`X2.1s1p1d.asd.mol <../../test/fscc_highspin/X2.1s1p1d.asd.mol>` and
`X2.1s1p1d.D2h.mol <../../test/fscc_highspin/X2.1s1p1d.D2h.mol>`.)

The affected value is the "Inactive energy (Output from ONEERG)" in the
output: :

    -15622.826231066265 (wrong value,    linear symmetry)
    -15622.826368346343 (correct value,  linear symmetry)
    -15622.826368346305 (correct value,  D2h symmetry)

We see differences in energies by about $`10^{-4}`$ a.u. what is
sufficient for detecting an error.

#### The DIRAC demonstration program

By tracking down the source of the problem we get to the 9x9 matrix of
the *X2.1s1p1d* test system above, *Jz_SS_matrix.fermirp2-2*, for the
diagonalization testing, which is saved in the LINSYM routine.

In the linear symmetry eigenvectors obtained with the lapack's DSYEVR
routine happen to be numerically inaccurate. This can be verified with
the abovementioned matrix.

With the standalone DIRAC-dependent diagonalization testing program,
*diag.x* (its source code is in the utils directory, together with the
text input file
`DIAGONALIZE_TESTS.INP <../../../test/diagonalization/DIAGONALIZE_TESTS.INP>`),
we found discrepancies in numerical accuracy of eigenvectors (U)
obtained with the DSYEVR routine in comparison to other diagonalization
methods. Eigenvalues (eps) are unaffected. A is the original matrix for
diagonalization, I is the unit matrix. The norm is defined as sum of
absolute values of elements divided by total number of elements. We
split matrix elements into diagonal and off-diagonal.

Values around $`10^{-6} - 10^{-5}`$ clearly indicate deviation from the
zero threshold for which one would expect numbers cca
$`10^{-15} - 10^{-14}`$.

### Eigenvalues and eigenvectors

By using the testing matrix, *Jz_SS_matrix.fermirp2-2*, we get properly
degenerate eigenvalues with DSYEVR and with other diagonalization
methods:

    1  -1.50000000000001
    2  -1.50000000000000
    3  -1.50000000000000
    4  0.499999999999999
    5  0.500000000000000
    6  0.500000000000000
    7  0.500000000000000
    8  0.500000000000002
    9   2.50000000000000

Now we can proceed to checking the numerical accuracy of obtained
eigenvectors.

### Intel+MKL-i8, DSYEVR:

    U^{+}*A*U - eps =  0 >  norm/diag:0.1684D-14    norm/offdiag:0.6623D-06
    U^{+}*U -I =  0      >  norm/diag:0.2097D-15    norm/offdiag:0.1325D-05
    U*U^{+} -I =  0      >  norm/diag:0.6293D-05    norm/offdiag:0.4174D-05

### GNU+ownmath-i8, DSYEVR:

    U^{+}*A*U - eps =  0 >  norm/diag:0.1739D-14   norm/offdiag:0.5091D-06
    U^{+}*U -I =  0      >  norm/diag:0.2097D-15   norm/offdiag:0.1018D-05
    U*U^{+} -I =  0      >  norm/diag:0.5239D-05   norm/offdiag:0.3407D-05

### Intel+MKL-i8, eispack RS:

    U^{+}*A*U - eps = 0  >  norm/diag:0.5058D-15   norm/offdiag:0.9620D-16
    U^{+}*U -I =  0      >  norm/diag:0.6168D-15   norm/offdiag:0.1400D-15
    U*U^{+} -I =  0      >  norm/diag:0.5674D-15   norm/offdiag:0.1586D-15

#### Independent demonstration program

For public purposes we are offering the standalone and DIRAC independent
demonstration program *dsyerv_check.F90* which does the same checks as
the above mentioned (DIRAC dependent) *diag.x* program. (Source codes
for downloading are
`dsyerv_check.F90 <../../../utils/dsyerv_check.F90>`,
`eispack.F <../../../src/pdpack/eispack.F>`, and the testing matrix is
`Jz_SS_matrix.fermirp2-2 <../../../test/diagonalization/Jz_SS_matrix.fermirp2-2>`.)

Results are as follows:

### GNU+/usr64/lib, DSYEVR:

    U^{+}*A*U - eps ?= 0> norm/diag:0.1783D-14  norm/offdiag:0.1419D-06
        U^{+}*U - I ?= 0> norm/diag:0.2591D-15  norm/offdiag:0.2838D-06
        U*U^{+} - I ?= 0> norm/diag:0.1047D-05  norm/offdiag:0.7950D-06

### Intel+MKL, DSYEVR:

    U^{+}*A*U - eps ?= 0> norm/diag:0.1684D-14  norm/offdiag:0.2746D-15
        U^{+}*U - I ?= 0> norm/diag:0.2097D-15  norm/offdiag:0.6923D-16
        U*U^{+} - I ?= 0> norm/diag:0.2467D-16  norm/offdiag:0.8681D-16

It is interesting to observe that obtained Intel+MKL results are in fact
correct and different that those from the *diag.x* DIRAC based
standalone program. This can be attributed to more complicated compiling
and linking flags of the *diag.x* code in the complex DIRAC's cmake
buildup apparatus.

### Intel+lapack 3.5.0, DSYEVR:

    U^{+}*A*U - eps ?= 0> norm/diag:0.1665D-14  norm/offdiag:0.9524D-07
        U^{+}*U - I ?= 0> norm/diag:0.2097D-15  norm/offdiag:0.1905D-06
        U*U^{+} - I ?= 0> norm/diag:0.7649D-06  norm/offdiag:0.5035D-06

The recent lapack 3.6.0 from netlib still suffers with non-orthogonal eigenvectors  
(see this [bug
report](https://github.com/Reference-LAPACK/lapack/issues/151) ).

## Solution of the problem

The simplest solution is to keep non-DSYEVR diagonalization routine in
the LINSYM part to avoid possibility of highly degenerate values with
numerically unstable eigenvectors. The eispack's RS routine, own DIRAC's
Householder or Jacobi methods do fit for this purpose because they can
properly handle eigenvectos of degenerate eigenvalues. The drawback is
the time efficiency, especially for large systems.

## Addendum

The dsyevr standalone testing program was placed on github,
<https://github.com/miroi/lapack-dsyevr-test>.

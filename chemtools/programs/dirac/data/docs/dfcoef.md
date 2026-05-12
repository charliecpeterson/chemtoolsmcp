orphan  

NOTE: *DFCOEF* file was replaced by *CHECKPOINT.h5* in DIRAC22, and the
information below is only relevant for older versions of DIRAC.

# DFCOEF

The binary *DFCOEF* file serves for storing molecular orbitals and
related information.

In the Fortran language it can be written like

    CHARACTER TEXT*74
    DIMENSION CMO(NCMOTQ),EIG(NORBT),IBEIG(NORBT),IDIM(3,2)
    DOUBLE PRECISION TOTERG

    DO I = 1,NFSYM
      IDIM(1,I) = NPSH(I)
      IDIM(2,I) = NESH(I)
      IDIM(3,I) = NFBAS(I,0)
    ENDDO
    WRITE(IUNIT) TEXT,NFSYM,((IDIM(I,J),I = 1,3),J=1,NFSYM),TOTERG

Let *i* be the integer size (4 or 8) and *d* the double precision size
(8). Size of the stored data, *Size*, given in first section above, is
*74+i*(1+3\*NFSYM)+d\*.

    i NFSYM Size
    4 1     74+4*4+8 = 98
    4 2     74+4*7+8 = 110
    8 1     74+8*4+8 = 114
    8 2              = 138

After calculating the *Size* we can store remaining data:

- molecular orbitals coefficients, CMO
- SCF eigenvalues, EIG
- specification of the irreducible representation per orbital, IBEIG

<!-- -->

    WRITE(IUNIT) CMO   Size d*NCMOTQ
    WRITE(IUNIT) EIG   Size d*NORBT
    WRITE(IUNIT) IBEIG Size i*NORBT

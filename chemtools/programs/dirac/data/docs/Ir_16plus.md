orphan  

# Getting excited states of $`Ir^{16+}`$

In the following we are interested in getting the ground
($`^{2}S_{1/2}`$) and two closest excited states ($`^{2}P_{1/2}`$,
$`^{2}P_{3/2}`$) of the $`Ir^{16+}`$ cation and their correlation
energies. The electronic configurations of these states are

    (^2S)     : [Xe] 4f(14) 5s(1) 5p1/2(0)  5p3/2(0)  
    (^2P_1/2) : [Xe] 4f(14) 5s(0) 5p1/2(1)  5p3/2(0)  
    (^2P_3/2) : [Xe] 4f(14) 5s(0) 5p1/2(0)  5p3/2(1) 

For simplicity, we will work with the wo-component Hamiltonian
(`HAMILTONIAN_.X2C`) and employ the smallest *v2z* decontracted basis
set by K.Dyall. Due to the convergence problem of the standalone AMFI
atomic SCF code we keep the +2 charge (`AMFI_.AMFICH`) for mean-field
orbitals.

There are two ways to obtain excited states - (i) from the converged SCF
state, and, (ii) only at the correlated level from the 2P_aver SCF
state.

For subsequent Coupled Cluster (CC) correlated calculations please
soften the DHOLU variable in the subroutine DENOM (file
*src/relccsd/cceqns.F*) to the value of 5.0D-4 and recompile DIRAC. Note
that this is not recommended approach as the *p32* states are in general
not well described at the CC level. Nevertheless, with this little trick
we can compare desired excited states calculated with both CC and
Fock-space CC methods.

## The $`^{2}S_{1/2}`$ ground state

Thanks to the linear symmetry, having two irreps, one can place the
unpaired electron the 1st irrep to get the $`^{2}S_{1/2}`$ ground state.
SCF calculations are followed by two CC calculations, where in the
second one uses orbital energies for denominators. :

    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2S12.scf_cc33e.2fs.inp
    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2S12.scf_cc33e_oe.2fs.inp

Input files to download are
`Ir.dyall_v2z.lsym.mol <Ir.dyall_v2z.lsym.mol>`,
`Z61.x2c.2S12.scf_cc33e.2fs.inp <Z61.x2c.2S12.scf_cc33e.2fs.inp>`,
`Z61.x2c.2S12.scf_cc33e_oe.2fs.inp <Z61.x2c.2S12.scf_cc33e_oe.2fs.inp>`.
Corresponding output files are
`Z61.x2c.2S12.scf_cc33e.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2S12.scf_cc33e.2fs_Ir.dyall_v2z.lsym.out>`,
`Z61.x2c.2S12.scf_cc33e_oe.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2S12.scf_cc33e_oe.2fs_Ir.dyall_v2z.lsym.out>`.

## The SCF $`^{2}P_{1/2}`$, $`^{2}P_{3/2}`$ excited states

By placing the unpaired electron into 2nd irrep one gets the
$`^{2}P_{1/2}`$ first excited state: :

    pam  --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2P12.scf_cc33e.2fs.inp --get "DFCOEF=DFCOEF.v2z.2P12"
    pam  --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2P12.scf_cc33e_oe.2fs.inp 

Input files to download
`Z61.x2c.2P12.scf_cc33e.2fs.inp <Z61.x2c.2P12.scf_cc33e.2fs.inp>`,
`Z61.x2c.2P12.scf_cc33e_oe.2fs.inp <Z61.x2c.2P12.scf_cc33e_oe.2fs.inp>`.
Corresponding output files are
`Z61.x2c.2P12.scf_cc33e.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2P12.scf_cc33e.2fs_Ir.dyall_v2z.lsym.out>`,
`Z61.x2c.2P12.scf_cc33e_oe.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2P12.scf_cc33e_oe.2fs_Ir.dyall_v2z.lsym.out>`.

How to obtain the $`^{2}P_{3/2}`$ second excited state at the SCF level,
and, consequently, at the CC level ? For that, we utilize the
`WAVE_FUNCTION_.REORDER MO` keyword with reading of the
*DFCOEF.v2z.2P12* file from the previous run. Likewise one uses overlap
selection in the SCF step: :

    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2P32.scf_cc33e.2fs.inp --put "DFCOEF.v2z.2P12=DFCOEF"
    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2P32.scf_cc33e_oe.2fs.inp --put "DFCOEF.v2z.2P12=DFCOEF"

Corresponding input files to download are
`Z61.x2c.2P32.scf_cc33e.2fs.inp <Z61.x2c.2P32.scf_cc33e.2fs.inp>`,
`Z61.x2c.2P32.scf_cc33e_oe.2fs.inp <Z61.x2c.2P32.scf_cc33e_oe.2fs.inp>`.
Output files are
`Z61.x2c.2P12.scf_cc33e.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2P12.scf_cc33e.2fs_Ir.dyall_v2z.lsym.out>`,
`Z61.x2c.2P12.scf_cc33e_oe.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2P12.scf_cc33e_oe.2fs_Ir.dyall_v2z.lsym.out>`.

## The CCSD(T) $`^{2}P_{1/2}`$, $`^{2}P_{3/2}`$ excited states

The other option is to start from the $`^{2}P_{aver}`$ averaged single
determinant state and distinguish between individual $`^{2}P_{1/2}`$ and
$`^{2}P_{3/2}`$ states at the Coupled Cluster correlated level thanks to
the linear symmetry.

First we test the averaged, $`^{2}P_{aver}`$, state: :

    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2Paver.scf_cc33e.2fs.inp
    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2Paver.scf_cc33e_oe.2fs.inp

Files to download are
`Z61.x2c.2Paver.scf_cc33e.2fs.inp <Z61.x2c.2Paver.scf_cc33e.2fs.inp>`,
`Z61.x2c.2Paver.scf_cc33e_oe.2fs.inp <Z61.x2c.2Paver.scf_cc33e_oe.2fs.inp>`.

Afterwards we can proceed to the individual spin-orbit distinguished
states, based on $`M_{J}`$ splitted occupation of each fermion irrep at
the Coupled Cluster level.

First the first excited state, $`^{2}P_{1/2}`$: :

    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2Paver.scf_cc33e_2P12.2fs.inp
    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2Paver.scf_cc33e_oe_2P12.2fs.inp

Input files to download are
`Z61.x2c.2Paver.scf_cc33e_2P12.2fs.inp <Z61.x2c.2Paver.scf_cc33e_2P12.2fs.inp>`,
`Z61.x2c.2Paver.scf_cc33e_oe_2P12.2fs.inp <Z61.x2c.2Paver.scf_cc33e_oe_2P12.2fs.inp>`.
Corresponding output files are
`Z61.x2c.2Paver.scf_cc33e_2P12.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2Paver.scf_cc33e_2P12.2fs_Ir.dyall_v2z.lsym.out>`,
`Z61.x2c.2Paver.scf_cc33e_oe_2P12.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2Paver.scf_cc33e_oe_2P12.2fs_Ir.dyall_v2z.lsym.out>`.

Then we proceed to the second excited state, $`^{2}P_{3/2}`$: :

    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2Paver.scf_cc33e_2P32.2fs.inp
    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.2Paver.scf_cc33e_oe_2P32.2fs.inp

Files to download are
`Z61.x2c.2Paver.scf_cc33e_2P32.2fs.inp <Z61.x2c.2Paver.scf_cc33e_2P32.2fs.inp>`,
`Z61.x2c.2Paver.scf_cc33e_oe_2P32.2fs.inp <Z61.x2c.2Paver.scf_cc33e_oe_2P32.2fs.inp>`.
Corresponding output files are
`Z61.x2c.2Paver.scf_cc33e_2P32.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2Paver.scf_cc33e_2P32.2fs_Ir.dyall_v2z.lsym.out>`,
`Z61.x2c.2Paver.scf_cc33e_oe_2P32.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.2Paver.scf_cc33e_oe_2P32.2fs_Ir.dyall_v2z.lsym.out>`.

## The $`^{2}S_{1/2}`$, $`^{2}P_{1/2}`$ and $`^{2}P_{3/2}`$ FSCCSD states

Simple and very stable approach to obtain ground and multiple excited
states in one step is through the Fock-space Coupled Cluster method.
Starting from the closed-shell system, $`Ir^{17+}`$, one gets - by
solving (01) sector all three correlated states of interest,
$`^{2}S_{1/2}`$, $`^{2}P_{1/2}`$ and $`^{2}P_{3/2}`$ :

    pam --noarch --mw=120  --mol=Ir.dyall_v2z.lsym.mol --inp=Z61.x2c.scf_fscc01_33ce_5s5p.2fs.inp

The input file to download is
`Z61.x2c.scf_fscc01_33ce_5s5p.2fs.inp <Z61.x2c.scf_fscc01_33ce_5s5p.2fs.inp>`.
Corresponding output file is
`Z61.x2c.scf_fscc01_33ce_5s5p.2fs_Ir.dyall_v2z.lsym.out <Z61.x2c.scf_fscc01_33ce_5s5p.2fs_Ir.dyall_v2z.lsym.out>`.

### Overview of excitation energies

In the following table we summarize excitation energies. All values are
in a.u. Energies in the Table are not rounded, the are cut to 8 decimal
places ("oe" means orbital energies used in CC denominators, otherwise
recalculated diagonal Fock matrix elements).

| Method    | ^2S\_{1/2}      | ^2P\_{1/2}      | ^P\_{3/2}       | 2S12-2P12 | 2S12-2P32 |
|-----------|-----------------|-----------------|-----------------|-----------|-----------|
| (SCF ref) |                 |                 |                 |           |           |
| SCF       | -17751.10181462 | -17749.67796221 | -17748.96107014 | 1.42385   | 2.14074   |
| CCSD      | -17751.90589433 | -17750.48885480 | -17749.77167089 | 1.41704   | 2.13422   |
| CCSDoe    | -17751.90589433 | -17750.48885479 | -17749.77167088 | 1.41704   | 2.13422   |
| CCSD(T)   | -17751.90998637 | -17750.49777405 | -17749.78015403 | 1.41221   | 2.12983   |
| CCSD(T)oe | -17751.90999205 | -17750.49779342 | -17749.78020119 | 1.41220   | 2.12979   |
| (CC ref)  |                 |                 |                 |           |           |
| CCSD      | -17751.90589433 | -17750.48895732 | -17749.77154045 | 1.41694   | 2.13435   |
| CCSDoe    | -17751.90589433 | -17750.48895728 | -17749.77154043 | 1.41694   | 2.13435   |
| CCSD(T)   | -17751.90998637 | -17750.49779338 | -17749.78012458 | 1.41219   | 2.12986   |
| CCSD(T)oe | -17751.90999205 | -17750.49784867 | -17749.78024342 | 1.41214   | 2.12974   |
| FSCCSD    | -17751.90324662 | -17750.48431436 | -17749.76770431 | 1.41893   | 2.13554   |

It seems that quality of computed excitation energies increases in the
line SCF-FSCCSD-CCSD-CCSD(T). Triple excitations (CCSD(T) results) are
significant.

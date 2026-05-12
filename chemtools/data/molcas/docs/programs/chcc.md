<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/chcc.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.6. CHCC

[previous](<ccsdt.html> "4.2.5. CCSDT") | [next](<cht3.html> "4.2.7. CHT3") | [index](<../../genindex.html> "General Index")

# 4.2.6. CHCC¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input




The CHCC is a Closed-Shell Coupled-Clusters Singles and Doubles program based exclusively on the Cholesky (or RI) decomposed 2-electron integrals aimed towards calculation of large systems on highly parallel architectures. Use of point-group symmetry is not implemented. Main advantage compared to the CCSDT module in Molcas is in its more efficient parallelization and dramatically lowered memory (and eventually disk) requirements.

## 4.2.6.1. Dependencies¶

CHCC requires a previous run of the RHF SCF program to produce molecular orbitals and orbital energies stored in RUNFILE. The SCF program (as well as SEWARD) must be run in Cholesky/RI mode.

The algorithm used for almost complete elimination of the CHCC limits in calculated system size due to the computer memory bottleneck relies on blocking of the virtual orbitals. Number of blocks (further also referred to as the “large” segmentation, LARGe), \\(N'\\), should be as small as possible, because increasing of the segmentation brings in more CPU and I/O overhead. Furthermore, blocking can be “fine tuned” by, so called, “small” segmentation (SMALl), \\(N''\\), which affects only the (typically) most demanding \\(O^2V^4\\) scaling terms. The “large” segmentation can range from 1 to 32, “small” segmentation from 1 to 8, but their product, i.e. “large” × “small” must be no more than 64.

Selected blocking also determines the number of “independent” parallel tasks that must be executed in each iteration of the CCSD equations. In other words, particular segmentation predetermines the optimal number of computational nodes (i.e., if the best possible parallelization is desired). If the requested “large” segmentation is \\(N'\\), then \\(N'^2\\) terms scaling as \\(O^3V^3\\) and \\(N'^2/2\\) terms scaling as \\(O^2V^4\\) result. Depending on which of these terms dominated in the calculations (\\(O^3V^3\\) is more demanding for systems with large number of occupied orbitals and rather small basis set, while \\(O^2V^4\\) dominated for relatively large basis sets, i.e. large number of virtual orbitals), number of these task should be divisible by the number of computational nodes for optimal performance. To make it simple, as a rule of thumb, \\(N'^2/2\\) should be divisible by the number of nodes, since the \\(O^3V^3\\) are typically twice less expensive then the \\(O^2V^4\\) step. Otherwise, any reasonable (i.e. the number of tasks is larger than the number of computational nodes, obviously) combination is allowed.

## 4.2.6.2. Files¶

### 4.2.6.2.1. Input files¶

CHCC will use the following input files: CHVEC, CHRED, CHORST, RUNFILE, and CHOR2F (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

### 4.2.6.2.2. Output files¶

L0xxxx, L1xxxx, L2xxxx
    

MO-transformed Cholesky vectors

T2xxxx
    

T2 \\((ij,a'b')\\) excitation amplitudes

RstFil
    

Communication file containing T1 amplitudes, restart informations, etc.

## 4.2.6.3. Input¶

The input for each module is preceded by its name like:
    
    
    &CHCC
    

Optional keywords

TITLe
    

This keyword is followed by one title line.

FROZen
    

Integer on the following line specifies number of inactive occupied orbitals in the CCSD calculation. (Default=0)

DELEted
    

Integer on the following line specifies number of inactive virtual orbitals in the CCSD calculation. (Default=0)

LARGe
    

Integer on the following line specifies the main segmentation of the virtual orbitals. Value must be between 1 (no segmentation) and 32. Product of Large and Small segmentation must be lower than 64. (Default=1)

SMALl
    

Integer on the following line specifies the auxiliary segmentation of the virtual orbitals. Value must be between 1 (no segmentation) and 8. Product of Large and Small segmentation must be lower than 64. Small segmentation doesn’t generate extra parallel tasks. (Default=1)

CHSEgmentation
    

Integer on the following line specifies the block size of the auxiliary (Cholesky/RI) index. Value must be lower than the minimal dimension of the auxiliary index on each computational node. (Default=100)

MHKEy
    

Integer on the following line specifies if library BLAS (MHKEy=1) or hard-coded fortran vector-vector, matrix-vector and matrix-matrix manipulation is used. (Default=1)

NOGEnerate
    

This keyword specifies that the pre-CCSD steps (regeneration of integrals from the Cholesky/RI vectors, etc.) are skipped. (Default=OFF)

ONTHefly
    

This keyword specifies that all integral types scaling steeper then \\(O^2V^2\\) are generated “on-the-fly” from the Cholesky/RI vectors. Use of this keyword leads to dramatically savings of the disk resources, but leads to significant arithmetic overhead. Keywords “ONTHefly” and “PRECalculate” are mutually exclusive. (Default=OFF)

PRECalculate
    

This keyword specifies that all integral are precalculated before the CCSD iterative procedure starts. Use of this keyword leads to significant consumption of the disk space, especially is single-processor runs. (Default=ON)

NODIstribute
    

This keyword (in combination with the “PRECalculate” keyword) specifies that all integral are stored on each computational node. In case of all integrals being stored on each node, extra permutation symmetry can be applied, thus leading to significant savings of the disk space. However, in case of massively parallel runs (i.e. more than ~8 nodes), savings from keeping only subset of integrals required on particular node are more significant than savings due to permutational symmetry. (Default=OFF)

JOINlkey
    

The parameter on the following line specifies, which algorithm is used for precalculation and of the integrals in parallel run. In parallel runs, SEWARD produces AO Cholesky/RI vectors segmented in auxiliary index over parallel nodes. Depending on the network bandwidth and computational power of each node, different algorithms can lead to optimal performance. Following options are available:

0 — None: no cumulation of Cholesky/RI vectors is needed (debug only).

1 — Minimal: Cholesky/RI vectors are cumulated prior to integral precalculation. Low network bandwidth is required.

2 — Medium: \\(O^2V^2\\) integrals are generated from local Cholesky/RI vectors and cumulated along with the Cholesky/RI vectors afterwards. Other integrals are calculated from cumulated intermediates.

3 — Full: All integrals are generated from local Cholesky/RI vectors and cumulated afterwards. High network bandwidth is required.

(Default=2)

MAXIterations
    

Integer on the following line specifies maximum number of CCSD iteration (Default=40)

RESTart
    

This keyword specifies that CCSD calculation is restarted from previous run. This keyword is currently under development, thus disabled. (Default=OFF)

THREshold
    

Double precision floating point number on the following line specifies the convergence threshold for the CCSD correlation energy. (Default=1.0d-6)

PRINtkey
    

The integer on the following line specifies the print level in output

1 — Minimal

2 — Minimal + timings of each step of the CCSD iterations

10 — Debug

(Default=1)

END of input
    

This keyword indicates that there is no more input to be read.
    
    
    &CHCC &END
    Title
    Benzene dimer
    Frozen
    12
    Deleted
    0
    Large
    4
    Small
    2
    CHSEgment
    100
    Precalculate
    Join
    2
    Maxiter
    50
    Threshold
    1.0d-6
    Print
    2
    End of Input
    

### Table of Contents

  * [1\. Introduction](<../../intro.html>)
  * [2\. Installation Guide](<../../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../../tutorials/tut.html>)
  * [4\. User’s Guide](<../ug.html>)
    * [4.1. The Molcas environment](<../env-main.html>)
    * [4.2. Programs](<../programs.html>)
      * [4.2.1. ALASKA](<alaska.html>)
      * [4.2.2. AVERD](<averd.html>)
      * [4.2.3. CASPT2](<caspt2.html>)
      * [4.2.4. CASVB](<casvb.html>)
      * [4.2.5. CCSDT](<ccsdt.html>)
      * 4.2.6. CHCC
      * [4.2.7. CHT3](<cht3.html>)
      * [4.2.8. CMOCORR ¤](<cmocorr.html>)
      * [4.2.9. CPF](<cpf.html>)
      * [4.2.10. DIMERPERT ¤](<dimerpert.html>)
      * [4.2.11. DMRGSCF](<dmrgscf.html>)
      * [4.2.12. DYNAMIX](<dynamix.html>)
      * [4.2.13. EMBQ ¤](<embq.html>)
      * [4.2.14. ESPF (+ QM/MM interface)](<espf.html>)
      * [4.2.15. EXPBAS](<expbas.html>)
      * [4.2.16. EXTF](<extf.html>)
      * [4.2.17. FALCON ¤](<falcon.html>)
      * [4.2.18. FALSE](<false.html>)
      * [4.2.19. FFPT](<ffpt.html>)
      * [4.2.20. GATEWAY](<gateway.html>)
      * [4.2.21. GENANO](<genano.html>)
      * [4.2.22. GEO ¤](<geo.html>)
      * [4.2.23. GRID_IT](<grid_it.html>)
      * [4.2.24. GUESSORB](<guessorb.html>)
      * [4.2.25. GUGA](<guga.html>)
      * [4.2.26. GUGACI](<gugaci.html>)
      * [4.2.27. GUGADRT](<gugadrt.html>)
      * [4.2.28. LEVEL](<level.html>)
      * [4.2.29. LOCALISATION](<localisation.html>)
      * [4.2.30. LOPROP](<loprop.html>)
      * [4.2.31. MBPT2](<mbpt2.html>)
      * [4.2.32. MCKINLEY (a.k.a. DENALI)](<mckinley.html>)
      * [4.2.33. MCLR](<mclr.html>)
      * [4.2.34. MCPDFT](<mcpdft.html>)
      * [4.2.35. MKNEMO ¤](<mknemo.html>)
      * [4.2.36. MOTRA](<motra.html>)
      * [4.2.37. MPPROP](<mpprop.html>)
      * [4.2.38. MPSSI](<mpssi.html>)
      * [4.2.39. MRCI](<mrci.html>)
      * [4.2.40. MULA](<mula.html>)
      * [4.2.41. NEMO ¤](<nemo.html>)
      * [4.2.42. NEVPT2](<nevpt2.html>)
      * [4.2.43. NUMERICAL_GRADIENT](<numerical_gradient.html>)
      * [4.2.44. POLY_ANISO](<poly_aniso.html>)
      * [4.2.45. QMSTAT](<qmstat.html>)
      * [4.2.46. QUATER](<quater.html>)
      * [4.2.47. RASSCF](<rasscf.html>)
      * [4.2.48. RASSI](<rassi.html>)
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
      * [4.2.52. SEWARD](<seward.html>)
      * [4.2.53. SINGLE_ANISO](<single_aniso.html>)
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<ccsdt.html> "4.2.5. CCSDT") | [next](<cht3.html> "4.2.7. CHT3") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/chcc.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/cht3.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.7. CHT3

[previous](<chcc.html> "4.2.6. CHCC") | [next](<cmocorr.html> "4.2.8. CMOCORR ¤") | [index](<../../genindex.html> "General Index")

# 4.2.7. CHT3¶

  * Dependencies

  * Files

    * Input files

    * Intermediate files

    * Output files

  * Input




CHT3 is a Closed-Shell Coupled-Clusters perturbative triples program based exclusively on the Cholesky (or RI) decomposed 2-electron integrals aimed towards calculation of large systems on highly parallel architectures. Use of point-group symmetry is not implemented. Main advantage compared to the CCSDT module is Molcas is in its more efficient parallelization and dramatically lowered memory (and eventually disk) requirements.

## 4.2.7.1. Dependencies¶

CHT3 requires previous run of the CHCC Cholesky/RI based CCSD program to produce T1 and T2 excitation amplitudes stored in T2xxxx and RstFil files. The CHCC program (as well as SEWARD and SCF) must be run in Cholesky/RI mode.

The algorithm used for almost complete elimination of the CHT3 limits in calculated system size due to the computer memory bottleneck relies on blocking of the virtual orbitals. Size of blocks is, unlike in CHCC program, determined automatically for optimal performance.

## 4.2.7.2. Files¶

### 4.2.7.2.1. Input files¶

RUNFILE
    

File for communication of auxiliary information.

L0xxxx, L1xxxx, L2xxxx
    

MO-transformed Cholesky vectors

T2xxxx
    

T2 \\((ij,a'b')\\) excitation amplitudes

RstFil
    

Communication file containing T1 amplitudes, restart informations, etc.

### 4.2.7.2.2. Intermediate files¶

All the intermediate files are created, used and removed automatically, unless you yourself create a link or a file with the specified name.

KMATAA, KMATBA, LMATAA, LMATBA
    

Temporary integral files

### 4.2.7.2.3. Output files¶

None

## 4.2.7.3. Input¶

The input for each module is preceded by its name like:
    
    
    &CHT3
    

TITLe
    

This keyword starts the reading of title lines, with the number of title lines limited to 10. Reading the input as title lines is stopped as soon as the input parser detects one of the other keywords, however only ten lines will be accepted. This keyword is _optional_.

FROZen
    

Integer on the following line specifies number of inactive occupied orbitals in the (T) calculation. This keyword is _optional_. (Default=0)

DELEted
    

Integer on the following line specifies number of inactive virtual orbitals in the (T) calculation. This keyword is _optional_. (Default=0)

LARGe
    

Integer on the following line specifies the main segmentation of the virtual orbitals used in previous CCSD run.

NOGEnerate
    

This keyword specifies that the pre-(T) steps (generation of integrals from the Cholesky/RI vectors, etc.) are skipped. This keyword can be used for restarting the (T) calculation if the required integrals were already generated. This keyword is _optional_. (Default=OFF)

NOTRiples
    

This keyword specifies that the post integral preparation steps, i.e. the real calculation of (T) contribution will not be done. Job can be restarted from this point using the NOGEnerate keyword. This keyword is _optional_. (Default=OFF)

ALOOp
    

Two integers on the following line specify first and last triplet of virtual orbitals blocks to be calculated in the first (“A loop”) of the two parts of the (T) calculation. Using this keyword enables user to split the (T) calculation into separate jobs. Information about the total number of triplets in the “A loop” can be found in the output of the “preparation” step of the (T) program. Values -1, -1 mean, that the whole “A loop” is either executed or skipped, depending on the parameters of the BLOOp keyword. This keyword is _optional_. (Default=-1,-1)

BLOOp
    

Two integers on the following line specify first and last triplet of virtual orbital block to be calculated in the second (“B loop”) of two parts of the (T) calculation. Using this keyword enables user to split the (T) calculation into separate jobs. Information about the total number of triplets in the “B loop” can be found in the output of the “preparation” step of the (T) program. Values -1, -1 mean, that the whole “B loop” is either executed or skipped, depending on the values of the ALOOp keyword. This keyword is _optional_. (Default=-1,-1)

PRINtkey
    

The integer on the following line specifies the print level in output

1 — Minimal

2 — Minimal + timings of each (T) step

10 — Debug

This keyword is _optional_. (Default=1)
    
    
    &CHT3
    Title  = Benzene dimer
    Frozen = 12
    Large  = 4
    ALOOp  = 20 120
    BLoop  = 1 250
    Print  = 2
    

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
      * [4.2.6. CHCC](<chcc.html>)
      * 4.2.7. CHT3
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

[previous](<chcc.html> "4.2.6. CHCC") | [next](<cmocorr.html> "4.2.8. CMOCORR ¤") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/cht3.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

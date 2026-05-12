<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/quater.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.46. QUATER

[previous](<qmstat.html> "4.2.45. QMSTAT") | [next](<rasscf.html> "4.2.47. RASSCF") | [index](<../../genindex.html> "General Index")

# 4.2.46. QUATER¶

  * Dependencies

  * Files

    * Input files

  * Input

    * Keywords

    * Input example




## 4.2.46.1. Dependencies¶

The QUATER is free-standing and does not depend on any other program.

## 4.2.46.2. Files¶

### 4.2.46.2.1. Input files¶

The calculation of vibrational wave functions and spectroscopic constants uses no input files (except for the standard input).

## 4.2.46.3. Input¶

This section describes the input to the QUATER program in the Molcas program system. The program name is
    
    
    &QUATER
    

### 4.2.46.3.1. Keywords¶

NOROtation
    

No rotation is performed by the program. Only the rotation matrix is printed out.

NOTRanslation
    

No translation is performed by the program.

DEBUg
    

Turn on DEBUG printout

AXIS
    

Define the old frame of reference

NEWAxis
    

Define the new frame of reference

GEO1
    

Define the first geometry

GEO2
    

Define the second geometry

XYZ1
    

Define the origin and two axes for the orientation of the first geometry by the index of three atoms of this geometry.

XYZ2
    

Define the origin and two axes for the orientation of the second geometry by the index of three atoms of this geometry.

END
    

End of input

QUATER will perform a vib-rot analysis and compute spectroscopic constants.

### 4.2.46.3.2. Input example¶
    
    
    &QUATER
    
    GEO1
       19
    titre
     C     0.000000     0.000000     0.000000
     O     0.000000     0.000000     1.400000
     H     0.895670     0.000000     1.716663
     C    -0.683537    -1.183920    -0.483333
     H    -0.513360     0.889165    -0.363000
     C    -0.683537    -1.183920    -1.933333
     H    -0.170177    -2.073085    -0.120333
     H    -1.710256    -1.183920    -0.120333
     C     0.683537    -1.183920    -2.416667
     H    -1.196896    -2.073085    -2.296333
     H    -1.196896    -0.294755    -2.296333
     C     1.367073     0.000000    -1.933333
     H     1.196896    -2.073085    -2.053667
     H     0.683537    -1.183920    -3.505667
     C     1.367073     0.000000    -0.483333
     H     2.393792     0.000000    -2.296333
     H     0.853714     0.889165    -2.296333
     H     1.880433    -0.889165    -0.120333
     H     1.880433     0.889165    -0.120333
    END
    GEO2
       23
    titre
     C     0.000000     0.000000     0.000000
     H     0.000000     0.000000     1.089000
     C     1.367075     0.000000    -0.483328
     H    -0.334267    -0.970782    -0.363000
     C     1.367081     0.000000    -1.933328
     H     1.880433     0.889165    -0.120326
     H     1.880433    -0.889165    -0.120326
     C     0.683546     1.183920    -2.416664
     H     2.393801     0.000000    -2.296324
     H     0.853722    -0.889165    -2.296330
     C    -0.683529     1.183920    -1.933336
     H     1.196904     2.073085    -2.053662
     O     0.683551     1.183920    -3.816664
     C    -0.683535     1.183920    -0.483336
     H    -1.196887     2.073085    -2.296338
     H    -1.196887     0.294755    -2.296338
     O    -0.023570     2.327015    -0.016667
     H    -1.710255     1.183920    -0.120340
     H     0.237132     1.957142    -4.132332
     C    -0.023576     2.327015     1.383333
     H     0.489783     3.216180     1.746335
     H    -1.050296     2.327015     1.746329
     H     0.489783     1.437850     1.746335
    END
    XYZ1
    15 12 9
    XYZ2
    11 14 1
    END
    

This input will perform the alignment of the second geometry (GEO2) on the first one (GEO1). Atom number 11 (C11) of the second geometry will be moved to the position of atom number 15 of the first geometry (C15). The vector C11 C14 in GEO1 will be aligned with the vector C15 C12 of GEO1. Finally the plane 11 14 1 of GEO1 will be aligned with the plane 15 12 9 of GEO2.

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
      * 4.2.46. QUATER
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

[previous](<qmstat.html> "4.2.45. QMSTAT") | [next](<rasscf.html> "4.2.47. RASSCF") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/quater.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

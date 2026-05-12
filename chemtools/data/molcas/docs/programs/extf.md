<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/extf.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.16. EXTF

[previous](<expbas.html> "4.2.15. EXPBAS") | [next](<falcon.html> "4.2.17. FALCON ¤") | [index](<../../genindex.html> "General Index")

  15. index:: single: Program; Extf single: Extf




# 4.2.16. EXTF¶

  * Input

    * General keywords

    * Input examples




This module calculates the contribution of an external force that is acting on the system. It applies the modification directly on the gradient and it needs to be called after the execution of ALASKA, in an optimization or molecular dynamics calculation. The keyword LINEAR applies a constant linear force between two atoms [[81](<../../references.html#id308> "A. Valentini, D. Rivero, F. Zapata, C. García-Iriepa, M. Marazzi, R. Palmeiro, I. Fdez. Galván, D. Sampedro, M. Olivucci, L. M. Frutos. Angew. Chem. Int. Ed., 56\[14\] \(2017\) 3842-3846.")].

## 4.2.16.1. Input¶

### 4.2.16.1.1. General keywords¶

MODULE
    

Module of the force to apply, in nanonewton. If it’s negative, the force is applied in opposite direction. See the other keywords for what is the direction of positive and negative forces. Note that this is the module of the total force, so for example, in the case of a force pair between two atoms, the force applied on each atom will be a factor of \\(\sqrt{2}\\) smaller than this value.

LINEAR
    

This keyword is followed by two integer values, specifying the atom numbers (following the numbering of the geometry) between which a force is applied along the vector joining them. A positive force (see the MODULE keyword) means an attractive (compression) force, a negative force is a repulsive (extension) force.

BENDING
    

This keyword is followed by three integer values, specifying the atom numbers (following the numbering of the geometry) between which a force is applied to open or close their planar angle. A positive force (see the MODULE keyword) tends to close the angle, a negative force opens it.

TORSIONAL
    

This keyword is followed by four integer values, specifying the atom numbers (following the numbering of the geometry) between which a force is applied to open or close their dihedral angle. A positive force (see the MODULE keyword) tends to close positive dihedrals (i.e. towards less positive values), a negative force opens positive dihedrals (towards more positive values).

GAUSSIAN
    

This keyword modulates the applied force with a Gaussian time profile. It is followed by two real values, indicating the time at which the force is maximum (i.e. the value specified by MODULE) and a sigma value for the Gaussian decay.

### 4.2.16.1.2. Input examples¶

The following input example is a semiclassical molecular dynamics with tully surface hop, where a linear force of about 2.9 nN is applied between atom 1 and atom 2.
    
    
    &Gateway
    coord=$Project.xyz
    basis=6-31G*
    group=nosym
    
    >> EXPORT MOLCAS_MAXITER=400
    >> DOWHILE
    
    &Seward
    
    &rasscf
     nactel = 6 0 0
     inactive = 23
     ras2 = 6
     ciroot = 2 2 1
     prwf = 0.0
     mdrlxroot = 2
    
    &alaska
    
    &surfacehop
     tully
     decoherence = 0.1
     psub
    
    &Extf
     LINEAR
     1 2
     MODULE
     -4.1
    
    &Dynamix
     velver
     dt = 41.3
     velo = 1
     thermo = 0
    >>> End Do
    

This example shows an excited state CASSCF MD simulation of a methaniminium cation using the Tully Surface Hop algorithm. In the simulation, the carbon and the nitrogen are pulled apart with a constant force of 1.5 nN (nanonewton) on each atom. Within the EXTF module the keyword LINEAR is used. Note EXTF needs to be called after the execution of ALASKA, inside the loop.
    
    
    &GATEWAY
     COORD
     6
     angstrom
     C  0.00031448  0.00000000  0.04334060
     N  0.00062994  0.00000000  1.32317716
     H  0.92882820  0.00000000 -0.49115611
     H -0.92846597  0.00000000 -0.49069213
     H -0.85725321  0.00000000  1.86103989
     H  0.85877656  0.00000000  1.86062860
     BASIS= 3-21G
     GROUP= nosym
    
    >> EXPORT MOLCAS_MAXITER=1000
    >> DOWHILE
    
    &SEWARD
    
    >> IF ( ITER = 1 )
    
    &RASSCF
      LUMORB
     FileOrb= $Project.GssOrb
     Symmetry= 1
     Spin= 1
     nActEl= 2 0 0
     Inactive= 7
     RAS2= 2
     CIroot= 3 3 1
    
    >> COPY $Project.JobIph $Project.JobOld
    
    >> ENDIF
    
    &RASSCF
     JOBIPH; CIRESTART
     Symmetry= 1
     Spin= 1
     nActEl= 2 0 0
     Inactive= 7
     RAS2= 2
     CIroot= 3 3 1
     MDRLXR= 2
    
    >> COPY $Project.JobIph $Project.JobOld
    
    &surfacehop
     TULLY
     SUBSTEP = 200
     DECOHERENCE = 0.1
     PSUB
    
    &ALASKA
    
    &extf
     LINEAR
     1 2
     MODULE
     -2.12132
    
    &Dynamix
     VELVer
     DT= 10.0
     VELO= 3
     THER= 2
     TEMP=300
    
    >> END DO
    

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
      * 4.2.16. EXTF
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

[previous](<expbas.html> "4.2.15. EXPBAS") | [next](<falcon.html> "4.2.17. FALCON ¤") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/extf.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

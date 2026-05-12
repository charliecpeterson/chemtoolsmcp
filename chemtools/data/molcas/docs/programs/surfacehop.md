<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/surfacehop.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.55. SURFACEHOP

[previous](<slapaf.html> "4.2.54. SLAPAF") | [next](<symmetrize.html> "4.2.56. SYMMETRIZE") | [index](<../../genindex.html> "General Index")

# 4.2.55. SURFACEHOP¶

  * Output files

    * Input

  * General keywords

  * Input examples




This module deals with surface hop semiclassical molecular dynamics (SHMD) and has to be used together with module DYNAMIX. Its purpose is the calculation of the relax root for the next step of the SHMD. The implemented algorithm under this module is the Tully’s fewest switches [[220](<../../references.html#id305> "J. C. Tully. J. Chem. Phys., 93\[2\] \(1990\) 1061-1071.")], using the Hammes-Schiffer/Tully scheme [[221](<../../references.html#id306> "S. Hammes-Schiffer, J. C. Tully. J. Chem. Phys., 101\[6\] \(1994\) 4657-4667.")] and the decoherence correction proposed by Granucci and Persico [[222](<../../references.html#id307> "G. Granucci, M. Persico. J. Chem. Phys., 126\[13\] \(2007\) 134114\(1–11\).")].

Under the Hammes-Schiffer/Tully scheme, the non-adiabatic population transfer between states of the same multiplicity is determined using the wavefunction overlap between the current timestep and the two previous timesteps, in an interpolation-extrapolation scheme. This is done in lieu of calculating explicitly the non-adiabatic coupling, and thus allows for surface-hopping when explicit non-adiabatic coupling is not available or is too expensive.

There are two methods to calculate the wavefunction overlap available through the SURFACEHOP module. The default implementation calls the RASSI module to obtain the overlap matrix between all states at the current and previous timestep. The alternative method (previously default) can be requested using the keyword NORASSI and uses instead a dot product of the CI vectors to approximate the overlap matrix.

## 4.2.55.1. Output files¶

RUNFILE
    

Surface hop information such as Amatrix and CI coefficients for previous steps are stored in this file.

$Project.md.xyz
    

Contains the geometry of every timestep in the dynamics, in standard xyz coordinates.

$Project.md.energies
    

Contains the Potential energy of the current active state, Kinetic energy, and Total energy of the system throughout the simulation, followed by the potential energies of all states in the dynamics.

### 4.2.55.1.1. Input¶
    
    
    &Gateway
    coord=$Project.xyz
    basis=6-31G*
    group=nosym
    
    >> EXPORT MOLCAS_MAXITER=400
    >> DOWHILE
    
    &Seward
    
    &rasscf
     jobiph
     cirestart
     nactel = 6 0 0
     inactive = 23
     ras2 = 6
     ciroot = 2 2 1
     prwf = 0.0
     mdrlxroot = 2
    
    &Surfacehop
     tully
     decoherence = 0.1
     psub
    
    &alaska
    
    &Dynamix
     velver
     dt = 41.3
     velo = 1
     thermo = 0
    >>> End Do
    

## 4.2.55.2. General keywords¶

TULLY
    

This keyword enables the Tully–Hammes-Schiffer integration of the TDSE for the Tully Surface Hop Algorithm. If you use this keyword you should not use the HOP keyword in DYNAMIX.

NORASSI
    

This keyword must be used after the TULLY keyword. It disables the use of RASSI to calculate wavefunction overlaps, instead using the dot product of CI vectors (previous default option).

DECOHERENCE
    

This keyword must be used after the TULLY keyword. It enables the decoherence correction in the population density matrix as reported by Persico and Granucci. The value is called decay factor and it is usually 0.1 hartree. It can be seen as how strongly this correction is applied. It is recommendable to leave it to 0.1, unless you really know what you’re doing.

SUBSTEP
    

This keyword must be used after the TULLY keyword. This keyword specifies how many steps of integration we use to interpolate/extrapolate between two Newton’s consecutive steps. The default is usually a good compromise between quickness and precision (200 substeps each femtoseconds of MD).

PSUB
    

This keyword must be used after the TULLY keyword. To print in Molcas output \\(\mat{D}\\) matrix, \\(\mat{A}\\) matrix, \\(\mat{B}\\) matrix, probabilities, randoms, population and energies at each substep (quite verbose, but gives you a lot of useful information).

DMTX
    

This keyword must be used after the TULLY keyword. With this keyword you can start your calculation with an initial \\(\mat{A}\\) matrix (population density matrix). It is a complex matrix. In the first line after the keyword you must specify its dimension \\(N\\). Then \\(N\\) lines (\\(N\\) values each line) with the real part of the matrix followed by \\(N\\) more lines with the imaginary part.

FRANDOM
    

This keyword must be used after the TULLY keyword. It fixes the random number to one provided by the user, in case a deterministic trajectory is needed

ISEED
    

This keyword must be used after the TULLY keyword. The initial seed number is read from the input file. Then, seed numbers are modified (in a deterministic way), saved in the RunFile and read in the next call to the module. This way, MD simulations are reproducible.

MAXHOP
    

This keyword must be used after the TULLY keyword. It specifies how many non-adiabatic transitions are allowed between electronic states.

H5RESTART
    

This keyword allows to restart a surface hopping trajectory calculation from an HDF5 file. The name of the restart file is given on the next line.

## 4.2.55.3. Input examples¶

This example shows an excited state CASSCF MD simulation of a methaniminium cation using the Tully Surface Hop algorithm. Within the SURFACEHOP module The keyword TULLY enables the TDSE integration. The options used in this case are: (SUBSTEP=200) to specify 200 substep of electronic integration between Newton’s, (DECOHERENCE=1) to deal with the decoherence using a decay constant of 0.1 hartree and (PSUB) to print the substeps matrices verbosely into the Molcas log.
    
    
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
      * 4.2.55. SURFACEHOP
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<slapaf.html> "4.2.54. SLAPAF") | [next](<symmetrize.html> "4.2.56. SYMMETRIZE") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/surfacehop.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

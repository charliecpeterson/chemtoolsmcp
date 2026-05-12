<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/vibrot.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.57. VIBROT

[previous](<symmetrize.html> "4.2.56. SYMMETRIZE") | [next](<wfa.html> "4.2.58. WFA") | [index](<../../genindex.html> "General Index")

# 4.2.57. VIBROT¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




The program VIBROT is used to compute a vibration-rotation spectrum for a diatomic molecule, using as input a potential computed over a grid. The grid should be dense around equilibrium (recommended spacing 0.05 au) and should extend to large distance (say 50 au) if dissociation energies are computed.

The potential is fitted to an analytical form using cubic splines. The ro-vibrational Schrödinger equation is then solved numerically (using Numerov’s method) for one vibrational state at a time and for a number of rotational quantum numbers as specified by input. The corresponding wave functions are stored on file VIBWVS for later use. The ro-vibrational energies are analyzed in terms of spectroscopic constants. Weakly bound potentials can be scaled for better numerical precision.

The program can also be fed with property functions, such as a dipole moment curve. Matrix elements over the ro-vib wave functions for the property in question are then computed. These results can be used to compute IR intensities and vibrational averages of different properties.

VIBROT can also be used to compute transition properties between different electronic states. The program is then run twice to produce two files of wave functions. These files are used as input in a third run, which will then compute transition matrices for input properties. The main use is to compute transition moments, oscillator strengths, and lifetimes for ro-vib levels of electronically excited states. The asymptotic energy difference between the two electronic states must be provided using the ASYMptotic keyword.

## 4.2.57.1. Dependencies¶

The VIBROT is free-standing and does not depend on any other program.

## 4.2.57.2. Files¶

### 4.2.57.2.1. Input files¶

The calculation of vibrational wave functions and spectroscopic constants uses no input files (except for the standard input). The calculation of transition properties uses VIBWVS files from two preceding VIBROT runs, redefined as VIBWVS1 and VIBWVS2.

### 4.2.57.2.2. Output files¶

VIBROT generates the file VIBWVS with vibrational wave functions for each \\(v\\) and \\(J\\) quantum number, when run in the wave function mode. If requested VIBROT can also produce files VIBPLT with the fitted potential and property functions for later plotting.

## 4.2.57.3. Input¶

This section describes the input to the VIBROT program in the Molcas program system. The program name is
    
    
    &VIBROT
    

### 4.2.57.3.1. Keywords¶

The first keyword to VIBROT is an indicator for the type of calculation that is to be performed. Two possibilities exist:

ROVIbrational spectrum
    

VIBROT will perform a vib-rot analysis and compute spectroscopic constants.

TRANsition moments
    

VIBROT will compute transition moment integrals using results from two previous calculations of the vib-rot wave functions. In this case the keyword Observable should be included, and it will be interpreted as the transition dipole moment.

Note that only one of the above keywords can be used in a single calculation. If none is given the program will only process the input section.

After this first keyword follows a set of keywords, which are used to specify the run. Most of them are optional.

The compulsory keywords are:

ATOMs
    

Gives the mass of the two atoms. Write mass number (an integer) and the chemical symbol Xx, in this order, for each of the two atoms in free format. If the mass numbers is zero for any atom, the mass of the most abundant isotope will be used. All isotope masses are stored in the program. You may introduce your own masses by giving a negative integer value to the mass number (one of them or both). The masses (in unified atomic mass units, or Da) are then read on the next (or next two) entry(ies). The isotopes of hydrogen can be given as H, D, or T.

POTEntial
    

Gives the potential as an arbitrary number of lines. Each line contains a bond distance (in au) and an energy value (in au). A plot file of the potential is generated if the keyword Plot is added after the last energy input. One more entry should then follow with three numbers specifying the start and end value for the internuclear distance and the distance between adjacent plot points. This input must only be given together with the keyword RoVibrational spectrum.

In addition you may want to specify some of the following optional input:

TITLe
    

One single title line

GRID
    

The next entries give the number of grid points used in the numerical solution of the radial Schrödinger equation. The default value is 199\. The maximum value that can be used is 4999.

RANGe
    

The next entry contains two distances Rmin and Rmax (in au) specifying the range in which the vibrational wave functions will be computed. The default values are 1.0 and 5.0 au. Note that these values most often have to be given as input since they vary considerably from one case to another. If the range specified is too small, the program will give a message informing the user that the vibrational wave function is large outside the integration range.

VIBRational
    

The next entry specifies the number of vibrational quanta for which the wave functions and energies are computed. Default value is 3.

ROTAtional
    

The next entry specifies the range of rotational quantum numbers. Default values are 0 to 5. If the orbital angular momentum quantum number (\\(m_\ell\\)) is non zero, the lower value will be adjusted to \\(m_\ell\\) if the start value given in input is smaller than \\(m_\ell\\).

ORBItal
    

The next entry specifies the value of the orbital angular momentum (0, 1, 2, etc.). Default value is zero.

SCALe
    

This keyword is used to scale the potential, such that the binding energy is 0.1 au. This leads to better precision in the numerical procedure and is strongly advised for weakly bound potentials.

NOSPectroscopic
    

Only the wave function analysis will be carried out but not the calculation of spectroscopic constants.

OBSErvable
    

This keyword indicates the start of input for radial functions of observables other than the energy, for example the dipole moment function. The next line gives a title for this observable. An arbitrary number of input lines follows. Each line contains a distance and the corresponding value for the observable. As for the potential, this input can also end with the keyword Plot, to indicate that a file of the function for later plotting is to be constructed. The next line then contains the minimum and maximum R-values and the distance between adjacent points. When this input is given with the top keyword RoVibrational spectrum the program will compute matrix elements for vibrational wave functions of the current electronic state. Transition moment integrals are instead obtained when the top keyword is Transition moments. In the latter case the calculation becomes rather meaningless if this input is not provided. The program will then only compute the overlap integrals between the vibrational wave functions of the two states. The keyword Observable can be repeated up to ten times in a single run. All observables should be given in atomic units.

TEMPerature
    

The next entry gives the temperature (in K) at which the vibrational averaging of observables will be computed. The default is 300 K.

STEP
    

The next entry gives the starting value for the energy step used in the bracketing of the eigenvalues. The default value is 0.004 au (88 \\(\text{cm}^{-1}\\)). This value must be smaller than the zero-point vibrational energy of the molecule.

ASYMptotic
    

The next entry specifies the asymptotic energy difference between two potential curves in a calculation of transition matrix elements. The default value is zero atomic units.

ALLRotational
    

By default, when the Transition moments keyword is given, only the transitions between the lowest rotational level in each vibrational state are computed. The keyword AllRotational specifies that the transitions between all the rotational levels are to be included. Note that this may result in a very large output file.

PRWF
    

Requests the vibrational wave functions to be printed in the output file.

DISTunit
    

Unit used for distances in the input potential. The default is BOHR. Other options include ANGSTROM and PICOMETER. The short form PM can also be used, instead of PICOMETER.

ENERunit
    

Unit used for energies in the input potential. The default is HARTREE. Other options include ELECTRONVOLT, KCAL/MOL, KJ/MOL, CM-1, and MEGAHERTZ. The short form EV can be used instead of ELECTRONVOLT and likewise MHZ can be used instead of MEGAHERTZ.

### 4.2.57.3.2. Input example¶
    
    
    &VIBROT
      RoVibrational spectrum
      Title = H2 (^1 Pi_u)
      Atoms = 0 H 0 H
      Potential
        0.4233417991952784    -93390.8116364055
        0.5291772489940979   -125520.5784258792
        0.5820949738935077   -135202.0740308874
        0.6350126987929174   -142230.7885620708
        0.6879304236923273   -147325.2117261678
        0.7408481485917370   -150985.4845047687
        0.7937658734911469   -153567.9481018878
        0.8466835983905567   -155331.6637865382
        0.8996013232899664   -156468.2460791877
        0.9525190481893763   -157121.6176632051
        1.0054367730887860   -157401.2568735270
        1.0583544979881960   -157391.4024626400
        1.1112722228876060   -157157.4776230008
        1.1641899477870150   -156750.6989542662
        1.2700253975858350   -155571.7997582064
        1.4816962971834740   -152450.7563927988
        1.6933671967811130   -149070.0021134733
        1.9050380963787530   -145873.2312217305
        2.1167089959763920   -143043.6172437684
        2.6458862449704900   -137805.7761879516
        3.1750634939645880   -134764.6588985511
        5.2917724899409790   -131360.0872323780
      DistUnit = angstrom
      EnerUnit = cm-1
      Grid = 450
      Range = 0.4 5.0
      Vibrations = 3
      Rotations = 1 4
      Orbital = 1
      Observable
        Dipole Moment
        0.4233417991952784           0.57938359
        0.5291772489940979           0.62852037
        0.5820949738935077           0.65216622
        0.6350126987929174           0.67506184
        0.6879304236923273           0.69709869
        0.7408481485917370           0.71821433
        0.7937658734911469           0.73833904
        0.8466835983905567           0.75741713
        0.8996013232899664           0.77538706
        0.9525190481893763           0.79219774
        1.0054367730887860           0.80778988
        1.0583544979881960           0.82211035
        1.1112722228876060           0.83510594
        1.1641899477870150           0.84672733
        1.2700253975858350           0.86565481
        1.4816962971834740           0.88532063
        1.6933671967811130           0.88056207
        1.9050380963787530           0.85474708
        2.1167089959763920           0.81515210
        2.6458862449704900           0.70549066
        3.1750634939645880           0.62103112
        5.2917724899409790           0.46501146
      Plot  = 1.0 10.0 0.1
      Scale
    

**Comments** : The vibrational-rotation spectrum for the \\(^1\Pi_u\\) state of \\(\ce{H2}\\) will be computed using the potential curve given in the input. The 3 lowest vibrational levels will be obtained and for each level for the rotational states in the range \\(J\\)=1 to 4. The mass for the most abundant isotope of \\(\ce{H}\\) will be used. The vib-rot matrix elements of the dipole function will also be computed. A plot file of the potential and the dipole function will be generated.

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
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * 4.2.57. VIBROT
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<symmetrize.html> "4.2.56. SYMMETRIZE") | [next](<wfa.html> "4.2.58. WFA") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/vibrot.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

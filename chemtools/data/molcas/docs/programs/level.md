<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/level.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.28. LEVEL

[previous](<gugadrt.html> "4.2.27. GUGADRT") | [next](<localisation.html> "4.2.29. LOCALISATION") | [index](<../../genindex.html> "General Index")

# 4.2.28. LEVEL¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




The LEVEL program is used to compute a vibration-rotation spectrum for a diatomic molecule, using as input a potential that is computed over a grid, or an analytic potential with its parameters specified. The grid should be dense around equilibrium (recommended spacing 0.05 au) and should extend to a large distance (say 50 au) if dissociation energies are computed.

The ro-vibrational Schrödinger equation is solved numerically (using Numerov’s method). The ro-vibrational energies are analyzed in terms of spectroscopic constants.

## 4.2.28.1. Dependencies¶

The LEVEL program is free-standing and does not depend on any other program.

## 4.2.28.2. Files¶

### 4.2.28.2.1. Input files¶

The calculation of vibrational wavefunctions and spectroscopic constants uses no input files (except for the standard input).

### 4.2.28.2.2. Output files¶

LEVEL generates a standard output file which ends with a summary of all levels found.

## 4.2.28.3. Input¶

This section describes the input to the LEVEL program in the Molcas program system. The program name is
    
    
    &LEVEL
    

### 4.2.28.3.1. Keywords¶

The compulsory keywords are:

IAN1
    

Integer Atomic Number of atom 1

IMN1
    

Integer Mass Number of atom 1

IAN2
    

Integer Atomic Number of atom 2

IMN2
    

Integer Mass Number of atom 2

CHARge
    

Charge of molecule

NUMPot
    

Number of potentials (1 for a single potential, 2 for two potentials and calculation of matrix elements coupling their levels).

RH
    

Step size, \\(\Delta R\\) for the numerical solution of the differential equation. Calculations should be done with smaller and smaller values of this variable (with all other variables kept the same) until convergence with respect to this variable is achieved.

RMIN
    

Minimum value of \\(R\\) for the numerical solution of the differential equation. Calculations should be done with smaller and smaller values of this variable (with all other variables kept the same) until convergence with respect to this variable is achieved.

pRV
    

The \\(p\\) value (power) for the “radial variable” used for numerically solving the differential equation

aRV
    

The real number \\(R\\) value around which the “radial variable” used for numerically solving the differential equation, is centered.

EPS
    

The real number \\(\epsilon\\) value indicating the convergence tolerance when numerically solving the differential equation.

NTP
    

The integer indicating the number of turning points when providing a pointwise potential in the input file.

LPPOt
    

The integer indicating how often to print the potential and its first two derivatives (they will all be printed to the output file at every LPPOTth point if LPPOT > 0, and only the potential will be printed in condensed format to file 8 at every |LPPOT|th point if LPPOT < 0).

IOMEg1
    

The integer angular momentum quantum number \\(\Omega\\).

VLIM
    

The real number indicating the limit of the potential \\(V(R)\\) as \\(R\rightarrow \infty\\).

IPOTl
    

The integer indicating the form of the analytic potential. Choose IPOTL = 1 for a Lennard–Jones potential, IPOTL = 3 for an EMO (extended Morse oscillator), IPOTL = 4 for the MLR (Morse/Long-range) potential.

PPAR
    

The integer power \\(p\\) used in an MLR potential.

QPAR
    

The integer power \\(q\\) used in an MLR potential.

NSR
    

The integer order of the polynomial function in an MLR potential’s exponent, for the short-range (SR) part of the potential.

NLR
    

The integer order of the polynomial function in an MLR potential’s exponent, for the long-range (LR) part of the potential.

IBOB
    

The integer flag specifying whether or not to include (IBOB > 0) or exclude (IBOB <= 0) Born–Oppenheimer Breakdown functions.

DSCM
    

The real number indicating the \\(\mathfrak{D}_e\\) value (the “depth at equilibrium” for the potential).

REQ
    

The real number indicating the \\(R_e\\) value (the equilibrium internuclear distance)

RREF
    

The real number indicating the \\(R_ref\\) value (the reference distance for the MLR model).

NCMM
    

Integer indicating the number of long-range terms used in the MLR model (e.g. if using C6, C8, C10, then NCMM = 3).

IVSR
    

Integer indicating the power used in the damping function for the MLR model.

IDSTt
    

Integer indicating the type of damping function used. Choose IDSTT = 1 for the Douketis–Scoles-type (DS) function, and IDSTT = 2 for the Tang–Toonies (TS) function.

RHOAb
    

Real number indicating the \\(\rho_{AB}\\) parameter for the damping function in an MLR model.

MMLR
    

Integer array containing NCMM elements, which indicate the inverse powers of the long-range terms in the MLR model. For example, if using C6, C8, C10, then MMLR = 6 8 10. If using C4, C6, C8 (for example, for the potential between a neutral atom and an ion) then use MMLR = 4 6 8.

CMM
    

Real number array containing NCMM elements, which indicate the coefficients of the inverse powers of the long-range terms in the MLR model. For example, if using C6, C8, C10, then CMM = C6 C8 C10. If using C4, C6, C8 (for example, for the potential between a neutral atom and an ion) then use CMM = C4 C6 C8.

PARM
    

Real number array containing NLR+1 elements, which indicate the exponent expansion coefficients for the MLR model.

NLEV1
    

Integer indicating the number of rovibrational levels to seek. If negative, the program will try to automatically find all levels from \\(v=0\\) to v=-|textrm{NLEV1}|.

AUTO1
    

Integer indicating whether or not to automatically generate trial energies for each vibrational level. If AUTO1 > 0, the trial energies are generated, wheras if AUTO1 <= 0, then the user can provide trial energies manually.

LCDC
    

Integer indicating whether or not to calculate inertial rotational constants: \\(B_v\\), and the first six centrifugal distortion constants: \\(-D_v\\), \\(H_v\\), \\(L_v\\), \\(M_v\\), \\(N_v\\), \\(O_v\\). If LCDC > 0, then these are calculated, and otherwise they are not.

LXPCt
    

Integer indicating whether or not to calculate expectation values or matrix elements using the ro-vibrational wavefunctions obtained from solving the Schrödinger equation. If LXPCT = 0, no expectation values or matrix elements are calculated, and otherwise they are.

NJM
    

Integer indicating how many rotational levels (and expectation values, if LXPCT > 0) to find for each vibrational level found.

JDJR
    

Integer indicating a step size for increasing of \\(J\\) when we ask for a calculation of rotational (J) eigenvalues from IJ(i) to NJM for each vibrational level \\(i\\). If JDJR = 1, then you are asking the program to automatically determine all possible rotational levels between IJ(i) to NJM.

LPRWf
    

Integer indicating whether or not to print the ro-vibrational wavefunction levels at every LPRWFth mesh point. If LPRWF = 0, no wavefunction is printed.

### 4.2.28.3.2. Input example¶
    
    
    &LEVEL
      IAN1 = 3
      IMN1 = 6
      IAN2 = 3
      IMN2 = 6
      CHARGE = 0
      NUMPOT = 1
      RH = 0.0005
      RMIN = 0.125
      PRV = 1
      ARV = 5.0d0
      EPS = 2.d-10
      NTP = -1
      LPPOT = 0
      IOMEG1 = 0
      VLIM = 0.0d0
      IPOTL = 4
      PPAR = 5
      QPAR = 3
      NSR = 3
      NLR = 3
      IBOB = -1
      DSCM = 3.337678701485D+02
      REQ = 4.170010583477D+00
      RREF = 8.0d0
      NCMM = 3
      IVSR = -2
      TDSTT = 1
      RHOAB = 0.54d0
      MMLR = 6 8 10
      CMM = 6.719000000d+06 1.126350000d+08  2.786940000d+09
      PARM = -5.156803528943D-01 -9.585070416286D-02 1.170797201140D-01 -2.282814434665D-02
      NLEV1 = -999
      AUTO1 = 1
      LCDC = 2
      LXPCT = 0
      NJM = 0
      JDJR = 1
      LPRWF = 0
    

**Comments** : The vibrational-rotation spectrum for the \\(1^3\Sigma_u(a)\\) state of \\(^{(6,6)}\ce{Li2}\\) will be computed using the MLR potential given in the input.

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
      * 4.2.28. LEVEL
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

[previous](<gugadrt.html> "4.2.27. GUGADRT") | [next](<localisation.html> "4.2.29. LOCALISATION") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/level.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/gugadrt.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.27. GUGADRT

[previous](<gugaci.html> "4.2.26. GUGACI") | [next](<level.html> "4.2.28. LEVEL") | [index](<../../genindex.html> "General Index")

# 4.2.27. GUGADRT¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




The GUGADRT program generates distict row table (DRT) used in the GUGACI in Direct CI calculations [[71](<../../references.html#id21> "B. Roos. Chem. Phys. Lett., 15 \(1972\) 153-159.")]. Only DRT in active space are generated because the hole-particle symmetry is used in GUGACI [[93](<../../references.html#id50> "Y. Wang, G. Zhai, B. Suo, Z. Gan, Z. Wen. Chem. Phys. Lett., 375 \(2003\) 134-140."), [94](<../../references.html#id92> "B. Suo, G. Zhai, Y. Wang, Z. Wen, X. Hu, L. Li. J. Comput. Chem., 26 \(2005\) 88-96.")]. These DRT are used to evaluated the coupling coefficients by the Graphical Unitary Group Approach [[72](<../../references.html#id59> "I. Shavitt. Int. J. Quantum Chem., 12-S11 \(1977\) 131-148."), [73](<../../references.html#id60> "I. Shavitt. Int. J. Quantum Chem., 14-S12 \(1978\) 5-32."), [74](<../../references.html#id98> "P. E. M. Siegbahn. J. Chem. Phys., 72 \(1980\) 1647-1656.")], for wavefunctions with at most two electrons excited from a set of reference configurations. The reference configurations can be specified as a list, where the occupation numbers are given for each active orbital (see below) in each reference configuration, or as a Full CI within the space defined by the active orbitals. In the GUGADRT and GUGACI the orbitals are classified as follows: Frozen, Inactive, Active, Secondary, and Deleted orbitals. Within each symmetry type, they follow this order. For the GUGADRT program, only the active orbitals are relevant.

  * **Inactive:** Inactive orbitals are doubly occupied in all reference configurations, but excitations out of this orbital space are allowed in the final CI wavefunction, i.e., they are correlated but have two electrons in all _reference_ configurations. Since only single and double excitations are allowed, there can be no more than two holes in the active orbitals.

  * **Active:** Active orbitals are those which may have different occupation in different reference configurations.




## 4.2.27.1. Dependencies¶

## 4.2.27.2. Files¶

### 4.2.27.2.1. Input files¶

TRAONE
    

Transformed one-electron integrals from MOTRA. Orbital information such as frozen, deleted orbitals will be read from this file.

### 4.2.27.2.2. Output files¶

GUGADRT
    

This file contains the DRT that is needed in subsequent CI calculations.

## 4.2.27.3. Input¶

This section describes the input to the GUGADRT program in the Molcas program system, with the program name:
    
    
    &GUGADRT
    

The first four characters of the keywords are decoded and the rest are ignored.

### 4.2.27.3.1. Keywords¶

Formally, there are no compulsory keyword. Obviously, some input must be given for a meaningful calculation.

TITLe
    

The lines following this keyword are treated as title lines, until another keyword is encountered.

SPIN
    

The spin degeneracy number, i.e. 2S+1. The value is read from the line following the keyword, in free format. The default value is 1, meaning a singlet wave function.

ELECtrons
    

The number of electrons to be correlated in the CI calculation. The value is read from the line following the keyword, in free format. Note that this number should include the nr of electrons in inactive orbitals. An alternative input specification is NACTEL. Default: Twice nr of inactive orbitals.

NACTel
    

The number of electrons in active orbitals in the reference configurations. The value is read from the line following the keyword, in free format. Note that this number includes only the of electrons in active orbitals. An alternative input specification is ELECTRONS. Default: Zero.

INACtive
    

The number of inactive orbitals, i.e. orbitals that have occupation numbers of 2 in all reference configurations. Specified for each of the symmetries. The values are read from the line following the keyword, in free format.

ACTIve
    

The number of active orbitals, i.e. orbitals that have varying occupation numbers in the reference configurations. Specified for each of the symmetries. The values are read from the line following the keyword, in free format.

At least one of the Inactive or Active keywords must be present for a meaningful calculation. If one of them is left out, the default is 0 in all symmetries.

REFErence
    

Specify selected reference configurations. The additional input that is required usually spans more than one line. The first line after the keyword contains the number of reference configurations, and the total number of active orbitals, and these two numbers are read by free format. Thereafter the input has one line per reference configuration, specifying the occupation number for each of the active orbitals, read by 80I1 format. Note that Reference and CIall are mutually exclusive.

SYMMetry
    

Specify the selected symmetry type (the irrep) of the wave function as a number between 1 and 8 (see SEWARD). Default is 1, which always denote the totally symmetric irrep.

CIALl
    

Use a Full CI within the subspace of the active orbitals as reference configurations. The symmetry of the wavefunction must be specified. Note that CIall and Reference are mutually exclusive. One of these two alternatives must be chosen for a meaningful calculation.

PRINt
    

Printlevel of the program. Default printlevel (0) produces very little output. Printlevel 5 gives some information that may be of interest. The value is read from the line following the keyword, in free format.

### 4.2.27.3.2. Input example¶
    
    
    &GUGADRT
    Title     =  CH2 molecule.
    Electrons =  8
    Spin      =  1
    Inactive  =  1    0    0    0
    Active    =  2    2    2    0
    Symmetry  =  1
    Ciall
    

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
      * 4.2.27. GUGADRT
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

[previous](<gugaci.html> "4.2.26. GUGACI") | [next](<level.html> "4.2.28. LEVEL") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/gugadrt.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

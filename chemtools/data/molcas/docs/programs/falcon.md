<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/falcon.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.17. FALCON ¤

[previous](<extf.html> "4.2.16. EXTF") | [next](<false.html> "4.2.18. FALSE") | [index](<../../genindex.html> "General Index")

# 4.2.17. FALCON ¤¶

Warning

This program is not available in OpenMolcas

  * Description

  * Input

    * Keywords

    * Input examples




## 4.2.17.1. Description¶

FALCON calculates total energy of the large system based on the fragment approach. Total energy of the whole system is calculated from total energies of fragments as follows,

\\[E^{\text{whole}}=\sum C_i^{\text{fragment}} E_i^{\text{fragment}},\\]

where \\(E_i^{\text{fragment}}\\) is the total energy of fragment \\(i\\), and \\(C_i^{\text{fragment}}\\) is its coefficient.

In addition to the total energy, FALCON can calculate orbitals of the whole system. Fock matrix and overlap matrix of the whole system are calculated from ones of fragments using following equations,

\\[\mat{F}^{\text{whole}}=\sum C_i^{\text{fragment}} \mat{F}_i^{\text{fragment}},\\]

and

\\[\mat{S}^{\text{whole}}=\sum C_i^{\text{fragment}} \mat{S}_i^{\text{fragment}},\\]

where \\(F_i\\) and \\(S_i\\) are the Fock matrix and overlap matrix, respectively, of fragment \\(i\\).

Then

\\[\mat{F}\mat{C}=\mat{S}\mat{C}\mat{\varepsilon}\\]

is solved to obtain the orbitals, \\(\mat{C}\\), and orbitals energies, \\(\mat{\varepsilon}\\).

## 4.2.17.2. Input¶

Below follows a description of the input to FALCON.

The input for each module is preceded by its name like:
    
    
    &FALCON
    

Argument(s) to a keyword, either individual or composed by several entries, can be placed in a separated line or in the same line separated by a semicolon. If in the same line, the first argument requires an equal sign after the name of the keyword.

### 4.2.17.2.1. Keywords¶

TITLe
    

One-line title.

FRAGment
    

Takes one, two or three argument(s). The first value (integer) defines the fragment number, the second value (real) determines coefficient, and the third value (integer) is the fragment number that is equivalent to this fragment when translational symmetry is used. A default for the second value is 1.0 where the first and third values have no default. Other keyword(s) specific to this fragment must follow this keyword.

OPERator
    

A real value following this keyword represents a coefficient, \\(C_i^{\text{fragment}}\\), of fragment \\(i\\) (current fragment), where \\(i\\) is a value specified by FRAGMENT keyword. This keyword is equivalent with the second value of keyword, FRAGMENT.

EQUIvalence
    

An integer, \\(j\\), following this keyword declares that current fragment is translationally equivalent with fragment \\(j\\), and information provided for fragment \\(j\\) are tranfered to current fragment. This keyword is equivalent with the third value of keyword, FRAGMENT.

TRANslate
    

Three real numbers following this keyword specifies the translational vector by which the current fragment is translated to give new coordinate. A unit of either bohr or angstrom can follow. The default unit is angstrom. This keyword takes effect only when the equivalent fragment is specified.

RUNFile
    

Following this keyword specifies the name of RunFile file for the corresponding fragment.

ONEInt
    

Following this keyword specifies the name of OneInt file for the corresponding fragment.

NFRAgment
    

An integer following this keyword specifies the number of fragments. If this keyword is not given, the largest fragment number given by FRAGMENT keyword is set to be the number of fragment.

NIRRep
    

An integer following this keyword specifies the number of irreducible representation of point group symmetry.

OCCUpation
    

A list of integer(s) following this keyword specifies the number of occupied orbitals in each symmetry representation in the unfragmented system.

DISTance
    

A real number following this keyword specifies the distance of two atoms that are equivalent to each other, followed by a unit that is eather angstrom or bohr. Default is angstrom.

NEAR
    

A real number following this keyword specifies the distance of two atoms within which atoms are considered to be too close each other. An unit that is eather angstrom or bohr can follow. Default is angstrom.

PRINt
    

An integer following this keyword specifies the format of orbital print out.

ORBEne
    

A real number follwing this keyword stands for the threshold for orbital print out. The orbitals with orbital energy below this value are print out.

ORBOcc
    

A real number follwing this keyword stands for the threshold for orbital print out. The orbitals with occupation number above this value are print out.

### 4.2.17.2.2. Input examples¶

Below shows an example of input file for the three fragment system of which energy, \\(E^{\text{whole}}\\), is written as

\\[E^{\text{whole}}= E_1^{\text{fragment}} + E_2^{\text{fragment}} - E_3^{\text{fragment}},\\]

by fragment energies, \\(E_1^{\text{fragment}}\\), \\(E_2^{\text{fragment}}\\), and \\(E_3^{\text{fragment}}\\).
    
    
    &FALCON
    Fragment=1,  1.0
    Fragment=2,  1.0
    Fragment=3, -1.0
    

which can be simplified as,
    
    
    &FALCON
    Fragment=3, -1.0
    

The next example is a two fragment system in which fragment 1 and fragment 2 are equivalent except for their positons. When their difference in position is described by a vector, (1.0, 1.0, -1.0), a translational symmetry can be used and the input becomes as follows,
    
    
    &FALCON
    Fragment=2, 1.0, 1
    Translate=1.0, 1.0, -1.0
    

If the total energy of the whole system is given by the sum of total energies of three fragment,

\\[E^{\text{whole}}= E_1^{\text{fragment}} + E_2^{\text{fragment}} + E_3^{\text{fragment}},\\]

input is simplly as follows,
    
    
    &FALCON
    nFragment=3
    

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
      * 4.2.17. FALCON ¤
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

[previous](<extf.html> "4.2.16. EXTF") | [next](<false.html> "4.2.18. FALSE") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/falcon.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

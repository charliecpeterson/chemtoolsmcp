<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/numerical_gradient.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.43. NUMERICAL_GRADIENT

[previous](<nevpt2.html> "4.2.42. NEVPT2") | [next](<poly_aniso.html> "4.2.44. POLY_ANISO") | [index](<../../genindex.html> "General Index")

# 4.2.43. NUMERICAL_GRADIENT¶

  * Dependencies

  * Files




The NUMERICAL_GRADIENT module is a program which numerically evaluates the gradient of the energy with respect to nuclear perturbations.

Note that this module is automatically invoked by the ALASKA module if the wave function method is MBPT2, CCSDT, CASPT2, MS-CASPT2, or a calculation using the Cholesky decomposition. The user should normally never request the execution of this module; instead it is advised to use the NUMErical keyword in Alaska, if it is necessary to force the use of numerical gradients rather than analytical ones.

The module is parallelized over the displacements, which in case of large jobs gives a linear speed up compared to a serial execution, although in order to obtain this it is important to choose the number of nodes such that the number of contributing perturbations is a multiple of the number of nodes. For a given molecule the number of perturbations equals the number of atoms times 6 (a perturbation with plus and minus delta for each of the three axes). Symmetry can of course reduce this number. If the request of execution originates from the SLAPAF module further reduction in perturbations is achieved due to the utilization of rotational and translational invariance.

## 4.2.43.1. Dependencies¶

The dependencies of the NUMERICAL_GRADIENT module is the union of the dependencies of the SEWARD, SCF, RASSCF, MBPT2, MOTRA, CCSDT, and CASPT2 modules.

## 4.2.43.2. Files¶

The files of the NUMERICAL_GRADIENT module is the union of the files of the SEWARD, SCF, RASSCF, MBPT2, MOTRA, CCSDT, and CASPT2 modules.

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
      * 4.2.43. NUMERICAL_GRADIENT
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

[previous](<nevpt2.html> "4.2.42. NEVPT2") | [next](<poly_aniso.html> "4.2.44. POLY_ANISO") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/numerical_gradient.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

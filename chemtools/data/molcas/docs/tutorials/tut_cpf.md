<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_cpf.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.15. CPF — A Coupled-Pair Functional Program

[previous](<tut_mrci.html> "3.3.14. MRCI — A Configuration Interaction Program") | [next](<tut_ccsdt.html> "3.3.16. CCSDT — A Set of Coupled-Cluster Programs") | [index](<../genindex.html> "General Index")

# 3.3.15. CPF — A Coupled-Pair Functional Program¶

The CPF program produces Single and Doubles Configuration Interaction (SDCI), Coupled-Pair Functional (CPF), Modified Coupled-Pair Functional (MCPF), and Averaged Coupled-Pair Functional (ACPF) wave functions (see CPF section of the user’s guide) from one reference configuration. The difference between the MRCI and CPF codes is that the former can handle Configuration Interaction (CI) and Averaged Coupled-Pair Functional (ACPF) calculations with more than one reference configuration. For a closed-shell reference the wave function can be generated with the SCF program. In open-shell cases the RASSCF has to be used.

The TITLe keyword behaviors in a similar fashion to the other Molcas modules. The CPF keyword requests an Coupled-Pair Functional calculation. This is the default and is mutually exclusive with keywords MCPF, ACPF, and SDCI which request different type of calculations. The input below lists the input files for the GUGA and CPF programs to obtain the MCPF energy for the lowest triplet state of \\(B_2\\) symmetry in the water molecule. The GUGA module computes the coupling coefficients for a triplet state of the appropriate symmetry and the CPF module will converge to the first excited triplet state. One orbital of the first symmetry has been frozen in this case (core orbital) in the MOTRA step.

## 3.3.15.1. CPF Output¶

The CPF section of the output lists the number of each type of orbital in each symmetry including pre-frozen orbitals that were frozen by the GUGA module. After some information concerning the total number of internal configurations used and storage data, it appears the single reference configuration in the MRCI format: an empty orbital is listed as “`0`” and a doubly occupied as “`3`”. The spin of a singly occupied orbital by “`1`” (spin up) or “`2`” (spin down). The molecular orbitals are listed near the end of the output.

Sample input requested by the GUGA and CPF modules to calculate the MCPF energy for the lowest \\(B_1\\) triplet state of the water in \\(C_{2v}\\) symmetry:
    
    
    &GUGA
    Title= H2O molecule. Triplet state.
    Electrons= 8; Spin= 3
    Inactive= 2 0 1 0; Active= 1 1 0 0
    CiAll= 2
    
    &CPF
    Title= MCPF of triplet state of C2v Water
    MCPF
    

There are four input files to the CPF module; CIGUGA from GUGA, TRAONE and TRAINT from MOTRA and ONEINT from SEWARD. The orbitals are saved in CPFORB.

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<tut.html>)
    * [3.1. Quickstart Guide for Molcas](<nutshell.html>)
    * [3.2. Problem Based Tutorials](<pbtutorials.html>)
    * [3.3. Program Based Tutorials](<tutorials.html>)
      * [3.3.1. Molcas Flowchart](<flowchart.html>)
      * [3.3.2. Environment and EMIL Commands](<tut_emil.html>)
      * [3.3.3. GATEWAY — Definition of geometry, basis sets, and symmetry](<tut_gateway.html>)
      * [3.3.4. SEWARD — An Integral Generation Program](<tut_seward.html>)
      * [3.3.5. SCF — A Self-Consistent Field program and Kohn–Sham DFT](<tut_scf.html>)
      * [3.3.6. MBPT2 — A Second-Order Many-Body PT RHF Program](<tut_mbpt2.html>)
      * [3.3.7. RASSCF — A Multi Configurational Self-Consistent Field Program](<tut_rasscf.html>)
      * [3.3.8. CASPT2 — A Many Body Perturbation Program](<tut_caspt2.html>)
      * [3.3.9. NEVPT2 — \\(n\\)-Electron Valence State Second-Order Perturbation Theory](<tut_nevpt2.html>)
      * [3.3.10. RASSI — A RAS State Interaction Program](<tut_rassi.html>)
      * [3.3.11. CASVB — A non-orthogonal MCSCF program](<tut_casvb.html>)
      * [3.3.12. MOTRA — An Integral Transformation Program](<tut_motra.html>)
      * [3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program](<tut_guga.html>)
      * [3.3.14. MRCI — A Configuration Interaction Program](<tut_mrci.html>)
      * 3.3.15. CPF — A Coupled-Pair Functional Program
      * [3.3.16. CCSDT — A Set of Coupled-Cluster Programs](<tut_ccsdt.html>)
      * [3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization](<tut_optimiza.html>)
      * [3.3.18. MCKINLEY — A Program for Integral Second Derivatives](<tut_mckinley.html>)
      * [3.3.19. MCLR — A Program for Linear Response Calculations](<tut_mclr.html>)
      * [3.3.20. GENANO — A Program to Generate ANO Basis Sets](<tut_genano.html>)
      * [3.3.21. FFPT — A Finite Field Perturbation Program](<tut_ffpt.html>)
      * [3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules](<tut_vibrot.html>)
      * [3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program](<tut_single_aniso.html>)
      * [3.3.24. POLY_ANISO — Semi-_ab initio_ Electronic Structure and Magnetism of Polynuclear Complexes Program](<tut_poly_aniso.html>)
      * [3.3.25. GRID_IT — A Program for Orbital Visualization](<tut_grid_it.html>)
      * [3.3.26. Writing MOLDEN input](<tut_molden.html>)
      * [3.3.27. Tools for selection of the active space](<tut_expbas.html>)
      * [3.3.28. Most frequent error messages found in Molcas](<tut_errors.html>)
      * [3.3.29. Some practical hints](<tut_hints.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<tut_mrci.html> "3.3.14. MRCI — A Configuration Interaction Program") | [next](<tut_ccsdt.html> "3.3.16. CCSDT — A Set of Coupled-Cluster Programs") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_cpf.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

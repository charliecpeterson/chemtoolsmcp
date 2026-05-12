<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_single_aniso.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program

[previous](<tut_vibrot.html> "3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules") | [next](<tut_poly_aniso.html> "3.3.24. POLY_ANISO — Semi-ab initio Electronic Structure and Magnetism of Polynuclear Complexes Program") | [index](<../genindex.html> "General Index")

# 3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program¶

The program SINGLE_ANISO calculates nonperturbatively the temperature- and field-dependent magnetic properties (Van Vleck susceptibility tensor and function, molar magnetization vector and function) and the pseudospin Hamiltonians for Zeeman interaction (the \\(g\\) tensor and higher rank tensorial components) and the zero-field splitting (the \\(D\\) tensor and higher rank tensorial components) for arbitrary mononuclear complexes and fragments on the basis of ab initio spin-orbit calculations. SINGLE_ANISO requires as input file the RUNFILE containing all necessary _ab initio_ information: spin–orbit eigenstates, angular momentum matrix elements, the states been mixed by the spin–orbit coupling in RASSI, etc. Usually, the SINGLE_ANISO runs after RASSI.

For a proper spin–orbit calculation the relativistic basis sets should be used for the whole calcualtion. For SEWARD, the atomic mean-field (AMFI), Douglas–Kroll (DOUG) must be employed. To ensure the computation of angular momentum integrals the ANGMOM should be also used, specifying the origin of angular momentum integrals as the coordinates of the magnetic center of the molecule, i.e. the coordinates of the atom where the unpaired electrons mainly reside. For program RASSI the necessary keywords are: SPIN, since we need a spin–orbit coupling calculation, and MEES, to ensure the computation of angular momentum matrix elements in the basis of spin-free states (SFS).

In the cases where spin–orbit coupling has a minor effect on the low-lying energy spectrum (most of the isotropic cases: \\(\ce{Cr^{3+}}\\), \\(\ce{Gd^{3+}}\\), etc.) the pseudospin is usually the same as the ground spin. For these cases the SINGLE_ANISO may run without specifying any keywords in the input file.
    
    
    &SINGLE_ANISO
    

In the cases when spin–orbit coupling play an important role in the low-lying energy spectrum, i.e. in the cases of e.g. octahedral \\(\ce{Co^{2+}}\\), most of the lanthanide complexes, the pseudospin differs strongly from the spin of the ground state. In these cases, the dimension of the pseudospin can be found by analysing the spin–orbit energy spectrum obtained at RASSI. The pseudospin is best defined as a group of spin–orbit states close in energy. Once specified, these eigenstates are further used by the SINGLE_ANISO to build proper pseudospin eigenfunctions. As an example of an input for SINGLE_ANISO requiring the computation of all magnetic properties (which is the default) and the computation of the \\(g\\) tensor for the ground Kramers doublet (i.e. pseudospin of a Kramers doublet is \\(\tilde{S}=1/2\\)).
    
    
    &SINGLE_ANISO
    MLTP
    1
    2
    

SINGLE_ANISO has implemented pseudospins: \\(\tilde{S}=1/2\\), \\(\tilde{S}=1\\), …, up to \\(\tilde{S}=7/2\\). The user can also ask for more pseudospins at the same time:
    
    
    &SINGLE_ANISO
    MLTP
    3
    2 4 2
    

For the above input example, the SINGLE_ANISO will compute the \\(g\\) tensor for the ground Kramers doublet (spin–orbit states 1 and 2), the \\(g\\) tensor, ZFS tensor and coefficients of higher rank ITO for the pseudospin \\(\tilde{S}=3/2\\) (spin orbit functions 3–6), and the \\(g\\) tensor for the third excited Kramers doublet (spin orbit functions 7 and 8).

## 3.3.23.1. SINGLE_ANISO Output¶

The SINGLE_ANISO section of the Molcas output is divided in four parts. In the first part, the \\(g\\) tensor and higher rank Zeeman tensors are computed. They are followed by \\(D\\) tensor and higher rank ZFS tensors. The program also computes the angular moments in the direction of the main magnetic axes.

In the second part, the paramaters of the crystal field acting on the ground atomic multiplet of lanthanides are calculated.

In the third part, the powder magnetic susceptibility is printed, followed by the magnetic susceptibility tensors with and without intermolecular interaction included.

In the fourth part, magnetization vectors (if required) are printed, and then the powder molar magnetization calculated for the TMAG temperature.

The keywords TINT and HINT control the temperature and field intervals for computation of magnetic susceptibility and molar magnetization respectively. Computation of the magnetic properties at the experimental temperature and field points with the estimation of the standard deviation from experiment is also possible via TEXP, defining the experimental temperature and measured magnetic susceptibility and HEXP, defining the experimental field and averaged molar magnetization.
    
    
    &SINGLE_ANISO
    TITLE
    g tensor and magnetic susceptibility
    TYPE
    4
    MLTP
    2
    3 3
    TINT
    0.0 100 101 0.001
    

The above input requires computation of the parameters of two pseudospins \\(\tilde{S}=1\\): the ground (spin–orbit functions 1–3) and first excited (spin–orbit functions 4–6) and the magnetic susceptibility in 101 steps equally distributed in the temperature domain 0.0–100.0 K.

## 3.3.23.2. SINGLE_ANISO — Basic and Most Common Keywords¶

MLTP
    

Specifies the number and dimension of the pseudospins Hamiltonians

TMAG
    

Sets the temperature for the computation of molar magnetization

MVEC
    

Number and radial coordinates of directions for which the magnetization vector will be computed

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
      * [3.3.15. CPF — A Coupled-Pair Functional Program](<tut_cpf.html>)
      * [3.3.16. CCSDT — A Set of Coupled-Cluster Programs](<tut_ccsdt.html>)
      * [3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization](<tut_optimiza.html>)
      * [3.3.18. MCKINLEY — A Program for Integral Second Derivatives](<tut_mckinley.html>)
      * [3.3.19. MCLR — A Program for Linear Response Calculations](<tut_mclr.html>)
      * [3.3.20. GENANO — A Program to Generate ANO Basis Sets](<tut_genano.html>)
      * [3.3.21. FFPT — A Finite Field Perturbation Program](<tut_ffpt.html>)
      * [3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules](<tut_vibrot.html>)
      * 3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program
      * [3.3.24. POLY_ANISO — Semi-_ab initio_ Electronic Structure and Magnetism of Polynuclear Complexes Program](<tut_poly_aniso.html>)
      * [3.3.25. GRID_IT — A Program for Orbital Visualization](<tut_grid_it.html>)
      * [3.3.26. Writing MOLDEN input](<tut_molden.html>)
      * [3.3.27. Tools for selection of the active space](<tut_expbas.html>)
      * [3.3.28. Most frequent error messages found in Molcas](<tut_errors.html>)
      * [3.3.29. Some practical hints](<tut_hints.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<tut_vibrot.html> "3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules") | [next](<tut_poly_aniso.html> "3.3.24. POLY_ANISO — Semi-ab initio Electronic Structure and Magnetism of Polynuclear Complexes Program") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_single_aniso.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

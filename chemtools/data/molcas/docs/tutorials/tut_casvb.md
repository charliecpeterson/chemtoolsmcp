<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_casvb.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.11. CASVB — A non-orthogonal MCSCF program

[previous](<tut_rassi.html> "3.3.10. RASSI — A RAS State Interaction Program") | [next](<tut_motra.html> "3.3.12. MOTRA — An Integral Transformation Program") | [index](<../genindex.html> "General Index")

# 3.3.11. CASVB — A non-orthogonal MCSCF program¶

CASVB is a program for carrying out quite general types of non-orthogonal MCSCF calculations, offering, for example, all the advantages associated with working within a valence bond formalism.

**Warning:** as for any general MCSCF program, one may experience convergence problems, (e.g., due to redundant parameters), and the non-orthogonal optimization of orbitals can furthermore give linear dependency problems. Several options in CASVB can help overcoming these difficulties.

This program can be used in two basic modes:

  1. fully variational optimization

  2. representation of CASSCF wavefunctions using overlap- (_relatively inexpensive_) or energy-based criteria.




CASVB executes the following logical steps: Setup of wavefunction information, starting guess generation, one, or several, optimization steps, various types of analysis of the converged solution.

## 3.3.11.1. CASVB Input¶

CASVB attempts to define defaults for as many input quantities as possible, so that in the simplest case no input to the CASVB module is required. Sample input for a CASVB calculation on the lowest singlet state of \\(\ce{CH_2}\\):
    
    
    &GATEWAY
    coord
    3
    ch2 molecule
    C 0.000000  0.000000 0.000000
    H 0.000000  0.892226 0.708554
    H 0.000000 -0.892226 0.708554
    
    group= x y; basis= sto-3g
    &SEWARD
    &SCF
    &RASSCF
    nactel= 6 0 0; inactive= 1 0 0 0; ras2= 3 1 2 0
    lumorb
    &CASVB
    

## 3.3.11.2. CASVB Output¶

The amount of output in CASVB depends heavily on the setting of the PRINT levels. In case of problems with convergence behaviour it is recommended to increase these from their rather terse default values.

In the following the main features of the output are outlined, exemplified by the job in the input above. Initially, all relevant information from the previous RASSCF calculation is recovered from the JOBIPH interface file, after which the valence bond wavefunction information is summarized, as shown below. Since spatial configurations have not been specified explicitly in this example, a single covalent configuration is chosen as default. This gives 5 spin-adapted VB structures.
    
    
    Number of active electrons :   6
              active orbitals  :   6
              Total spin       : 0.0
              State symmetry   :   1
    
    Spatial VB configurations
    -------------------------
        Conf. =>   Orbitals
          1   =>    1  2  3  4  5  6
    
    Number of VB configurations :     1
              VB structures     :     5
              VB determinants   :    20
    

The output from the following optimization steps summarizes only the most relevant quantities and convergence information at the default print level. For the last optimization step, for example, The output below thus states that the VB wavefunction was found by maximizing the overlap with a previously optimized CASSCF wavefunction (output by the RASSCF program), and that the spin adaptation was done using the Yamanuchi–Kotani scheme. Convergence was reached in 7 iterations.
    
    
    -- Starting optimization - step  3 --------
    
    Overlap-based optimization (Svb).
    
    Optimization algorithm:            dFletch
    Maximum number of iterations:           50
    Spin basis:                         Kotani
    
    -------------------------------------------
    Optimization entering local region.
    Converged ... maximum update to coefficient:  0.59051924E-06
    Final Svb :    0.9978782695
    Number of iterations used:   7
    

Finally in the output below the converged solution is printed; orbital coefficients (in terms of the active CASSCF MOs) and structure coefficients. The overlap between orbitals are generally of interest, and, as also the structures are non-orthogonal, the structure weights in the total wavefunction. The total VB wavefunction is not symmetry-adapted explicitly (although one may ensure the correct symmetry by imposing constraints on orbitals and structure coefficients), so its components in the various irreducible representations can serve to check that it is physically plausible (a well behaved solution generally has just one non-vanishing component).

Next follows the one-electron density with natural-orbital analysis, again with quantities printed in the basis of the active CASSCF MOs.
    
    
    Orbital coefficients :
    ----------------------
              1           2           3           4           5           6
      1  0.43397359 -0.43397359 -0.79451779 -0.68987187 -0.79451780 -0.68987186
      2 -0.80889967  0.80889967 -0.05986171 -0.05516284 -0.05986171 -0.05516284
      3  0.00005587 -0.00005587  0.20401015 -0.20582094  0.20401016 -0.20582095
      4  0.39667145  0.39667145  0.00000000  0.00000000  0.00000000  0.00000000
      5 -0.00000001 -0.00000001 -0.53361427 -0.65931951  0.53361425  0.65931952
      6  0.00000000  0.00000000  0.19696124 -0.20968879 -0.19696124  0.20968879
    
    Overlap between orbitals :
    --------------------------
              1           2           3           4           5           6
      1  1.00000000 -0.68530352 -0.29636622 -0.25477647 -0.29636623 -0.25477647
      2 -0.68530352  1.00000000  0.29636622  0.25477647  0.29636623  0.25477646
      3 -0.29636622  0.29636622  1.00000000  0.81994979  0.35292419  0.19890631
      4 -0.25477647  0.25477647  0.81994979  1.00000000  0.19890634  0.04265679
      5 -0.29636623  0.29636623  0.35292419  0.19890634  1.00000000  0.81994978
      6 -0.25477647  0.25477646  0.19890631  0.04265679  0.81994978  1.00000000
    
    Structure coefficients :
    ------------------------
         0.00000000  0.00000001  0.09455957  0.00000000 -0.99551921
    
    Saving VB wavefunction to file VBWFN.
    
    Saving VB CI vector to file JOBIPH.
    
    Svb :          0.9978782695
    Evb :        -38.4265149062
    
    Chirgwin-Coulson weights of structures :
    ----------------------------------------
    VB spin+space (norm   1.00000000) :
         0.00000000  0.00000000 -0.00211737  0.00000000  1.00211737
    VB spin only  (norm   0.38213666) :
         0.00000000  0.00000000  0.00894151  0.00000000  0.99105849
    
    Symmetry contributions to total VB wavefunction :
    -------------------------------------------------
    Irreps 1 to 4 :  0.10000000E+01  0.15118834E-17  0.17653074E-17  0.49309519E-17
    
    Energies for components > 1d-10 :
    ---------------------------------
    Irreps 1 to 4 : -0.38426515E+02  0.00000000E+00  0.00000000E+00  0.00000000E+00
    
    One-electron density :
    ----------------------
              1           2           3           4           5           6
      1  1.98488829 -0.00021330  0.00011757  0.00000000  0.00000000  0.00000000
      2 -0.00021330  1.90209222 -0.00006927  0.00000000  0.00000000  0.00000000
      3  0.00011757 -0.00006927  0.02068155  0.00000000  0.00000000  0.00000000
      4  0.00000000  0.00000000  0.00000000  0.09447774  0.00000000  0.00000000
      5  0.00000000  0.00000000  0.00000000  0.00000000  1.97572540 -0.00030574
      6  0.00000000  0.00000000  0.00000000  0.00000000 -0.00030574  0.02213479
    
    Natural orbitals :
    ------------------
              1           2           3           4           5           6
      1 -0.99999668  0.00000000  0.00257629  0.00000000  0.00000000  0.00005985
      2  0.00257628  0.00000000  0.99999668  0.00000000  0.00000000 -0.00003681
      3 -0.00005995  0.00000000 -0.00003666  0.00000000 -0.00000001 -1.00000000
      4  0.00000000  0.00000000  0.00000000  1.00000000  0.00000001  0.00000000
      5  0.00000000  0.99999999  0.00000000  0.00000000  0.00015650  0.00000000
      6  0.00000000 -0.00015650  0.00000000 -0.00000001  0.99999999 -0.00000001
    
    Occupation numbers :
    --------------------
              1           2           3           4           5           6
      1  1.98488885  1.97572545  1.90209167  0.09447774  0.02213475  0.02068154
    

## 3.3.11.3. Viewing and plotting VB orbitals¶

In many cases it can be helpful to view the shape of the converged valence bond orbitals. Molcas therefore provides two facilities for doing this. For the Molden program, an interface file is generated at the end of each CASVB run (see also [Section 4.3.1](<../users.guide/tools.html#ug-sec-molden>)). Alternatively a CASVB run may be followed by RASSCF ([Section 4.2.47](<../users.guide/programs/rasscf.html#ug-sec-rasscf>)) and GRID_IT ([Section 4.2.23](<../users.guide/programs/grid_it.html#ug-sec-gridit>)) with the VB specification, in order to generate necessary files for viewing with LUSCUS.

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
      * 3.3.11. CASVB — A non-orthogonal MCSCF program
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

[previous](<tut_rassi.html> "3.3.10. RASSI — A RAS State Interaction Program") | [next](<tut_motra.html> "3.3.12. MOTRA — An Integral Transformation Program") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_casvb.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

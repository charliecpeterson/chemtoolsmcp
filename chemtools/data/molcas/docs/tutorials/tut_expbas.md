<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_expbas.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.27. Tools for selection of the active space

[previous](<tut_molden.html> "3.3.26. Writing MOLDEN input") | [next](<tut_errors.html> "3.3.28. Most frequent error messages found in Molcas") | [index](<../genindex.html> "General Index")

# 3.3.27. Tools for selection of the active space¶

Selecting an active space is sometimes easy. For a small molecule, an active space for the ground and the lowest valence excited states is usually the valence orbitals, i.e. orbitals composed of atomic orbitals belonging to the usual “valence shells” (there are some exceptions to this rule). Problems arise for medium or large molecules, for higher excited states, and for molecules including transition, lanthanide or actinide elements. A good wish list of orbitals will give a CASSCF/CASPT2 calculation that demand unrealistically large computer resources and time. Compromises must be made. Any smaller selection of active orbitals can in general affect your results, and the selection should be based on the specific calculations: see [Section 3.3.29](<tut_hints.html#tut-sec-hints>) for advise.

The following three tools may be help in the process:

LOCALISATION
    

is a program that can take a (subrange of) orbitals from an orbital file, and produce a new orbital file where these orbitals have been transformed to become localized, while spanning the same space as the original ones.

EXPBAS
    

can take an orbital file using a smaller basis set, and “expand” it into a new orbital file using a larger basis.

LUSCUS
    

(is of course also described elsewhere) is the orbital viewer.

It is of course best to have a good perception of the electronic structure of the molecule, including all states of interest for the calculation. If it is a larger system, where lots of ligands can be assumed not to partake in non-dynamic correlation, it is a good idea to run some simple exploratory calculations with a much smaller model system. Check the literature for calculations on similar systems or model systems.

First of all, you need to know how many orbitals (in each symmetry) that should be active. Their precise identity is also good to know, in order to have a good set of starting orbitals, but we come to that later. **Necessary** active orbitals are: Any shells that may be open in any of the states or structures studied. Breaking a bond generally produces a correlated bond orbital and a correlating antibonding orbital, that must both be active (Since it is the **number** of orbitals we are dealing with as yet, you may as well think of the two radical orbitals that are produced by completely breaking the bond). You probably want to include one orbital for each aromatic carbon. **Valuable correlated** active orbitals are: Oxygen lone pair, \\(\ce{CC}\\) \\(\pi\\) bonds. **Valuable correlating** active orbitals are: the antibonding \\(\pi^*\\) \\(\ce{CC}\\) orbitals, and one additional set of correlating d orbitals for most transition elements (sometimes called the “double d-shell effect”).

The valuable correlated orbitals can be used as Ras-1 orbitals, and correlating ones can be used as Ras-3 orbitals, if the active space becomes too large for a casscf calculation.

Assuming we can decide on the number of active orbitals, the next task is to prepare starting orbitals that enables CASSCF to converge, by energy optimization, to the actual starting orbitals for your calculation. Use a very small basis set to begin with: This will usually be one of the minimal bases, e.g. ANO-S-MB. This is not just to save time: the small basis and the large energy spacings make it much easier to get well-defined correlating orbitals.

Performing the actual casscf (or rasscf) calculation may give you the active space you want: Viewing the orbitals by LUSCUS may confirm this, but very often the orbitals are too mixed up (compared to one’s mental picture of what constitutes the best orbitals). Using localisation program solves this problem. In order to localise without mixing up orbitals from different subspaces may require to produce the new orbital file through several runs of the program; however, for the present perpose, it may be best not to have so very strict restrictions, for example: Allow mixing among a few high inactive and the most occupied orbitals; and also among the weakly occupied and some virtual orbitals.

Running the localisation program, and viewing the localised orbitals, is a great help since directly in LUSCUS one can redefine orbitals as being inactive, or ras3 , or whatever, to produce a new orbital file. The resulting annotated localised orbitals can be used in a new run.

Once a plausible active space has been found, use the expbas tool to obtain starting orbitals using, e.g. ANO-VDZP basis, or whatever is to be used in the bulk of the production run.

It is also a good idea to, at this point, “waste” a few resources on a single-point calculation for a few more states than you are really interested in, and maybe look at properties, etc. There may be experimental spectra to compare with.

And please have a look at [Section 3.3.29](<tut_hints.html#tut-sec-hints>).

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
      * [3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program](<tut_single_aniso.html>)
      * [3.3.24. POLY_ANISO — Semi-_ab initio_ Electronic Structure and Magnetism of Polynuclear Complexes Program](<tut_poly_aniso.html>)
      * [3.3.25. GRID_IT — A Program for Orbital Visualization](<tut_grid_it.html>)
      * [3.3.26. Writing MOLDEN input](<tut_molden.html>)
      * 3.3.27. Tools for selection of the active space
      * [3.3.28. Most frequent error messages found in Molcas](<tut_errors.html>)
      * [3.3.29. Some practical hints](<tut_hints.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<tut_molden.html> "3.3.26. Writing MOLDEN input") | [next](<tut_errors.html> "3.3.28. Most frequent error messages found in Molcas") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_expbas.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

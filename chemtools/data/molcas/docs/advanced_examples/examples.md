<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/advanced.examples/examples.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

5.1. Examples

[previous](<ae.html> "5. Advanced Examples and Annexes") | [next](<ex-x2.html> "5.1.1. Computing high symmetry molecules") | [index](<../genindex.html> "General Index")

# 5.1. Examples¶

  * [5.1.1. Computing high symmetry molecules](<ex-x2.html>)
    * [5.1.1.1. A diatomic heteronuclear molecule: \\(\ce{NiH}\\)](<ex-x2.html#a-diatomic-heteronuclear-molecule-ce-nih>)
    * [5.1.1.2. A diatomic homonuclear molecule: \\(\ce{C2}\\)](<ex-x2.html#a-diatomic-homonuclear-molecule-ce-c2>)
    * [5.1.1.3. A transition metal dimer: \\(\ce{Ni2}\\)](<ex-x2.html#a-transition-metal-dimer-ce-ni2>)
    * [5.1.1.4. High symmetry systems in Molcas](<ex-x2.html#high-symmetry-systems-in-molcas>)
  * [5.1.2. Geometry optimizations and Hessians](<ex-op.html>)
    * [5.1.2.1. Ground state optimizations and vibrational analysis](<ex-op.html#ground-state-optimizations-and-vibrational-analysis>)
    * [5.1.2.2. Excited state optimizations](<ex-op.html#excited-state-optimizations>)
    * [5.1.2.3. Restrictions in symmetry or geometry](<ex-op.html#restrictions-in-symmetry-or-geometry>)
      * [5.1.2.3.1. Optimizing with geometrical constraints](<ex-op.html#optimizing-with-geometrical-constraints>)
      * [5.1.2.3.2. Optimizing with symmetry restrictions](<ex-op.html#optimizing-with-symmetry-restrictions>)
    * [5.1.2.4. Optimizing with Z-Matrix](<ex-op.html#optimizing-with-z-matrix>)
    * [5.1.2.5. CASPT2 optimizations](<ex-op.html#caspt2-optimizations>)
  * [5.1.3. Computing a reaction path](<ex-rp.html>)
    * [5.1.3.1. Studying a reaction](<ex-rp.html#studying-a-reaction>)
      * [5.1.3.1.1. Reactant and product](<ex-rp.html#reactant-and-product>)
      * [5.1.3.1.2. Transition state optimization](<ex-rp.html#transition-state-optimization>)
    * [5.1.3.2. Finding the reaction path – an IRC study](<ex-rp.html#finding-the-reaction-path-an-irc-study>)
  * [5.1.4. High quality wave functions at optimized structures](<ex-hi.html>)
  * [5.1.5. Excited states](<ex-ex.html>)
    * [5.1.5.1. The vertical spectrum of thiophene](<ex-ex.html#the-vertical-spectrum-of-thiophene>)
      * [5.1.5.1.1. Planning the calculations](<ex-ex.html#planning-the-calculations>)
      * [5.1.5.1.2. Generating Rydberg basis functions](<ex-ex.html#generating-rydberg-basis-functions>)
      * [5.1.5.1.3. SEWARD and CASSCF calculations](<ex-ex.html#seward-and-casscf-calculations>)
      * [5.1.5.1.4. CASPT2 calculations](<ex-ex.html#caspt2-calculations>)
      * [5.1.5.1.5. Transition dipole moment calculations](<ex-ex.html#transition-dipole-moment-calculations>)
    * [5.1.5.2. Influence of the Rydberg orbitals and states. One example: guanine](<ex-ex.html#influence-of-the-rydberg-orbitals-and-states-one-example-guanine>)
    * [5.1.5.3. Other cases](<ex-ex.html#other-cases>)
  * [5.1.6. Solvent models](<ex-rc.html>)
    * [5.1.6.1. Kirkwood model](<ex-rc.html#kirkwood-model>)
    * [5.1.6.2. PCM](<ex-rc.html#pcm>)
    * [5.1.6.3. Calculation of solvent effects: Kirkwood model](<ex-rc.html#calculation-of-solvent-effects-kirkwood-model>)
    * [5.1.6.4. Solvation effects in ground states. PCM model in formaldehyde](<ex-rc.html#solvation-effects-in-ground-states-pcm-model-in-formaldehyde>)
    * [5.1.6.5. Solvation effects in excited states. PCM model and acrolein](<ex-rc.html#solvation-effects-in-excited-states-pcm-model-and-acrolein>)
  * [5.1.7. Computing relativistic effects in molecules](<ex-so.html>)
    * [5.1.7.1. Scalar relativistic effects](<ex-so.html#scalar-relativistic-effects>)
    * [5.1.7.2. Spin–orbit coupling (SOC)](<ex-so.html#spin-orbit-coupling-soc>)
    * [5.1.7.3. The \\(\ce{PbO}\\) molecule](<ex-so.html#the-ce-pbo-molecule>)



### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<ae.html>)
    * 5.1. Examples
      * [5.1.1. Computing high symmetry molecules](<ex-x2.html>)
      * [5.1.2. Geometry optimizations and Hessians](<ex-op.html>)
      * [5.1.3. Computing a reaction path](<ex-rp.html>)
      * [5.1.4. High quality wave functions at optimized structures](<ex-hi.html>)
      * [5.1.5. Excited states](<ex-ex.html>)
      * [5.1.6. Solvent models](<ex-rc.html>)
      * [5.1.7. Computing relativistic effects in molecules](<ex-so.html>)
    * [5.2. Annexes](<annexes.html>)



### Search

[previous](<ae.html> "5. Advanced Examples and Annexes") | [next](<ex-x2.html> "5.1.1. Computing high symmetry molecules") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/advanced.examples/examples.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

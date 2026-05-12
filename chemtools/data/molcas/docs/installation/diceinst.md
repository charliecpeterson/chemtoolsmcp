<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/installation.guide/diceinst.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

2.4. Installation of Dice–Molcas interface for HCI calculations

[previous](<dmrginst.html> "2.3. Installation of CheMPS2–Molcas interface for DMRG calculations") | [next](<stochcas.html> "2.5. Installation of Molcas for Stochastic-CASSCF calculations") | [index](<../genindex.html> "General Index")

# 2.4. Installation of Dice–Molcas interface for HCI calculations¶

The Dice–Molcas interface allows one to use heat-bath configuration interaction (HCI) implemented in Dice as an FCI solver in CASSCF calculations, referred to as HCI-CASSCF [[16](<../references.html#id359> "S. Sharma, A. A. Holmes, G. Jeanmairet, A. Alavi, C. J. Umrigar. J. Chem. Theory Comput., 13\[4\] \(2017\) 1595-1604."), [17](<../references.html#id360> "A. A. Holmes, N. M. Tubman, C. J. Umrigar. J. Chem. Theory Comput., 12\[8\] \(2016\) 3674-3680.")]. A large active space, up to around 100 active orbitals, can be calculated with HCI-CASSCF. Currently, the interface supports ground state HCI-CASSCF calculations.

The interface requires the Dice 1.0 binary (<https://github.com/sanshar/Dice>). For installation of Dice, consult <https://sanshar.github.io/Dice/installation.html>. The interface supports both parallel Dice and Molcas.

The Dice–Molcas interface is built by activating in CMake:
    
    
    -D DICE=ON
    

Before runing HCI-CASSCF calculations with the Dice–Molcas interface, make sure to increase stack size; and export the Dice binary and all the required libraries for Dice.
    
    
    ulimit -s unlimited
    export PATH=/path/to/dice/binary:$PATH
    

To run parallel Dice, export the environment variable MOLCAS_DICE, for example when running on 16 nodes use:
    
    
    export MOLCAS_DICE=16
    

Verify the installation:
    
    
    molcas verify .all -w dice
    

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<ig.html>)
    * [2.1. Installation](<install.html>)
    * [2.2. Parallel Installation](<parainst.html>)
    * [2.3. Installation of CheMPS2–Molcas interface for DMRG calculations](<dmrginst.html>)
    * 2.4. Installation of Dice–Molcas interface for HCI calculations
    * [2.5. Installation of Molcas for Stochastic-CASSCF calculations](<stochcas.html>)
    * [2.6. Maintaining the package](<maintain.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<dmrginst.html> "2.3. Installation of CheMPS2–Molcas interface for DMRG calculations") | [next](<stochcas.html> "2.5. Installation of Molcas for Stochastic-CASSCF calculations") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/installation.guide/diceinst.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

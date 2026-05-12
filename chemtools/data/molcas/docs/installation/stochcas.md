<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/installation.guide/stochcas.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

2.5. Installation of Molcas for Stochastic-CASSCF calculations

[previous](<diceinst.html> "2.4. Installation of Dice–Molcas interface for HCI calculations") | [next](<maintain.html> "2.6. Maintaining the package") | [index](<../genindex.html> "General Index")

# 2.5. Installation of Molcas for Stochastic-CASSCF calculations¶

  * Uncoupled Form

  * Embedded form




The Stochastic-CASSCF method is based on the interface of the RASSCF program of Molcas, responsible for the orbital rotations via Super-CI, and the NECI program, responsible for the FCIQMC dynamics, replacing the deterministic Direct-CI based algorithm for large active space selection. In principle, two installation protocols can be adopted that are referred to as embedded and uncoupled forms. In the embedded form, the NECI program is treated as a dependent subroutine of the RASSCF program. This form effectively leads to an automatized version of the Stochastic-CASSCF within the OpenMolcas software. In the uncoupled form of Stochastic-CASSCF, NECI is installed as a stand-alone program and the Molcas-NECI interface is controlled manually by the user. In this guide the uncoupled form will be discussed. It is the form preferred by the developers of the method due to the non-black-box nature of the approach.

## 2.5.1. Uncoupled Form¶

The necessary routines in Molcas are installed automatically, but for improved communication between Molcas and NECI it is recommended to compile with HDF5.

The NECI code is available at <https://github.com/fkfest/NECI_STABLE>.

The NECI code requires some external software and libraries:

  * MPI: For builds intended to be run in parallel. OpenMPI, MPICH2 and its derivatives (IBM MPI, Cray MPI, and Intel MPI) have been tested.

  * Linear algebra: ACML, MKL, BLAS/LAPACK combination.

  * HDF5: To make use of the structured HDF5 format for reading/writing POPSFILES
    

(files storing the population of walkers, and other information, to restart calculations). This library should be built with MPI and fortran support.




For configuring and compiling NECI cmake is recommended:
    
    
    cmake -DENABLE_BUILD_HDF5=ON -DENABLE_HDF5=ON $path_to_neci/
    make -j hdf5
    make -j neci dneci
    

Cmake flag `-DENABLE_BUILD_HDF5=ON` builds the HDF5 library from source, and use that instead of one provided by the system. Cmake flag `-DENABLE_HDF5=ON` makes use of HDF5 for popsfiles (default=on).

Two executable files will be generated: neci and dneci in /bin. The latter is compulsory for sampling one- and two-body density matrices necessary for performing the orbital optimization. More details about configuration/installation of the NECI code can be found in the NECI documentation: https://www2.fkf.mpg.de/alavi/neci/stable/

There are currently no default verification tests for the Stochastic-CASSCF method. However, after installation of Molcas one test is possible to verify that MO integrals are correctly dumped into the FCIDUMP file. Simply use:
    
    
    molcas verify limannig
    

## 2.5.2. Embedded form¶

For the embedded form the NECI source code has to be downloaded into the Molcas source directory. Just execute in the Molcas repository:
    
    
    git submodule update --init External/NECI
    

Then compile Molcas with the `-DNECI=ON` cmake flag.

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<ig.html>)
    * [2.1. Installation](<install.html>)
    * [2.2. Parallel Installation](<parainst.html>)
    * [2.3. Installation of CheMPS2–Molcas interface for DMRG calculations](<dmrginst.html>)
    * [2.4. Installation of Dice–Molcas interface for HCI calculations](<diceinst.html>)
    * 2.5. Installation of Molcas for Stochastic-CASSCF calculations
      * 2.5.1. Uncoupled Form
      * 2.5.2. Embedded form
    * [2.6. Maintaining the package](<maintain.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<diceinst.html> "2.4. Installation of Dice–Molcas interface for HCI calculations") | [next](<maintain.html> "2.6. Maintaining the package") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/installation.guide/stochcas.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

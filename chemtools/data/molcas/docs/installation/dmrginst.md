<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/installation.guide/dmrginst.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

2.3. Installation of CheMPS2–Molcas interface for DMRG calculations

[previous](<parainst.html> "2.2. Parallel Installation") | [next](<diceinst.html> "2.4. Installation of Dice–Molcas interface for HCI calculations") | [index](<../genindex.html> "General Index")

# 2.3. Installation of CheMPS2–Molcas interface for DMRG calculations¶

The CheMPS2–Molcas interface requires the following components: The CheMPS2–Molcas interface [[13](<../references.html#id331> "Q. M. Phung, S. Wouters, K. Pierloot. J. Chem. Theory Comput., 12\[9\] \(2016\) 4352-4361."), [14](<../references.html#id330> "S. Wouters, V. Van Speybroeck, D. Van Neck. J. Chem. Phys., 145\[5\] \(2016\) 054120.")], based on the Block–Molcas interface [[15](<../references.html#id332> "N. Nakatani, S. Guo. J. Chem. Phys., 146\[9\] \(2017\) 094102.")], can support DMRG-SS-CASPT2 and DMRG-SA-CASPT2 calculations.

It requires the CheMPS2 binary. For installation of CheMPS2, consult <http://sebwouters.github.io/CheMPS2/index.html> if it is not already available in your OS.

Note that only the version with the Open Multi-Processing (OpenMP) is supported, thus build CheMPS2 with:
    
    
    -D WITH_MPI=OFF
    

In order to efficiently run the CheMPS2–Molcas interface, it is advisible to compile either serial or parallel Molcas with MPI. An example:
    
    
    ./configure -compiler intel -parallel -64 -mpiroot /path/to/mpi/root \
                -mpirun /path/to/mpi/bin/mpirun -blas MKL -blas_lib -mkl=sequential \
                -hdf5_inc /path/to/hdf5/include \
                -hdf5_lib /path/to/hdf5/lib \
                -chemps2 /path/to/chemps2/binary
    

The CheMPS2–Molcas interface can also be activated with CMake:
    
    
    -D CHEMPS2=ON -D CHEMPS2_DIR=/path/to/chemps2/binary
    

Before testing the CheMPS2–Molcas interface, make sure to increase stack size, export OMP_NUM_THREADS, the CheMPS2 binary, and all the required libraries for CheMPS2.
    
    
    ulimit -s unlimited
    [export OMP_NUM_THREADS=...]
    export PATH=/path/to/chemps2/binary:$PATH
    

Verify the installation:
    
    
    molcas verify extra:850,851
    molcas verify benchmark:970
    

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<ig.html>)
    * [2.1. Installation](<install.html>)
    * [2.2. Parallel Installation](<parainst.html>)
    * 2.3. Installation of CheMPS2–Molcas interface for DMRG calculations
    * [2.4. Installation of Dice–Molcas interface for HCI calculations](<diceinst.html>)
    * [2.5. Installation of Molcas for Stochastic-CASSCF calculations](<stochcas.html>)
    * [2.6. Maintaining the package](<maintain.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<parainst.html> "2.2. Parallel Installation") | [next](<diceinst.html> "2.4. Installation of Dice–Molcas interface for HCI calculations") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/installation.guide/dmrginst.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

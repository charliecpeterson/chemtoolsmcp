<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/installation.guide/parainst.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

2.2. Parallel Installation

[previous](<install.html> "2.1. Installation") | [next](<dmrginst.html> "2.3. Installation of CheMPS2–Molcas interface for DMRG calculations") | [index](<../genindex.html> "General Index")

# 2.2. Parallel Installation¶

  * Supported MPI implementations

  * Using an external Global Arrays installation (optional step)

  * General overview of the procedure with the configure script (alternative 1)

  * General overview of the procedure with cmake (alternative 2)

  * Running Molcas in parallel

    * Example of a submit script

    * Memory

    * I/O

    * Pinning

    * GA specific issues




Installation of Molcas for execution in multi-processor environments can be a bit more involved than the standard installation, this chapter considers those particulars not covered previously.

The parallelization of Molcas is achieved through the use of the Global Arrays (GA) API and some direct MPI calls. The API can be linked to an external GA library or to our own DGA library (an internal PGAS framework built upon MPI-2).

Warning

The DGA library is not available in OpenMolcas.

When using DGA (the default), the current list of supported MPI-2.2 implementations is given below: MPICH2/MPICH3, MVAPICH2, OpenMPI, Intel MPI.

When one wants to use an external GA library, it has to be configured and compiled separately. In that case, please read the section on using an external GA installation to properly configure and install GA first!!!

**IMPORTANT** : not all modules support distribution of work and/or resources through parallel execution, and even if they do it might be that some functionality is limited to serial performance. This is a list of core modules which _can_ benefit from parallel execution: gateway, seward, scf, rasscf, caspt2. More detailed information regarding parallel behaviour can be found in the documentation of the respective module and in the table at the beginning of the manual about supported parallelism. If no information is available, you should conclude that there is nothing to be gained from parallel execution.

## 2.2.1. Supported MPI implementations¶

Most probably, you will use a free MPI-2 implementation such as MPICH2/MPICH3, MVAPICH2, or Open MPI.

  * MPICH2: https://www.mpich.org/

  * MPICH3: https://www.mpich.org/

  * MVAPICH2: http://mvapich.cse.ohio-state.edu/

  * Open MPI: https://www.open-mpi.org/




**NOTE** : Open MPI versions older than v1.6.5 are not supported. More specifically, only Open MPI v1.6.5, and v1.8.1 are tested and known to work correctly with Molcas.

It is a very good idea to verify that the correct compiler environment is present before configuring Molcas. You should therefore check that the backend compiler of the wrappers is correct by running /path/to/mpif77 -show (MPICH2/MPICH3 and MVAPICH2) or /path/to/mpif77 –showme (Open MPI), which will list the actual executed command. If the backend compiler seems to be correct, also try to run it to see if it is properly detected (on some clusters you will need to load the appropriate module for the compiler). If all is well, you should be able to configure Molcas without any problems.

It is highly recommended to use the compiler that was used for the MPI library to build GA (optional) and Molcas to avoid compatibility issues. However, if you really want to use a different compiler than the compiler which was used for building the MPI library, you can do so by passing the -fc and -cc command line arguments (MPICH2/MPICH3 and MVAPICH2) to the wrappers, or setting the environment variables OMPI_F77/OMPI_F90 and OMPI_CC (Open MPI).

Several commercial MPI implementations exist such as HP-MPI, IBM’s MPI-F, Intel MPI, SGI’s MPT. Currently we only support Intel MPI. For the others that are not (yet) supported, it is recommended to either configure Molcas without parallel options and change the Symbols file after the serial configuration, or rely on cmake to use the correct options.

Please refer to the documentation of your MPI implementation for details on how to build programs, i.e. which wrappers to use and if necessary what libraries you need to link in.

## 2.2.2. Using an external Global Arrays installation (optional step)¶

If you wish to use an external GA library, it has to be installed before you build Molcas. You could e.g. use this if you have trouble with the built-in DGA solution. The installation instructions may be found at the Global Arrays home page: http://hpc.pnl.gov/globalarrays/

Note that any problems with installation or other issues specific to GA are best resolved by contacting the GA authors directly, rather than the Molcas group. It is therefore a very good idea to run the GA testing code as a job on the cluster where you want to use Molcas to make sure that it works properly before continuing to install Molcas.

Global Arrays needs to be installed with 8-byte integer support using the flag(s) –enable-i8 –with-blas8[=…] [–with-scalapack8[=…]], and for infiniband clusters you probably need to use the –with-openib flag. When linking to an external library, e.g. the Intel MKL, do not forget to include the proper ilp64 library versions.

Please read the documentation of GA for more details about installation.

## 2.2.3. General overview of the procedure with the configure script (alternative 1)¶

In the simplest case, the parallel version of Molcas may be installed simply by specifying the flag -parallel to configure. For example:
    
    
    ./configure -parallel
    

When using an external GA, pass the location of the installation to Molcas configure:
    
    
    ./configure -parallel -ga /opt/ga-5.1
    

When the locations of the MPI lib and include directories is set incorrectly, you can specify them by setting their common root directory with the par_root flag or if they are in different directories you can use the separate par_inc and par_lib flags:
    
    
    ./configure -parallel -par_root /usr/lib/openmpi
    ./configure -parallel -par_inc /usr/lib/openmpi/include -par_lib /usr/lib/openmpi/lib
    

More likely, some individual tailoring will be required, the following summarizes the necessary steps:

  1. Check that the correct wrapper compilers were detected, as specified in $MOLCAS/Symbols.

  2. If needed, change the F77/F90 and CC variables in the Symbols file for any custom modifications you made to the wrappers.

  3. Optionally install (and test) the external Global Arrays library.

  4. Check the command for executing binaries in parallel, as specified by RUNBINARY in $MOLCAS/molcas.rte.

  5. Install (and test) Molcas.




Provided that steps 1–4 can be successfully accomplished, the installation of Molcas itself is unlikely to present many difficulties.

## 2.2.4. General overview of the procedure with cmake (alternative 2)¶

CMake accepts two main flags for parallel installation, one to specify the use of parallelization -DMPI=ON, and one to specify a “true” GA library -DGA=ON instead of DGA (the default is -DGA=OFF, meaning no external GA is used, so do not confuse the option `-DGA` which means “define GA” with DGA). When using the latter -DGA=ON flag, there are two further options: using a precompiled GA library or compile GA together with the rest of Molcas. To use a precompiled GA, make sure the GAROOT environment variable is exported and contains the path of the GA installation, before running cmake. To compile GA as part of molcas, use the flag -DGA_BUILD=ON (in addition to -DGA=ON)

CMake will determine an appropriate MPI libary based on the compiler it finds, so in order to use a specific MPI library, just make sure the CC and FC variables point to the correct MPI wrappers!

The whole procedure is summarized below (square brackets showing optional commands):
    
    
    [export GAROOT=/path/to/external/GA]
    [CC=/path/to/mpicc] [FC=/path/to/mpifort] cmake -DMPI=ON [-DGA=ON [-DGA_BUILD=ON]] /path/to/molcas
    make [-j4]
    

## 2.2.5. Running Molcas in parallel¶

A few comments on running on a cluster:

The very old MPICH versions sometimes need a file with a list of the nodes the job at hand is allowed to use. At default the file is static and located in the MPICH installation tree. This will not work on a workstation cluster, though, because then all jobs would use the same nodes.

Instead the queue system sets up a temporary file, which contains a list of the nodes to be used for the current task. You have to make sure that this filename is transfered to $mpirun. This is done with the -machinefile flag. On a Beowulf cluster using PBS as queue system the RUNBINARY variable in $MOLCAS/molcas.rte should look something like:
    
    
    RUNBINARY='/path/to/mpirun -machinefile $PBS_NODEFILE -np $MOLCAS_NPROCS $program'
    

The newer MPICH2/MPICH3 as well as MVAPICH2, which works through the use of the HYDRA daemons and does not need this command line argument, as well as Open MPI most likely only need the -np $MOLCAS_NPROCS command line option. They use mpiexec instead of mpirun.

Parallel execution of Molcas is achieved by exporting the environment variable MOLCAS_NPROCS, for example when running on 4 nodes use:
    
    
    export MOLCAS_NPROCS=4
    

and continuing as usual.

In this section, we assume you will be using PBS on a cluster in order to submit jobs. If you don’t use PBS, please ask your system administrator or consult the cluster documentation for equivalent functionality.

### 2.2.5.1. Example of a submit script¶
    
    
    #!/bin/sh
    #PBS -l walltime=10:00:00
    #PBS -l nodes=4
    #PBS -l pmem=3000mb
    
    ######## Job settings ###########
    export MOLCAS_MEM=800
    export SUBMIT=/home/molcasuser/project/test/
    export Project=test000
    export MOLCAS_NPROCS=4
    
    ######## modules ###########
    . use_modules
    module load intel/11.1
    module load openmpi/1.4.1/intel/11.1
    
    ######## molcas settings ###########
    export MOLCAS=/usr/local/molcas80.par/
    export WorkDir=/disk/local/
    
    ######## run ###########
    cd $SUBMIT
    molcas $Project.input -f
    

### 2.2.5.2. Memory¶

The maximum available memory is set using the PBS option pmem. Typically, MOLCAS_MEM will then be set to around 75% of the available physical memory. So for a parallel run, just divide the total physical memory by the number of processes you will use and take a bit less. For example, for a system with 2 sockets per node and 64 GB of memory, running 1 process per socket, we would set pmem to 30000 MB.

### 2.2.5.3. I/O¶

The important thing to consider for I/O is to have enough scratch space available and enough bandwidth to the scratch space. If local disk is large enough, this is usually preferred over network-attached storage. Molcas requires the absolute pathname of the scratch directory to be the same across nodes.

### 2.2.5.4. Pinning¶

Process pinning is sometimes required to achieve maximum performance. For CASPT2 for example, processes need to be pinned to their socket or NUMA domain.

The pinning configuration can usually be given as an option to the MPI runtime. With Intel MPI for example, one would set the I_MPI_PIN_DOMAIN variable to socket. Alternatively, you can use a third-party program to intervene on your behalf, e.g. <https://code.google.com/p/likwid/>. Please ask your system administrator how to correctly pin your processes.

### 2.2.5.5. GA specific issues¶

When using GA, several problems can occur when trying to run jobs with a large amount of memory per process. A few example error messages are given here with their proposed solution.
    
    
    (rank:0 hostname:node1011 pid:65317):ARMCI DASSERT fail.
     src/devices/openib/openib.c:armci_pin_contig_hndl():1142
     cond:(memhdl->memhndl!=((void *)0))
    

The error output in the Molcas errfile (stderr) then says:
    
    
    Last System Error Message from Task 2:: Cannot allocate memory
    

Related messages that display a problem with armci_server_register_region instead of armci_pin_contig_hndl can also occur, and point to similar problems.

This can have two causes:

  * Some parameters of the Mellanox mlx4_core kernel module were set too low, i.e., log_num_mtt and log_mtts_per_seg. These should be set according to the instructions on https://community.mellanox.com/docs/DOC-1120. Values of 25 and 0 respectively, or 24 and 1 should be fine.

  * The “max locked memory” process limit was set too low. You can check this value by running ulimit -a or ulimit -l. Make sure you check this through an actual job! Easiest is to start an interactive job and then execute the command. The value should be set to unlimited, or at least to the amount of physical memory available.



    
    
    0: error ival=4 (rank:0 hostname:node1011 pid:19142):ARMCI DASSERT fail.
     src/devices/openib/openib.c:armci_call_data_server():2193
     cond:(pdscr->status==IBV_WC_SUCCESS)
    

This error is related to the value of the variable ARMCI_DEFAULT_SHMMAX, try setting it at least to 2048. If this is still too low, you should consider patching GA to allow higher values.

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<ig.html>)
    * [2.1. Installation](<install.html>)
    * 2.2. Parallel Installation
      * 2.2.1. Supported MPI implementations
      * 2.2.2. Using an external Global Arrays installation (optional step)
      * 2.2.3. General overview of the procedure with the configure script (alternative 1)
      * 2.2.4. General overview of the procedure with cmake (alternative 2)
      * 2.2.5. Running Molcas in parallel
    * [2.3. Installation of CheMPS2–Molcas interface for DMRG calculations](<dmrginst.html>)
    * [2.4. Installation of Dice–Molcas interface for HCI calculations](<diceinst.html>)
    * [2.5. Installation of Molcas for Stochastic-CASSCF calculations](<stochcas.html>)
    * [2.6. Maintaining the package](<maintain.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<install.html> "2.1. Installation") | [next](<dmrginst.html> "2.3. Installation of CheMPS2–Molcas interface for DMRG calculations") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/installation.guide/parainst.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

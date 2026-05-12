<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/installation.guide/install.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

2.1. Installation

[previous](<ig.html> "2. Installation Guide") | [next](<parainst.html> "2.2. Parallel Installation") | [index](<../genindex.html> "General Index")

# 2.1. Installation¶

  * Prerequisites

    * Prerequisite hardware

    * Prerequisite software

      * Windows

      * MacOS

    * Preparing the installation

  * Configuring Molcas

    * Simple configuration with the setup script

    * Advanced configuration with the configure script (alternative 1)

    * Advanced configuration with CMake (alternative 2)

      * Summary of main options for CMake

      * Example 1: GCC C/Fortran compilers with GA/OpenMPI and OpenBLAS

      * Example 2: Intel C/Fortran compilers with GA/IntelMPI and MKL

  * Building Molcas

    * Verifying the Molcas installation

  * GUI and documentation




The present installation guide describes the necessary steps for installing and tailoring Molcas. It also describes the steps for applying updates whenever necessary.

The installation procedure can be reduced to a few simple steps:

  1. Extract the contents of the tar

  2. Configure the package

  3. Build the package

  4. Build GUI and documentation (optional)

  5. Make the package generally available




## 2.1.1. Prerequisites¶

### 2.1.1.1. Prerequisite hardware¶

In general, Molcas can be built on any hardware that runs under a UNIX operating system. Some of these variants of hardware and software have been tested by us, and you should not have any problems to install Molcas on any of these. For other platforms you will most likely need to put some extra effort into the installation. In many cases the only effort on your part is setting some compiler flags, paths to system software etc. For a list of the platforms where we have successfully installed Molcas see our homepage: <http://www.molcas.org>.

To load the executables resident, sufficient memory is required. In addition, the programs are enabled to allocate work space dynamically. To avoid excessive paging we recommend that your machine should be equipped with at least 2 GB of memory per running application. Note, that Molcas will run faster with more memory.

To build Molcas you may need up to 2 GB of free disk space depending on what you choose to install and how (e.g. CMake keeps the object files around). The actual size of a Molcas installation is around 300-600 MB. To run the verification tests of Molcas you should have a scratch disk with up to 1 GB of free disk space, depending on the suite you run. For the “small” set about 400 MB will suffice. To perform larger calculations, ample amount of scratch disk space is necessary. The exact amount varies with the type of systems studied, but a general recommendation is at least 4 GB of disk space, per production run.

### 2.1.1.2. Prerequisite software¶

If you obtain the source code of Molcas, then you need to make certain that the necessary software is available to build Molcas. The minimum requirements are:

  * A Fortran compiler (with support for Fortran 2003 and later)

  * A C compiler

  * GNU make See URL <https://www.gnu.org> and navigate to the gnumake page or go directly to <https://www.gnu.org/software/make/>

  * Perl (version 5.008 or higher)




Also, you can benefit from following optional dependencies:

  * CMake (version 2.8.11 or higher, recommendeded for easier configuration)

  * an optimized BLAS/LAPACK library (alternative for the slow built-in library based on Netlib)

  * MPI-2 (to enable parallelization features of Molcas)

  * Global Arrays (version 5 or higher, an alternative to the built-in DGA library)




Warning

The DGA library is not available in OpenMolcas.

The Graphical User Interface codes in Molcas require additional software, including OpenGL and glut library. However, in most of the cases there is no need to install these libraries, since executables for GUI are included into the distribution, or they can be downloaded from Molcas webpage (<http://www.molcas.org>).

In order to get TaskFarm working, the following packages should be installed on your system:

  * Debian/Ubuntu: libipc-shareable-perl, libparallel-forkmanager-perl

  * RedHat/CentOS: perl-IPC-Shareable, perl-Parallel-ForkManager

  * Other OS: Shareable.pm, and ForkManager.pm should be available from $PERL5LIB




#### 2.1.1.2.1. Windows¶

To install Molcas under MS Windows (98/NT/XP/7/8) one should install Cygwin (freeware from RedHat Inc., which can be downloaded from <https://www.cygwin.com>). The minimal installation of Cygwin to run Molcas includes:

  * check that user name (under Windows) does not contain spaces

  * select a disk, which has enough space for installation of Cygwin and Molcas

  * install Cygwin to the root of selected disk with all defaults

  * run setup again and install the following packages: Devel\\(\rightarrow\\)gcc-fortran, Devel\\(\rightarrow\\)make, Devel\\(\rightarrow\\)gcc-gcc, Utils\\(\rightarrow\\)time, Perl\\(\rightarrow\\)perl

  * optionally install editors: Editors\\(\rightarrow\\)mc, Editors\\(\rightarrow\\)vim

  * run cygwin.bat to create Cygwin environment for the user

  * copy Molcas tar file into your home directory in Cygwin, and proceed with installation in the same way as under Linux.




#### 2.1.1.2.2. MacOS¶

Installation of Molcas under MacOS requires installation of the Apple Developer Tools (Xcode) and a Fortran compiler. These programs could be downloaded from:

<https://developer.apple.com/xcode/downloads/>

<https://opensource.apple.com/>

<https://gcc.gnu.org/wiki/GFortranBinaries#MacOS>

<http://hpc.sourceforge.net/>

<https://www.macports.org>

However, if you are looking for an out of the box solution, you can download a Free PGI for Mac OS X distribution available at <https://www.pgroup.com/products/freepgi/index.htm>

### 2.1.1.3. Preparing the installation¶

In order to install Molcas you need to choose a directory where the Molcas driver script is to be installed. The driver executes scripts and programs form the Molcas package and must be located in a directory included into the PATH variable. Usually this will be /usr/bin when installing as root, and ~/bin when installing as an unprivileged user.

The driver script molcas uses the value of the environment variable MOLCAS to identify which version to use. The major advantage with this mechanism is that it is easy to switch between different versions of Molcas by simply changing the environment variable MOLCAS. However if the current directory is a subdirectory (up to 3rd level) of a Molcas tree, the latter will be used regardless of the value of the MOLCAS variable.

Molcas itself can be located in any place on the disk. The installation can be done by root, or by an unprivileged user. In the later case you can copy the molcas driver script to an appropriate location, e.g. /usr/local/bin, after the installation.

All files are contained in a tar archive file with the name molcasXX.tar.gz, where XX depends on the version number, you need to uncompress the file with the command gunzip molcasXX.tar.gz, and untar the package with tar -xvf molcasXX.tar.

## 2.1.2. Configuring Molcas¶

Before you can build Molcas you have to configure it. Most common platforms have been setup by the Molcas developers, so for a serial installation with default settings for compiler and compiler flags configuration of Molcas can be done without specifying any special extra options.

There are two ways to configure Molcas: using the configure script (alternative 1) or using cmake (alternative 2). The latter is more recent and does not support all the original platforms, but it supports most commonly used environments and is easier as it is usually able to autodetect the necessary libraries. The CMake alternative also makes it easier to synchronize different installations based on the same source. Therefore, we recommend you to use alternative 2. If you are already familiar with building Molcas 8.0 or an earlier version, it might be easier to keep using alternative 1, as you can then port your exisiting configuration options. Note that for certain external software (e.g. DMRG), CMake is mandatory, and thus alternative 2 is needed.

For new users, use of the setup script is recommended. It will also allow you to choose between the two configuration alternatives if CMake is detected.

### 2.1.2.1. Simple configuration with the setup script¶

If you are new to building Molcas, it is recommended that you use the setup script to configure Molcas. Advanced users and/or users that need further customization should skip this section and use one of the alternatives in the next sections to configure Molcas.

The setup script will prompt you interactively about the most important settings. To get started, first unpack the Molcas source and then run:
    
    
    ./setup
    

in the main Molcas directory. Answer the questions and then proceed to Section 2.1.3 to build Molcas.

For advanced users that need further customization, one of the alternatives in the next sections will be needed.

### 2.1.2.2. Advanced configuration with the configure script (alternative 1)¶

You can start the configuration by running the configure script:
    
    
    ./configure
    

To know which flags can be used, run {./configure -h. The above command (without any flags) will use a default configuration, i.e. a serial Molcas installation using built-in BLAS/LAPACK.

When configuration is finished, you should review the log file configure.log to see if everything is ok. There is no harm in running the configuration script even if it should fail, you simply rerun it with correct parameters.

If the configuration step was not successful, you probably are missing some prerequisite software, or this software is located in an unusual location on the disk. In the later case you might need to update your PATH, or use flag -path in configure.

Molcas supports out-of-source installation. If for some reason, you would like to install molcas under a separate tree, you can create a directory, and call configure with appropriate flags, e.g.
    
    
    mkdir $HOME/molcas
    cd $HOME/molcas
    /sw/molcas_dist/configure -speed safe
    

The list all the options for configure, run
    
    
    ./configure -help
    

### 2.1.2.3. Advanced configuration with CMake (alternative 2)¶

Start configuration by creating a build directory and then running the cmake program with the location of the Molcas source. You can create the build directory as a subdirectory of the Molcas root directory, but we recommend you create it outside. For example, suppose the Molcas root is located at molcas in your home directory and you want to build it at molcas-build:
    
    
    mkdir ~/molcas-build
    cd ~/molcas-build/
    cmake ~/molcas/
    

After the first run, CMake keeps a cache of the configuration around as a file CMakeCache.txt. If you want to change certain options (e.g. use a different compiler), you need to remove this file first. As an example, if we wish to reconfigure with the intel compilers instead we do:
    
    
    rm CMakeCache.txt
    CC=icc FC=ifort cmake ~/molcas/
    

Once the cache file is there, subsequent additional options do not require pointing to the Molcas source, just to the build directory itself. Here we add an option to build parallel Molcas:
    
    
    cmake -DMPI=ON .
    

When using MPI, CMake will pick up the correct compiler and MPI library from the wrappers. So if you configure with MPI or you later wish to use a different MPI library/compiler wrapper, it is better to remove the cache file first, then (re)configure pointing to the wrappers:
    
    
    rm CMakeCache.txt
    CC=mpicc FC=mpifort cmake -DMPI=ON ~/molcas/
    

#### 2.1.2.3.1. Summary of main options for CMake¶

To see all the CMake options, you can run the ccmake command to interactively select the options and their values. The following is a list of some of the most important options.

  * `BUILD_SHARED_LIBS` — (`ON`/`OFF`): Enable building shared libraries (reduced disk space).

  * `CMAKE_BUILD_TYPE` — (`Debug`/`Garble`/`RelWithDebInfo`/`Release`/`Fast`): Normally use `Release`. `RelWithDebInfo` may be useful for reporting a problem. `Fast` in unsupported and may give wrong results in some cases.

  * `CMAKE_INSTALL_PREFIX` — Specify the directory where Molcas will be installed when running make install.

  * `MPI` — (`ON`/`OFF`): Enable multi-process parallelization.

  * `GA` — (`ON`/`OFF`): Enable interface with Global Arrays (see [Section 2.2](<parainst.html#sec-parallel-installation>)).

  * `HDF5` — (`ON`/`OFF`): Enable HDF5 files (portable binary format) in some programs.

  * `LINALG` — (`Internal`/`Runtime`/`MKL`/`ACML`/`OpenBLAS`): Select the linear algebra library (BLAS + LAPACK) against which to link Molcas. `Internal` uses the default (and slow) Netlib version included with Molcas. `Runtime` (experimental) offers the possibility of choosing the library at run time.

  * `OPENMP` — (`ON`/`OFF`): Enable multi-thread parallelization (usually restricted to using a multi-threaded version of linear algebra libraries.

  * `TOOLS` — (`ON`/`OFF`): Compile the tools that have CMake support.




#### 2.1.2.3.2. Example 1: GCC C/Fortran compilers with GA/OpenMPI and OpenBLAS¶

  * 64-bit Linux OS

  * MPI preinstalled (install OpenMPI or MPICH with package manager)



    
    
    # OpenBLAS
    tar zxvf OpenBLAS-v0.2.15.tar.gz
    cd OpenBLAS-0.2.15/
    make USE_OPENMP=1 NO_LAPACK=0 INTERFACE64=1 BINARY=64 DYNAMIC_ARCH=1 libs netlib shared
    [sudo] make PREFIX=/opt/openblas-lapack-ilp64 install
    # GA
    tar zxvf /path/to/ga-5-4b.tgz
    cd ga-5-4b/
    ./configure --enable-i8 --with-blas8 --with-lapack8 --with-scalapack8 --prefix=/opt/ga54b-ilp64.OpenMPI
    make
    [sudo] make install
    # Molcas
    tar zxvf molcas.tgz
    cd molcas
    mkdir build && cd build/
    export GA=/opt/ga54b-ilp64.OpenMPI
    export OPENBLASROOT=/opt/openblas-lapack-ilp64
    CC=mpicc FC=mpifort cmake -DMPI=ON -DGA=ON -DLINALG=OpenBLAS ../
    make -j4
    

Any other configurations can be easily derived from the above by simply removing the unneeded optional stuff.

#### 2.1.2.3.3. Example 2: Intel C/Fortran compilers with GA/IntelMPI and MKL¶

  * 64-bit Linux OS

  * Intel Compilers/MPI preinstalled (usually on a cluster as a module)

  * Infiniband (OpenIB)



    
    
    # make sure Intel compilers/MPI are loaded
    module load ...
    # check compilers/environment
    type mpiicc
    type mpiifort
    echo $MKLROOT
    # GA
    tar zxvf /path/to/ga-5-4b.tgz
    cd ga-5-4b/
    ./configure --prefix=/opt/ga54b-ilp64.IntelMPI --enable-i8 --with-openib \
                --with-blas8="-L$MKLROOT/lib/intel64 -lmkl_intel_ilp64
                              -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm" \
                --with-scalapack8="-L$MKLROOT/lib/intel64 -lmkl_scalapack_ilp64
                                   -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core
                                   -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread -lm"
    make
    [sudo]make install
    # Molcas
    tar zxvf molcas.tgz
    cd molcas
    mkdir build && cd build/
    export GA=/opt/ga54b-ilp64.IntelMPI
    CC=mpiicc FC=mpiifort cmake -DMPI=ON -DGA=ON -DLINALG=MKL ../
    make -j4
    

## 2.1.3. Building Molcas¶

When the configuration step (Section 2.1.2) is completed successfully, you can build Molcas. This is simply done by typing make in the Molcas root directory. It is recommended that you save the output from make in a log file for tracing of potential problems.
    
    
    make > make.log 2>&1
    

In order to speed up the build process, you can perform a parallel compilation. When you configure Molcas with CMake, a simple
    
    
    make -jN
    

will suffice, while if you configured Molcas with the configure script, you need to build in two steps:
    
    
    make -jN build
    make install
    

In the above commands, `N` should be replaced with the number of threads you wish to use for building. Typically, a number equal to the number of cores will be fine.

When Molcas is being compiled some compilers give a lot of warnings. These are not serious in most cases. We are working on eliminating them, but the job is not yet completely finished.

After Molcas has been built correctly, you should absolutely run a basic verification to ensure that the installation is sane. See the next section for details on verification.

### 2.1.3.1. Verifying the Molcas installation¶

After a successful build of Molcas you should verify that the various modules run correctly. Directory test/ contains various subdirectories with test inputs for Molcas. Use the command molcas verify [parameters] to start verification. Running this command without parameters will check the main modules and features of Molcas and we recommend this default for verifying the installation. You can also specify a keyword as argument that translates into a sequence of test jobs, or you can specify a list of test jobs yourself. The script has extended documentation, just run molcas verify –help.

To generate a report after performance tests you should execute a command molcas timing. The report is then located in the file test/timing/user.timing. The results of benchmark tests for some machines are collected on our webpage <http://www.molcas.org/benchmark.html> At the completion of the test suite a log of the results is generated in the file test/results. If installation was performed by another user (e.g. root), you can redefine the location of output files by adding the flag -path PATH. Each test job is signaled as either ok of failed. If there are any failed jobs, the outputs are saved in Test/Failed_Tests. Each test job tests for a resulting checksum for the modules tested. This checksum is typically the energy for a wavefunction program such as RASSCF, whereas other types of codes use other checksums.

The checksums will not match exactly with our reference values since different machines use different arithmetics. We have tried to make the acceptable tolerances as small as possible and at the same time make all tests pass successfully. It might be the case that your particular platform will produce one or more results that are just outside our tolerances, and in such a case the test is most likely ok.

## 2.1.4. GUI and documentation¶

Normally, there is no need to build the GUI used in Molcas, since we provide executables for most common platforms. You can download executables for the GUI from the Molcas webpage (<http://www.molcas.org>).

In order to build documentation in various formats, use the command make doc. Alternatively the documentation is also available no the web page (<http://www.molcas.org>).

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<ig.html>)
    * 2.1. Installation
      * 2.1.1. Prerequisites
      * 2.1.2. Configuring Molcas
      * 2.1.3. Building Molcas
      * 2.1.4. GUI and documentation
    * [2.2. Parallel Installation](<parainst.html>)
    * [2.3. Installation of CheMPS2–Molcas interface for DMRG calculations](<dmrginst.html>)
    * [2.4. Installation of Dice–Molcas interface for HCI calculations](<diceinst.html>)
    * [2.5. Installation of Molcas for Stochastic-CASSCF calculations](<stochcas.html>)
    * [2.6. Maintaining the package](<maintain.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<ig.html> "2. Installation Guide") | [next](<parainst.html> "2.2. Parallel Installation") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/installation.guide/install.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

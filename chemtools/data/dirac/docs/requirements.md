orphan  

# Requirements

DIRAC requires CMake, Fortran 90, C, and C++ compilers and a Python
environment for the setup and run scripts. The program is designed to
run on a unix-like operating system, and uses MPI for parallel
calculations. Optionally DIRAC can (and should) be linked to platform
specific BLAS and LAPACK libraries. Installing HDF5 is strongly
recommened as not all functionality will work without it.

## Minimum requirements

- CMake 3.23 - or later
- Python 3.6 - or later
- GFortran/GCC/G++ 4.9 - 5.2 or Intel compilers 13.0 - 15.0.6
- HDF5 1.10 - or later (note that patch levels 1.14.0 and 1.14.1 should
  be avoided due to a bug). The installation of the h5py Python package
  is not mandatory but recommended.

If your distribution does not provide a recent CMake, you can install it
without root permissions within few minutes. Download the binary
distribution from <https://cmake.org/download/>, extract it, and set the
environment variables PATH and LD_LIBRARY_PATH to point to the bin/ and
lib/ subdirectories of the extracted path, respectively. Verify that
your new CMake is recognized with `cmake --version`.

## Requirements for building with the parallel library ExaTensor

- CMake 3.23 - or later
- Python 3.6 - or later
- GFortran/GCC/G++ 8.x or 11+ (so neither 9.x nor 10.x), Intel compilers
  18 or later
- OpenMPI 4.1.5 or later, or MPICH 3.2.1 or later. Compilation with
  Intel MPI is possible but should be checked for performance.

Further informaton on supported environments can be found at the [DIRAC
fork of ExaTensor website](https://github.com/RelMBdev/ExaTENSOR). MPI
is not required for building only the TAL-SH (single-node) component of
ExaTensor.

MacOS X users are `advised to use GCC 13 instead of GCC 14/15` until an
[issue with compiler optimization level affecting the ExaCorr code is
resolved](https://gitlab.com/dirac/dirac/-/issues/155).

At runtime you need to set environment variables to inform the library
about the optimal parallel setup in both the installation as well as the
execution stage. This is explained in detail at the [DIRAC fork of
ExaTensor website](https://github.com/RelMBdev/ExaTENSOR).

The setup script of DIRAC should usually take care of the installation
step, but the appropriate set up at runtime should be carefully
considered by the user to obtain optimal performance. Examples of setup
for running the standalone ExaTENSOR (`Qforce.x`) test utility for
different architectures are given
[here](https://github.com/RelMBdev/ExaTENSOR/blob/master/run.sh).

Users interested in this functionalty should read the section on [tensor
libraries](installation/tensor_libraries.html) for DIRAC-specific
instructions and known issues.

## Supported, but optional dependencies

- OpenMPI 1.6.2 - 1.8.5 (MPI parallelization)
- IntelMPI 4.1 - 5.0 (MPI parallelization)
- Boost 1.54.0 - 1.60.0 (PCMSolver; installed along DIRAC if below
  1.54.0)
- Alps 2.2 (DMRG code)
- [TBLIS](https://github.com/MatthewsResearchGroup/tblis) 2+ (ExaCorr
  code, see the [Tensor libraries section](tensor_libraries.html)). As
  for TAL-SH/ExaTensor, MacOS X users are
  `advised to use GCC 13 instead of GCC 14/15` until an [issue with
  compiler optimization level affecting the ExaCorr code is
  resolved](https://gitlab.com/dirac/dirac/-/issues/155).

## Additional dependencies due to PCMSolver

It is optionally possible to enable the polarizable continuum model
functionality. The functionality is provided using the external module
[PCMSolver](http://pcmsolver.github.io/pcmsolver-doc/). The module
requires additional dependencies:

- the zlib compression library, version 1.2 or higher;
- the Boost set of libraries, version 1.54 or higher.

The module performs checks on the system to find an installation of
Boost that has the suitable version. The search is directed towards
conventional system directories. If the search fails, the module will
build a copy of Boost on the spot. If your system has Boost installed in
a nonstandard location, e.g. /boost/location, you have to direct the
search by adding the following flags to the `setup` script:

    $ ./setup -DBOOST_INCLUDEDIR=/boost/location/include -DBOOST_LIBRARYDIR=/boost/location/lib

By default, compilation of the module is enabled. CMake will check that
the additional dependencies are met. In case they are not, compilation
of PCMSolver will be disabled. CMake will print out which dependencies
were not satisfied. Passing the option `-DENABLE_PCMSOLVER=OFF` to the
`setup` script will disable compilation of the module and skip detection
of additional dependencies.

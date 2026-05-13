orphan  

# Notes on setting up and using tensor libraries in Dirac

The ExaCorr code (see `**EXACC`) is built around the use of tensor
contract libraries.

**ExaCorr will not be available in compilations with 64-bit integers**

Originally based on the tensor libraries
[TAL-SH](https://github.com/DmitryLyakh/TAL_SH) and
[ExaTENSOR](https://github.com/ORNL-QCI/ExaTENSOR) by Dmitry Lyakh
(which support both CPU and GPU architectures), through the Tensor
Algebra Processing Primitives ([TAPP](https://arxiv.org/abs/2601.07827))
API the code now supports additional tensor libraries:
[TBLIS](https://github.com/MatthewsResearchGroup/tblis) (CPU
architectures), and the [TAPP reference
implementation](https://github.com/TAPPorg/reference-implementation)
(CPU architectures, testing only).

**Important** Only the [fork maintained by the DIRAC
team](https://github.com/RelMBdev/ExaTENSOR) (which contains both
ExaTENSOR and the TAL-SH codes) should be used with DIRAC, since it
incorporates enhancements to the build system not found in the original,
no longer maintaned repositories by Dmitry Lyakh.

We are currently working towards supporting other distributed memory
tensor libraries such as the [Cyclops Tensor
Framework](https://github.com/cyclops-community/ctf), to provide users
with additional choice.

The choice of tensor library used as computational backend is made at
compile time, via the `TENSOR_EXECUTOR` environment variable.

Configuring for TBLIS (build will fetch TAPP and TBLIS from their
repositories):

    export TENSOR_EXECUTOR=1 # enables TBLIS via the TAPP API

The variable `ENABLE_TBLIS` can be toggled (=ON/OFF) through CMake.

Configuring for the TAPP reference implementation (build will fetch TAPP
and the reference implementation from its repository):

    export TENSOR_EXECUTOR=4 # enables the TAPP reference implementation via the TAPP API

Configuring for TAL-SH (build will fetch the code from the ExaTENSOR
repository):

    export TENSOR_EXECUTOR=2 # enables TAL-SH 

Configuring for ExaTENSOR (build will fetch the code from its
repository):

    export TENSOR_EXECUTOR=3 # enables ExaTENSOR 

**Important** By default, both TAL-SH / ExaTENSOR and TBLIS will be
fetched and built, even if e.g. TBLIS is the default executor.

# Configuring and building TAL-SH / ExaTENSOR for ExaCorr

For TAL-SH / ExaTENSOR, it is possible to avoid fetching via the network
by setting the `EXATENSOR_GIT_REPO_LOCATION` variable on CMake to the
path to a local clone of the ExaTENSOR git repository. This is aimed as
a workaround in systems which do not allow network access, such as some
supercomputer centers. We are currently working to [support a similar
capability for TAPP and
TBLIS](https://gitlab.com/dirac/dirac/-/issues/161).

The setup for building TAL-SH / ExaTENSOR in this version of DIRAC is
driven by DIRAC's CMake system but not fully integrated with it, with
environment variables being passed on to the TAL-SH / ExaTENSOR Make
system.

Upon starting the build for TAL-SH / ExaTENSOR, the file `Exatensor_ENV`
containing TAL-SH / ExaTENSOR environment variables is created by
DIRAC's CMake system. The file `Exatensor_ENV_UP` is also created as a
backup of the last configuration, as `Exatensor_ENV` is re-generated and
overwritten at each CMake configuration. The values of the variables in
`Exatensor_ENV` can be changed (e.g. `BUILD_TYPE=OPT` to
`BUILD_TYPE=DEV` to toggle debug flags on), or valid variables in the
the ExaTENSOR build system (found under `exatensor/src/exatensor` in the
build directory) can be added, for additional customization.

For example, the contents of `Exatensor_ENV` for a Mac OSX build with
GNU compilers and the OpenBLAS library could look like the following:

    WRAP=NOWRAP              # relevant for Cray compilers 
    BUILD_TYPE=OPT           # optimization configuration
    EXA_TALSH_ONLY=YES       # only build the TAL-SH component, for a sequential build
    # DIRAC compilers, passed on
    CMAKE_Fortran_COMPILER=/opt/local/bin/gfortran-mp-14 
    CMAKE_C_COMPILER=/opt/local/bin/gcc-mp-14 
    CMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-14 
    TOOLKIT=GNU              # GNU compilers
    EXA_OS=NO_LINUX          # Mac OSX 
    GPU_CUDA=NOCUDA          # CPU-only configuration
    MPILIB=NONE              # no MPI 
    BLASLIB=OPENBLAS 
    PATH_BLAS_OPENBLAS=/opt/local/lib       
    EXA_NO_BUILD=YES         # skips building ExaTENSOR and TAL-SH

**Note** The variable `EXA_NO_BUILD=YES` is added to `Exatensor_ENV` at
the end of complete build, so if one wishes to change any configuration
variables and recompile, this has to be removed.

Before configuring, the environment variable `EXA_GPUS` should be set in
order to enable GPU support for TAL-SH or ExaTENSOR.

For NVIDIA GPUs:

    export EXA_GPUS=NVIDIA

It is important for NVIDIA architectures to see whether the
`GPU_SM_ARCH` is correctly set in `Exatensor_ENV`. Examples are
`GPU_SM_ARCH=70` for V100 (sm_70) GPUs, `GPU_SM_ARCH=80` for A100 GPUs
(sm_80), and `GPU_SM_ARCH=90` for H100 GPUs (sm_90). For additional
information please consult the NVIDIA documentation, or [other online
resources](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

For AMD GPUs:

    export EXA_GPUS=AMD

For AMD GPUs, `Exatensor_ENV` will contain `GPU_CUDA=CUDA` and
`USE_HIP=YES`. The `GPU_SM_ARCH` will be disregarded if present, and the
build system should detect the target architecture from environment
variables; if not, the GPU type can be passed on via the environment
variable `HIP_TARGET`.

# Running ExaCorr in parallel and on GPU nodes

DIRAC can be compiled and run both in sequential and in parallel with
all of the above tensor libraries, provided that the code is compiled
with 32-bit integers.

**However, unless the ExaTENSOR library is used to distribute tensors
over multiple nodes, the tensor libraries will only use a single node**.
Furthermore, ExaTENSOR-based parallel calculations with less than four
(4) MPI ranks will actually run as a single node calculation.

In the case ExaCorr has been compiled with GPU support for TAL-SH, and
ExaTENSOR is not in use, the environment variable `TALSH_GPUS` can be
used to control how many of the GPUs accessible to TAL-SH runs will be
used (if `TALSH_GPUS` is not defined, all GPUs available to the process
will be used).

OpenMP parallelization is by default enabled in TAL-SH / ExaTENSOR, and
is controlled by the appropriate environment variables. The use of a
threaded math library is highly recommended.

ExaTENSOR runs will require the definition of another set of environment
variables, for example as in:

    export QF_NUM_PROCS=4             # total number of MPI ranks in the calculation. Must be at least 4 for ExaTENSOR to work properly. 
    export QF_PROCS_PER_NODE=4        # number of MPI ranks executing per node. in thi case all MPI ranks run on the same node.
    export QF_CORES_PER_PROCESS=2     # Number of CPU cores attached to each MPI rank
    export QF_NVMEM_PER_PROCESS=0     #
    export QF_HOST_BUFFER_SIZE=1000   # Memory available on the node, divided by the number of MPI ranks
    export QF_MEM_PER_PROCESS=750     # about 75% of QF_HOST_BUFFER_SIZE

    export QF_GPUS_PER_PROCESS=2 # Number of NVIDIA or AMD GPUs to be used per MPI rank; here set to 2 GPUs per rank
    export QF_MICS_PER_PROCESS=0 # this should always be set to zero
    export QF_AMDS_PER_PROCESS=0 # this should always be set to zero, even in the case of AMD GPU systems

    export OMP_NUM_THREADS=$QF_CORES_PER_PROCESS
    export OMP_DYNAMIC=false
    export OMP_MAX_ACTIVE_LEVELS=3
    export OMP_THREAD_LIMIT=256
    export OMP_WAIT_POLICY=PASSIVE
    export OMP_PROC_BIND="spread"
    export OMP_PLACES="{0:4},{4:4},{8:4},{12:4}"

please consult the examples for runtime configurations in the source
tree, and [the documentation of
ExaTENSOR](https://github.com/RelMBdev/ExaTENSOR/tree/master/DOC) for
further details.

These environment variables can also be added to the `~/.diracrc` file
for convenience (see `pam --help`).

**Important** In addition to no longer being supported by its original
authors, ExaTENSOR exploits MPI-3 features that are not always
well-supported by the underlying software stack (MPI implementation,
communication API etc). As such it can be rather non-trivial to arrive
at a configuration suitable for production runs. For example, at the
time of the DIRAC26 release, we have been able to compile and run
ExaCorr with ExaTENSOR over multiple computing nodes at the [Zeus HPC
cluster (Lille/Fr)](https://hpc-doc.univ-lille.fr/) using GNU 11
compilers and OpenMPI 4.1.5; runs on other resources with similar
software stacks were not successful.

With that, users are encouraged to periodically consult the collection
of build and runtime configuration examples for the [main
development](https://gitlab.com/dirac/dirac/-/tree/master/configuration-examples?ref_type=heads)
and [DIRAC26
release](https://gitlab.com/dirac/dirac/-/tree/release-26/configuration-examples?ref_type=heads)
branches in the [DIRAC Git repository](https://gitlab.com/dirac/dirac/)
for hints on their systems.

# OpenMP and GPU bindings

GPU bindings or OpenMP thread placement are not necessarily easy to get
right, so users should consult the documentation of the machine the jobs
will run on. Additionally, tools such as
[hello_jobstep](https://code.ornl.gov/olcf/hello_jobstep) or
[hello_cpu_binding or
hello_gpu_binding](https://dci-gitlab.cines.fr/external/adastra-examples/-/tree/main)
may be used, or serve as a starting point to be adapted to your needs.

orphan  

# How to set good environment variables for the Intel MKL library

By default DIRAC links to the multi-threaded Intel MKL library.

The pam script sets MKL_NUM_THREADS=1, MKL_DYNAMIC="FALSE",
OMP_NUM_THREADS=1, and OMP_DYNAMIC="FALSE", unless these variables are
set by the user. In order to benefit from the parallelization of MKL,
the user should provide appropriate environment variables.

Be very careful when running MPI calculations and using threaded MKL. If
all cores are already taken by the MPI (or by other users on the same
node), you may observe a significant slow down of the code's execution
run.

Currently DIRAC cannot benefit from `OMP_NUM_THREADS` other than 1 so
this variable should always be set to 1 for DIRAC calculations.

## Examples

Allow for 4 threads per core in MKL routines:

    export MKL_NUM_THREADS=4
    export MKL_DYNAMIC="FALSE"

Allow for 4 threads per core in MKL/BLAS routines:

    export MKL_NUM_THREADS=4
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=4"
    export MKL_DYNAMIC="FALSE"

As an example of proper splitting of CPUs between MPI and MKL, assume a
16 core node where 8 cores are assigned to MPI and the rest to MKL
threads:

    export MKL_NUM_THREADS=2
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=2"
    export MKL_DYNAMIC="FALSE"

If you would like to run a sequential job on the same node it would then
read:

    export MKL_NUM_THREADS=16
    export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=16"
    export MKL_DYNAMIC="FALSE"

You can find recommended settings for calling Intel MKL routines from
multi-threaded applications on the [Intel web
page](http://software.intel.com/en-us/articles/recommended-settings-for-calling-intelr-mkl-routines-from-multi-threaded-applications/).

## Troubleshooting

Sometimes you could get this error:

    OMP: Error #18: Setting environment variable "__KMP_REGISTERED_LIB_12973" failed:
    OMP: Hint: Seems application required too much memory.

This can happen with statically linked multi-threaded MKL library, see
[here](http://software.intel.com/en-us/forums/showthread.php?t=104947).

orphan  

# Testing the installation

It is very important that you verify that your DIRAC installation
correctly reproduces the reference test set before running any
production calculations. But also with all tests passing you are
strongly advised to always carefully verify your results since we cannot
guarantee a complete test coverage of all possible keyword combinations.

The test set driver is CTest which can be invoked with "make test" after
building the code.

## Environment variables for testing

Before testing with "make test" you should export the following
environment variables:

    $ export DIRAC_TMPDIR=/scratch            # scratch space (adapt the path of course)
    $ export DIRAC_MPI_COMMAND="mpirun -np 8" # only relevant if you compile with MPI

Note that if you set the DIRAC_MPI_COMMAND the pam script will assume
you that this is a parallel calculation. Before you test an MPI build,
you have to set DIRAC_MPI_COMMAND otherwise the tests will be executed
without MPI environment and many of the tests will fail.

**Tests based on the ExaTENSOR library**

For an MPI build and if the ExaCorr module is activated (which is the
case by default), additional environment variables have to be set in
order to run the tests checking the ExaTENSOR based functionality, see
[tensor_libraries.html](tensor_libraries.html) section for additional
details. Note a minimal MPI configuration to execute ExaCorr in this
case requires 4 (four) MPI ranks.

## Running the test set

You can run the whole test set either using:

    $ make test

or directly through CTest:

    $ ctest

Both are equivalent ("make test" runs CTest) but running CTest directly
makes it easier to run sequential tests on several cores:

    $ ctest -j4        # only for sequential tests

You can select the subset of tests by matching test names to a regular
expression:

    $ ctest -R dft

or matching a label:

    $ ctest -L short

To print all labels:

    $ ctest --print-labels

## How to get information about failing tests

We do our best to make sure that the release tarball is well tested on
our machines. However, it is impossible for us to make sure that all
functionalities and tests will work on all systems and all compiler
versions. Therefore, some tests may fail and this is how it can look:

    Total Test time (real) = 258.17 sec

    The following tests FAILED:
             34 - dft_response (Failed)
             73 - xyz_input (Failed)
             75 - dft_ac (Failed)
    Errors while running CTest

The first place to look for the reasons is build/Testing/Temporary:

    $ ls -l Testing/Temporary/
    total 108
    -rw-rw-r-- 1 bast bast  2247 Dec 11 10:19 CTestCostData.txt
    -rw-rw-r-- 1 bast bast 94798 Dec 11 10:19 LastTest.log
    -rw-rw-r-- 1 bast bast    39 Dec 11 10:19 LastTestsFailed.log

LastTest.log contains a log of all tests that were run. Search for the
test name or "failed". If tests do not produce any output, then this is
the place to look for reasons (environment variables, problems with
pam).

In addition, for each test (whether passed or failed), a directory is
created under build/test. So in this case you can go into
build/test/dft_response to figure out what went wrong. The interesting
files are the output files:

    -rw-rw-r-- 1 bast bast 89538 Dec 11 10:17 lda_hf.out
    -rw-rw-r-- 1 bast bast  1570 Dec 11 10:17 lda_hf.out.diff
    -rw-rw-r-- 1 bast bast   841 Dec 11 10:17 lda_hf.out.reference
    -rw-rw-r-- 1 bast bast   841 Dec 11 10:17 lda_hf.out.result

The `*.out` file is the output file of the test. The `*.result` file
contains numbers extracted by the test script. The `*.reference` file
contains numbers extracted from the reference file. The `*.diff` file is
empty for a test where result and reference match. If they do not match,
the `*.diff` file will list the lines of output where the results
differ.

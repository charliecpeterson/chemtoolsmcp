orphan  

# Installation instructions for system administrators

We recommend using EasyBuild to compile DIRAC in HCP systems. However,
if you still want to compile it by yourself, you can follow the
following instructions.

Please read the other installation sections for details but the
installation procedure is typically this:

    $ ./setup --mpi --qmkl=parallel --prefix=/full/install/path/
    $ cd build
    $ make [-j]
    $ export DIRAC_TMPDIR=/full/path/scratch
    $ export DIRAC_MPI_COMMAND="mpirun -np 8" # make test will run with MPI using 8 processes
    $ make test
    $ make install

This will install binaries, run scripts, the basis set library, as well
as tools into the install path.

Advise users to always set a suitable scratch directory:

    $ export DIRAC_TMPDIR=/full/path/scratch

If your system runs MPI jobs with mpirun, the pam script can be called
with the `--mpi` flag:

    $ pam [other flags] --mpi=8      # will run "mpirun -np 8 dirac.x"

You can define a custom MPI launcher with DIRAC_MPI_COMMAND, in this
case do not use `--mpi flag`:

    $ export DIRAC_MPI_COMMAND="mpirun -np 8"
    $ pam [other flags]              # no --mpi flag

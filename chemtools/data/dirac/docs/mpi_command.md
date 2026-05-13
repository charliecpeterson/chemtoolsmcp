orphan  

# How to define an alternative MPI launcher

The pam script by default launches MPI runs with mpirun. If you cannot
use mpirun because you are on a system which does not support it or
which offers alternative MPI launchers (SGI, Cray), you can define your
own DIRAC_MPI_COMMAND like this:

    $ export DIRAC_MPI_COMMAND="mpirun -np 8"
    $ export DIRAC_MPI_COMMAND="mpprun"
    $ export DIRAC_MPI_COMMAND="aprun -n 24"

This will then launch:

    $DIRAC_MPI_COMMAND dirac.x

instead of:

    mpirun -np N dirac.x

Note that `--mpi` overrides DIRAC_MPI_COMMAND.

# Passing arguments to MPI launcher

You can pass arguments to the MPI launcher, for example:

    ./pam --scratchfull="/tmp/milias/TEST" --noarch --mpiarg="-x PATH -x LD_LIBRARY_PATH --wdir '/tmp/milias/TEST'" --mpi=4 --inp=cc.inp  --mol=N2.ccpVDZ.mol

Note that you have provide proper workdir (the `--wdir` flag) for your
MPI launcher, otherwise it ends with error.

Concerning passing of environmental variables (through the `-x` flag),
if the variable is not defined, the MPI launcher gives warning (like
`Warning: could not find environment variable "LD_LIBRARY_PATHx"`).

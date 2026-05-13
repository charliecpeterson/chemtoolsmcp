orphan  

The scratch directory is the place where DIRAC will write temporary
files. At the beginning of the run, DIRAC copies input files and the
binary into the scratch space and at the end of the calculation, the
output is copied back, possibly together with other useful (restart)
files.

# How you can set the scratch directory

You can set the scratch space like this:

    $ pam --scratch=/somepath

If `--scratch` is not set, DIRAC will look for the environment variabls
`DIRAC_TMPDIR`.

If these variables are not set, DIRAC defaults to
`$HOME/DIRAC_scratch_directory`.

On your local computer you typically want to use the same scratch space
every time and you may be tired of typing `pam --scratch` every time. In
this case, put `--scratch=/somepath` in your `~/.diracrc` or export
`DIRAC_TMPDIR` in your `~/.bashrc` or equivalent.

DIRAC will always append `$USER/DIRAC_inp_mol_pid` to the scratch path.
This is to prevent users from accidentally wiping out their home
directory.

# What if you don't want DIRAC to append anything to your scratch path

For this use:

    $ pam --scratchfull=/somepath

In this case DIRAC will use the path as it is and not append anything to
it. Be careful with this since DIRAC may remove the path after the
calculation has finished.

# How to set the scratch path on a cluster

On a cluster you typically want to be able to find the scratch directory
based on some job id. If you are on a cluster with a Torque/PBS
scheduler then it can be useful to set:

    $ pam --scratch=/somepath/$USER/$PBS_JOBID

## Cluster with a global scratch disk

This means that all nodes write temporary files to the same disk.

No additional flags are needed for this run.

## Cluster with local scratch disks

This means that each node writes temporary files to its own disk.

For this situation the input files have to be distributed over the nodes
based on `--machfile`:

    $ pam --scratch=/somepath/$USER/$PBS_JOBID --machfile=/path/to/machfile

If you don't give `--machfile`, DIRAC assumes that you run using a
global scratch disk.

The list of machines is typically generated when the scheduler launches
the job but you can also set the list of machines manually.

The input files are by default distributed using rsh/rcp. If your
cluster does not support these you can change to ssh/scp protocol:

    $ pam --scratch=/somepath/$USER/$PBS_JOBID --machfile=/path/to/machfile --rsh=ssh --rcp=scp

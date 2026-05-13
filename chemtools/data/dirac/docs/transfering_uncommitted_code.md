orphan  

# Transfering uncommitted code between machines

## As regular tarball

For this make sure that you also take the .git directory otherwise
compilation will complain with:

    fatal: Not a git repository (or any of the parent directories): .git
    CMake Error at cmake/ConfigGitRevision.cmake:9 (string):
      string sub-command STRIP requires two arguments.

Also make sure that you get all the external sources. In other words if
you manually tar up a fresh DIRAC clone without .git that has never been
built, externals will be missing.

## Using scp

This should work. But even better is git clone over ssh (below).

## Git clone

You can git clone from one machine to another. Note that you cannot push
to a non-bare repository. In other words if you clone a regular clone,
you cannot push to it.

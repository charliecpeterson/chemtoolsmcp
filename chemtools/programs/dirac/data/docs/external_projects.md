orphan  

# Including external projects using Git submodules

See also this nice
[tutorial](http://chrisjean.com/2009/04/20/git-submodules-adding-using-removing-and-updating/)
and these [tips and
tricks](http://blogs.atlassian.com/2013/03/git-submodules-workflows-tips/).

We can use CMake's support for external projects together with the Git
submodule functionality to fetch, compile, and link source code from
external repositories.

When such external libraries/repositories are fetched, they are put in
the directory `external`. Later when everything is set up, you can go in
there, make changes, but you are not making changes to the parent code
repository, but to the respective external repository. This way you can
work on both codes at the same time, while making sure that changes to
modules are communicated to the respective repositories.

This mechanism takes some time to get used to but turns out to be
extremely powerful and convenient. Below are some typical scenarios.

## Adding a new external project

First we need to set it up on the Git side. Let's assume the external
project is hosted on `git@repo.ctcc.no:pcmsolver` and we want to put it
in `external/pcmsolver`:

    $ git submodule add git@repo.ctcc.no:pcmsolver external/pcmsolver

This will clone the external project and with `git status` we see:

    $ git status

    # On branch master
    # Changes to be committed:
    #   (use "git reset HEAD <file>..." to unstage)
    #
    # modified:   .gitmodules
    # new file:   external/pcmsolver

Now we commit this:

    $ git commit

Great. Now the external project is fetched by Git and we need to tell
CMake to build and link it. For this we open `CMakeLists.txt`, define
some variables that we want to pass to the external project and we use
the macro `add_external`:

    set(ExternalProjectCMakeArgs
        -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/external
        -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCBLAS_ROOT=${CBLAS_ROOT}
        -DEIGEN3_ROOT=${EIGEN3_ROOT}
        )

    add_external(pcmsolver "")

The `add_external` macro takes two arguments. The name of the external
project and the command to be used for testing it. The name of the
external project will be used as target name. If the testing command
string is empty, as in the example above, the external project will not
be tested after it had been built.

Finally we have to make sure that the library is linked.

If DIRAC f90 modules depend on f90 modules provided by the external
library, we have to impose a compilation order:

    add_dependencies(dirac pcmsolver)

## You just want to build DIRAC

In this case you don't have to know anything about Git submodules:

    $ ./setup
    $ cd build
    $ make

However, you need to have access to all repositories that the DIRAC will
fetch from.

## You want to check out the external sources without building DIRAC

Do this:

    $ git submodule init
    $ git submodule update

You will find the external sources in `external`.

## You want to update external sources

For each external module DIRAC will reference a specific commit. This
pointer (HEAD) will not move when external modules change until you tell
DIRAC to reference some other commit. In other words, if I commit a bug
to an external project repository and I don't tell DIRAC to use the new
commit, I will not see this bug in DIRAC.

But sometimes you want to update the reference. For this go to the
external repository (example: xcint):

    $ cd dirac/external/xcint

Switch to the branch that you want to reference and update:

    $ git checkout master
    $ git pull origin master

Now register the new reference in DIRAC:

    $ cd dirac
    $ git add external/xcint

## You want to commit and push modifications to external sources

The nice thing about Git submodules is that you can work on several
projects or modules within one parent project. This is what we do very
often. So I can change things on the DIRAC side, change things on the
external project side, and commit changes to the respective
repositories.

Before you modify external sources switch to the branch that you want to
commit to. In the initial state external source repositories do not
point to any branch (detached HEAD state).

Here is a typical work-flow:

    $ cd dirac/external/xcint
    $ git checkout master          # switch to the branch that you want to commit to
    $ vi xcint-source.F90          # edit xcint source
    $ git commit xcint-source.F90  # commit
    $ git push origin master       # push to xcint

    $ cd dirac
    $ git commit external/xcint    # move the DIRAC reference to xcint
    $ vi dirac-source.F90          # edit DIRAC source
    $ git commit dirac-source.F90  # commit
    $ git push origin master       # push to DIRAC

## External project tracks different remotes on different DIRAC branches

This sounds involved but it might happen more often than you think.
Let's again take PCMSolver as an example. The version released in DIRAC
lives on GitHub and is publicly accessible, the remote URL being
[git@github.com:PCMSolver/pcmsolver.git](git@github.com:PCMSolver/pcmsolver.git).
On some other DIRAC branches though, we would maybe like to work with
the development version of the module. This version is not publicly
accessible and resides at another remote URL.

Each DIRAC branch has its own .gitmodules file, correctly configured to
track the right remote. When switching DIRAC branches though, the same
switch of remotes does not happen for submodules. To be more explicit,
assume you have `DIRAC_branch-public` tracking GitHub and
`DIRAC_branch-private` tracking repo.ctcc.no. Moreover, assume you are
currently working on `DIRAC_branch-public`. As explained
[here](http://blogs.atlassian.com/2013/03/git-submodules-workflows-tips/#tip1),
the correct workflow to switch from one to the other is:

    $ git checkout DIRAC_branch-private
    $ git submodule sync

The `git submodule sync` command synchronizes submodules' remote URL
configuration setting to the value specified in .gitmodules (taken from
[this
page](https://www.kernel.org/pub/software/scm/git/docs/git-submodule.html))

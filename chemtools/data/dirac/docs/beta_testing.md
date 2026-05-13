orphan  

# Checklist for beta-testers

## Verify that the release installs correctly

Check out the release branch:

    $ git checkout release-25  # adapt name of branch

Try to configure and build the code:

    $ ./setup [--fc=ifx --cc=icx --cxx=icpx]

Please report problems. Pay attention whether math libraries are
correctly detected.

## Run the test set

Of course one should test the code regularly: to check for errors in
your new implementations that may break functionality in other parts of
the code, and to check that modifications that you pulled from the main
repository work on your machine. The latter tests should typically not
reveal problems, because the code is daily and nightly tested on
different architectures, but one is of course never sure.

The easiest way to test is to type ctest inside the build directory.
This will run a standard test set (without additional ctest parameters)
and check for errors. You may also run a subset by specifying regular
expressions: e.g. :

    $ ctest -R dft 

where the "-R" means regular expression, so this will run all tests that
contain the substring "dft".

Since tests are provided with descriptive tags (labels), you can run
group of tests based on label. For example, this command selects only
short tests from the whole suite: :

    $ ctest -L short 

Good approach is to run the (selected) test suite with the full CDash
report, so that other developers can see the actual status (before
accepting code change): :

    $ ctest -jN  (-L short) -D ExperimentalUpdate -D ExperimentalConfigure -D ExperimentalBuild -D ExperimentalTest -D ExperimentalSubmit

You can also choose to run the test set, for instance, using the
extracted tarball (see further up), and with simpler command: :

    $ ./setup [--flags]
    $ cd build
    $ ctest -jN -D Experimental

The test results also appears automatically on the [DIRAC
Dashboard](https://testboard.org/cdash/index.php?project=DIRAC). And we
can all see which tests fail and why.

## Verify the author list

Verify the author list below the logo in the output. Check versions.
Check typos.

## Run tutorials from the website

Verify that they actually run.

## Verify the manual and other documentation

Correct typos and mark sections that are wrong or not clear with a
warning:

    .. warning::

       Something wrong here! (replace this text)

This will create a red box with the warning inside.

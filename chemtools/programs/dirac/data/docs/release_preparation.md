orphan  

# Release preparation

## Feature freeze

Two months before the release (example: DIRAC21), a release branch
(example: release-21) is created from master. This is a feature freeze
and from this moment on the release branch ideally only receives
cosmetics and bugfixes, no major new features, no merges from master or
other branches (except cherry-picks; more about it later). You as
developer are responsible to commit or merge the to-be-released code to
master, before the release branch is created.

## Can I commit bleeding edge code to master after the feature freeze?

Yes you can! This is one of the reasons we have the release branch and
exactly for this reason we will never merge from master to the release
branch, only from the release branch to master.

## How to check out the release branch (example release-21 for DIRAC21)

Check out the release branch:

    $ git checkout release-21

Now you have it next to master (verify this):

    $ git branch

You can switch back to master:

    $ git checkout master

And then switch back to the release branch:

    $ git checkout release-21

## How to commit changes and bugfixes that are relevant for the released code

Commit all such changes to the release branch, not to master. This way
no commit or bugfix will get lost. Please do *not* transfer commits from
the release branch to master manually. This is not only unnecessary work
but it is harmful (conflicts)! Again, do not commit to master, commit to
the release branch:

    $ git checkout release-21
    $ git pull                      # update the local release-21 branch
    $ git commit                    # commit your modifications
    $ git push                      # push your changes to release-21 branch
    $ git checkout master           # switch to the master branch
    $ git merge --no-ff release-21  # merge changes to master, fix conflicts if any
    $ git push                      # push your changes to master

Note that by merging your changes to master you might get conflicts. Git
points them out clearly. In such cases open conflicting file(s) and fix
discrepancies inside manually. They are marked with "\<\<\<\<\<\<" and
"\>\>\>\>\>\>" strings.

What if you accidentally committed something to master which belongs to
the release? In this case do *not* merge master to the release branch,
but rather cherry-pick the individual commit(s) to the release branch:

    $ git checkout release-21
    $ git cherry-pick [hash]        # with [hash] that corresponds
                                    # to the commit

## How to exclude code from being released

Code that is part of `master` is meant to be part of the next release.
There is no mechanism anymore to remove code from `master` with
preprocessing or other scripts. This mechanism got removed when we
changed the license to LGPL.

- Do not send merge requests towards `master` for code that you do not
  with to be released.
- As a reviewer, please raise a question if a merge request towards
  `master` looks like something that should not be released quite yet.
- We will not release any of your private development branches on
  <https://gitlab.com/dirac/dirac-private>. Place your code there if you
  do not wish to release it quite yet.
- By sending a merge request towards `master` you signal that this code
  is meant to be released.
- Do not send merge requests towards `master` of code written by others
  without asking them.

## Check copyright headers and version

1.  Verify copyright headers.
2.  Verify logo (list of authors).
3.  Bump the version (the version number is kept in file VERSION; but
    also git grep -i for "orphaned" version numbers in the source code).

## Copyright statements and markers

In the development source code you can find the following copyright
statement markers:

    !dirac_copyright_start
    !dirac_copyright_end
    !dalton_copyright_start
    !dalton_copyright_end
    \* dirac_copyright_start \*
    \* dirac_copyright_end \*
    \* dalton_copyright_start \*
    \* dalton_copyright_end \*

They are there so that we do not have to update copyrights manually. You
can update them with the `maintenance/update_copyright.py` script. This
script will overwrite whatever is between these markers so it is a bad
idea to modify text between the markers manually.

## Create the final tarball

In principle we could just let people download the tarball directly from
GitLab. However, since we include external code through Git submodules,
this has the following disadvantages:

- Users would require Git
- Tarball is not self-contained and not fully reproducible since we have
  little control over the external repositories and whether they won't
  disappear in few years

For this reason we create a self-contained tarball using the
`maintenance/create-release-tarball.sh` script and this is then the
tarball that we upload to Zenodo.

We then direct users to download the tarball from Zenodo and not from
GitLab for two reasons:

- More self-contained and reproducible
- We get download metrics from Zenodo which we would not get from GitLab

orphan  

# Patch preparation

Patches are *not* intended for new features, rather bug fixes and
polish.

## How to check out the release branch (example release-21 for DIRAC21)

Check out the release branch:

    $ git checkout release-21

Now you have it next to master (verify this):

    $ git branch

You can switch back to master:

    $ git checkout master

And then switch back to the release branch:

    $ git checkout release-21

## How to prepare the patch

1.  Bump the version (the version number is kept in file VERSION). Also
    git grep -i for "orphaned" version numbers in the source code),
    e.g.:

        $ git grep "21\." src/

2.  Verify that CHANGELOG.rst is up to date (you may want to check
    history since previous version using git log).

At this point, you will have to create a branch in order to commit these
changes, e.g.:

    $ git checkout -b saue_patch21.1
    $ git add CHANGELOG.rst VERSION
    $ git commit
    $ git push

Next create a merge request to merge these changes into the current
release branch.

Once the merge request has been accepted, we are now ready to download
the tarball from GitLab and upload it to Zenodo (later we will automate
this further).

As an example, for the 21.1 patch, visit
<https://gitlab.com/dirac/dirac/-/tree/release-21> and click on the
Download icon next to the Clone button. Rename the tarball to
DIRAC-21.1-Source.tar.gz.

Before uploading the tarball, it is useful to compare to the existing
tarball. We shall use the powers of git to do this. First unpack the
tarball we just created, e.g.:

    $ tar xvf DIRAC-21.1-Source.tar.gz

Next, go to the DIRAC homepage and click on the download button. This
will take you to the proper zenodo page. From here you download the
current release tarball. We unpack it and create a temporary git
repository of it, e.g.:

    $ tar xvf DIRAC-21.0-Source.tar.gz
    $ cd DIRAC-21.0-Source/
    $ git init
    $ git add .
    $ git commit

We have now staged and committed the entire content of the current
release tarball. This information is stored in the .git subdirectory.
The trick is now to copy this over to the directory containing the
contents of the patch tarball, e.g.:

    $ cp -R .git $HOME/Dirac/build/DIRAC-21.1-Source
    $ cd $HOME/Dirac/build/DIRAC-21.1-Source
    $ git status
    $ git diff

This directory has now been activated as git repository, allowing us to
check what files have been modified (git status) and in what manner (git
diff). If things are okay, we are ready to upload the patch tarball to
the zenodo page of the current release.

orphan  

# Working with multiple remotes

## There are two repositories

With the move to LGPL, the repository was split into two:

- Public repository: <https://gitlab.com/dirac/dirac/>
  - Contains `master` and `release-*` branches
  - Everything that is on `master` is meant to become the next release
- Private repository: <https://gitlab.com/dirac/dirac-private/>
  - Nothing is public, you decide when to share
  - No `master` branch, no `release-*` branches

## How to start a new private development branch

Before starting a new private development branch on
<https://gitlab.com/dirac/dirac-private/> you want to start from the
most up-to-date public `master`.

We assume you have cloned from `git@gitlab.com:dirac/dirac-private.git`
and your `origin` therefore refers to
`git@gitlab.com:dirac/dirac-private.git`.

You can verify that this is the case:

    $ git remote -v

    origin  git@gitlab.com:dirac/dirac-private.git (fetch)
    origin  git@gitlab.com:dirac/dirac-private.git (push)

To start from public `master`, you first want to add a new remote
pointing to the public repository. You can call it `upstream`, or give
it a different name (e.g. `public`):

    $ git remote add upstream https://gitlab.com/dirac/dirac.git
    $ git remote -v
    $ git fetch upstream

The above fetches all commits from `upstream` which you did not have yet
locally but will not merge anything.

Now you can create a new local branch `myownbranch` from `master`:

    $ git branch myownbranch upstream/master

And then push the new branch to the private repository (`origin`):

    $ git push origin -u myownbranch

## Updating your development with upstream changes

This is something you will want to do relatively often, maybe one per
month.

To merge the public `master` into your development branch `myownbranch`,
you first want to add a new remote pointing to the public repository,
like described further above:

    $ git remote add upstream https://gitlab.com/dirac/dirac.git
    $ git remote -v
    $ git fetch upstream

The above fetches all commits from `upstream` which you did not have yet
locally but will not merge anything yet.

We merge in a separate step:

    $ git checkout myownbranch
    $ git merge upstream/master

Your local branch `myownbranch` contains now commits from public
`master`.

## Contributing your changes upstream

This describes how to get your changes from your own development branch
into public `master`. Typically this happens once the feature is
implemented, and submitted for publication and you want it to become
part of the next DIRAC release.

- Push the branch to <https://gitlab.com/dirac/dirac.git>
- Create merge request towards `master`
- This code will be in the next release (no more code removal in the
  release process)
- Then delete the branch on <https://gitlab.com/dirac/dirac-private/>
  (or keep it, up to you)

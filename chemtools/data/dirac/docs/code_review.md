orphan  

# Code review

## Master and release branches are protected

This means that nobody can push to them directly, all contributions have
to be submitted using merge requests (see below).

## How to submit a merge request

- Before you do anything, create a new branch from the branch that you
  wish to modify.
- Implement your change and commit to the new branch.
- Push your new branch.
- Go to <https://gitlab.com/dirac/dirac-private> and click on the blue
  button "Create merge request" (top right).
- Describe the change unless it's evident from the commit message.
- Check the box next to "Remove source branch when merge request is
  accepted".
- Make sure it is directed to the right branch.
- Click green button "Submit merge request".
- If you commit does not modify code execution you may use the command
  "git push -o ci.skip" to avoid invoking runners.

## How to ask for a code review

- Either assign to somebody.
- Or mention somebody via their @username in the discussion. Then they
  get notified.
- Or let the merge request sit and somebody will pick it up.

## How to do code review

- Be polite.
- Check that code is submitted to the right branch.
- Check that commit messages are clear.
- Changelog entry needed?
- Documentation needed?
- Tests preserved and new functionality tested?
- Wait until the test runner reports before merging or click "merge when
  pipeline succeeds".
- If test runner result is red, investigate what happened.

## What if you committed locally to master or a release branch?

Then you cannot push directly but do not worry and follow these steps:

- Create a new branch from the branch that you changed and tried to
  push.

- Push the newly created branch to
  <https://gitlab.com/dirac/dirac-private>.

- Create a merge request and point it to the branch you originally meant
  to modify.

- Finally clean up your local branch by rewinding to either
  `origin/master` or `origin/release-N` :

      $ git checkout master
      #WARNING! next step will delete commits that came after origin/master
      $ git reset --hard origin/master

If you do not clean up your local master or release branch, it can start
to diverge next time you pull since your local history may be different
from remote history.

## How to get changes from a release branch to master?

Visit <https://gitlab.com/dirac/dirac-private/-/merge_requests/new>,
select the release branch as "Source branch", verify that "Target
branch" is the <span class="title-ref">master</span> branch, then click
the green button "Compare branches and continue".

Scroll down and have a look at the commits whether the changes to be
merged are what you expected. Abort if this looks unexpected. You can
also browse the "Changes" to see what is part of this merge request.

Then fill out the rest (title, description) like any other merge
request. Note that you cannot and should not mark the source branch to
be deleted upon merge and that's good, since we want to keep the release
branches forever (these branches are protected against deletion).

Finally green button: "Submit merge request".

## How to create a new release branch?

If you create a new release branch (e.g. "release-25" in the year 2025)
locally using the terminal, you will experience that the repository will
refuse to accept your `git push origin -u release-25`. This is because
we protect `master` and `release-*` against (accidental) direct pushes.

But in this case it will be in your way. You (repository/group admins)
can temporarily unprotect the matching wild-card pattern at GitLab
(Settings -\> Repository -\> Protected branches).

## Further reading

- <https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html>
- <https://about.gitlab.com/2017/03/17/demo-mastering-code-review-with-gitlab/>

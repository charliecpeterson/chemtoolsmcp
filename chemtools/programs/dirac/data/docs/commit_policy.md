orphan  

# Commit policy after the release is out

Since DIRAC12 we have two novelties:  
- We have put a patching/updating mechanism in place for patches and
  bugfixes.
- We have a versioned documentation.

Both the patching and the documentation **will** break down if we do not
follow the guidelines outlined here.

1.  Commit patches and bugfixes (large or small) that relate to DIRAC14
    to the release-14 branch and not to master.
2.  Commit changes to the DIRAC14 documentation to the release-14 branch
    and not to master. In the days and weeks following the release we
    will modify the DIRAC14 documentation a lot. For this simply commit
    to release-14. The documentation is generated from that branch. This
    branch continues to live until we release the next version and
    discontinue support of DIRAC14.

To summarize the above two points: all commits that go to master will
**never** make it to this documentation:
<http://diracprogram.org/doc/release-14/> and will **never** be part of
patches to DIRAC14 if you don't carefully plan your commits, the patches
**will** get lost (from the user's point of view).

**All** commits that go to release-14 can be integrated to master. This
means that all changes to release-14 will get integrated into the main
line development. **Never** merge from master to release-14.

If, by accident, you committed to the wrong branch, you can
`git cherry-pick` the commit to the release-14 branch.

3.  Do not change the file VERSION. This file is changed when we release
    a new patch. But we will not release a patch from each commit to
    release-14

To summarize, ask yourself the following questions:  
- Is this code or documentation change useful for DIRAC14 users? If yes,
  commit to release-14.
- Is this code or documentation change only useful for future versions
  of DIRAC? If yes, commit to master or some other branch.

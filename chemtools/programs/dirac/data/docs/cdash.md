orphan  

# Nightly testing dashboard

Please contribute to the [DIRAC
Dashboard](https://testboard.org/cdash/index.php?project=DIRAC). All you
need is a desktop which is idle and lonely over night or a cluster with
some free CPU hours, a deploy key, and a script that you launch via
crontab (desktop) or that you submit to the batch system (cluster).
Below we show you how to do that.

# Creating a deploy key

A deploy key is a single-purpose ssh-key without a passphrase which is
not attached to your repository user account and which only can read
(clone) and not write (push).

We create one with the following command:

    $ ssh-keygen -t rsa -b 4096

We need to specify its name (for instance
"/home/user/.ssh/nightly_rsa"):

    Enter file in which to save the key (/home/user/.ssh/id_rsa): /home/user/.ssh/nightly_rsa

And we use an empty passphrase (we just hit enter twice):

    Enter passphrase (empty for no passphrase):
    Enter same passphrase again:

And we are done with the key:

    Your identification has been saved in /home/user/.ssh/nightly_rsa.
    Your public key has been saved in /home/user/.ssh/nightly_rsa.pub.

# Uploading the public key to GitLab

Now upload the public key to gitlab. If are not sure how to do this,
please consult the documentation at <https://docs.gitlab.com/ee/ssh/>.

# Writing a CDash script

A CDash script can in principle be as brief as this:

    $ ./setup [--flags]
    $ cd build
    $ ctest -jN -D Nightly --track master

The result of appears automatically on the [DIRAC
Dashboard](https://testboard.org/cdash/index.php?project=DIRAC).

However, typically we need a little bit more because we want it to clone
the code in an unsupervised fashion or because we want to also test the
tarball creation. This is the script that Radovan uses:
<https://gitlab.com/dirac/dirac-private/blob/master/maintenance/cdash/cdash_crontab.sh>
This script takes care of using a specific deploy key and integrates
nicely with the crontab example given below.

# Launching your script automatically via crontab

Edit your crontab with:

    $ crontab -e

Here is an example:

    # m  h  dom mon dow command
      00 04 *   *   *   /home/user/nightly/cdash_crontab.sh -b 'master' -j 4  -c 'Intel' -n 'Intel-Mac' -t 'master' -k /home/user/.ssh/nightly_rsa
      55 04 *   *   *   killall -9 dirac.x

You can verify your crontab file changes by :

    $ crontab -l

You can also verify that a crontab file with your username exists by:

    $ sudo  ls -l /var/spool/cron/crontabs

# CDash tracks

We currently have the following tracks installed: master, release-14,
release-15, Nightly, and Experimental. It is no problem to install
additional tracks - just ask Radovan. The tracks are there to organize
the various builds. So instead of submitting a build to the track
"Nightly" with the build name "master-foo" it is clearer and more
concise to submit a build "foo" to the track "master". We recommend to
use "master" or "release-15" depending on the context and not to use
"Nightly" or "Experimental". But it is OK to use "Nightly" or
"Experimental" to calibrate your scripts.

The semantic difference between "Nightly" and "Experimental" is that
"Nightly" corresponds to a Git hash closest to a certain configurable
time whereas "Experimental" corresponds to "now". This means that for
extremely actively developed projects it may be important to test all
nightly tests using a well defined version and then one should prefer
"Nightly" over "Experimental".

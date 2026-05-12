orphan  

# What is the difference between DFCOEF and DFPCMO?

Both files hold the MO coefficients (and some other information) and can
be used to restart SCF calculations.

DFCOEF is always generated. To generate DFPCMO, you need the
`GENERAL_.PCMOUT` keyword.

To see the difference between the two files, run a calculation with:

    **GENERAL
    .PCMOUT

and save both files from the scratch directory:

    $ pam ... --get "DFCOEF DFPCMO"

and then inspect both files. You will see that only DFPCMO is
human-readable. Both contain the **same** information. The DFCOEF file
is machine-readable and machine-dependent which means that DIRAC cannot
restart from a DFCOEF file generated on a different architecture but you
can always restart from DFPCMO on any machine. The advantage of the
DFCOEF file is that it is smaller in size (but with gzip you can
compress DFPCMO files to the size of DFCOEF files). The advantage of
DFPCMO files is that you can read them yourself (try to understand what
the numbers mean) and use them in external scripts and programs.

You do not need any keywords for restarting, simply copy DFPCMO or
DFCOEF to the scratch directory and DIRAC will restart automatically.

Try it. It is important that you know how to restart. This will save you
many troubles and CPU hours.

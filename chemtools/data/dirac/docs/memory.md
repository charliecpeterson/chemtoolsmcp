orphan  

# Help with memory problems

DIRAC will allocate the following memory segments:

- Static allocation (approx. 130 MB; you can verify it using "make
  info")
- WORK array for F77 "dynamic" allocation (specified by `./pam --mb`)
- Additional F90 dynamic allocations (depends on the calculation)

## MEMGET errors

They look like this:

    --- SEVERE ERROR, PROGRAM WILL BE ABORTED ---

    Date and time (Linux) : Fri Nov 26 10:48:19 2010
    MEMGET ERROR, insufficient work space in memory

This means that you need to increase the WORK array (increase `--mb`).

## MEMCHK errors

They look like this:

    MEMCHK ERROR, not a valid memget id in work(kfree-1)
    Text from calling routine : PSIDHF.DHFSCF (called from MEMREL)
    KFIRST,KFREE,IALLOC =         1         2         1
    found memory check :                        0
    expected           :               1234567890

This means that you have found a bug in the code. Please contact the
developers.

## Other memory allocation errors

If you see a error message about memory problems in the queuing system
output (and not in the DIRAC output) it typically means that there is
not enough memory for F90 dynamic allocations.

You need to increase the memory limit that you specify to the queuing
system or perhaps decrease `--mb` (if possible).

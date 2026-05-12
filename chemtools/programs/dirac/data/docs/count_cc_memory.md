orphan  

# Counting the RelCC memory consumption

It is of great importance to know the exact number of the memory needed
for a high memory and time demanding RelCC job.

## SCF

The first step is the SCF run with saving molecular orbitals, preferably
as DFPCMO text file. Afterwards, using obtained molecular orbitals,
prepare the active space for MOLTRA and RelCC.

## MOLTRA & RelCC

Equip the MOLTRA input with the `MOLTRA_.NO4IND` keyword and RelCC input
with the `RELCC_.COUNTMEM` keyword. Run SCF part with previously saved
converged molecular orbitals, with one iteration. The job passes quickly
through MOLTRA integrals transformations as it produces only two-index
quantities in the MRCONEE file.

In the RelCC module, the program counts the memory need and leaves.

## Example

Demonstration of this feature is in the DIRAC *test/count_cc_memory*,
see `test  <test>`

We found the almost minimal setting - via the *--ag* parameter flag -
for the memory counting run:

    pam --noarch --gb=0.20 --ag=0.366 --inp=H2O.ae4z.x2c.scf_countmem-relcc.inp  --mol=H2O.xyz --put "DFPCMO.H2O.x2c.ae4z=DFPCMO" 

Input files are
`H2O.ae4z.x2c.scf_countmem-relcc.inp  <H2O.ae4z.x2c.scf_countmem-relcc.inp>`,
`H2O.xyz  <H2O.xyz>`, producing the output
`file  <H2O.ae4z.x2c.scf_countmem-relcc_H2O.out>`.

We are to look the "Peak memory usage" to get reasonable estimate of
dynamic memory:

    Predicted RelCC memory demand:           0.335 GB
    Peak memory usage    (Gb) :   0.363

Next step is the sharp run based on the previous RelCC memory conuting.
The serial job needs at least 0.366 GB, rounding to 0.4 GB of the total
memory:

    pam --noarch --gb=0.20 --ag=0.366 --inp=H2O.ae4z.x2c.scf_relcc.inp  --mol=H2O.xyz --put "DFPCMO.H2O.x2c.ae4z=DFPCMO"

We also perform the parallel run, which consumes at least 16\*0.366 =
5.856 GB of the memory, rounding to 5.9 GB. This amount of memory must
be free at the running machine equiped with at least 16 CPUs:

    pam --noarch --mpi=16 --gb=0.20 --ag=0.366 --inp=H2O.ae4z.x2c.scf_relcc.inp  --mol=H2O.xyz --put "DFPCMO.H2O.x2c.ae4z=DFPCMO"  

Have a look at the `output  <H2O.ae4z.x2c.scf_relcc_H2O.out>` file. We
search there the peak memory usage which is cca 99.2% of the *--ag*
parameter.

    Predicted RelCC memory demand:        0.335 GB
                Peak memory usage:        0.363 GB

Therefore in practical RelCC memory demanding calculations watch for the
**Peak memory usage** value and set the **--ag** parameter accordingly.
This may take few iterations, especially when you want to find the real
memory minimum. Don't forget to multiply your memory amount with the
number of threads in parallel calculations.

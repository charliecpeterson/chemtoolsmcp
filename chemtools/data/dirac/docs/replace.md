orphan  

# Replace strings in input files

The "pam --replace" is a fantastic option which we encourage you to use
it. It replaces any string in input files by user defined string.

Try this:

    **DIRAC
    .WAVE FUNCTION
    **HAMILTONIAN
    .hamiltonian
    **WAVE FUNCTION
    .SCF
    *END OF

and this:

    DIRAC
    .
    .
    C 1
    10. 1
    Ne 0.0 0.0 0.0
    LARGE BASIS basis
    FINISH

now run it with:

    pam --replace hamiltonian=LEVY-LEBLOND --replace basis=cc-pVDZ

it will write to the output file:

    inp_mol_hamiltonian=LEVY-LEBLOND_basis=cc-pVDZ.out

you get the point: you can now run whole tables with just two input
files!

A nice application of this feature is to make a PES scan, written as a
script:

    for r in `seq 1.00 0.05 2.00`; do
        pam --replace distance=$r ..
    done

# Transfer renamed files

The pam scripts enables putting/getting renamed files. Thanks to this
feature we are able to save DIRAC working files under own original names
and ensure their renaming upon moving them to and from the scratch
directory.

This feature is great for running many calculation at the same time
without risking that files will be overwritten by concurrent runs.

Here we extract the DFCOEF file from the scratch directory and save it
as DFCOEF.tl2.v3z in the home directory:

    pam --get "DFCOEF=DFCOEF.tl2.v3z"

In next run we copy files to the scratch directory (where calculations
occur) with their proper names:

    pam --put "DFCOEF.tl2.v3z=DFCOEF"
    pam --copy="/home/milias/my_scratch/Tl2_fscc02/MDCINT.18corr_el.v3z=MDCINT" 

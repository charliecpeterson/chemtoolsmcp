orphan  

# Testing basis set files in DIRAC

The DIRAC program suite hosts more than different 300 basis set input
files. These are either relativistic basis sets (all-electron, ECP) or
nonrelativistic basis sets, collected from other sources (Dalton, EMSL
etc). Therefore it is necessary to have an (automated) tool for testing
all basis set files distributed with DIRAC.

Currently, there are few tests focused at checking basis set files:

> - test/basis_input_scripted (only reads basis sets, but all of them)
> - test/basis_input (reads few basis sets)
> - test/basis_contraction (run few SCFs in contracted basis)
> - test/basis_automatic_augmentation (run few SCFs in augmented basis)

## Basis set files testing

The test **test/basis_input_scripted** is dedicated for automated
checking of (almost all) basis set sets files sitting in the DIRAC
directory.

### How it works

The tests performs reading of multiple basis set files, and merely
checks that the input parser does not choke on picked basis set. No
reference outputs are involved.

The principal Python script
`test <../../../test/basis_input_scripted/test>` is employing the
runtest library (<https://runtest.readthedocs.io>).

When the typical string sequence is found in the basis set file, the
script extracts the proton number of a given element. Script then
launches run(s) with the
`input <../../../test/basis_input_scripted/input_test_basis.inp>`,
`molecule <../../../test/basis_input_scripted/element.mol>` files and
with pam parameters containing selected basis set name and element's
proton number as strings for replacements by Python.

### Default run

Test's default run checks all basis set files in given day once per
month. This is achieved by running through basis set directories where
each file with several basis sets inside is read. The total number of
individual runs makes more than 5000, and it takes about 20 minutes.

On other days the test makes empty run.

### Examples of usage

User can specify by environment variables (BSFILE, ZELEM, RANDOM_BSF_Z)
on what basis files/elements to run.

Read all elements basis sets from selected files:

    export BSFILE="basis_ecp/ECPDS28SDFSF basis/dyall*"
    ./test -b <BUILDIR> 

Run only selected elements in selected basis set file(s):

    export BSFILE="basis_ecp/ECPDS28SDFSF"
    export ZELEM="30 35"
    ./test -b <BUILDIR> 

Run all elements in two randomly selected basis set files:

    export RANDOM_BSF_Z="2"
    ./test -b <BUILDIR> 

Run three elements in three randomly selected basis set file:

    export RANDOM_BSF_Z="3 3"
    ./test -b <BUILDIR> 

### Nonworking cases

Because of many basis set files (more than 300) one can not expect all
of them working properly.

There are cases of wrong basis sets. For example, the proton number can
not be extracted from "basis_dalton/aug-pc-0_emsl". For more info about
that, see the DIRAC
[issue](https://gitlab.com/dirac/dirac-private/issues/250).

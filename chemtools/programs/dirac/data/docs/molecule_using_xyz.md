orphan  

# How to specify the molecule and basis set using xyz files

The actual xyz file only contains the number of centers, the atoms, and
the corresponding coordinates (by default in angstroms). It could look
like this one:

    6
    Methanol, identifying the alcoholic proton as H1
    C      -0.000000010000       0.138569980000       0.355570700000
    O      -0.000000010000       0.187935770000      -1.074466460000
    H       0.882876920000      -0.383123830000       0.697839450000
    H      -0.882876940000      -0.383123830000       0.697839450000
    H      -0.000000010000       1.145042790000       0.750208830000
    H1     -0.000000010000      -0.705300580000      -1.426986340000

The first line is the number of atoms. The second line can be blank or
an arbitrary comment. It is not read by the program. If you ommit this
line, then your molecule will be wrong, one atom will be missing! The
following lines contain atom types and coordinates. The first entry on
each line is the name of the atom. Any string up to eight characters
will do, but if the name is present in the periodic table DIRAC will
recognize the charge and can find the corresponding basis set.

From the xyz file DIRAC cannot know what basis set you want to use. For
this reason, basis set and symmetry specification is then done in the
input file using the keywords given below. This is in contrast to the
traditional mol files. This means that the keywords below do not make
sense when combined with a traditional mol file.

## \*BASIS

Here the basis set information is specified. .DEFAULT specifies the
default large component basis set for all atoms. This can be modified
for specific atom by the .SPECIAL keyword. For example, if we use the
methanol.xyz file listed above, the following input specifies a cc-pVDZ
on carbon and the 3 hydrogens labeled H, while oxygen and H1 will be
treated with a cc-pVTZ basis:

    **MOLECULE
    *BASIS
    .DEFAULT
     cc-pVDZ
    .SPECIAL
     O BASIS cc-pVTZ
    .SPECIAL
     H1 BASIS cc-pVTZ

## \*CENTERS

Here the physical properties of the centers in our molecule can be
specified. This is useful if the name of the center differs from the
IUPAC element name or in case we want to use nuclei with fractional
charges. This is controlled by the keyword nucleus that can be repeated
as often as is needed. For the geometry file methanol.xyz we need to
specify:

    *CENTERS
    .NUCLEUS
     H1 1.0

Another reason to change the charge of the nucleus is to be able to run
counterpoise calculations in which a ghost basis set is to be placed
without adding a charge. This can be achieved by defining the ghost
center as (for instance) O.Gh and then putting the nuclear charge to
zero using:

    *CENTERS
    .NUCLEUS
     O.Gh 0.0 8.0

the first number is the charge to be used, the second corresponds to the
element number and is used to search the basis set library. In this
example you can therefore simply specify the basis set name using the
*DEFAULT* or *SPECIAL* keywords and the program will then find the
corresponding oxygen basis.

## \*COORDINATES

Here you can specify the units to be used when reading the coordinates.
If nothing is specified, angstrom is the default (this is then a real
xyz file). If we specify 'AU' we use atomic units instead:

    *COORDINATES
    .UNITS
    AU

We can also specify our own unit, by typing the name and the conversion
factor when going from this new unit to atomic units:

    *COORDINATES
    .UNITS
    nm 18.8972

Now all coordinates are entered in nanometer. If we add 'A' at the end,
we can enter the new unit in angstrom:

    *COORDINATES
    .UNITS
    nm 10.0 A

## \*SYMMETRY

With this keyword we can control symmetry recognition. To turn symmetry
detection off and to force C1 symmetry use:

    *SYMMETRY
    .NOSYM

# Including ghost centers and point charges

In some situations you may want to add a ghost center, for instance in
counterpoise calculations (see `bsse`) or to lower the symmetry detected
by DIRAC (see `case_CmF`). An example of the latter case is to take out
inversion symmetry for an atom:

    2

    Ne    0.0 0.0 0.0
    foo   0.0 0.0 10.0

We give a non-standard name to the ghost center, so that DIRAC does not
find a nuclear charge. This has to be accompanied by telling DIRAC not
to add a basis to the ghost center:

    **MOLECULE
    *BASIS
    .DEFAULT
    dyall.2zp
    .SPECIAL
    foo NOBASIS

In a counterpoise calculation, on the other hand, you want the center to
carry a basis set. For a ghost neon atom you would then specify:

    **MOLECULE
    *CENTERS
    .NUCLEUS
    foo 0.0 10.0

As already explained, the first number on the last line is the charge of
the center, and the second the nuclear charge used to find the basis. If
you actually want to add a point charge, let us say of charge -2.3, you
can use the xyz-file above, but now you want charge, but not basis, and
write:

    **MOLECULE
    *BASIS
    .DEFAULT
    dyall.2zp
    .SPECIAL
    foo NOBASIS
    *CENTERS
    .NUCLEUS
    foo -2.3

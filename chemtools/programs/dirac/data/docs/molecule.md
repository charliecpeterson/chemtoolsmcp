orphan  

<div class="index">

\*\*MOLECULE

</div>

# \*\*MOLECULE

## Specification of molecular symmetry and basis sets

This section is needed for calculations using the simple .xyz input
format for the molecular structure. For calculations with the .mol input
format, basis set information is to be specified together with the
atomic coordinates.

<div class="index">

\*CHARGE

</div>

# \*CHARGE

## Molecular charge

Allows explicit specification of the molecular charge.

<div class="index">

.CHARGE

</div>

### .CHARGE

Sets the value of the molecular charge. For example:

    .CHARGE
    -2

will set the molecular charge to -2. The default value is 0.

<div class="index">

\*SYMMETRY

</div>

# \*SYMMETRY

## Molecular symmetry

Allows explicit specification of the molecular symmetry that is to be
used in the calculations. This will then overrule the automatically
detected point group.

<div class="index">

.NOSYM

</div>

### .NOSYM

Switches off all use of symmetry (this is currently the only option).

<div class="index">

.LINEAR

</div>

### .LINEAR

Forces to use the linear symmetry even for the one-atom case. When you
want to remove the inversion symmetry (e.g., an atom under an external
electric field), a ghost center must be included.

<div class="index">

\*BASIS

</div>

# \*BASIS

## Basis set

This section should contain the keywords needed to specify the basis
set. The minimum specification is a default basis set for all atoms in
the system.

<div class="index">

.DEFAULT

</div>

### .DEFAULT

The name of the basis set file that is to be searched for basis sets.

Example: use the double zeta basis sets optimized by Dyall:

    .DEFAULT
    dyall.cv2z

<div class="index">

.SPECIAL

</div>

### .SPECIAL

Special basis sets for specific atom types.

The simplest is to specify an alternative basis set library file. One
then lists the atomtype, the keyword BASIS, and the name of the
alternate file.

Example: use Dunnings cc-pVDZ basis for the fluorine atom:

    .SPECIAL
    F BASIS cc-pVDZ

Basis sets may also be explicitly typed using the EXPLICIT keyword and
the format that is also used in the .mol file.

Example: specification of a basis set for the atomtype HB:

    .SPECIAL
    HB EXPLICIT
        2    1    1
        4    0    3
       13.0100000
        1.9620000
        0.4446000
        0.1220000
        1    0    3
       0.7270000

Note that you may use different basis sets for the same element, in the
example above you may have most hydrogen atoms labeled as H and have a
few special ones labeled as HB.

We can also introduce point charges that have no basis set attached by
specifying the keyword NOBASIS after the atomtype.

Example: a point charge at the position of an oxygen:

    .SPECIAL
    O.PC NOBASIS

The keyword .SPECIAL can be repeated as often as you like, atomtypes not
listed as special will receive the default basis set.

Example: EuF3 with dyall set on Eu and the cc-pVDZ on the fluorides:

    **MOLECULE
    *BASIS
    .DEFAULT
    dyall.cv2z
    .SPECIAL 
    F BASIS cc-pVDZ

<div class="index">

\*CENTERS

</div>

# \*CENTERS

## Centers

This optional section allows modification of the properties of a center.
This makes it possible to specify fractional charges (for embedding
point charges) or zero charge (for ghost centers ised in counterpoise
calculations).

<div class="index">

.NUCLEUS

</div>

### .NUCLEUS

The atom type followed by the modified nuclear charge.

Example: a fractional charge of -0.6 for an oxygen point charge:

    .NUCLEUS
    O.PC -0.6

In case the center with the modified charge should contain a basis set
there is a second keyword to specify the real nuclear charge that is to
be used when searching the basis set file for the element-specific basis
set.

Example: an oxygen ghost center (zero charge):

    .NUCLEUS
    O.Gh 0.0 8.0

Like the .SPECIAL keyword described above also the .NUCLEUS keyword can
be specified as often as necessary. Note that all atomtypes should have
exactly the same name as they appear in the xyz file, it is thereby good
practice to use a logical naming scheme like the one used above (with a
suffix Gh to indicate ghost atoms and a suffix pc to indicate point
charges) but this is not enforced by the program.

<div class="index">

\*COORDINATES

</div>

# \*COORDINATES

## Coordinates

<div class="index">

.UNITS

</div>

### .UNITS

With xyz type input coordinates are expected to be defined in terms of
Angstrom units. If you want to need to deviate from this standard you
can alter the unit type by specifying the keyword UNITS.

Example: specification of coordinates in atomic units:

    .UNITS
     AU

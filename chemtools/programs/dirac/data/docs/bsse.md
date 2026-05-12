orphan  

# Basis set superposition error in the DFT

The basis set superposition error (BSSE, also known as counterpoise
correction, see Boys:1970) can be estimated by performing subsystem
calculations using ghost atoms which carry basis sets of the full
system.

This tutorial describes how to use ghost atoms in DIRAC and how to avoid
certain pitfalls when working with ghost centers. As an example we shall
consider the helium dimer, having linear symmetry $`D_{\infty h}`$. For
the counterpoise correction calculation one beryllium atom is replaced
by a ghost atom without charge, but carrying the full beryllium basis.
The symmetry is now reduced to $`C_{\infty v}`$. DIRAC can automatically
detect and exploit the linear symmetry, thus providing computational
savings. In the case of DFT there is, however, a complication: The ghost
center needs not only the basis of a beryllium atom, but also the same
numerical grid. This will force us to reduce the symmetry of the
molecular calculation to $`C_{\infty v}`$. When carrying out the
counterpoise correction we shall furthermore have to make sure that
atoms are in the same place as in the molecular calculation. Below we
see how to handle these issues, first with the use of xyz-files, next
using mol-files.

Another matter is that conventional DFT based on local functionals is
not a good choice for the description of a van der Waals complex like
the helium dimer. We shall therefore combine short-range DFT with MP2.

## Using mol-files

We start with the molecular calculation. Without consideration of
counterpoise we would use

<div class="literalinclude">

He2.mol

</div>

However, DIRAC will not detect the full molecular symmetry
$`D_{\infty h}`$, which is different from the counterpoinse system and
so the same numerical grid can not be used. We therefore add a ghost
center, with no basis, to break inversion symetry.

<div class="literalinclude">

He2Gh.mol

</div>

The input file for our calculation combining MP2 with short-range LDA
reads

<div class="literalinclude">

MP2srLDA.inp

</div>

Note a final touch: We use the keyword
<span class="title-ref">NORTSD</span> to avoid that our molecule is
moved (placing origin at center of mass) during symmetry detection. We
run the molecular calculation using:

    pam --inp=MP2srLDA --mol=He2Gh --get=numerical_grid

Note that we save the file containing the numerical grid. Naming the
ghost center 'Gh' assures that no grid is generated for it, as can be
seen from the output:

    Atom  Deg   Rmin        Rmax        Step size     #rp     #tp
    =============================================================
    He1     1   0.998E-05   0.157E+02   0.123E+00      99    8552
    He2     1   0.998E-05   0.157E+02   0.123E+00      99    8552

We now turn to the counterpoise calculation. We replace one helium atom
by a ghost atom, but this time we want a helium basis set for it. We
therefore specify

<div class="literalinclude">

HeGh.mol

</div>

DIRAC uses nuclear charge to find basis sets from libraries so we have
to add "Q=2.0" for the ghost atom in the mol-file to provide this
information. Note also that we again call the ghost atom 'Gh', but it
will get a grid since we import it from the previous calculation. The
corresponding menu file reads

<div class="literalinclude">

MP2srLDA_cp.inp

</div>

We have adjusted the electron number to that of the atom and also ask
for the import of the numerical grid. We run the calculation using:

    pam --inp=MP2srLDA_cp --mol=HeGh --put=numerical_grid

## Using xyz-files

write me...

### Symmetry recognition

If you leave the symmetry text in the molecule specification blank,
DIRAC will try to evaluate the symmetry of the system automatically.
Note that the symmetry recognition uses the atom types and coordinates
but not the respective basis sets. This means that if you use ghost
atoms with different basis sets, you should give the symmetry explicitly
or check the symmetry recognized by DIRAC.

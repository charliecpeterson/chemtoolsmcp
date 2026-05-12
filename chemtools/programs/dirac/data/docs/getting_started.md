orphan  

# Getting started with DIRAC

You have installed and tested DIRAC and now it's time to run your first
DIRAC calculation!

You have two possibilities to run DIRAC. The traditional uses two input
files: the `*.inp` file, which determines what should be calculated,
together with the `*.mol` file, which defines the molecular geometry and
the basis set. Alternatively you can use the `*.inp` file, file together
with a geometry file (`*.xyz` file,). In this case the `*.xyz` file
provides the nuclear coordinates and the `*.inp` file contains in
addition to the job description also the basis set information (see
`**MOLECULE`).

DIRAC writes its output to a text file, named after the molecule and
input file names. If this file already exists, for example from a
previous calculation, the old file is renamed to make place for the new
output. The python script `pam` that launches the main DIRAC executable
also writes some general information to standard output.

## Dirac-Coulomb SCF

Let's try it out and run our first DIRAC calculation with the following
minimal input (`hf.inp`) to calculate the Dirac-Coulomb Hartree-Fock
ground state (with the default [simple Coulombic
correction](http://dx.doi.org/10.1007/s002140050280) to approximate the
contribution from (SS\|SS) integrals:

    **DIRAC
    .WAVE FUNCTION
    **WAVE FUNCTION
    .SCF
    **MOLECULE
    *BASIS
    .DEFAULT
     cc-pVDZ
    *END OF INPUT

together with the following example geometry file (`methanol.xyz`):

    6
    my first DIRAC calculation # anything can be in this line
    C       0.000000000000       0.138569980000       0.355570700000
    O       0.000000000000       0.187935770000      -1.074466460000
    H       0.882876920000      -0.383123830000       0.697839450000
    H      -0.882876940000      -0.383123830000       0.697839450000
    H       0.000000000000       1.145042790000       0.750208830000
    H       0.000000000000      -0.705300580000      -1.426986340000

Now start the calculation by executing:

    pam --mol=methanol.xyz --inp=hf.inp

If everything works fine the results of the calculation will be written
to `hf\_methanol.out`. In addition a
[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
checkpoint file `hf\_methanol.h5` will be created (see `checkpoint` for
detailed information), To see the actual content of the file you may
execute:

    h5dump -n hf_methanol.h5

## Closed shell CCSD

Running CCSD(T) calculations is straightforward if the ground state of
the molecule can be well-described by a single closed shell determinant.
This is the case for many molecules.

The input requires the specification of a basis set (TZ or better is
recommended, but for this example we will take DZ to reduce the run
time). We take the inter halogen molecule ClF as an example and use the
default cut-offs for correlating electrons (include all valence
electrons with energy above -10 hartree) and truncation of virtual space
(delete virtuals above 20 hartree). The geometry file (`clf.xyz`) is
very simple and only requires to specify the coordinates. The program
will then identify the symmetry as $`C_{\infty v}`$:

    2
    ClF molecule at equilibrium distance taken from NIST
    Cl   0.0  0.0  0.0
    F    0.0  0.0  1.628

We specify the wave function type (SCF, followed by RELCCSD) and basis
set in the input file (`cc.inp`) to calculate the CCSD(T) energy with
the Dirac-Coulomb Hamiltonian again with the contribution from (SS\|SS)
integrals approximated by a simple Coulombic correction:

    **DIRAC
    .WAVE FUNCTION
    **WAVE FUNCTION
    .SCF
    .RELCCSD
    **MOLECULE
    *BASIS
    .DEFAULT
     cc-pVDZ
    *END OF INPUT

Now start the calculation by executing:

    pam --mol=clf.xyz --inp=cc.inp

If everything works fine the results of the calculation will be written
to `cc_clf.out`. In addition an HDF5 checkpoint file `cc_clf.h5` will be
created, as mentioned above.

You can suppress having the checkpoint file copied back to your home
directory using:

    pam --noh5

## Cheap relativistic calculations

Relativistic calculations are more expensive than non-relativistic ones,
but this is accentuated by the fact that relativistic effects are
associated with heavy elements, which also implies many electrons and
increased computational cost. However, as explained in Dubillard2006, we
can induce strong relativistic effects in molecules of light atoms by
playing with the speed of light. The starting point is the observation
that a diagnostic of relativistic effects is provided by the Lorentz
factor

``` math
\gamma = \frac{1}{\sqrt{1-(v/c)^2}},
```

where $`v`$ is the speed of the particle and $`c`$ is the speed of
light. In atomic units we have:

    * The speed of light :        137.0359992

as shown in the DIRAC output. Scalar relativistic effects are associated
with the high speeds of electrons in the vicinty of heavy nuclei. In
fact, for a one-electron atom the average speed of the *1s* electron in
atomic units is $`v_{1s}=Z`$. This explains why relativistic effects
become so pronounced for heavy atoms; what counts is the nuclear charge,
not the atomic mass. On the other hand, we see that relativistic effects
are sensitive to the ratio $`(v/c)`$, so one way to accentuate
relativistic effects is to *reduce* the speed of light.

Let us look at an example of this. We shall look at the geometry of the
water molecule as a function of the speed of light. The molecular
geometry is given by `h2o.xyz`

<div class="literalinclude">

h2o.xyz

</div>

For a standard geometry optimization we use the input file `geo.inp`

<div class="literalinclude">

geo.inp

</div>

We run DIRAC using :

    pam --inp=geo --mol=h2o.xyz

From the DIRAC output we find that the initial geometry is:

    Bond distances (angstroms):
    ---------------------------

                    atom 1     atom 2                           distance
                    ------     ------                           --------
    bond distance:    H  1       O                              0.957897
    bond distance:    H  2       O                              0.957897


    Bond angles (degrees):
    ----------------------

                    atom 1     atom 2     atom 3                   angle
                    ------     ------     ------                   -----
    bond angle:       H  2       O          H  1                 104.120

whereas at the end of optimization we have:

    Bond distances (angstroms):
    ---------------------------

                    atom 1     atom 2                           distance
                    ------     ------                           --------
    bond distance:    H  1       O                              0.942905
    bond distance:    H  2       O                              0.942905


    Bond angles (degrees):
    ----------------------

                    atom 1     atom 2     atom 3                   angle
                    ------     ------     ------                   -----
    bond angle:       H  2       O          H  1                 105.667

Now let's play. We now do a geometry optimization using the file
`geo_ultra.inp`

<div class="literalinclude">

geo_ultra.inp

</div>

where the speed of light is reduced all the way down to 10
$`a_0 E_h/\hbar`$. In the DIRAC output the final geometry is now:

    Bond distances (angstroms):
    ---------------------------

                    atom 1     atom 2                           distance
                    ------     ------                           --------
    bond distance:    H  1       O                              1.066701
    bond distance:    H  2       O                              1.066701


    Bond angles (degrees):
    ----------------------

                    atom 1     atom 2     atom 3                   angle
                    ------     ------     ------                   -----
    bond angle:       H  2       O          H  1                  89.505

That is quite a change ! The reader is encouraged to play with other
speeds of light. It is also possible to turn off spin-orbit interaction
by setting `HAMILTONIAN_.SPINFREE`.

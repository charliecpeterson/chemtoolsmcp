orphan  

# Electronic structure of CmF

We want to investigate the structure of the diatomic molecule CmF
(Curium Fluoride). Following Mochizuki_JCP2003 we assume that curium has
valence configuration in $`5f^77s^2`$ in the molecule, in particular
since curium is a late actinide and thus quite similar to lanthanides.
The molecule will therefore have an atomic-like open $`5f^7`$ shell.
This may cause convergence problems since DIRAC by default assumes
open-shell orbitals to have energies above close-shell ones, which is
likely to not be the case here. In order to guide DIRAC we shall
therefore employ the following strategy:

1.  We calculate the curium atom.
2.  We import the $`5f^7`$ orbitals into the molecular calculation,
    place them in the open shell and keep them frozen in a first SCF
    calculation.
3.  We then freeze all closed-shell orbitals, allowing the $`5f^7`$
    orbitals to relax in the molecule.

## The curium atom

For ease of analysis we shall impose linear symmetry also in the atomic
calculation. We achieve this through the introduction of a ghost center

<div class="literalinclude">

Cm.xyz

</div>

The ground state configuration of curium (Cm, Z=96) is
$`[Rn]5f^76d^17s^2`$ (see [here](https://www.webelements.com/curium/)
for general information and [also
here](http://physics.nist.gov/PhysRefData/Handbook/Tables/curiumtable5.htm)),
so with two open shells. We specify this configuration by using the
keyword `SCF_.KPSELE`:

<div class="literalinclude">

Cm_KPSELE.inp

</div>

We run the calculation using:

    pam --inp=Cm --mol=Cm.xyz --get "DFCOEF=cf.Cm"

This calculation converges smoothly in 13 iterations, and we are now
ready to attack the molecule.

## The molecular calculation:

### 1. Importing curium $`5f`$ orbitals

The menu file for the first step of the molecular calculation reads

<div class="literalinclude">

CmF_5f_frz.inp

</div>

In the `*SCF` section it may be noted that we specify 49 closed-shell
orbitals, followed by seven open-shell ones, with seven active
electrons. Next we specify that we will take orbitals 45 to 51 from the
coefficient file of the cerium atom, that we name AFCMXX, put them in
position 50 to 56 and keep them frozen. This also implies that we are
effectively doing a closed-shell calculation, which helps convergence as
well. We run the calculation using the command:

    pam --mw=200 --inp=CmF_5f_frz --mol=CmF.xyz --put "cf.Cm=AFCMXX"  --get "DFCOEF=cf.CmF_5f_frz"

The calculation converges in 18 iterations. Here is a snippet from the
population analysis:

<div class="literalinclude">

CmF_5f_frz.pop

</div>

It shows that the upper closed-shell orbital (49) is dominated by cerium
$`(7)s`$. The open-shell cerium $`5f`$ orbitals are frozen and keep
their energies from the atomic calculation, with $`5f_{5/2}`$ and
$`5f_{7/2}`$ orbitals having energies -0.5359 and -0.4840 $`E_h`$,
respectively.

### 2. Relaxing the curium $`5f`$ orbitals

We now want to relax the curium $`5f`$ orbitals in the molecule whilst
keeping the other orbitals frozen. This step is not necessarily requred,
but allows us to see how the $`5f`$ orbital energies evolve. We use the
following input file

<div class="literalinclude">

CmF_5f_relax.inp

</div>

It will be seen that we now import the coefficient file of the previous
run and freeze all closed-shell orbitals. We use the command:

    pam --mw=200 --inp=CmF_5f_frz --mol=CmF.xyz --put "cf.CmF_5f_frz=DFCOEF"  --get "DFCOEF=cf.CmF_5f_relax"            

The calculation converges in 14 iterations. Again a snippet from the
Mulliken population analysis:

<div class="literalinclude">

CmF_5f_relax.pop

</div>

We note that the curium $`5f`$ orbital energies are split by the
molecular field, also that they are somewhat lowered. We also note that
orbitals 46 - 48, dominated by fluorine $`2p`$ orbitals, are almost
degenerate with the curium $`5f`$ orbitals, whereas orbital 50 is an
almost pure curium $`7s`$ orbital.

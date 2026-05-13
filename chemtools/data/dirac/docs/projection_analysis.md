orphan  

# Projection analysis

## Theory

A serious flaw of Mulliken population analysis is its basis set
sensitivity. This is well known from the non-relativistic domain. In the
relativistic domain a further disadvantage is that the expansion of
large and small components in separate basis sets means large and small
component populations come separately and it is not easy to see how they
are connected. Furthermore, if the large and small component are
expanded in scalar basis functions, as is the case of the DIRAC code,
then the population analysis can not distinguish distributions from
different spin-orbit components of atomic orbitals, for instance,
$`p_{1/2}`$ and $`p_{3/2}`$.

Projection analysis Dubillard2006 eliminates all these inconveniences.
It is based on the simple concept of molecular orbitals as linear
combination of atomic orbitals (LCAO) and allows analysis of electronic
structure in terms of well-defined atomic (or fragment) orbitals.

Projection analysis requires as input a set of pre-calculated orbitals
{$`\psi^A_p`$} for the constituent atoms (or fragment) of the molecule.
Here $`A`$ refers to the atomic center (or fragment) and $`p`$ is on
orbital index for that center. The molecular orbitals are then expanded
as

``` math
\left|\psi^{MO}_i\right> = \sum_A \sum_{p\in A} \psi^A_p c^A_{pi} + \left|\psi^{pol}_i\right>
```

Typically one will select the occupied orbitals of the constituent
atoms. They are not guaranteed to fully span the molecular orbitals and
therefore the expansion includes their orthogonal complement
$`\left|\psi^{pol}_i\right>`$, denoted the *polarization contribution*.
The expansion coefficients $`c^A_{pi}`$ are found by solving the
equation

``` math
\left<\psi^A_p|\psi^{MO}_i\right> = \sum_B\sum_{q\in B}\left<\psi^A_p|\psi^B_q\right>c^B_{qi}
```

obtained by projection.

For projection analysis, arbitrary fragments can in principle be used.
There is, however, an important restriction in that DIRAC can only do
analysis in terms of fragments complying to the molecular point group.
For instance, the two hydrogens in water are related by symmetry in the
full $`C_{2v}`$ molecular point group. They then have to be treated as a
single fragment, or they can be separated by lowering the symmetry.
Starting from DIRAC14 projection analysis can be carried out in terms of
individual atoms using the `PROJECTION_.ATOMS` keyword, as shown in the
example below.

## Example 1: Methane

### Introduction

As a first simple example we shall consider the methane molecule. The
molecular input file `CH4.xyz`
(`download <../../../../test/tutorial_projection_analysis/CH4.xyz>`) is

<div class="literalinclude">

../../../../test/tutorial_projection_analysis/CH4.xyz

</div>

Suppose that we have run a DFT(PBE) calculation using:

    pam --inp=PBE  --mol=CH4.xyz 

and the menu file `PBE.inp`
(`download <../../../../test/tutorial_projection_analysis/PBE.inp>`) is

<div class="literalinclude">

../../../../test/tutorial_projection_analysis/PBE.inp

</div>

### Preparing for analysis

We would now like to analyze the electronic structure of the methane
molecule in terms of the orbitals of the constituent atoms. We first
generate the atomic fragments. We recommend calculating the atoms in the
highest possible symmetry and the converting the coefficients to no
($`C_1`$) symmetry. This allows a very precise identification of the
individual atomic orbitals (including $`m_j`$ quantum number). We
accordingly run calculations where we save MO files, each as the unique
file:

    pam --inp=H  --mol=H 
    pam --inp=C  --mol=C 

with corresponding input files `H.inp`
(`download <../../../../test/tutorial_projection_analysis/H.inp>`),
`H.mol`
(`download <../../../../test/tutorial_projection_analysis/H.mol>`),
`C.inp`
(`download <../../../../test/tutorial_projection_analysis/C.inp>`), and
`C.mol`
(`download <../../../../test/tutorial_projection_analysis/C.mol>`).

### Running the analysis

We are now ready for the projection analysis itself. We prepare the
input file `prj.inp`
(`download <../../../../test/tutorial_projection_analysis/prj.inp>`)

<div class="literalinclude">

../../../../test/tutorial_projection_analysis/prj.inp

</div>

In the `*PROJECTION` section we specify the two atomic types identified
by the name of their coefficient files as well as an orbital string (see
`orbital_strings`) listing what orbitals to include for each atom. DIRAC
will assume that each atom has been calculated in its proper basis.

We now run the projection analysis using:

    pam --inp=prj --mol=CH4.xyz --put "C.h5 H.h5 PBE_CH4.h5=CHECKPOINT.h5" --noh5

Note that we set the pam flag `--noh5` to avoid generating a new HDF5
checkpoint file, since nothing new is added to the existing one.

### Interpreting the output

The first thing to look for in the output
(`download <../../../../test/tutorial_projection_analysis/result/prj_CH4.out>`)
is the section:

    * Total gross contributions:
       AFCX 1               6.4301   E -     6.4301   P -     0.0000
       AFHX 1               0.8716   E -     0.8716   P -     0.0000
       AFHX 2               0.8716   E -     0.8716   P -     0.0000
       AFHX 3               0.8716   E -     0.8716   P -     0.0000
       AFHX 4               0.8716   E -     0.8716   P -     0.0000
       Polarization:     0.0834

It shows the gross population (in Mulliken sense, but in terms of the
fragment orbitals) of each atom and then, most importantly, the gross
population of the polarization contribution. In this example it is small
and perfectly acceptable. If it becomes too large the analysis looses
meaning because it impies that significant electron density has not been
assigned to any fragment. In the case of significant polarization
contribution one can either increase the number of atomic orbitals used
in the analysis or turn on *repolarization*. For the latter it suffices
to add the `PROJECTION_.POLREF` keyword (see the corresponding
`input file <../../../../test/tutorial_projection_analysis/prj_polref.inp>`
) which activates *Intrinsic Atomic Orbitals (IAOs)*, Knizia2013, thus
eliminating completely the polarization contribution. We then get (see
the corresponding
`output <../../../../test/tutorial_projection_analysis/result/prj_polref_CH4.out>`
file ):

    * Total gross contributions:
       AFCX 1               6.4782   E -     6.4782   P -     0.0000
       AFHX 1               0.8804   E -     0.8804   P -     0.0000
       AFHX 2               0.8804   E -     0.8804   P -     0.0000
       AFHX 3               0.8804   E -     0.8804   P -     0.0000
       AFHX 4               0.8804   E -     0.8804   P -     0.0000
       Polarization:     0.0000

from which we can conclude that at this level of theory the carbon and
hydrogen charges in methane are -0.48e and +0.12e, respectively.

Having concluded that the polarization contribution is acceptable, we
can proceed with the analysis. We turn to this section:

    * Total reference orbital contributions:
        AFCX 1  E     1       1.99999     -0.10660951E+02
        AFCX 1  E     2       1.27874     -0.97828469E+00
        AFCX 1  E     3       1.05077     -0.65580701E+00
        AFCX 1  E     4       1.05030     -0.65541446E+00
        AFCX 1  E     5       1.05030     -0.65541446E+00
        AFHX 1  E     1       0.87162     -0.23118508E+00
        AFHX 2  E     1       0.87162     -0.23118508E+00
        AFHX 3  E     1       0.87162     -0.23118508E+00
        AFHX 4  E     1       0.87162     -0.23118508E+00

which gives the accumulated gross population of each fragment orbital.
We can identify the fragments from the name of the coefficient file and
its orbital energy. For instance the first orbital on the list is
clearly carbon *1s* followed by carbon *2s*. Next comes three orbitals
that are almost degenerate. These are cleary the carbon *2p* orbitals,
the first one being $`2p_{1/2}`$ followed by the two $`2p_{3/2}`$. By
furthermore looking into the output of the carbon atom calculation (note
that we asked for Mulliken population analysis in the input) we find
that the first $`2p_{3/2}`$ orbital has $`m_j=1/2`$ and the second
$`m_j=-3/2`$. From these gross population we can deduce the atomic
configurations of the atoms in the molecule. For carbon we find
$`[He]2s^{1.3}2p^{3.2}`$ and for hydrogen $`1s^{0.9}`$.

There is also detailed output for each molecular orbital in the output.
For the third molecular orbital we find for instance:

    * Electronic eigenvalue nr.  3: -0.3411330794649       (Occupation : f = 1.0000)
    ================================================
        Orbital            Total           Eigenvalue       Kramers partner 1   Kramers partner 2
        AFCX 1  E     3       0.57945     -0.65580701E+00   (-1.0000,-0.0000)   ( 0.0000, 0.0000)
        AFHX 1  E     1       0.31559     -0.23118508E+00   ( 0.0000,-0.5774)   (-0.5774,-0.5774)
        AFHX 2  E     1       0.31559     -0.23118508E+00   ( 0.0000, 0.5774)   ( 0.5774,-0.5774)
        AFHX 3  E     1       0.31559     -0.23118508E+00   ( 0.0000, 0.5774)   (-0.5774, 0.5774)
        AFHX 4  E     1       0.31559     -0.23118508E+00   ( 0.0000,-0.5774)   ( 0.5774, 0.5774)
    * Gross contributions:
       AFCX 1               0.5254   E -     0.5254   P -     0.0000
       AFHX 1               0.1165   E -     0.1165   P -     0.0000
       AFHX 2               0.1165   E -     0.1165   P -     0.0000
       AFHX 3               0.1165   E -     0.1165   P -     0.0000  
       AFHX 4               0.1165   E -     0.1165   P -     0.0000
       Polarization:     0.0085

To fully understand the output we have to keep in mind that all
calculations are Kramers-restricted such that all orbitals come in
pairs. The expansion of the molecular orbitals should therefore rather
be written as

``` math
\left|\psi^{MO}_i\right> = \sum_A \sum_{p\in A} \left(\psi^A_p c^A_{pi} + \psi^A_\overline{p} c^A_{\overline{p}i}\right) + \left|\psi^{pol}_i\right>
```

where $`c^A_{pi}`$ and $`c^A_{\overline{p}i}`$ is the expansion
coefficient of Kramers partner 1 and 2, respectively. Let us now define
the absolute value

``` math
\left|c^A_{Pi}\right| = \sqrt{\left|c^A_{pi}\right|^2+\left|c^A_{\overline{p}i}\right|^2}
```

This is the real number reported under the heading `Total`. Under the
heading `Kramers partner 1` is reported the complex number

``` math
z_1 = \left(Re(z_1),Im(z_1)\right) = \frac{c^A_{pi}}{\left|c^A_{Pi}\right|},
```

and analogously for Kramers partner 2, from which we get the detailed
decomposition of the molecular orbital. We find that this molecular
orbital is composed of the carbon $`2p_{1/2}`$ orbital and the hydrogen
$`1s`$ orbitals.

### Exercises

1.  What is the effect of increasing the C-H bond distance on the atomic
    configurations of the C,H atoms in the methane molecule and on the
    atomic charges of the constituent atoms ? Collect your calculated
    data in a table.
2.  How different basis sets affect the atomic configurations of the C,H
    atoms in the methane molecule and the atomic charges of the same
    constituent atoms ?
3.  Investigate the effect of various DFT functionals (see `*DFT`) on
    the atomic configurations of the C and H atoms in the methane
    molecule and on atomic charges of these constituent atoms.

## Example 2: Uranyl

### Introduction

We next turn to a molecule with a heavy atom and a more complicated
electronic structure, namely uranyl $`UO_2^{2+}`$. Here the two oxygens
are related by symmetry, and so one would at first consider running the
projection analysis with lower (or no) symmetry or using the
`PROJECTION_.ATOMS` keyword, as in the methane example above. However,
we can reduce the symmetry from $`D_{\infty h}`$ to $`C_{\infty v}`$ by
introducing a ghost center. Our molecular input file then looks like

<div class="literalinclude">

UO2.mol

</div>

Now the oxygen atoms are no longer connected by symmetry and we can
carry out the projection analysis in $`C_{\infty v}`$ symmetry.

### Preparing the analysis

Using the menu file `HF.inp`

<div class="literalinclude">

UO2_HF.inp

</div>

we run 4-component Hartree-Fock:

    pam --mw=150 --inp=HF --mol=UO2 

Coefficients are saved on the h5 checkpoint file. Next we generate
atomic reference orbitals. For oxygen this is quite straightforward:

    pam --inp=O --mol=O 

using molecule input `O.mol`

<div class="literalinclude">

O.mol

</div>

and the menu file `O.inp`

<div class="literalinclude">

O.inp

</div>

For the uranium we must be a bit more careful since the ground state
configuration is an open-shell one: $`[Rn]5f^36d^17s^2`$. Using the
molecular input `U.mol`

<div class="literalinclude">

U.mol

</div>

and the inp file `U.inp`

<div class="literalinclude">

U_KPSELE.inp

</div>

Note that `SCF_.KPSELE` is needed for getting the convergence in this
case. Convergence is now straightforward :

    pam --inp=U --mol=U 

and the correct configuration is confirmed by Mulliken population
analysis.

### Running the analysis

We are now ready to do the projection analysis. We prepare the input
file `prj.inp`

<div class="literalinclude">

prj_UO2.inp

</div>

and prepare the coefficient files:

    $ ln -s O.h5 O1.h5
    $ ln -s O.h5 O2.h5

We run the projection analysis:

    pam --inp=prj --mol=UO2 --put="U.h5 O1.h5 O2.h5 HF_UO2.h5=CHECKPOINT.h5"

### Interpreting the output

As in the methane case above we first look at the polarization
contribution:

    * Total gross contributions:
       AFUXXX              89.0635   E -    89.0635   P -     0.0000
       AFO1XX               8.3098   E -     8.3098   P -     0.0000
       AFO2XX               8.3098   E -     8.3098   P -     0.0000
       Polarization:     0.3169

We see that it is more sizable than before. If we eliminate the
polarization contribution using the `PROJECTION_.POLREF` keyword, we
obtain:

    * Total gross contributions:
       AFUXXX              89.1586   E -    89.1586   P -     0.0000
       AFO1XX               8.4207   E -     8.4207   P -     0.0000
       AFO2XX               8.4207   E -     8.4207   P -     0.0000
       Polarization:     0.0000

The atomic gross populations tell us that the oxygen atoms has a charge
-0.42, whereas the uranium atom has charge +2.84, considerably far from
the formal oxidation state +VI.

We next turn to the electron configuration of the atoms in the molecule
by looking at the section:

    * Total reference orbital contributions:
        AFUXXX  E     1       2.00000     -0.42818128E+04   1s
        AFUXXX  E     2       2.00000     -0.80663682E+03   2s
        AFUXXX  E     3       2.00000     -0.77703498E+03   2p-
        AFUXXX  E     4       2.00000     -0.63578312E+03   2p+
        AFUXXX  E     5       2.00000     -0.63578312E+03   2p+
        AFUXXX  E     6       2.00000     -0.20673001E+03   3s
        AFUXXX  E     7       2.00000     -0.19325143E+03   3p-
        AFUXXX  E     8       2.00000     -0.16037784E+03   3p+
        AFUXXX  E     9       2.00000     -0.16037784E+03   3p+
        AFUXXX  E    10       2.00000     -0.13906994E+03   3d-
        AFUXXX  E    11       2.00000     -0.13906994E+03   3d-
        AFUXXX  E    12       2.00000     -0.13242590E+03   3d+
        AFUXXX  E    13       2.00000     -0.13242590E+03   3d+
        AFUXXX  E    14       2.00000     -0.13242590E+03   3d+
        AFUXXX  E    15       2.00000     -0.54355461E+02   4s
        AFUXXX  E    16       2.00000     -0.48231971E+02   4p-
        AFUXXX  E    17       2.00000     -0.39554187E+02   4p+
        AFUXXX  E    18       2.00000     -0.39554187E+02   4p+
        AFUXXX  E    19       2.00000     -0.29743891E+02   4d-
        AFUXXX  E    20       2.00000     -0.29743891E+02   4d-
        AFUXXX  E    21       2.00000     -0.28130201E+02   4d+
        AFUXXX  E    22       2.00000     -0.28130201E+02   4d+
        AFUXXX  E    23       2.00000     -0.28130201E+02   4d+
        AFUXXX  E    24       2.00000     -0.15202067E+02   4f-
        AFUXXX  E    25       2.00000     -0.15202067E+02   4f-
        AFUXXX  E    26       2.00000     -0.15202067E+02   4f-
        AFUXXX  E    27       2.00000     -0.14785575E+02   4f+
        AFUXXX  E    28       2.00000     -0.14785575E+02   4f+
        AFUXXX  E    29       2.00000     -0.14785575E+02   4f+
        AFUXXX  E    30       2.00000     -0.14785575E+02   4f+
        AFUXXX  E    31       2.00001     -0.12603340E+02   5s 
        AFUXXX  E    32       1.99989     -0.10135890E+02   5p-
        AFUXXX  E    33       1.99952     -0.80948982E+01   5p+
        AFUXXX  E    34       1.99993     -0.80948982E+01   5p+
        AFUXXX  E    35       1.99983     -0.43524477E+01   5d-
        AFUXXX  E    36       1.99914     -0.43524477E+01   5d-
        AFUXXX  E    37       1.99872     -0.40407150E+01   5d+
        AFUXXX  E    38       1.99956     -0.40407150E+01   5d+
        AFUXXX  E    39       1.99990     -0.40407150E+01   5d+
        AFUXXX  E    40       1.98417     -0.21392050E+01   6s
        AFUXXX  E    41       1.93819     -0.13442976E+01   6p-
        AFUXXX  E    42       1.75124     -0.98481977E+00   6p+ (1/2)
        AFUXXX  E    43       1.98199     -0.98481977E+00   6p+ (3/2)
        AFUXXX  E    44       0.03906     -0.20241412E+00   7s
        AFUXXX  E    45       0.82745     -0.34596371E+00   5f- (1/2)
        AFUXXX  E    46       0.15661     -0.34596371E+00   5f- (3/2)
        AFUXXX  E    47       0.00000     -0.34596371E+00   5f- (5/2)
        AFUXXX  E    48       0.00000     -0.31822326E+00   5f+ (7/2)
        AFUXXX  E    49       0.00000     -0.31822326E+00   5f+ (5/2)
        AFUXXX  E    50       0.33927     -0.31822326E+00   5f+ (3/2)
        AFUXXX  E    51       0.94161     -0.31822326E+00   5f+ (1/2)
        AFUXXX  E    52       0.42621     -0.19266767E+00   6d- (1/2)
        AFUXXX  E    53       0.09667     -0.19266767E+00   6d- (3/2)
        AFUXXX  E    54       0.34518     -0.18309790E+00   6d+ (1/2)
        AFUXXX  E    55       0.33432     -0.18309790E+00   6d+ (3/2)
        AFUXXX  E    56       0.00010     -0.18309790E+00   6d+ (5/2)
        AFO1XX  E     1       2.00005     -0.20696087E+02   1s
        AFO1XX  E     2       1.89709     -0.12503665E+01   2s
        AFO1XX  E     3       1.51690     -0.61398417E+00   2p-
        AFO1XX  E     4       1.46077     -0.61259798E+00   2p+
        AFO1XX  E     5       1.54590     -0.61259798E+00   2p+
        AFO2XX  E     1       2.00005     -0.20696087E+02   1s
        AFO2XX  E     2       1.89709     -0.12503665E+01   2s
        AFO2XX  E     3       1.51690     -0.61398417E+00   2p-
        AFO2XX  E     4       1.46077     -0.61259798E+00   2p+
        AFO2XX  E     5       1.54590     -0.61259798E+00   2p+

We can to a large extent deduce the identity of the various atomic
orbitals by just looking at orbital energies and I have added it by hand
to the above output in a hopefully obvious notation. Even more precise
information, such as $`m_j`$ value (indicated in parenthesis) can be
obtained from looking at the Mulliken population analysis of each atom.
From this list of gross population we extract the atomic valence
configuration $`5f^{2.25}6d^{1.16}7s^{0.04}`$ for the uranium atom and
$`2s^{1.90}2p^{4.41}`$ for oxygen. An interesting feature is that the
uranium *6p* population adds up to *5.64* and not six electrons. This is
a manifestation of the so-called *6p-hole*, due to overlap with the
oxygen ligands, and mostly located to the $`6p_{3/2.1/2}`$ orbital.

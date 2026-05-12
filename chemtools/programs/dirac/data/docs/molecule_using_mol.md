orphan  

# How to specify the molecule and basis set using the traditional mol files

The procedure used to read mol files is as follows: first, the basis set
for the large component will be read in from the the mol file; then the
small component basis set is generated automatically via the kinetic
balance prescription, but can (by experienced users) also be specified
explicitly. Try this only if you have studied the theory and know what
the criteria for small component basis sets are, there are many pitfalls
that one needs to be aware of!

> [!WARNING]
> The traditional mol file is entered in fixed format (with few
> exceptions). This means that exact position and spacing matters. If
> you misplace the charge, the number of centers, etc., then DIRAC will
> not be able to read the structure and basis set information correctly.
> Until you get familiar with the format, we recommend to copy existing
> mol files and adapt them.

We will learn the mol file format using the following example:

    DIRAC


    C   2
            1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE BASIS cc-pVDZ
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE BASIS cc-pVDZ
    FINISH

This is the water molecule using the cc-pVDZ basis for the large
component basis set. The first 3 lines are arbitrary comment lines. You
can leave them blank or write some note to yourself but they have to be
there. Line 4 specifies a cartesian GTO basis set (C) and 2 atom types
(basis set types). So after line 4 DIRAC expects 2 basis set types and
here they are - first hydrogen:

    1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE BASIS cc-pVDZ

then oxygen:

    8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE BASIS cc-pVDZ

and we finish off with a FINISH in the final line.

You can already guess the meaning of the numbers. We have 2 hydrogens
with a nuclear charge of 1.0 and one oxygen with a nuclear charge of
8.0. Then come the atom symbols and the coordinates in bohr followed by
the large component basis set label. DIRAC will look up this label in
the basis set library and complain if it does not exist. Note that the
atom label (H1, H2, O) is only for you so that you find it in the
output. It does not matter to the code. The code will identify basis
sets based on the nuclear charge, not the label. The cartesian
coordinates may be given in free format. However, the name of the atom
must still be left four places, and no coordinates must enter the four
first positions.

DIRAC typically runs in cartesian GTO. Note that by default the
transformation to spherical harmonic basis (modified for the small
components) is done in the transformation to the orthogonal basis.
Spherical Gaussians (S) are allowed in 2-component calculations.

## Default coordinates are in bohr, how can I use angstrom instead?

See the "A" in line 4 in position 20? If it is there, DIRAC will read in
angstrom, if this position is blank, then DIRAC will read in bohr
(default):

    DIRAC
             1         2         3           # this is a comment line
    1234567890123456789012345678901234567890 # this is a comment line
    C   2              A
            1.    2
    [...]

## Nuclear exponent

You can modify the default gaussian exponent for the nuclear charge
distribution by giving a floating point value somewhere within the space
specified below:

    DIRAC
             1         2         3           # this is a comment line
    1234567890123456789012345678901234567890 # this is a comment line
    C   2              A
            1.    2--------here--------
    [...]

As an example, here is the Ne atom with a custom nuclear exponent of
10.0D8:

    DIRAC

                   --------here--------
    C   1
          10.     1   10.0D+08
    Ne    0.0  0.0  0.0
    LARGE BASIS cc-pVDZ
    FINISH

## Fitting basis set

In the development version of DIRAC it is now possible to make use of
density fitting. In such calculations, it is mandatory to include a
basis set used to expand the density. This is done adding a line with
the FTSET keyword after the coordinates for a given atom type. Basically
all the options available in the definition of large or small basis sets
are applicable to the fit set (i.e. the use of a set from the library,
explicitly typed basis, even/well-tempered etc). Here is an example:

    DIRAC


    C   2
            1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE BASIS cc-pVDZ
    FTSET BASIS Ahlrichs-Coulomb-Fit
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE BASIS cc-pVDZ
    FTSET BASIS Ahlrichs-Coulomb-Fit
    FINISH

## Point charges

You can add point charges with the "LARGE NOBASIS" basis set
specification (or, equivalently, with "LARGE POINTCHARGE"). For example
adding two point charges, both of -1.6, to the water example above:

    DIRAC


    C   3
            1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE BASIS cc-pVDZ
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE BASIS cc-pVDZ
          -1.6    2
    XX1    .0000000000        1.0000000000         .0000000000
    XX2    .0000000000       -1.0000000000         .0000000000
    LARGE NOBASIS
    FINISH

## Multiple basis sets from the basis set library

As an alternative to the BASIS option described above, it is possible to
use specify different basis set files via the MULTIBASIS keyword. The
MULTIBASIS option is present to allow easier inclusion of different sets
of diffuse and/or polarization functions to a reference basis set. The
syntax for this keyword is very similar to that of the BASIS:

    DIRAC


    C   2
            1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE MULTIBASIS 2 cc-pVDZ cc-pVDZ-diffuse
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE MULTIBASIS 2 cc-pVDZ cc-pVDZ-diffuse
    FINISH

After the keyword an integer with the number of files to be read should
be specified. It is followed by the name of the different basis set
files, each separated by a whitespace. The only limitation for the
number of basis set files is that the total length of this line should
not exceed Fortran's maximum allowed line size (80 characters).

## Explicitly typed basis sets

Explicitly typed basis sets are best described using an explicit
example:

    DIRAC


    C   2
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE EXPLICIT  3    1    1    1
    # s functions
    f  10    4
      11720.0000000  0.00071000 -0.00016000  0.00000000  0.00000000
       1759.0000000  0.00547000 -0.00126300  0.00000000  0.00000000
        400.8000000  0.02783700 -0.00626700  0.00000000  0.00000000
        113.7000000  0.10480000 -0.02571600  0.00000000  0.00000000
         37.0300000  0.28306200 -0.07092400  0.00000000  0.00000000
         13.2700000  0.44871900 -0.16541100  0.00000000  0.00000000
          5.0250000  0.27095200 -0.11695500  0.00000000  0.00000000
          1.0130000  0.01545800  0.55736800  0.00000000  0.00000000
          0.3023000 -0.00258500  0.57275900  1.00000000  0.00000000
          0.0789600  0.00000000  0.00000000  0.00000000  1.00000000
    # p functions
    f   5    3
         17.7000000  0.04301800  0.00000000  0.00000000
          3.8540000  0.22891300  0.00000000  0.00000000
          1.0460000  0.50872800  0.00000000  0.00000000
          0.2753000  0.46053100  1.00000000  0.00000000
          0.0685600  0.00000000  0.00000000  1.00000000
    # d functions
    f   2    2
          1.1850000  1.00000000  0.00000000
          0.3320000  0.00000000  1.00000000
    [...]

Following EXPLICIT is the highest angular quantum number plus one. In
this case it is 3, since we are using a *spd* basis set. Following this
number are the number of blocks for each *l*-value. The memory
requirements grow rapidly with the number of basis functions in a block
(note for instance that four *g* functions actually are 60 basis
functions, as there are 15 cartesian components of each *g* function).
Memory requirements can therefore be reduced by splitting basis
functions of the quantum number into different blocks. This will,
however, decrease the performance of the integral calculation.

Following the LARGE EXPLICIT line we see a line with a comment. Lines
starting with either !, \$, or \# are comments and are ignored by the
code.

The next line starts with a single character describing the input format
of the basis set in this block.

The default format is 8F10.4 which will be used if left blank. **Be very
careful when using this default format as it will miss any exponential
parameter standing to the right of the 10 characters**. In this format
the first column is the orbital exponent and the seven last columns are
contraction coefficients. If no numbers are given, a zero is assumed. If
more than 7 contracted functions occur in a given block, the contraction
coefficients may be continued on the next line, but the first column
(where the orbital exponents are given) must then be left blank.

An F or f in the first position (like in the example above) will
indicate that the input is in free format. This will of course require
that all contraction coefficients need to be typed in, as all numbers
need to be present on each line. However, note that this options is
particularly handy together with completely decontracted basis sets, as
described below. Note that the program reads the free format input from
an internal file that is 80 characters long, and no line should
therefore exceed 80 characters.

The numbers in the line "f 10 4" are the number of of primitive
Gaussians in this block (10) and the number of contracted Gaussians in
this block (4). If a zero is given for he number of contracted
Gaussians, an uncontracted basis set will be assumed, and only orbital
exponents need to be given.

One may also give the format H or h. This corresponds to high precision
format 4F20.8, where the first column again is reserved for the orbital
exponents, and the three next columns are designated to the contraction
coefficients. If no number is given, a zero is assumed. If there are
more than three contracted orbitals in a given block, the contraction
coefficients may be continued on the next line, though keeping the
column of the orbital exponents blank.

## Specification of how to generate small component functions

Coming back to the example above we can add another integer following "f
10 4", which will specify how to generate small component functions:

    DIRAC


    C   2
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE EXPLICIT  3    1    1    1
    # s functions
    f  10    4    0
      11720.0000000  0.00071000 -0.00016000  0.00000000  0.00000000
       1759.0000000  0.00547000 -0.00126300  0.00000000  0.00000000
        400.8000000  0.02783700 -0.00626700  0.00000000  0.00000000
        113.7000000  0.10480000 -0.02571600  0.00000000  0.00000000
         37.0300000  0.28306200 -0.07092400  0.00000000  0.00000000
         13.2700000  0.44871900 -0.16541100  0.00000000  0.00000000
          5.0250000  0.27095200 -0.11695500  0.00000000  0.00000000
          1.0130000  0.01545800  0.55736800  0.00000000  0.00000000
          0.3023000 -0.00258500  0.57275900  1.00000000  0.00000000
          0.0789600  0.00000000  0.00000000  0.00000000  1.00000000
    # p functions
    f   5    3    0
         17.7000000  0.04301800  0.00000000  0.00000000
          3.8540000  0.22891300  0.00000000  0.00000000
          1.0460000  0.50872800  0.00000000  0.00000000
          0.2753000  0.46053100  1.00000000  0.00000000
          0.0685600  0.00000000  0.00000000  1.00000000
    # d functions
    f   2    2    0
          1.1850000  1.00000000  0.00000000
          0.3320000  0.00000000  1.00000000
    [...]

If a 0 (as in this example) or no number is given, the small component
functions are generated both upwards and downwards (default). If the
number is 1, the small component functions are generated upwards, If the
number is 2, the small component functions are generated downwards. For
other values no small components functions generated.

## MOLFDIR-type basis sets

The MOLFDIR basis set file(s) (here: Oxygen-xyz.bas) must be copied to
the scratch area, for example using the pam script:

    DIRAC


    C   2
            1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE MOLFBAS Hydrogen-xyz.bas
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE MOLFBAS Oxygen-xyz.bas
    FINISH

## Even-tempered basis sets (geometric progressions)

The exponents are generated in an even-tempered series:

``` math
\eta_{N - k + 1} = \alpha \beta^{k - 1}
```

with

``` math
k = 1, \ldots, N
```

    LARGE EVENTEMP 0.05 3.0 11 3 2 1 1
    1..5
    6..11
    7..11
    9..10

This means that alpha is 0.05, beta is 3.0, N is 11. Highest angular
quantum number plus one is 3 (s, p, and d functions), we will have 2
blocks for the s, 1 block for p, and one block for d. The s blocks go
from exponent 1 to 5 and from 6 to 11, p goes from 7 to 11, d goes from
9 to 10.

## Well-tempered basis sets

This works very much like LARGE EVENTEMP, except that the exponents are
generated in an well-tempered series:

``` math
\eta_N = \alpha
```

and

``` math
\eta_{N - k + 1} = \eta_{N - k + 2} \beta [ 1 + \gamma (\frac{k}{N})^n ]
```

with

``` math
k = 1, \ldots, N
```

    LARGE WELLTEMP 0.05 2.5 2.0 6.0 11 3 2 1 1
    1..5
    6..11
    7..11
    9..10

Here alpha is 0.05, beta is 2.5, gamma is 2.0, n is 6.0, and N is 11.

## Explicit small component basis set

This is for experts. Following the line or block specifying the LARGE
component basis set you can override the default kinetic balance
prescription and specify the small component basis set either using
SMALL EXPLICIT or using SMALL MOLFBAS, in analogy to LARGE EXPLICIT and
LARGE MOLFBAS After the large component basis set the small component
basis set can be specified. If nothing is specified it is equivalent to
specifying SMALL KINBAL (see below). There are three possibilities for
giving the basis set.

## Family basis sets

Input for basis sets where the same set of exponents are used for all
functions. This is analogous to the well- and even-tempered basis sets
except that the exponents are not calculated from a formula, but must be
given in the file. These exponents may come from a basis set
optimization with GRASP:

    DIRAC


    C   2
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE FAMILY   10    3    1    1    1
    # exponents
     6665.0000000
     1000.0000000
      228.0000000
       64.7100000
       21.0600000
        7.4950000
        2.7970000
        0.5215000
        0.1596000
        0.0469000
    # ranges
    1..10
    6..10
    8..9
    [...]

## Dual family basis sets

A basis set analogous to the family basis set, except one set of
exponents are used for s, d, g, ... (gerade) functions, and another set
is used for p, f, h, ... (ungerade) functions:

    DIRAC


    C   2
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE DUALFAMILY   10    5    3    1    1    1
    # s, d, g, ...
     6665.0000000
     1000.0000000
      228.0000000
       64.7100000
       21.0600000
        7.4950000
        2.7970000
        0.5215000
        0.1596000
        0.0469000
    # p, f, h, ...
        9.4390000
        2.0020000
        0.5456000
        0.1517000
        0.0404100
    # ranges
    1..10
    1..5
    8..9
    [...]

## How to set the charge of the system

By default the charge of the atom/molecule is zero. You can change this
in the mol file:

    DIRAC

         *** <- you can use these positions on next line
    C   2  1
            1.    2
    H1   -1.4523499293         .0000000000         .8996235720
    H2    1.4523499293         .0000000000         .8996235720
    LARGE BASIS cc-pVDZ
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE BASIS cc-pVDZ
    FINISH

This is water with charge plus 1. Alternatively you can specify the
occupation explicitly in the inp file and leave it blank in the mol
file.

## Automatic augmentation with diffuse functions

By using a augmentation prefix you can use automatic augmentation to
extend Dunning, Dyall, and Turbomole basis sets with diffuse exponents.

The following augmentation prefixes are recognized:

    d-aug-cc-p...
    t-aug-cc-p...
    q-aug-cc-p...
    5-aug-cc-p...
    6-aug-cc-p...
    7-aug-cc-p...

    s-aug-dyall...
    d-aug-dyall...
    t-aug-dyall...
    q-aug-dyall...

    s-a-Turbomole...
    d-a-Turbomole...
    t-a-Turbomole...
    q-a-Turbomole...

What the code does is that it sorts exponents for each angular momentum,
calculates the factor between the two most diffuse exponents, and adds a
new exponent (or new exponents) by keeping this factor (even tempered
augmentation). If an angular momentum only has one exponent, then the
progression cannot be calculated (obviously), and a factor 3.5 is used.
This is a choice.

Automatic augmentation can be useful but it should not be used in a
black box manner. You are strongly encouraged to define DEBUG_PRIMITIVES
in abacus/herrdn.F and to examine which exponents have been added. The
code simply augments and does not consider whether it uses the original
SCF set exponents or correlating exponents.

## How to force specific symmetry

In all the above examples we have not specified symmetry explicitly. In
this case DIRAC will detect and use the highest symmetry available in
the code. In doing so it may translate and rotate the molecule (check
output).

There are two alternatives to this. The first is that we can turn off
symmetry completely and force C1 (by placing a 0 at the right place; the
0 means zero symmetry generators):

    DIRAC


    C   2    0
    [...]

    DIRAC

Alternative two is to specify an explicit symmetry other than C1. For
this we need to remove all symmetry-generated centers and specify the
coordinates only for the symmetry-unique centers:

    DIRAC


    C   2    2  X  Y
            8.    1
    O      .0000000000        0.0000000000        -.2249058930
    LARGE BASIS cc-pVDZ
            1.    1
    H     1.4523499293         .0000000000         .8996235720
    LARGE BASIS cc-pVDZ
    FINISH

We give only one hydrogen center, the other is generated by symmetry
operations. This molecule has two symmetry generators (second 2 in line
4).

> [!WARNING]
> If you fail to remove the centers generated by symmetry operations,
> DIRAC will apply symmetry operations to them which will re-generate
> the symmetry-unique centers. You will get more than one center at the
> same position and DIRAC will stop with the error "nuclei too close".

All available symmetry groups and the corresponding symmetry group
specification strings are given in the following list:

    C   ?    3  Z  Y  X   # group:      D2h
                          # operations: reflection(xy)
                          #             reflection(xz)
                          #             reflection(yz)

    C   ?    2XY  YZ      # group:      D2
                          # operations: rotation(z)
                          #             rotation(x)

    C   ?    2 Y X        # group:      C2v
                          # operations: reflection(xz)
                          #             reflection(yz)

    C   ?    2  ZXYZ      # group:      C2h
                          # operations: reflection(xy)
                          #             inversion

    C   ?    1XY          # group:      C2
                          # operations: rotation(z)

    C   ?    1  Z         # group:      Cs
                          # operations: reflection(xy)

    C   ?    1XYZ         # group:      Ci
                          # operations: inversion

    C   ?    0            # group:      C1

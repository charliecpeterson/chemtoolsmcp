orphan  

# How to specify ECP parameters in mol files

Effective core potential (ECP) is an efficient method in many quantum
mechanical calculations. It reduces basis set demands in heavy elements
by replacing core electrons with an effective potential. To account of
spin-orbit and other relativistic effects, many recently developed ECP
parameters are based on atomic Dirac-Fock calculations, and are
sometimes called the relativistic effective core potentials (RECP).

Following the usual practice the point nucleus model is used by default
for ECP calculations.

## Theory

Originally, the spin-orbit term is included in the RECP (SOREP) as
following Lee1977,

``` math
U^{SOREP} = U_{LJ}^{SOREP}(r) + \sum_{l=0}^{L-1}\sum_{j=|l-1/2|}^{l+1/2}\sum_{m=-j}^{j}
           [U_{ij}^{SOREP}(r)- U_{LJ}^{SOREP}(r)]|ljm\rangle \langle ljm|
```

The SOREP is divided into two types of potential - averaged relativistic
effective core potential (AREP) and spin-orbit potential.
($`U^{SOREP} = U^{AREP} + U^{SO}`$) Ermler1981

``` math
U^{AREP} = U_{L}^{AREP}(r) + \sum_{l=0}^{L-1}\sum_{m=-l}^{l}
            [U_{l}^{AREP}(r)- U_{L}^{AREP}(r)]|lm\rangle \langle lm|
```

where

``` math
U_{l}^{AREP} = \frac{1}{2l+1}[l\cdot U_{l,l-1/2}^{SOREP}(r) 
                              + (l+1)\cdot U_{l,l+1/2}^{SOREP}(r)]
```

The spin-orbit potential ($`U^{SO}`$) is defined as,

``` math
U^{SO} = s \cdot \sum_{l=1}^{L} \frac{2}{2l+1}\Delta U_{l}^{SOREP}(r)
         \sum_{m=-l}^{l} \sum_{m^{\prime}=-l}^{l}|lm\rangle 
         \langle lm|l|lm^{\prime}\rangle \langle lm^{\prime}|
```

with

``` math
\Delta U_{l}^{SOREP(r)} = U_{l,l+1/2}^{SOREP(r)} - U_{l,l-1/2}^{SOREP(r)}
```

## ECP parameters from the library

See the simple example file fro the ecp test
(`HI_ecplib.mol <../../test/ecp/HI_ecplib.mol>`) :

<div class="literalinclude">

../../test/ecp/HI_ecplib.mol

</div>

1\. Basis set library for ECP (line 7): The format is same as the large
component basis set. (see large component basis set) However, one should
use the basis set parameters corresponding to the RECP.

2\. ECP parameters from the library (line 8): ECPLIB keyword should be
used for the use of RECP parameters from the library.

Note: In DIRAC program, some ECP libraries are provided in basis_ecp
directory for your conveniences. However, it is safe to check ECP
parameters from authors' webpages. The name of ECP parameters used here
is following, ECP(XX)(YY)(ZZ)(SO/SF).

- XX: representative authors, (CE=Christiansen, Ermler, and coworkers),
  (DS=Dolg, Stoll, and coworkers), (HW=Hay, Wadt, and coworkers),
  (SB=Stevens, Basch, and coworkers)
- YY: number of core electrons replaced by potential
- ZZ: further information (if exist),
- (SO/SF) : spin-orbit(SOREP) or spin-free(AREP), The difference of SO
  and SF is the existance of SO parameters.

For example, ECPDS60MDFSO indicates Stuttgart RECP from Dolg, Stoll, and
coworkers with 60 electrons replaced by potential which is fitted from
relativistic calculation (spin-orbit potential is included).

Correlation-consistent basis sets by Kirk Peterson and co-workers that
can be used with the Stuttgart-Cologne ECPs are available
[here](http://tyr0.chem.wsu.edu/~kipeters/basis.html) .

## Explicitly typed ECP

    INTGRL
    HI MOLECULE
    I:Christiansen RECP, H:Aug-cc-pVTZ
    C   2    2 Y  Z    A
           53.    1
    I     1.59900000           0.00000000        0.00000000
    LARGE BASIS ECPCE46_TZ
    ECP 46 4 3
    # AREP
    # f
    4
      2        .922500       -1.447005
      2       2.569100      -14.188832
      2       7.908600      -43.263306
      1      25.061100      -27.740856
    # s-f
    6
      2       1.503600     -124.037542
      2       1.874600      235.545458
      2       2.682800     -261.475363
      2       3.446600      144.184313
      1       1.124200       31.003269
      0      11.458300        6.512373
    # p-f
    6
      2       1.245400      -95.368794
      2       1.582600      188.963297
      2       2.242900     -221.153567
      2       2.930100      104.680452
      1        .950300       30.809252
      0      12.782000        5.414640
    # d-f
    6
      2        .668500      -50.340552
      2        .828000      102.276429
      2       1.115700     -133.296812
      2       1.419000       75.010821
      1        .503000       19.150171
      0       4.557600        8.099959
    # spin-orbit
    # p-f
    6
      2       1.245400        5.416752
      2       1.582600        3.711418
      2       2.242900      -25.208180
      2       2.930100       27.780892
      1        .950300       -2.699622
      0      12.782000         .351446
    # d-f
    6
      2        .668500        -.406356
      2        .828000        2.040806
      2       1.115700       -3.283400
      2       1.419000        1.842046
      1        .503000        -.126998
      0       4.557600         .008442
    # f
    4
      2        .922500        -.019085
      2       2.569100         .035451
      2       7.908600        -.007995
      1      25.061100         .096830
         1.    1
    H     0.00000000           0.00000000        0.00000000
    LARGE BASIS aug-cc-pVTZ
    FINISH

For the explicitly typed ECP, the ECP keyword is used. Three numbers
after the ECP keyword denote the following:

1.  Number of core electrons: This number of core electrons is
    substituted by RECP. In this case, 46 core electrons in iodine atom
    are omitted and 7 electrons in the valence are described.
2.  Number of AREP blocks: The numbers of AREP blocks to be read.
3.  Number of SO blocks: The numbers of SOREP blocks to be read. In the
    case of AREP (spin-free) calculation, the number of SOREP blocks is
    set to 0.

After reading block numbers, AREP (and SOREP) blocks are read.

1.  AREP blocks: Three parameters in each line in AREP blocks are
    $`n_{li}`$, $`\alpha_{li}`$, and $`C_{li}`$ respectively in the
    following equation.

``` math
U_{l} = \sum_{l}C_{li}r^{n_{li}-2}e^{-\alpha_{li}r^{2}}
```

2.SO blocks: Three parameters in each line in SO blocks are same as in
AREP blocks.

Note: coefficients in the spin-orbit block are sometimes defined
differently by RECP developing groups. Spin-orbit potential in DIRAC is
defined as Park2012,

``` math
U_{l}^{SO,DIRAC} = \frac{2}{2l+1}\Delta U_{l}^{SOREP}
```

For example, factors (2/2l+1) are already multiplied in SO parameters at
the [Stuttgart RECP
webpage](http://www.tc.uni-koeln.de/PP/index.en.html). One can use it
directly without the modification. There are two kinds of SO factors in
the [RECPs of Christiansen and
coworkers](http://people.clarkson.edu/~pchristi/reps.html). The SO
factor in DIRAC is the same as the ones defined for Columbus program.

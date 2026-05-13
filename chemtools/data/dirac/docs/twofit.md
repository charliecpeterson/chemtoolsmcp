orphan  

# TWOFIT

TWOFIT is a utility program for the extraction of spectroscopic
constants ($`r_e`$, $`\omega_e`$, $`x_e\omega_e`$ ) of diatomic
molecules. It can also extract equivalent information from normal modes
provided the reduced mass $`\mu`$ is given.

## Theory

Consider the following Hamiltonian

``` math
\hat{H}=\hat{H}_{o}+\hat{H}_{1}+\hat{H}_{2}
```

where $`\hat{H}_{0}`$ is the Hamiltonian for the harmonic oscillator

``` math
\hat{H}_{0}=\frac{\hat{p}^{2}}{2\mu}+\frac{1}{2}V_{2}x^{2};\quad E_{v}^{(0)}=\hbar\omega(v+\frac{1}{2})
```

and $`\hat{H}_{1}`$ and $`\hat{H}_{2}`$ consists of cubic and quartic
contributions, respecetively, to a quartic force field

``` math
\hat{H}_{1}=\frac{1}{6}V_{3}x^{3};\quad\hat{H}_{2}=\frac{1}{24}V_{4}x^{4}
```

From second order perturbation theory we get the following energy

``` math
E_{v}=\hbar\omega(v+\frac{1}{2})+\frac{1}{24}V_{4}\left\langle v\left|x^{4}\right|v\right\rangle -\frac{V_{3}^{2}}{36}\left[\frac{\left|\left\langle v\left|x^{3}\right|v+1\right\rangle \right|^{2}-\left|\left\langle v\left|x^{3}\right|v-1\right\rangle \right|^{2}}{\hbar\omega}+\frac{\left|\left\langle v\left|x^{3}\right|v+3\right\rangle \right|^{2}-\left|\left\langle v\left|x^{3}\right|v-3\right\rangle \right|^{2}}{3\hbar\omega}\right]
```

Further manipulations then give the final expression

``` math
\begin{aligned}
\begin{array}{lcl}
E_{v} & = & \hbar\omega(v+\frac{1}{2})+\frac{1}{16}V_{4}\left(\frac{\hbar}{\mu\omega}\right)^{2}(v^{2}+v+\frac{1}{2})-\frac{5V_{3}^{2}}{48\hbar\omega}\left(\frac{\hbar}{\mu\omega}\right)^{3}(v^{2}+v+\frac{11}{30})\\
 & = & \hbar\omega(v+\frac{1}{2})-\left[\frac{5V_{3}^{2}}{48\hbar\omega}\left(\frac{\hbar}{\mu\omega}\right)^{3}-\frac{V_{4}}{16}\left(\frac{\hbar}{\mu\omega}\right)^{2}\right](v+\frac{1}{2})^{2}+\frac{1}{64}V_{4}\left(\frac{\hbar}{\mu\omega}\right)^{2}-\frac{7V_{3}^{2}}{576\hbar\omega}\left(\frac{\hbar}{\mu\omega}\right)^{3}\end{array}
\end{aligned}
```

to be compared with the general expression

``` math
E_{v}=(v+\frac{1}{2})\hbar\omega-(v+\frac{1}{2})^{2}\hbar\omega x_{e}
```

The anharmonic constant can accordingly be calculated as

``` math
\hbar\omega x_{e} = \left[\frac{5V_{3}^{2}}{48\hbar\omega}\left(\frac{\hbar}{\mu\omega}\right)^{3}-\frac{V_{4}}{16}\left(\frac{\hbar}{\mu\omega}\right)^{2}\right]
```

The average displacement is given as the standard deviation

``` math
\Delta x=\sigma =\sqrt{\left\langle x^{2}\right\rangle -\left\langle x\right\rangle ^{2}}=\sqrt{\left\langle x^{2}\right\rangle}=\sqrt{\frac{\hbar\left(\nu+\frac{1}{2}\right)}{m\omega}}
```

## Usage

The usage of TWOFIT is rather self-explanatory. TWOFIT will read a
formatted file of distances and energies prepared by the user. It will
then, through a dialogue with the user, perform a polynomial fit to
specified degree of the potential curve, find the equilibrium distance
$`r_e`$ by a Newton-Raphson search and from the extracted force
constants $`V_2`$, $`V_3`$ and $`V_4`$ calculate the harmonic frequency
$`\omega_e`$ as well as the anharmonic constant $`\omega_e x_{e}`$. For
a diatomic molecule, the program can look up masses of the most abundant
isotope for specified nuclear charge or accept atomic masses from the
user in order to calculate the reduced mass $`\mu`$. For normal modes,
the reduced mass must be given directly.

## Example1: Spectroscopic constants of HgF

The potential curve of mercury monofluoride HgF has been calculated at
the X2C/B3LYP level. We extract the following data into the file
`B3LYPpot` :

    1.94    -19746.664565273648
    1.96    -19746.665970540136
    1.98    -19746.667076184542
    2.00    -19746.667907574058
    2.02    -19746.668483409372
    2.04    -19746.668818993279
    2.06    -19746.668929911564

We fire up TWOFIT:

    twofit.x


             ***************************************************************
             ********** TWOFIT for diatomics : Written by T. Saue **********
             ***************************************************************

     Select one of the following:
       1. Spectroscopic constants.
       2. Spectroscopic constants + properties.

We chose the first option:

    1
    Name of input file with potential curve with " R E(R)" values (A40)

We provide the name of the file containing the potential curve:

    B3LYPpot
    Select one of the following:
      1. Bond lengths in Angstroms.
      2. Bond lengths in atomic units.
     (Note that all other quantities only in atomic units !)

The bond lengths are in this case in Angstroms:

    1
    * Number of points read:    7

    ** SPECTROSCOPIC CONSTANTS:

    Polynomial fit: Give order of polynomial

With seven points the maximal order of the polynomial is six. We select
order five, but the user is encouraged to try different options to see
how the calculated spectroscopic constants converge with the number of
points, their spacing as well as order of the polynomial:

    5
       * Polynomial fit of order:  5
       * Coefficients:
       c(  0):   -1.943938E+04
       c(  1):   -4.003912E+02
       c(  2):    2.091186E+02
       c(  3):   -5.469999E+01
       c(  4):    7.162816E+00
       c(  5):   -3.754896E-01
            X     Predicted Y Relative error
        1.940     -1.974666456527E+04 -2.5608E-14
        1.960     -1.974666597054E+04  1.3265E-13
        1.980     -1.974666707618E+04 -3.4212E-13
        2.000     -1.974666790758E+04  4.4879E-13
        2.020     -1.974666848340E+04 -3.4194E-13
        2.040     -1.974666881900E+04  1.3228E-13
        2.060     -1.974666892991E+04 -2.6345E-14
       * Chi square :  .1840E-15
       * Chi square per point:  .2628E-16
       * Number of singularities(SVD):   0
    * Local minimum   :           2.06044 Angstroms =            3.89366 Bohrs
    * Expected energy : -1.9746668930E+04 Hartrees
    Select one of the following:
      1. Select masses of the most abundant isotopes.
      2. Employ user-defined atomic masses.
      3. Normal modes: Give reduced mass.

We go for the easy solution:

    1
    * Give charge of atom A:

The order of atoms does not matter. We select mercury first:

    80
    * Mass     :    201.9706
    * Abundance:     29.8600
    * Give charge of atom B :

and then fluorine:

    9
    * Mass     :     18.9984
    * Abundance:    100.0000
    * Force constant  :        2.2407E+02 N/m
    * Frequency       : 1.40297E+13 Hz
                        4.67981E+02 cm-1
    * Mean displacement in harmonic ground state:    0.0644 Angstroms
      corresponding to interval: [    1.9960,    2.1248]
    * WARNING :     3 points lie outside interval...
    This may reduce the quality of your spectroscopic constants.
    * Omega*x_e       : 1.64740E+01 cm-1
     Do you want to calculate spectroscopic constants with other isotope(s) (y/n)?

Note that we get a warning, because the program checks whether all
points lie inside the standard deviation of the ground state (harmonic
approximation), corresponding to the mean displacement $`\pm\Delta x`$
(see above). Such a warning does not disqualify the fit, but the user
has to judge whether he has selected his opints wisely. Various options
follow, but we will simply say no to all of them:

    n
     Do you want to calculate spectroscopic constants
      with another polynomial order (y/n)?
    n
     Do you want to calculate spectroscopic constants at another geometry (y/n)?
    n

## Example 2: Anharmonicity of C-F stretch in CHFClBr

We start from a set of CCSD(T) energies (using scalar relativistic
pseudopotentials) along the normal coordinate $`Q`$ associated with the
C-F stretch of the chiral CHFClBr:

    -0.50  -610.12399429
    -0.40  -610.62520179
    -0.30  -610.86649287
    -0.20  -610.97769875
    -0.10  -611.02238183
     0.00  -611.03293760
     0.10  -611.02641387
     0.20  -611.01190951
     0.50  -610.95789506
     0.75  -610.91897869
     1.00  -610.89433750

The normal coordinates and associated normal coordinate $`\mu = 9.7032`$
was obtained from vibrational analysis (not using DIRAC). The TWOFIT run
proceeds as before:

    twofit.x


             ***************************************************************
             ********** TWOFIT for diatomics : Written by T. Saue **********
             ***************************************************************

    Select one of the following:
      1. Spectroscopic constants.
      2. Spectroscopic constants + properties.
    1
    Name of input file with potential curve with " R E(R)" values (A40)
    chfclbr.pes.dat
    Select one of the following:
      1. Bond lengths in Angstroms.
      2. Bond lengths in atomic units.
     (Note that all other quantities only in atomic units !)
    1
    * Number of points read:   11

    ** SPECTROSCOPIC CONSTANTS:

    Polynomial fit: Give order of polynomial 
    9
       * Polynomial fit of order:  9
       * Coefficients:
       c(  0):   -6.110329E+02
       c(  1):    6.084737E-05
       c(  2):    2.297324E-01
       c(  3):   -2.947082E-01
       c(  4):    2.477551E-01
       c(  5):   -1.473893E-01
       c(  6):    8.141347E-02
       c(  7):   -7.485671E-02
       c(  8):    4.956602E-02
       c(  9):   -1.180208E-02
            X     Predicted Y Relative error
       -0.500     -6.101239943742E+02  1.3793E-10
       -0.400     -6.106252010277E+02 -1.2483E-09
       -0.300     -6.108664959044E+02  4.9673E-09
       -0.200     -6.109776918295E+02 -1.1327E-08
       -0.100     -6.110223916741E+02  1.6111E-08
        0.000     -6.110329287639E+02 -1.4461E-08
        0.100     -6.110264185901E+02  7.7249E-09
        0.200     -6.110119083146E+02 -1.9564E-09
        0.500     -6.109578950951E+02  5.7392E-11
        0.750     -6.109189786861E+02 -6.3733E-12
        1.000     -6.108943375002E+02  4.0123E-13
       * Chi square :  .2564E-09
       * Chi square per point:  .2331E-10
       * Number of singularities(SVD):   0
    * Local minimum   :          -0.00007 Angstroms =           -0.00013 Bohrs
    * Expected energy : -6.1103292877E+02 Hartrees
    Select one of the following:
      1. Select masses of the most abundant isotopes.
      2. Employ user-defined atomic masses.
      3. Normal modes: Give reduced mass.

but now we select the third option:

    3
    * Give reduced mass in Daltons:
    9.7032
    * Force constant  :        7.1570E+02 N/m
    * Frequency       : 3.35432E+13 Hz
                     1.11888E+03 cm-1
    * Mean displacement in harmonic ground state:    0.0557 Angstroms
    corresponding to interval: [   -0.0558,    0.0557]
    * WARNING :    10 points lie outside interval...
    This may reduce the quality of your spectroscopic constants.
    * Omega*x_e       : 9.10592E+00 cm-1

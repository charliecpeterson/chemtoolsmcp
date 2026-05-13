orphan  

# VIBCAL

VIBCAL is a utility program allowing the calculation of vibrationally
averaged properties.

## Theory

Suppose that we have optimized the geometry of our molecule and have
carried out a harmonic vibrational analysis. For a selected normal mode
we expand the electronic energy $`E`$ as function of the associated
normal coordinate $`q`$ about the equilibrium position $`q=0`$.

``` math
E(q) = E^{[0]} + E^{[1]}q + \frac{1}{2}E^{[2]}q^2 + \frac{1}{6}E^{[3]}q^3 + \ldots;\quad
       E^{[n]} = \left.\frac{\partial^n E}{\partial q^n}\right|_{q=0}
```

We note that $`q=0`$ implies $`E^{[1]}=0`$ and that the second
derivative $`E^{[2]}`$ corresponds to the force constant $`k`$. We
furthermore expand some property $`P\equiv P(q)`$ in the same manner.

From perturbation theory we now obtain an expression for the expectation
value of the property $`P`$ for vibrational level $`\nu`$ of the
selected normal mode

``` math
P_{\nu}=\left<\nu\left|P(q)\right|\nu\right>_q = P^{[0]} 
+ \frac{1}{2}P^{[2]}\left(\frac{\hbar}{\mu\omega_e}\right)\left(\nu+\frac{1}{2}\right)
   -\frac{1}{2}P^{[1]}E^{[3]}\left(\frac{\hbar}{\mu\omega_e}\right)^2\frac{\left(\nu+\frac{1}{2}\right)}{\hbar\omega_e}+\ldots
```

where appears the reduced mass $`\mu`$ and harmonic frequency
$`\omega_e=\sqrt{\frac{k}{\mu}}`$ of the selected normal mode.

Experimentally one can observe the change in this expectation value
between two vibrational levels, given by

``` math
P_{\nu'} - P_{\nu} = \frac{1}{2}\frac{\hbar}{\mu\omega_e}\left[P^{[2]}-\frac{1}{\mu\omega_e^2}P^{[1]}E^{[3]}\right]\Delta\nu;\quad \Delta\nu=\nu' - \nu
```

We note that the purely electronic contribution $`P^{[0]}`$ does not
contribute to the vibrational shift. Furthermore, we may note that for a
purely harmonic mode ($`E^{[3]}=0`$) and a property purely linear in the
normal coordinate ($`P^{[2]}=0`$) the vibrational shift is strictly
zero.

## Usage

VIBCAL will read a formatted file of distances, energies and property
values. The distances may be in Angstroms, but all other quantities have
to be in atomic units. VIBCAL will next6, through a dialogue, with the
user generate the necessary derivatives for the calculation of
vibrational shifts of the property.

## Example: Parity violation shift associated with the C-F stretch of CHFClBr

We start from a set of CCSD(T) energies and B3LYP parity violation
energies calculated along the normal coordinate $`Q`$ associated with
the C-F stretch of the chiral CHFClBr:

    -0.5000000000000000  -0.61012399429E+03     2.633089925380E-17
    -0.2000000000000000  -0.61097769875E+03     9.078015651573E-18
    -0.1000000000000000  -0.61102238183E+03     4.933686545396E-18
     0.0000000000000000  -0.61103293760E+03     1.544047622021E-18
     0.1000000000000000  -0.61102641387E+03    -1.026099838285E-18
     0.2000000000000000  -0.61101190951E+03    -2.755550508570E-18
     0.5000000000000000  -0.61095789506E+03    -1.870247865432E-18

The coordinates are in this case given in Angstroms. We fire up VIBCAL:

    $vibcal.x


                       *******************************************
                       ********** VIBCAL for properties **********
                       *******************************************

    Set print level [default:0]:
          0. Basic print level 
          1. Print additional information

    Reduced mass (in Daltons)

Here we chose the default print level. For the reduced mass you have to
check with the preceeding vibrational analysis. We give:

    9.7032
    Name of the input file (A30) with potential/property curve

It is a good idea to have a telling name of the input file :

    CCpot_B3LYPprp
    Select one of the following:
      1. Bond lengths in Angstroms.
      2. Bond lengths in atomic units.
     (Note that all other quantities only in atomic units !)
    1

Here we choose Angstroms for the normal coordinate. :

    Select one of the following:
      1. Numerical derivatives (assumes fixed step length).
      2. Polynomial fits    
    2

In this example we do not have a fixed step length for our distances, so
we select the second option which uses polynomials fits. :

    We have    7 points
    Give order of polynomial:
    5
    Select point for the calculation of derivatives
    0.0

The distances refer to the normal coordinate associated with the C--F
stretch, so we set $`Q=0.0`$ since this refers to the equilibrium
structure. :

    Results have been written to CCpot_B3LYPprp.vibcal

Looking into this file we find :

    ******************************
    ****  SOME INFORMATION    ****
    ******************************
    Reduced mass in Daltons:    9.7032000000000007     
    2nd potential derivative:   0.42167678827523780     
    3rd potential derivative:   -1.6807622640206681     
    4th potential derivative:    9.1374786165355175     
    0th property derivative:   1.55000030970252530E-018
    1st property derivative:  -1.58073567198751495E-017
    2nd property derivative:   2.23189000229251784E-017

    ******************************
    ********   RESULTS    ********
    ******************************
    Difference : -0.24366752E-18 au =       -1.6 mHz
    PV shift:      -3.207 mHz
    Harmonic model:
    Difference :  0.12921573E-18 au =        0.9 mHz
    PV shift:       1.700 mHz

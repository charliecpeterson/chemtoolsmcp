orphan  

# Core electron excitations and ionization in H2O at the HF and DFT levels

## Introduction

We want to study excitation of the oxygen 1s electron of water.

## Restricted excitation window TD-HF (REW-TD-HF)

Starting from the molecular input file
<span class="title-ref">H2O.mol</span>

<div class="literalinclude">

H2O.mol

</div>

and the menu file <span class="title-ref">H2O.inp</span>

<div class="literalinclude">

H2O.inp

</div>

we first run a Hartree-Fock calculation of the ground electronic state:

    pam --mol=H2O --inp=H2O --outcmo

From Mulliken population analysis we confirm that the oxygen 1s orbital
is the first orbital:

    * Electronic eigenvalue no.  1: -20.581089118212       (Occupation : f = 1.0000)           
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L A1 O  s   
    --------------------------------------
     alpha    1.0000  |      1.0000
     beta     0.0000  |      0.0000

We now focus on electric dipole allowed transitions. The molecular
symmetry is $`C_{2v}`$ where the components of the electric dipole
operator $`-e(x,y,z)`$ span irreps $`(B_1,B_2,A_1)`$. This leads us to
set up the following input

<div class="literalinclude">

H2O_O1s.inp

</div>

We run our calculation:

    pam --inp=H2O_O1s --mol=H2O --incmo

and find the isotropically averaged oscillator strenghts

<div class="literalinclude">

H2O_1s_spectrum

</div>

where we have added by hand the most important virtual orbitals. For
reference, orbital densities of the fifteen first canonical HF orbitals
of water are given below

![alternate text](H2Oorbs.png)

We may simulating the spectrum using a Lorentzian lineshape with
half-width at half-maximum (HWHM) equal to 0.005 a.u.

![alternate text](water_O1s_tdhf.png)

## Complex response

We can alternatively obtain the above spectrum from complex response.
The isotropically averaged ocillator strength associated within the
electric dipole approximation is related to the isotropic electric
dipole polarizability through the relation

``` math
f^{iso}\left(\omega\right)=\frac{2m\omega}{\pi e^2} Im\left[\alpha^{iso}\left(\omega+i\gamma\right)\right]
```

where $`\gamma`$ is a damping parameter corresponding to the half-width
at half-maximum (HWHM) of the Lorentzian lineshape.

We can calculate the real and imaginary parts of the polarizability by
complex response. By choosing a frequency window we can directly access
the region of the spectrum of interest.

We use the input file

<div class="literalinclude">

cpp.inp

</div>

and the command:

    pam --incmo --mol=H2O --inp=cpp

To get enough data points we do a second run using:

    .FREQ INTERVAL
    20.01 21.0 0.02

Extracting the imaginary part of the frequency-dependent electric dipole
polarizatibility and converting to oscillator strength we can directly
plot the spectrum as below

![alternate text](water_O1s_cpp.png)

In the above graph we have also included the results from REW-TDDFT and
they completely overlap, except that a keen eyemay note that REW-TDDFT
is missing the peak around 570 eV, simply because not enough excitations
were specified in the input.

## Localizing the K edge

An interesting question is how to localize the K edge in the XANES
spectrum. A first approximation to the 1s ionization energy is provided
by Koopmans' theorem. We find $`IP_{1s}\approx -\varepsilon_{1s} =`$
20.581089 $`E_h`$ = 560.04 eV. This is singificantly off the value 539.7
eV reported by Kai Siegbahn and co-workers \[*ESCA applied to free
molecules* (1969)\] Siegbahn1969 ( see also
[here](http://srdata.nist.gov/xps/XPSDetailPage.aspx?AllDataNo=21090) ,
but here the work function of the reference metal must be subtracted).
reported by Kai Siegbahn and co-workers in 1977(?). We know that
Koopman's theorem ignores correlation and orbital relaxation. For
valence ionization Koopman's theorem often provides a reasonable
approximation since the errors tend to cancel each other. For core
excitations orbital relaxation dominates such that Koopman's theorem
greatly overestimates ionization energies.

To see this we carry our a average-of-configuration (AOC) calculation of
the 1s core-ionized system. Starting from the coefficients from the
neutral system and the input

<div class="literalinclude">

H2O_1s.inp

</div>

and the command:

    pam --incmo --mol=H2O --inp=H2O_1s

we find a total energy of -56.287847 $`E_h`$ for the core ionized system
compared to -76.115149 $`E_h`$ for the neutral system. This corresponds
to a $`\Delta`$ SCF value of 539.52 eV for the ionization energy,
tantalizingly close to experiment.

In passing we that note that in the first iteration of this calculation
we obtain a total energy of -55.534060 $`E_h`$. This is the energy of
the core-ionized system obtained using the orbitals of the neutral
system. Koopman's theorem is obtained by subtracting this energy from
the energy of the neutral system. For the neutral system we obtained
-76.115149 $`E_h`$, so by taking the difference we obtain 20.581089
$`E_h`$, which is exactly the *1s* orbital energy in the neutral system.

You should also note that we easily converge to the core-ionized system
using reordering and overlap selection: In the input we have specified
eight electrons in four inactive (closed) orbitals, followed by a single
electron in an active (open) orbital. We want the O1s to be the active
orbital, but this is not achieved automatically since DIRAC will
normally order orbitals according to their energy. We therefore start be
reordering the orbitals such that the O1s orbital from the previous
calculation on the neutral system comes out on top of the occupied
orbitals. However, this is not enough to converge to the desired state
since after the first diagonalization DIRAC will again by default order
orbitals according to their energy. This is why we use *overlap
selection*, that is, we ask DIRAC to rather order orbitals according to
their overlap with some reference orbitals. By default (dynamic overlap
selection) this will be the orbitals from the previous iteration.
However, in this case we activate *non-dynamic* overlap selection, which
means that we order orbitals according to their overlap with the
starting orbitals.

> > [!NOTE]
> > Overlap selection is nowadays marketed hard as MOM (Maximum Orbital
> > Method, see Gilbert_JPCA2008), but this method has been included in
> > DIRAC for at least two decades and goes back to the pioneering work
> > of [Paul Bagus](http://cascam.unt.edu/people/psbagus.htm) It was
> > used in Bagus_JCP1971, but not reported explicitly. However, it is
> > for instance documented in the 1970 manual of the ALCHEMY program
> > ALCHEMY1970 (in French ! On pdf page 9 you find a description of
> > keyword MOORDR using a "maximum overlap criterion").

The K edge obtained by $`\Delta SCF`$ does not correspond to that of our
TD-HF or complex reponse calculations since they only allow linear
reponse (orbital relaxation).

In the figure below we have plotted oscillator strength per atom for 1s
core excitation spectrum for water, taken from the [Gas Phase Core
Excitation Database](http://unicorn.mcmaster.ca/corex/cedb-title.html)
and recorded at 0.7 eV fwhm.

![alternate text](H2O_O1s_exp.png)

The vertical orange line corresponds to the O1s binding energy reported
by Siegbahn and co-workers and seems to be too early. We have also
plotted the spectrum obtained by REW-TDHF with a Lorentzian wideshape
corresponding to the fwhm of the experiment. In green we plot the
original spectrum, whereas in red it has been shifted so that our linear
response estimate for the ionization energy has been aligned with the
experimental O1s binding energy. We can see that the shift improves
agreement, but the spacing and relative intensities of peaks do not
agree with experiment.

## Static Exchange Approximation

In order to incorporate orbital relaxation we carry out a STEX
calculation. At the moment symmetry is not implemented, so we turn off
symmetry in the nolecular input file

<div class="literalinclude">

H2O_C1.mol

</div>

We then first run the ground state:

    pam --inp=H2O --mol=H2O_C1 --outcmo

followed by the 1s core-ionized state:

    pam --mol=H2O_C1 --inp=H2O_1s --incmo 

We now run STEX using the input

<div class="literalinclude">

stex.inp

</div>

and the command:

    pam --inp=stex --mol=H2O_C1 --incmo --put "H2O_1s_H2O_C1.h5=ION.h5"

We have plotted the STEX spectrum below together with the experimental
one

![alternate text](stex.png)

## Switching to DFT

Let us now look at what we can do with DFT. The first thing to note is
that Koopman's theorem does not hold:

<table style="width:96%;">
<colgroup>
<col style="width: 15%" />
<col style="width: 22%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<tbody>
<tr>
<td></td>
<td></td>
<td><blockquote>
<p>HF(AOC)</p>
</blockquote></td>
<td><blockquote>
<p>HF(focc)</p>
</blockquote></td>
<td><blockquote>
<p>LDA</p>
</blockquote></td>
<td><blockquote>
<p>PBE</p>
</blockquote></td>
<td><blockquote>
<p>PBE0</p>
</blockquote></td>
</tr>
<tr>
<td>Neutral</td>
<td>Energy</td>
<td>-76.115149</td>
<td>-76.115149</td>
<td>-75.962033</td>
<td>-76.438943</td>
<td>-76.437412</td>
</tr>
<tr>
<td></td>
<td><span
class="math inline"><em>ε</em><sub>1<em>s</em></sub></span></td>
<td>-20.581089</td>
<td>-20.581089</td>
<td>-18.620721</td>
<td>-18.766629</td>
<td>-19.223080</td>
</tr>
<tr>
<td><span class="math inline">1<em>s</em><sup>−1</sup></span></td>
<td>Energy</td>
<td>-56.287847</td>
<td>-55.070646</td>
<td>-55.980781</td>
<td>-56.300366</td>
<td>-56.069079</td>
</tr>
<tr>
<td></td>
<td>Energy(0)</td>
<td>-55.534060</td>
<td>-54.347068</td>
<td>-55.231846</td>
<td>-55.540835</td>
<td>-55.319556</td>
</tr>
<tr>
<td><span class="math inline"><em>Δ</em><em>E</em></span></td>
<td>relax</td>
<td>-19.827303</td>
<td>-21.044503</td>
<td>-19.981252</td>
<td>-20.138577</td>
<td>-20.368333</td>
</tr>
<tr>
<td></td>
<td>norelax</td>
<td>-20.581089</td>
<td>-21.768082</td>
<td>-20.730186</td>
<td>-20.898108</td>
<td>-21.117856</td>
</tr>
</tbody>
</table>

<table style="width:96%;">
<colgroup>
<col style="width: 18%" />
<col style="width: 25%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 12%" />
</colgroup>
<tbody>
<tr>
<td></td>
<td></td>
<td><blockquote>
<p>BP86</p>
</blockquote></td>
<td><blockquote>
<p>BLYP</p>
</blockquote></td>
<td><blockquote>
<p>B3LYP</p>
</blockquote></td>
<td><blockquote>
<p>CAMB3LYP</p>
</blockquote></td>
</tr>
<tr>
<td>Neutral</td>
<td>Energy</td>
<td>-76.525093</td>
<td>-76.508410</td>
<td>-76.487092</td>
<td>-76.496078</td>
</tr>
<tr>
<td></td>
<td><span
class="math inline"><em>ε</em><sub>1<em>s</em></sub></span></td>
<td>-18.786866</td>
<td>-18.791935</td>
<td>-19.145210</td>
<td>-19.219699</td>
</tr>
<tr>
<td><span class="math inline">1<em>s</em><sup>−1</sup></span></td>
<td>Energy</td>
<td>-56.366800</td>
<td>-56.340103</td>
<td>-56.148003</td>
<td>-56.115449</td>
</tr>
<tr>
<td></td>
<td>Energy(0)</td>
<td>-55.606191</td>
<td>-55.582410</td>
<td>-55.398934</td>
<td>-55.366886</td>
</tr>
<tr>
<td><span class="math inline"><em>Δ</em><em>E</em></span></td>
<td>relax</td>
<td>-20.158293</td>
<td>-20.168307</td>
<td>-20.339089</td>
<td>-20.380629</td>
</tr>
<tr>
<td></td>
<td>norelax</td>
<td>-20.918902</td>
<td>-20.926000</td>
<td>-21.088158</td>
<td>-21.129191</td>
</tr>
</tbody>
</table>

In the above table the entry <span class="title-ref">Energy(0)</span> is
the total energy of the core-ionized system calculated using the
orbitals of the neutral system. When we calculate the energy difference
between the neutral and core-ionized system using the orbitals of the
neutral system we reproduce the 1s orbital energy to the cited decimals,
but this is not the case of any other method, including HF using
fractional occupation.

Below we give the $`\Delta SCF`$ numbers is eV. Interestingly AOC-HF
at539.5 eV easily comes closest to the experimental value 539.7 eV. It
can furthermore be seen that $`\Delta SCF`$ values obtained with DFT
functionals show the trend LDA \< GGA \< hybrid with LDA closest, but
still far from experiment. Switching from average-of-configuration to
fractional occupation at the HF level leads to a dramatic deterioration
of the agreement with experiment.

<table style="width:95%;">
<colgroup>
<col style="width: 18%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 7%" />
</colgroup>
<tbody>
<tr>
<td></td>
<td><blockquote>
<p>HF(AOC)</p>
</blockquote></td>
<td><blockquote>
<p>HF(focc)</p>
</blockquote></td>
<td><blockquote>
<p>LDA</p>
</blockquote></td>
<td><blockquote>
<p>PBE</p>
</blockquote></td>
<td><blockquote>
<p>PBE0</p>
</blockquote></td>
<td><blockquote>
<p>BP86</p>
</blockquote></td>
<td><blockquote>
<p>BLYP</p>
</blockquote></td>
<td><blockquote>
<p>B3LYP</p>
</blockquote></td>
<td><blockquote>
<p>CAMB3LYP</p>
</blockquote></td>
</tr>
<tr>
<td><span
class="math inline"><em>ε</em><sub>1<em>s</em></sub></span></td>
<td><blockquote>
<p>560.0</p>
</blockquote></td>
<td><blockquote>
<p>560.0</p>
</blockquote></td>
<td><blockquote>
<p>506.7</p>
</blockquote></td>
<td><blockquote>
<p>510.7</p>
</blockquote></td>
<td><blockquote>
<p>523.1</p>
</blockquote></td>
<td><blockquote>
<p>511.2</p>
</blockquote></td>
<td><blockquote>
<p>511.4</p>
</blockquote></td>
<td><blockquote>
<p>521.0</p>
</blockquote></td>
<td><blockquote>
<p>523.0</p>
</blockquote></td>
</tr>
<tr>
<td><span class="math inline"><em>Δ</em><em>E</em></span> (relax)</td>
<td><blockquote>
<p>539.5</p>
</blockquote></td>
<td><blockquote>
<p>572.7</p>
</blockquote></td>
<td><blockquote>
<p>564.1</p>
</blockquote></td>
<td><blockquote>
<p>568.7</p>
</blockquote></td>
<td><blockquote>
<p>574.6</p>
</blockquote></td>
<td><blockquote>
<p>569.2</p>
</blockquote></td>
<td><blockquote>
<p>569.4</p>
</blockquote></td>
<td><blockquote>
<p>573.8</p>
</blockquote></td>
<td><blockquote>
<p>575.0</p>
</blockquote></td>
</tr>
<tr>
<td><span class="math inline"><em>Δ</em><em>E</em></span> (no
relax)</td>
<td><blockquote>
<p>560.0</p>
</blockquote></td>
<td><blockquote>
<p>592.3</p>
</blockquote></td>
<td><blockquote>
<p>543.7</p>
</blockquote></td>
<td><blockquote>
<p>548.0</p>
</blockquote></td>
<td><blockquote>
<p>554.3</p>
</blockquote></td>
<td><blockquote>
<p>548.5</p>
</blockquote></td>
<td><blockquote>
<p>548.8</p>
</blockquote></td>
<td><blockquote>
<p>553.5</p>
</blockquote></td>
<td><blockquote>
<p>554.6</p>
</blockquote></td>
</tr>
</tbody>
</table>

At the DFT level we have employed an energy expression based on
fractional occupation, which is the model that leads to Janak's theorem,
namely that the derivative of the energy with respect to occupation
number $`n_i`$ gives the energy of the corresponding orbital, that is

``` math
\frac{dE}{dn_i} = \varepsilon_i
```

We may investigate Janak's theorem numerically. We set up a script to do
DFT calculations with fractional occupation from 1.0 to 1.9 of the
oxygen 1s orbital

<div class="literalinclude">

janak.sh

</div>

and also add the energy of the neutral system to our data set. We then
carry out polynomial fits to various orders and calculate the derivative
of the energy at occupation 2.0 with respect to 1s occupatio number. We
then obtain

<table style="width:95%;">
<colgroup>
<col style="width: 17%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
</colgroup>
<tbody>
<tr>
<td></td>
<td><blockquote>
<p>HF(AOC)</p>
</blockquote></td>
<td><blockquote>
<p>HF(focc)</p>
</blockquote></td>
<td><blockquote>
<p>LDA</p>
</blockquote></td>
<td><blockquote>
<p>PBE</p>
</blockquote></td>
<td><blockquote>
<p>PBE0</p>
</blockquote></td>
<td><blockquote>
<p>BP86</p>
</blockquote></td>
<td><blockquote>
<p>BLYP</p>
</blockquote></td>
<td><blockquote>
<p>B3LYP</p>
</blockquote></td>
<td><blockquote>
<p>CAMB3LYP</p>
</blockquote></td>
</tr>
<tr>
<td>1</td>
<td>-19.824314</td>
<td>-21.044193</td>
<td>-19.981185</td>
<td>-20.138613</td>
<td>-20.368425</td>
<td>-20.158286</td>
<td>-20.168248</td>
<td>-20.339084</td>
<td>-20.380650</td>
</tr>
<tr>
<td>2</td>
<td>-18.196672</td>
<td>-20.579045</td>
<td>-18.618588</td>
<td>-18.765178</td>
<td>-19.222619</td>
<td>-18.785047</td>
<td>-18.789831</td>
<td>-19.143977</td>
<td>-19.218655</td>
</tr>
<tr>
<td>3</td>
<td>-18.220409</td>
<td>-20.581507</td>
<td>-18.619168</td>
<td>-18.764949</td>
<td>-19.221929</td>
<td>-18.785159</td>
<td>-18.790354</td>
<td>-19.144055</td>
<td>-19.218529</td>
</tr>
<tr>
<td>4</td>
<td>-18.220112</td>
<td>-20.581073</td>
<td>-18.620943</td>
<td>-18.766866</td>
<td>-19.223251</td>
<td>-18.787109</td>
<td>-18.792168</td>
<td>-19.145388</td>
<td>-19.219883</td>
</tr>
<tr>
<td>5</td>
<td>-18.220135</td>
<td>-20.581093</td>
<td>-18.620699</td>
<td>-18.766601</td>
<td>-19.223061</td>
<td>-18.786846</td>
<td>-18.791909</td>
<td>-19.145191</td>
<td>-19.219680</td>
</tr>
<tr>
<td>6</td>
<td>-18.220131</td>
<td>-20.581090</td>
<td>-18.620724</td>
<td>-18.766633</td>
<td>-19.223084</td>
<td>-18.786872</td>
<td>-18.791940</td>
<td>-19.145213</td>
<td>-19.219703</td>
</tr>
<tr>
<td>7</td>
<td>-18.220130</td>
<td>-20.581089</td>
<td>-18.620720</td>
<td>-18.766628</td>
<td>-19.223080</td>
<td>-18.786869</td>
<td>-18.791934</td>
<td>-19.145209</td>
<td>-19.219699</td>
</tr>
<tr>
<td>8</td>
<td>-18.220130</td>
<td>-20.581089</td>
<td>-18.620721</td>
<td>-18.766629</td>
<td>-19.223080</td>
<td>-18.786857</td>
<td>-18.791935</td>
<td>-19.145210</td>
<td>-19.219699</td>
</tr>
<tr>
<td>9</td>
<td>-18.220130</td>
<td>-20.581089</td>
<td>-18.620721</td>
<td>-18.766629</td>
<td>-19.223080</td>
<td>-18.786879</td>
<td>-18.791935</td>
<td>-19.145210</td>
<td>-19.219699</td>
</tr>
<tr>
<td><span
class="math inline"><em>ε</em><sub>1<em>s</em></sub></span></td>
<td>-20.581089</td>
<td>-20.581089</td>
<td>-18.620721</td>
<td>-18.766629</td>
<td>-19.223080</td>
<td>-18.786866</td>
<td>-18.791935</td>
<td>-19.145210</td>
<td>-19.219699</td>
</tr>
</tbody>
</table>

We see that a linear fit is clearly is insufficient, whereas a quadratic
fit is reasonable. However, a 8th order fit is in general needed to
converge the energy derivative to the 1s orbital energy with the cited
number of decimals.

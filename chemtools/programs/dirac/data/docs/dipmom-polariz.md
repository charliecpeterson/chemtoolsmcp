orphan  

# Finite-field Coupled Cluster calculations of dipole moment and polarizability

## BeH electric properties

This toy molecule in the small STO-2G decontracted basis demonstrates
the use of finite-field perturbation theory for calculating the dipole
moment ($`\mu_{z}`$) and the polarizability ($`\alpha_{zz}`$) both at
the Hartree-Fock and at the highly correlated Coupled Cluster level. Our
molecule is oriented in the z-axis.

Because BeH is open-shell system, we can not employ MP2 and CCSD
gradients, which are available only for closed-shell systems. We have to
resort to the finite-field perturbation scheme, where we add the
z-dipole moment operator as small perturbation to the X2C relativistic
Hamiltonian. The symmetry of heterodiatomic molecule must be lowered
from linear to C2v because of the z-oriented dipole perturbation
operator.

### Molecule in electric field

When a molecular system is placed in a homogenous static electric field,
F=\[Fx,Fy,Fz\] (p,q-components), its energy may be expanded in a Taylor
serie as follows:

<span label="expansion">
``` math
E=E_{0}-\mu_{p}F_{p} - \frac{1}{2}\alpha_{pq}F_{p}F_{q} ...
```
</span>

We keep only z-component of the field, F=Fz, and the dipole moment is
obtained as the first numerical derivative of the energy according to
the electric field at zero field strength:

<span label="mu">
``` math
\mu_{z}=-\Biggl( \frac{\partial E(F_{z})}{\partial F_{z}} \Biggr)_{F_{z}=0}
```
</span>

The polarizability (for simplicity, here the zz-component) is calculated
via second numerical derivative of the energy according to the electric
field at zero field strength:

<span label="alpha">
``` math
\alpha_{zz}=-\Biggl( \frac{\partial^{2} E(F_{z})}{\partial F^{2}_{z}} \Biggr)_{F_{z}=0}
```
</span>

### Method of computation

Let us first compute the total energies of few electric field strength.
This can be done through replacing the string in the DIRAC input file by
the field strength:

    pam --noarch  --replace zff=+0.0005 --mol=BeH.sto-2g.C2v.mol --inp=BeH.x2c_scf_relcc_ffz.inp --put "DFPCMO.BeH.x2c_scf_sto-2g.C2v=DFPCMO"

(Input files for download are
`BeH.x2c_scf_relcc_ffz.inp  <../../../test/ffpt_dipmom_polariz_relcc/BeH.x2c_scf_relcc_ffz.inp>`,
`BeH.sto-2g.C2v.mol <../../../test/ffpt_dipmom_polariz_relcc/BeH.sto-2g.C2v.mol>`
and the text MO file,
`DFPCMO.BeH.x2c_scf_sto-2g.C2v <../../../test/ffpt_dipmom_polariz_relcc/DFPCMO.BeH.x2c_scf_sto-2g.C2v>`
.)

The total SCF and CCSD(T) energies varrying on field strengths are
sorted in the following Table:

> | Fz      | E(SCF,Fz)           | E(CCSD(T),Fz)       |
> |---------|---------------------|---------------------|
> | +0.0010 | -14.424976681856371 | -14.461324673372751 |
> | +0.0005 | -14.425860788235255 | -14.462193434302819 |
> | +0.0000 | -14.426746572346620 | -14.463063980838802 |
> | -0.0005 | -14.427634036448282 | -14.463936314225093 |
> | -0.0010 | -14.428523182834038 | -14.464810435757153 |

NOTE: In this quick DIRAC test, only two output files are provided for
Coupled Cluster method,
`BeH.x2c_scf_relcc_-0.0005_BeH.sto-2g.C2v.out  <../../../test/ffpt_dipmom_polariz_relcc/result/BeH.x2c_scf_relcc_-0.0005_BeH.sto-2g.C2v.out>`
,
`BeH.x2c_scf_relcc_+0.0005_BeH.sto-2g.C2v.out  <../../../test/ffpt_dipmom_polariz_relcc/result/BeH.x2c_scf_relcc_+0.0005_BeH.sto-2g.C2v.out>`.

#### Spreadsheet

In the attached spreadsheet (accesible via Gnumerics, Libre Office Calc
or MS Excel), we interpolate five pairs, \[F,E(F)\], of the Table
`mytable` with the 4th-degree polynomial:

<span label="expansion4">
``` math
E(F)=a_{0}+a_{1}F+a_{2}F^{2}+a_3F{3}+a_{4}F^{4}
```
</span>

The negative first derivative of the polynomial `expansion4` at zero
field is the dipole moment, based on the equation `mu`:

<span label="mu_polyn">
``` math
\mu_{z} = - \Biggl( \frac{\partial E(F)}{\partial F} \Biggr)_{F=0} = -a_{1}
```
</span>

The second derivative (with minus sign) is the polarizability, following
prescription in `alpha`:

<span label="alpha_polyn">
``` math
\alpha_{zz} =- \Biggl( \frac{\partial^{2} E(F)}{\partial F^{2}} \Biggr)_{F=0} = -2a_{2}
```
</span>

In the spreadsheet environment we employ the function LINEST. For that
you have to prepare columns with finite-field stregths powered to 1, 2,
3 and 4.

The binary spreadsheet file available for download is
`data.ods <../../../test/ffpt_dipmom_polariz_relcc/data.ods>`.

### Dipole moment

We calculate here the electronic part of the z-component of the dipole
moment as the perturbing field goes in the z-direction.

#### Expectation value

For the SCF method, we can obtain the dipole moment - both for closed
and open-shell systems - via the expectation value, which gives all
three components, $`\mu_{x}`$, $`\mu_{y}`$ and $`\mu_{z}`$ . A good
practise is that dipole moment finite-field calculations are verified
against the SCF/DFT expectation value.

The SCF expectation value dipole moment - its zz-electronic
contribution - is $`\mu_{z}`$ =-1.77324745 au. (The input file for
download,
`BeH.x2c_scf_dipmom.inp  <../../../test/ffpt_dipmom_polariz_relcc/BeH.x2c_scf_dipmom.inp>`
and the the corresponding output file,
`BeH.x2c_scf_dipmom_BeH.sto-2g.C2v.out  <../../../test/ffpt_dipmom_polariz_relcc/result/BeH.x2c_scf_dipmom_BeH.sto-2g.C2v.out>`.)

#### Finite-field values

From the spreadsheet function table, we obtain the value of $`a_{1}`$
=-1.7732474 a.u., which is the expansion coefficient of the first order
field.

Simple first numerical derivation of the SCF perturbed energy according
to the applied electric field, equation `mu`,

<span label="mu_deriv">
``` math
\mu_{z} = - \Biggl( \frac{E(+F)-E(-F)}{2F} \Biggr),
```
</span>

gives -1.771568 a.u. for F=0.0005. For the field strength of F=0.001 it
is -1.76989 a.u.

The zero field derivative of the fourth order polynomial, equation
`mu_polyn` is more precise when comparing against the above mentioned
SCF expectation value, though finite field evaluation is still capable
to give value accurate to two decimal places.

### Polarizability

For closed-shell systems, the polarizability at the SCF and DFT levels
can be calculated via linear response method, which gives all components
of the polarizability tensor. For systems with unpaired electrons like
this one, however, we have to resort to finite-field perturbative
calculations. For simplicity, we focus on the zz-tesnor component of the
polarizability, $`\alpha_{zz}`$

#### SCF

For our molecule, the simple numerical second derivate calculation for
given F=0.0005 at SCF level, equation `alpha`

<span label="alpha_deriv">
``` math
\alpha_{zz} = - \Biggl( \frac{E(+F)+E(-F)-2E(0)}{F^{2}} \Biggr),
```
</span>

gives 6.71996 a.u.. This is close to the second order expansion
coefficient, multiplied by 2, giving the value of
$`\alpha_{zz}=-2a_{2}`$ =(-2)\*(-3.3599745995852954)=6.71994919 a.u.

#### CCSD(T)

Similar numerical derivation for perturbed CCSD(T) energies and for
F=0.0005, according to equation `alpha_deriv`, gives the value of
$`\alpha_{zz}`$ =7.147401232 a.u. The corresponding expansion
coefficient (multiplied by the factor of two) from equation
`alpha_polyn`, produces the value of $`\alpha_{zz}`$ =7.14738420 a.u.

One can conclude that the F=0.0005 field strength is sufficient to
obtain polarizability accurate to 3 decimal places by simple numerical
second derivation, equation `alpha_deriv`.

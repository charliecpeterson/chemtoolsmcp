orphan  

<div class="index">

\*XAMFI

</div>

# \*XAMFI

Options for the (extended) atomic-mean field aka **(e)amf** module for
the `HAMILTONIAN_.X2C` Hamiltonian, developed and written by S. Knecht,
M. Repisky, H. J. Aa. Jensen and T. Saue, 2013-2022. The HDF5 interface
for XAMFI was developed and implemented by K. N. Spauszus (U
Bonn/Germany) and S. Knecht.

> [!WARNING]
> The XAMFI module requires a working HDF5 installation as well as h5py
> in your python env!

The **(e)amf** module has been described in Knecht2022. Based on atomic
calculation it provides two-electron scalar-relativistic **and**
spin-orbit picture-change corrections for the two-component X2C
Hamiltonian (in a molecular or atomic) framework. Since it is completely
integrated in the SCF framework of DIRAC, it can explot all its
capabilities for efficient atomic calculations, for example the use of
atomic supersymmetry.

For a detailed example of how to use the `\*XAMFI` module, see the
**x_amfi** test of the DIRAC test suite or have a look at the
[tutorial](http://www.diracprogram.org/doc/release-23/tutorials/two_component_hamiltonians/xamfX2C.html).

The keywords below are vaild for both, atomic (mean-field) as well
molecular calculations.

> [!WARNING]
> Do not use approximations like .LVCORR for the parent 4-component
> Hamiltonian. Always use .DOSSSS in the atomic molecular-mean field
> calculations.

<div class="index">

.AMF

</div>

## .AMF

Activates the atomic mean-field model for two-electron
scalar-relativistic **and** spin-orbit picture-change corrections. See
also Algorithm 1 in Knecht2022 for further information.

<div class="index">

.EAMF

</div>

## .EAMF

Activates the **extended** atomic mean-field model for two-electron
scalar-relativistic **and** spin-orbit picture-change corrections. See
also Algorithm 2 in Knecht2022 for further information.

<div class="index">

.ATOMIC

</div>

## .ATOMIC

Activates a `deprecated(!)` simplified version of the atomic mean-field
model for two-electron scalar-relativistic **and** spin-orbit
picture-change corrections. Used only for testing/debugging purposes.

> [!WARNING]
> Do not use this option in production calculations.

<div class="index">

.DEBUG

</div>

## .DEBUG

Activates debug output in the \*XAMFI module.

> [!WARNING]
> **CAUTION**: This will lead to extremely verbose output

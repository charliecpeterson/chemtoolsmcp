orphan  

# Spin-orbit states from the COSCI method

This tutorial demonstrates the importance of the effective mean-field
spin-orbit screening on spin-orbit states of open-shell systems. Several
two-component Hamiltonians are employed.

## Spin-orbit states of the F atom

In the DIRAC test we calculate the energy difference between spin-orbit
splitted states of the $`^{2}P`$ state of Fluorine, using the COSCI
wavefunction and with several different Hamiltonians. All input files
for download (together with output files) are in the corresponding test
directory of DIRAC, `test/cosci_energy`.

The following table shows the energy difference betweem
$`X ^{2}P_{3/2}`$ and $`A ^{2}P_{1/2}`$ states:

<table style="width:56%;">
<colgroup>
<col style="width: 27%" />
<col style="width: 27%" />
</colgroup>
<thead>
<tr>
<th>Hamiltonian</th>
<th><blockquote>
<p>Splitting/cm-1</p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr>
<td>DC</td>
<td><blockquote>
<p>434.511758</p>
</blockquote></td>
</tr>
<tr>
<td>BSS+MFSSO</td>
<td><blockquote>
<p>438.792872</p>
</blockquote></td>
</tr>
<tr>
<td>BSS_RKB+MFSSO(*)</td>
<td><blockquote>
<p>438.793184</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2+MFSSO</td>
<td><blockquote>
<p>438.792782</p>
</blockquote></td>
</tr>
<tr>
<td>BSSsfBSO1+MFSSO</td>
<td><blockquote>
<p>438.868634</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2sfBSO1+MFSSO</td>
<td><blockquote>
<p>438.868738</p>
</blockquote></td>
</tr>
<tr>
<td>BSSsfESO1+MFSSO</td>
<td><blockquote>
<p>438.866098</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2sfESO1+MFSSO</td>
<td><blockquote>
<p>438.866201</p>
</blockquote></td>
</tr>
<tr>
<td>BSS</td>
<td><blockquote>
<p>583.459766</p>
</blockquote></td>
</tr>
<tr>
<td>BSS_RKB(**)</td>
<td><blockquote>
<p>583.459995</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2</td>
<td><blockquote>
<p>583.459700</p>
</blockquote></td>
</tr>
<tr>
<td>BSSsfESO1</td>
<td><blockquote>
<p>583.533060</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2sfESO1</td>
<td><blockquote>
<p>583.533187</p>
</blockquote></td>
</tr>
<tr>
<td>BSSsfBSO1</td>
<td><blockquote>
<p>583.535908</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2sfBSO1</td>
<td><blockquote>
<p>583.536036</p>
</blockquote></td>
</tr>
<tr>
<td>DC2BSS_RKB(DF)</td>
<td><blockquote>
<p>585.906861</p>
</blockquote></td>
</tr>
</tbody>
</table>

> (*) Known as X2C. (*\*) Known as X2C-NOAMFI.

Calculated values can be devided into two categories: those with the
mean-field spin-orbit term (MFSSO) and those without. Results matching
the four-component Dirac-Coulomb (DC) Hamiltonian are those containing
the MFSSO screening term.

For more information, see Refs. Ilias2001, Ilias2007 .

## Spin-orbit states of the $`Rn^{77+}`$ cation

Let us proceed with the isoelectronic, but heavier system: the
Fluorine-like (9 electrons), highly charged $`Rn^{77+}`$ cation (Z=86).
All input files for download (together with output files) are in the
corresponding test directory of DIRAC, `test/cosci_energy`. Calculated
energy differences between the ground, $`X ^{2}P_{3/2}`$, and the first
excited state, $`A ^{2}P_{1/2}`$, are in the following table:

<table style="width:56%;">
<colgroup>
<col style="width: 27%" />
<col style="width: 27%" />
</colgroup>
<thead>
<tr>
<th>Hamiltonian</th>
<th><blockquote>
<p>Splitting/eV</p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr>
<td>DC</td>
<td><blockquote>
<p>3700.081</p>
</blockquote></td>
</tr>
<tr>
<td>BSS+MFSSO</td>
<td><blockquote>
<p>3796.844</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2+MFSSO</td>
<td><blockquote>
<p>3777.837</p>
</blockquote></td>
</tr>
<tr>
<td>DC2BSS_RKB(DF)</td>
<td><blockquote>
<p>3810.190</p>
</blockquote></td>
</tr>
<tr>
<td>BSS</td>
<td><blockquote>
<p>3808.859</p>
</blockquote></td>
</tr>
<tr>
<td>BSS_RKB (*)</td>
<td><blockquote>
<p>3810.273</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2</td>
<td><blockquote>
<p>3790.044</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2sfBSO1+MFSSO</td>
<td><blockquote>
<p>4047.324</p>
</blockquote></td>
</tr>
<tr>
<td>DKH2sfBSO1</td>
<td><blockquote>
<p>4056.349</p>
</blockquote></td>
</tr>
</tbody>
</table>

> (\*) Known as X2C-NOAMFI.

## Excercises

1.  Why is the MFSSO term more important for the ligher element (F) than
    for the heavy $`Rn^{77+}`$ ?
2.  The one-electron spin-orbit term, SO1, is sufficient for
    representing spin-orbital effects in the Flourine atom, but not of
    the <span class="title-ref">Rn^{77+}</span> cation. Why ?
3.  For the Flourine atom, increase the speed of light
    (`GENERAL_.CVALUE`) in four-component calculations to emulate
    non-relativistic description. What is the effect on the spin-orbit
    splitting ? What artificial value of the speed of light generates
    the DC-SCF energy identical with nonrelativistic SCF energy up to 5
    decimal places ?
4.  To "increase" relativistic effects in Flourine, decrease the speed
    of light in four-component calculations. How does it affect the
    spin-orbit splitting ?
5.  Change the symmetry from D2h to automatic symmetry detection in the
    F mol file and add molecular spinors analysis to the input file
    (`**ANALYZE`). Identify molecular spinors (orbitals) of Flourine
    according to the extra quantum number in linear symmetry.

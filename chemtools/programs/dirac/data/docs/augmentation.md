orphan  

# Basis sets augmentation

DIRAC suite offers possibility to easily modify employed basis sets.

The purpose of this tutorial, which is reflected in the test
*basis_automatic_augmentation*, is to show nonstandard automatic
augmentation of provided standard basis sets, like cc-pVXZ, Turbomole
and Dyall's.

In the following table we demonstrate the influence of the (contracted)
basis set extension on the total nonrelativistic energy and on the
dipole moment, which is simply calculated through SCF wave-function
expectation value.

Two molecules were chosen: HF and HBr. For the former molecule, the
pre-cc-pVXZ and Turbomole basis sets are used for both F and H atoms. In
the latter molecule, the heavy Br atom is described with Dyall's
relativistic basis set, while the H atom is provided with the
nonrelativistic basis set.

<table style="width:97%;">
<colgroup>
<col style="width: 20%" />
<col style="width: 30%" />
<col style="width: 19%" />
<col style="width: 14%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr>
<th><blockquote>
<p>Basis set name</p>
</blockquote></th>
<th>Atom Basis size</th>
<th>Atom Basis size</th>
<th>E(NR-SCF)/a.u.</th>
<th>dip. mom/D</th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p>aug-cc-pVDZ</p>
</blockquote></td>
<td>F [10s5p2d|4s3p2d]</td>
<td><blockquote>
<p>H [5s2p|3s2p]</p>
</blockquote></td>
<td>-100.0337931</td>
<td>-1.89747844</td>
</tr>
<tr>
<td><blockquote>
<p>d-aug-cc-pVDZ</p>
</blockquote></td>
<td>F [11s6p3d|5s4p3d]</td>
<td><blockquote>
<p>H [6s3p|4s3p]</p>
</blockquote></td>
<td>-100.0339787</td>
<td>-1.88741792</td>
</tr>
<tr>
<td><blockquote>
<p>t-aug-cc-pVDZ</p>
</blockquote></td>
<td>F [12s7p4d|6s5p4d]</td>
<td><blockquote>
<p>H [7s4p|5s4p]</p>
</blockquote></td>
<td>-100.0341087</td>
<td>-1.88839962</td>
</tr>
<tr>
<td>Turbomole-DZP</td>
<td>F [8s4p1d|4s2p1d]</td>
<td><blockquote>
<p>H [4s1p|2s1p]</p>
</blockquote></td>
<td>-100.0028441</td>
<td>-1.94418011</td>
</tr>
<tr>
<td>s-a-Turbomole-DZP</td>
<td>F [9s5p2d|5s3p2d]</td>
<td><blockquote>
<p>H [5s2p|3s2p]</p>
</blockquote></td>
<td>-100.0180563</td>
<td>-1.91810891</td>
</tr>
<tr>
<td><blockquote>
<p>Turbomole-TZVPP</p>
</blockquote></td>
<td>F [11s6p2d1f|5s3p2d1f]</td>
<td><blockquote>
<p>H [5s2p1d|3s2p1d]</p>
</blockquote></td>
<td>-100.0657904</td>
<td>-1.91940698</td>
</tr>
<tr>
<td>s-a-Turbomole-TZVPP</td>
<td>F [12s7p3d2f|6s4p3d2f]</td>
<td><blockquote>
<p>H [6s3p2d|4s3p2d]</p>
</blockquote></td>
<td>-100.0668454</td>
<td>-1.89121376</td>
</tr>
<tr>
<td><blockquote>
<p>dyall.v2z</p>
</blockquote></td>
<td>Br [15s11p7d1f|15s11p7d1f]</td>
<td>(H [4s1p|2s1p])</td>
<td>-2572.6361195</td>
<td>-0.97693933</td>
</tr>
<tr>
<td>d-aug-dyall.v2z</td>
<td>Br [17s13p9d3f|17s13p9d3f]</td>
<td>(H [4s1p|2s1p])</td>
<td>-2572.6425365</td>
<td>-0.76521484</td>
</tr>
</tbody>
</table>

(H-atom in parenthesis has cc-pVDZ basis sets.)

Note that the nonrelativistic Hamiltonian with contracted basis sets was
used in all these examples. Contracted basis sets were set for light
elements (F,H), while for Br atom uncontracted scheme with Dyall's basis
was preferred.

For relativistic calculation with either 2-component of 4-component
Hamiltonians the nonrelativistic contraction of light elements basis
sets is no longer suitable, especially if the molecule contains heavy
elements. The user is adviced to resort to uncontracting his basis sets.
For example, for the HBr molecule you can combine decontracted cc-pVDZ
basis set for hydrogen with the decontracted (as is) dyall.v2z basis for
bromine.

Dyall's basis sets, which are constructed without contractions, can be
used also with the X2C Hamiltonian.

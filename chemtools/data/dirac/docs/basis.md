orphan  

# Pick the right basis for your calculation

The development of basis sets suitable for use in relativistic
calculations reflects the relative lateness of the field's development.
Because the more consistent efforts in method development started at
about the mid 1980's, it wasn't until well into the late 1990's that the
pioneering works of the early and mid 1990's were substantially
complemented and improved upon.

The availability of basis sets has dramatically improved in recent
years, notably with the work of K. G. Dyall, and Dirac users are
strongly advised to use Dyall's basis sets whenever they are available.
These sets follow roughly the "correlation-consistent" philosophy
introduced by Dunning and coworkers Dunning1989, so they already contain
polarization and correlation functions, but the SCF sets are designed
for an adequate SCF representation rather than to match correlating sets
for the valence shells.

## Dyall basis sets

We recommend that you use the Dyall basis set repositories for [double-,
triple-, and quadruple-zeta
sets](https://doi.org/10.5281/zenodo.7574628) and [quintuple-zeta
set](https://zenodo.org/records/17088050) whenever the basis sets are
available for the elements of interest. In order to make that usage as
convenient as possible, the following files, containing all sets
currently available at the URL above (published or to be published), are
made available:

<table style="width:96%;">
<colgroup>
<col style="width: 25%" />
<col style="width: 16%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th><blockquote>
<p>quality</p>
</blockquote></th>
<th><blockquote>
<p>valence</p>
</blockquote></th>
<th><blockquote>
<p>core-valence</p>
</blockquote></th>
<th><blockquote>
<p>all-electron</p>
</blockquote></th>
<th><blockquote>
<p>DFT</p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p>double-zeta</p>
</blockquote></td>
<td><blockquote>
<p>dyall.v2z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.cv2z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.ae2z</p>
</blockquote></td>
<td>dyall.2zp</td>
</tr>
<tr>
<td><blockquote>
<p>+diffuse functions</p>
</blockquote></td>
<td><blockquote>
<p>dyall.av2z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.acv2z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.aae2z</p>
</blockquote></td>
<td></td>
</tr>
<tr>
<td><blockquote>
<p>triple-zeta</p>
</blockquote></td>
<td><blockquote>
<p>dyall.v3z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.cv3z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.ae3z</p>
</blockquote></td>
<td>dyall.3zp</td>
</tr>
<tr>
<td><blockquote>
<p>+diffuse functions</p>
</blockquote></td>
<td><blockquote>
<p>dyall.av3z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.acv3z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.aae3z</p>
</blockquote></td>
<td></td>
</tr>
<tr>
<td><blockquote>
<p>quadruple-zeta</p>
</blockquote></td>
<td><blockquote>
<p>dyall.v4z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.cv4z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.ae4z</p>
</blockquote></td>
<td>dyall.4zp</td>
</tr>
<tr>
<td><blockquote>
<p>+diffuse functions</p>
</blockquote></td>
<td><blockquote>
<p>dyall.av4z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.acv4z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.aae4z</p>
</blockquote></td>
<td></td>
</tr>
<tr>
<td><blockquote>
<p>quintuple-zeta</p>
</blockquote></td>
<td><blockquote>
<p>dyall.v5z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.cv5z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.ae5z</p>
</blockquote></td>
<td></td>
</tr>
<tr>
<td><blockquote>
<p>+diffuse functions</p>
</blockquote></td>
<td><blockquote>
<p>dyall.av5z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.acv5z</p>
</blockquote></td>
<td><blockquote>
<p>dyall.aae5z</p>
</blockquote></td>
<td></td>
</tr>
</tbody>
</table>

The basis sets with diffuse functions are available for the s, p, and d
blocks. The diffuse functions consist of one additional function in each
symmetry that has functions for valence correlation.

While the division into "valence" and "core-valence" can be at times not
so clear-cut as for lighter elements, the option was made to stick to
the usual jargon of non-relativistic theory, particularly in relation to
the "correlation-consistent" family of basis sets.

The valence basis sets are defined differently for each block, and
include functions for the correlation of the following shells:

<table>
<colgroup>
<col style="width: 16%" />
<col style="width: 83%" />
</colgroup>
<thead>
<tr>
<th><blockquote>
<p>block</p>
</blockquote></th>
<th><blockquote>
<p>shells included</p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p>s</p>
</blockquote></td>
<td><blockquote>
<p>ns shell, (n-1)s, p shells</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>p</p>
</blockquote></td>
<td><blockquote>
<p>ns, np shells</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>d</p>
</blockquote></td>
<td><blockquote>
<p>ns, np, nd shells, (n+1)s and p shells</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>f</p>
</blockquote></td>
<td><blockquote>
<p>ns, np, nd, nf, (n+1)s, p, d shells, (n+2) s, p shells</p>
</blockquote></td>
</tr>
</tbody>
</table>

The choice for the f block is necessary to cover correlation of the open
f shell, which becomes a semicore shell towards the end of the row. The
reason for including the outer core shells for the s and d blocks is
that the correlation of these shells is usually necessary for accurate
results.

The core-valence basis sets include the (n-2) shell for the s elements,
the (n-1) shell for the p elements, the (n-1) shell for the d elements,
and nothing extra for the f elements.

The all-electron basis sets include correlating functions for all
shells, down to the 1s for all elements. These are intended for use when
correlating all electrons.

The basis sets also include functions for dipole polarization of the
outer core shells, as this is important for many elements. For the s
block these functions are included for the outer core (n-1)s and p
shells; for the p block, an f function is included to polarize the
(n-1)d shell; for the d block, polarizing f and higher functions are
included for the nd shell; none are added for the f block as the f is
very compact.

The DFT basis sets do not contain the correlating functions as these are
not necessary for DFT (or Hartree-Fock) calculations, except for the
outermost shells where the functions with one unit more of angular
momentum are included from the correlating sets as polarization
functions. The dipole polarization functions for the outer core are also
included. These basis sets are the most economical choice for DFT
calculations.

You are encouraged to look in the basis set files and in the original
archives published in Theor. Chem. Acc. to get a feel for what is
included in each case. The archive files are available on zenodo
[here](https://doi.org/10.5281/zenodo.7606546).

With the recent addition of Dyall basis sets for the light elements, it
is no longer necessary to use the standard non-relativistic basis sets,
such as the correlation-consistent sets of Dunning and coworkers.
However, because the Dyall basis sets are quite a bit larger, you might
want to continue using these basis sets. It is advisable that, in order
to have a balanced description when light and heavy elements are
present, that one uses either contracted or uncontracted sets thoughout.
For 4-component calculations it is strongly recommended to use the basis
sets uncontracted (which is the case for the files listed above).

See the [Dyall basis set
repository](https://doi.org/10.5281/zenodo.7574628) for the latest
updates and the appropriate basis set references. In case of errors or
omissions on any of the files in this directory, users are kindly asked
to contact the authors of DIRAC.

## Other relativistic basis sets

Apart from Dyall's sets, you can choose several different basis sets
based upon geometric progressions of exponents. One such set is that of
K. Faegri, also available in the basis set library, but with the
drawback that you may need to extend it by adding polarization
functions.

## Non-relativistic and scalar-relativistic basis sets

The DIRAC distribution shares a large library of standard
non-relativistic and scalar-relativistic basis sets with the
[Dalton](http://www.kjemi.uio.no/software/dalton) program. These basis
sets can be found in the directory **basis_dalton** of the DIRAC
distribution.

These basis sets are not all suitable for relativistic calculations,
especially not for the heavier elements. Basis sets developed for full
relativistic calculations (including spin-orbit coupling) can be found
in the directory **basis**.

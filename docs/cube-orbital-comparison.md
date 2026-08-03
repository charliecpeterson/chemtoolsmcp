# Phase-aligned orbital CUBE comparison

`compare_cube_orbitals` compares one pair of explicitly matched,
non-degenerate molecular-orbital CUBE fields. The caller supplies the two
orbital labels because file metadata cannot safely decide that correspondence.
For example, a program-specific orbital number may differ after symmetry,
spin, basis, or ordering choices even when the fields are the intended pair.

The fields must have the same atom identities and positions, origin, grid
shape, and voxel vectors within the documented tolerances. Chemtools does not
interpolate or rotate either field. It rejects a CUBE identified as density,
spin density, or potential. Unidentified metadata remains a warning, so a
caller can compare plainly labelled external fields while retaining that
uncertainty in the response.

Orbital phase is arbitrary. The tool reports the signed normalized overlap
`S`, whether it would flip the candidate sign, `abs(S)`, and the L2 distance
after that phase alignment:

```text
sqrt(2 - 2 * abs(S))
```

The integration uses uniform trapezoidal weights on the declared CUBE grid.
Zero-norm fields are refused. These values compare sampled real-space fields;
they are not an AO-basis overlap and do not by themselves establish equivalent
methods, occupations, energies, or correctness.

For degenerate or near-degenerate spaces, use
[`compare_cube_orbital_subspaces`](cube-orbital-subspace-comparison.md). It
uses a caller-declared set and principal-angle/SVD comparison, so arbitrary
unitary rotations are handled explicitly rather than hidden by a single-orbital
score.

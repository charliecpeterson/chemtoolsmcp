# Orbital-subspace CUBE comparison

`compare_cube_orbital_subspaces` compares two caller-declared equal-dimension
sets of two through eight orbital CUBE fields. The labels are
provenance for each supplied basis vector; Chemtools does not infer membership
or pair individual orbitals across the two sets.

Every field must be identified as orbital-like and share the exact same
nuclear geometry, CUBE origin, grid shape, and voxel vectors within the
existing CUBE tolerances. No interpolation, grid rotation, or coordinate
alignment is performed. The implementation uses uniform trapezoidal weights.

Finite CUBE boxes can leave sampled orbitals slightly non-orthonormal. Rather
than assuming the raw cross-overlap matrix is a principal-angle matrix,
Chemtools builds the reference and candidate Gram matrices and orthonormalizes
both sampled subspaces before taking the singular values of their cross
overlap. Those singular values are the principal overlaps; their arccosines
are the principal angles. They are invariant to sign changes and unitary
rotations within either supplied subspace.

The response retains the raw Gram and cross-overlap matrices for diagnosis,
then reports principal overlaps, angles, the least principal overlap, and a
projection Frobenius distance. A linearly dependent or numerically
rank-deficient set is refused. The tool does not decide whether two nearby
energy levels are scientifically degenerate, establish method equivalence, or
compare AO-basis coefficients.

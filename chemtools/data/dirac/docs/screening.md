orphan  

# Notes on screening

We have some number of old density matrices *D*<sub>\*k\* -
1</sub>,\*D\*<sub>\*k\* - 2</sub> etc, and want to write the new density
matrix as

..figure:: /dirac/images/math/6/c/8/6c86ef3df163c5e325edbc447abb31d9.png
:align: center :alt: D_k = \Delta D + \sum\_{i=1}^N c_i D\_{k-i},

> D_k = \Delta D + \sum\_{i=1}^N c_i D\_{k-i},

where Δ\*D\* will be sent to `TWOFCK` and should give efficient
screening. Currently we are minimizing the Frobenius norm (sum of
squares of elements) of Δ\*D\*, using one old density matrix. In the
general case we have to solve *Sc* = *y*, where

*S*<sub>\*i\*\*j\*</sub> = (*D*<sub>\*k\* -
\*i\*</sub>,\*D\*<sub>\*k\* - \*j\*</sub>) and *y*<sub>\*i\*</sub> =
(*D*<sub>\*k\*</sub>,\*D\*<sub>\*k\* - \*i\*</sub>).

Here "(cdot,\cdot)"(FIX) is the inner product between the *D* "vectors".
However, the use of the Frobenius norm gives the larges weight to the
large elements of Δ\*D\*. This is not really what we want, because we
care more about producing as many small elements in Δ\*D\* as possible.
Therefore we can imagine to introduce a metric matrix *M*, so that

..figure:: /dirac/images/math/8/7/7/877cd0b8aba2f7fcefddaa52ff414fdd.png
:align: center :alt: (A,B) = \sum\_{ij} A^\*\_{ij}M\_{ij}B\_{ij},

> (A,B) = \sum\_{ij} A^\*\_{ij}M\_{ij}B\_{ij},

and set *M* to emphasize the elements that have a chance of being made
small. In practice one could imagine

- Calculate Δ\*D\* using a unit (*M*<sub>\*i\*\*j\*</sub> = 1) metric.
- Find the largest elements (with some definition of largest) of Δ\*D\*
  and set the corresponding elements in *M* to zero.
- Recalculate *S* using the new metric.
- Solve for the new Δ\*D\*

Doing this for the water molecule SCF in test 1 shrinks the elements in
the final iteration Δ\*D\* by four orders of magnitude. In this case I
used five old density matrices. Just using a large number of old
densities without modifying the metric did very little. A simple test
inside Dirac shows that it is not enough to use one historical density
matrix with a modified metric. The reason is that at the end of the SCF
the change in density is almost orthogonal to the last density matrix.

It would also be possible to give larger weight to certain elements of
Δ\*D\*, for example those that come from different integral blocks or
centers.

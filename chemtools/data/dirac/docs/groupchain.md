orphan  

# Symmetry-handling at the correlated level

## Introduction

The DIRAC code can handle symmetries corresponding to $`D_{2h}`$ and
subgroups (denoted binary groups) as well as linear supersymmetry. At
the SCF level DIRAC employs a unique quaternion symmetry scheme for the
fermion symmetry of the spinors which combines time reversal and spatial
symmetry (Saue1999). A particularity of this scheme is that symmetry
reductions due to spatial symmetry is translated into a reduction of
algebra, from quaternion down to complex and possibly real algebra. This
leads to a classification of the binary groups as:

- Quaternion groups: $`C_1`$ , $`C_i`$
- Complex groups: $`C_2`$ , $`C_s`$ , $`C_{2h}`$
- Real group: $`D_2`$ , $`C_{2v}`$ , $`D_{2h}`$

In general, at the correlated level, the highest Abelian subgroup of the
point group under consideration is used. All quaternion and complex
subgroups are Abelian. The fermion irreps of the real groups are
non-Abelian and so an Abelian complex subgroup is chosen for
calculations on a correlated level.

``` math
\begin{aligned}
\begin{array}{lcl}D_2, C_{2v}&\rightarrow &C_2\\D_{2h}&\rightarrow &C_{2h}\end{array}
\end{aligned}
```

## Character tables for Abelian double groups

Below we give character tables for the Abelian double groups handled by
DIRAC. The irreps are given in the same order as internally in the
correlation modules of the program. This means that fermion irreps come
before boson irreps.

The tables are given according to the conventions of S. L. Altmann and
P. Herzig, *Point-Group Theory Tables*, Clarendon Press, Oxford, 1994 (A
second corrected edition is now available free of charge
[here](http://phaidra.univie.ac.at/o:104731).). The final column of the
tables give the irrep labels employed by DIRAC.

- Real groups :

|                  |       |       |
|------------------|-------|-------|
| $`\mathbf{C_1}`$ | $`E`$ |       |
| $`A_{1/2}`$      | $`1`$ | $`A`$ |
| $`A`$            | $`1`$ | $`a`$ |

|                  |       |                  |        |
|------------------|-------|------------------|--------|
| $`\mathbf{C_i}`$ | $`E`$ | $`i`$            |        |
| $`A_{1/2,g}`$    | $`1`$ | $`\phantom{-}1`$ | $`AG`$ |
| $`A_{1/2,u}`$    | $`1`$ | $`-1`$           | $`AU`$ |
| $`A_g`$          | $`1`$ | $`\phantom{-}1`$ | $`ag`$ |
| $`A_u`$          | $`1`$ | $`-1`$           | $`au`$ |

- Complex groups:

|                  |       |                  |        |
|------------------|-------|------------------|--------|
| $`\mathbf{C_2}`$ | $`E`$ | $`C_2`$          |        |
| $`\,^1E_{1/2}`$  | $`1`$ | $`\phantom{-}i`$ | $`1E`$ |
| $`\,^2E_{1/2}`$  | $`1`$ | $`-i`$           | $`2E`$ |
| $`A`$            | $`1`$ | $`\phantom{-}1`$ | $`a`$  |
| $`B`$            | $`1`$ | $`-1`$           | $`b`$  |

|                      |       |                  |        |
|----------------------|-------|------------------|--------|
| $`\mathbf{C_s}`$     | $`E`$ | $`\sigma_h`$     |        |
| $`\,^1E_{1/2}`$      | $`1`$ | $`\phantom{-}i`$ | $`1E`$ |
| $`\,^2E_{1/2}`$      | $`1`$ | $`-i`$           | $`2E`$ |
| $`A^{\prime}`$       | $`1`$ | $`\phantom{-}1`$ | $`a`$  |
| $`A^{\prime\prime}`$ | $`1`$ | $`-1`$           | $`b`$  |

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| $`\mathbf{C_{2h}}`$ | $`E`$ | $`C_2`$ | $`i`$ | $`\sigma_h`$ |  |
| $`\,^1E_{1/2,g}`$ | $`1`$ | $`\phantom{-}i`$ | $`\phantom{-}1`$ | $`\phantom{-}i`$ | $`1Eg`$ |
| $`\,^2E_{1/2,g}`$ | $`1`$ | $`-i`$ | $`\phantom{-}1`$ | $`-i`$ | $`2Eg`$ |
| $`\,^1E_{1/2,u}`$ | $`1`$ | $`\phantom{-}i`$ | $`-1`$ | $`-i`$ | $`1Eu`$ |
| $`\,^2E_{1/2,u}`$ | $`1`$ | $`-i`$ | $`-1`$ | $`\phantom{-}i`$ | $`2Eu`$ |
| $`A_g`$ | $`1`$ | $`\phantom{-}1`$ | $`\phantom{-}1`$ | $`\phantom{-}1`$ | $`ag`$ |
| $`B_g`$ | $`1`$ | $`-1`$ | $`\phantom{-}1`$ | $`-1`$ | $`bg`$ |
| $`A_u`$ | $`1`$ | $`\phantom{-}1`$ | $`-1`$ | $`-1`$ | $`au`$ |
| $`B_u`$ | $`1`$ | $`-1`$ | $`-1`$ | $`\phantom{-}1`$ | $`bu`$ |

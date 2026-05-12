orphan  

<div class="index">

\*LOCALIZATION

</div>

# \*LOCALIZATION

Performs localization of molecular orbitals localization using the
Pipek-Mezey criterion Pipek:Mezey. Since localized orbitals in general
do not span single irreps, this module only works with $`C_1`$ symmetry.

The implementation uses an exponential parametrization combined with
second-order minimization using the Newton trust-region method (see
Sections 10.8 and 12.3 of Helgaker:book for general principles.

In short, we have some function $`T(\mathbf{\kappa})`$ that we seek to
minimize with respect to variational parameters $`\mathbf{\kappa}`$,
where $`T^{[0]}=T(\mathbf{0})`$ is our current expansion point. In
iteration $`n`$ we consider a second-order expansion of the target
function

``` math
Q_n(\mathbf{\kappa}_n) = T_n^{[0]} + \mathbf{T}_n^{[1]T}\mathbf{\kappa}_n + \frac{1}{2}\mathbf{\kappa}_n^T T_n^{[2]}\mathbf{\kappa}_n
```

To the extent this expansion is valid, the (Newton) step to be taken is
given by the linear equation

``` math
T_n^{[2]}\mathbf{\kappa}_n=-\mathbf{T}_n^{[1]}
```

We assume the second-order expansion to be valid within a trust radius
$`h`$. If the proposed Newton step goes outside this radius, we will
instead take a step onto the border of our trust region in an optimal
direction. To find it, we set up a Lagrangian

``` math
L(\mathbf{\kappa},\mu)=Q(\mathbf{\kappa}) -\frac{1}{2}\mu\left(\Vert\mathbf{\kappa}\Vert^2-h^2\right)
```

where minimization in iteration \$n\$ now gives

``` math
\left(T_n^{[2]}-\lambda_n I\right)\mathbf{\kappa}_n=-\mathbf{T}_n^{[1]}
```

An important quantity in the algorithm is the ration between the
observed and predicted changes in the target function

``` math
r = \frac{T_{n+1}^{[0]}-T_{n}^{[0]}}{Q_{n}-T_{n}^{[0]}} = \frac{\mbox{DIFFG}}{\mbox{DIFFQ}}
```

In Helgaker:book the following scheme for the update of the trust radius
is proposed (we quote):

> 1.  If $`r>0.75`$ , increase the trust radius $`h_{n+1}=1.2h_{n}`$.
> 2.  If $`0.25 < r <0.75`$, do not change the trust radius
>     $`h_{n+1}=h_{n}`$.
> 3.  If $`r <0.25`$ , reduce the trust radius $`h_{n+1}=0.7h_{n}`$.
> 4.  If $`r<0`$ , reject the step $`\mathbf{\kappa}_n`$ and generate a
>     new step of trust radius $`h_n\rightarrow 0.7h_n`$.

<div class="index">

.PRJLOC

</div>

## .PRJLOC

Localization from projection analysis instead of Mulliken analysis.
\[Default: False\]:

    .PRJLOC

<div class="index">

.OWNBAS

</div>

## .OWNBAS

Calculate fragments in their own basis. For now it is the only option
which can be used with projection analysis option (.PRJLOC) \[Default:
False\]:

    .OWNBAS

<div class="index">

.C1READ

</div>

## .C1READ

Read C1 coefficients for fragments.

<div class="index">

.VECREF

</div>

## .VECREF

First give number of fragments to project onto. Then for each fragment
give filename of MO coefficients and string specifying which orbitals
will be used in the projection (`orbital_strings`):

    .VECREF
     3
     DFO1AT
     1..6
     DFH1AT
     1
     DFH2AT
     1

<div class="index">

.HESLOC

</div>

## .HESLOC

Define model of the Hessian \[Default: FULL\].

Use full Hessian:

    .HESLOC
     FULL

Use diagonal approximation of the Hessian. If using this approximation,
there is danger of converging to the stationary point instead of minima:

    .HESLOC
     DIAG

First the diagonal approximation of the Hessian will be used and after
reaching convergence criterion the localization will switch to the
construction of the full Hessian:

    .HESLOC
     COMB

<div class="index">

.CHECK

</div>

## .CHECK

This keyword can be used only in combination with .HESLOC = COMB. The
full Hessian will be calculated only at the last iteration. This way one
can check if the procedure converged to the minimum.

<div class="index">

.THFULL

</div>

## .THFULL

Convergence threshold for the localization process. It applies on the
functional value of the localization criteria when full Hessian is
calculated. It is used in .HESLOC = FULL scheme and in the second part
of the .HESLOC = COMB scheme. When not specified gradient criterion or
criterion of the number of negative eigenvalues equals zero will be
used:

    .THFULL
     1.0D-13

<div class="index">

.THGRAD

</div>

## .THGRAD

Convergence threshold for the localization process. It applies on the
gradient of the localization functional. It is used in .HESLOC = FULL or
DIAG scheme and in the second part of the .HESLOC = COMB scheme. When
not specified the functional value criterion or criterion of the number
of negative eigenvalues equals zero will be used:

    .THGRAD
     1.0D-17

<div class="index">

.THDIAG

</div>

## .THDIAG

Convergence threshold for the localization process. It applies on the
functional value of the localization criteria when diagonal Hessian is
calculated. It can be used with the .HESLOC = DIAG keyword. If .HESLOC =
COMB it is used to switch from the first to the second stage of the
convergence process:

    .THDIAG
     1.0D-05

<div class="index">

.SELECT

</div>

## .SELECT

Select subset of MOs for localization (use `orbital_strings` syntax).
Currently only Pipek-Mezey localization is implemented, therefore
localize only occupied orbitals. Localization of virtual orbitals is not
recommended.

    .SELECT
     1..5

<div class="index">

.MAXITR

</div>

## .MAXITR

Maximum number of iterations. Default:

    .MAXITR
     100

<div class="index">

.PRINT

</div>

## .PRINT

Set print level. Default:

    .PRINT
     1

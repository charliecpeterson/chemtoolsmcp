orphan  

# Basis sets for relativistic calculations

## Gaussian Type Orbitals (GTOs)

*Cartesian Gaussians* are defined as

``` math
G^{\alpha}_{ijk}=G^{\alpha}_i(x)G^{\alpha}_j(y)G^{\alpha}_k(z)
```

with

``` math
G^{\alpha}(x)=\left(\frac{2\alpha}{\pi}\right)^{1/4}\sqrt{\frac{\left(4\alpha\right)^i}{\left(2i-1\right)!!}}
               x^i\exp\left[-\alpha x^2\right]
```

Alternatively one may express a Cartesian Gaussian as

``` math
G^{\alpha}_{ijk}=N^{\alpha}_{ijk}x^iy^jz^k\exp\left[-\alpha r^2\right];\quad N^{\alpha}_{ijk}=\left(\frac{2\alpha}{\pi}\right)^{3/4}\sqrt{\frac{2^l}{F_{ijk}}}\left(\sqrt{2\alpha}\right)^l
```

where the sum $`i+j+k`$ is associated with orbital angular momentum
$`l`$. The factor

``` math
F_{ijk}=\left(2i-1\right)!!\left(2j-1\right)!!\left(2k-1\right)!!
```

shows that the Cartesian components of a given shell may have different
normalization constants. In the HERMIT integral code a single
normalization is chosen for each shell by ignoring $`F_{ijk}`$ meaning
that Cartesian Gaussians are normalized to

``` math
\left<G^{\alpha}_{ijk}|G^{\alpha}_{ijk}\right>=F_{ijk}
```

In practice this means that s- and p-functions are normalized to one. So
are $`d110`$, $`d101`$ and $`d011`$, whereas $`d200`$, $`d020`$ and
$`d002`$ are normalized to three. $`f111`$ is normalized to one,
$`f210`$, $`f201`$, $`f120`$, $`f102`$ and $`f012`$ are normalized to
three, and $`f300`$, $`f030`$ and $`f003`$ are normalized to 15.

*Spherical Gaussians* are defined by

``` math
G^{\alpha}_{lm} = R^{\alpha}_l\left(r\right)Y_{lm}\left(\theta,\phi\right);\quad
```

where the angular part is given by spherical harmonics $`Y_{lm}`$ and
the radial part by

> R^{alpha}\_l=N^{alpha}\_lr^lexpleft\[-alpha r^2right\];quad
> N^{alpha}\_l =
> frac{2left(2alpharight)^{3/4}}{pi^{1/4}}sqrt{frac{2^l}{left(2l+1right)!!}}left(sqrt{2alpha}right)

In passing we note that

``` math
N^{\alpha}_l=2\sqrt{\frac{\alpha}{2l+1}}N^{\alpha}_{l-1}
```

For given $`l`$ there are
$`\frac{1}{2}\left(l+2\right)\left(l+1\right)`$ $`\left(2l+1\right)`$
spherical Gaussians. The latter basis functions therefore provide more
compact basis set expansions. However, in 4-component relativistic
calculations the use of spherical Gaussians is somewhat more complicated
since the coupling of large and small component basis functions needs to
be taken into account.

## Kinetic balance

Kinetic balance corresponds to the non-relativistic limit of the exact
coupling of large and small component basis functions for
positive-energy orbitals. Starting from the radial function of a large
component spherical Gaussian basis function one obtains

``` math
R^S\propto -\left[\frac{\partial}{\partial r}+\frac{\left(1+\kappa^L\right)}{r}\right]R^{\alpha}_l
= \sqrt{\alpha\left(2l+3\right)}R^{\alpha}_{l+1}-2\sqrt{\frac{\alpha}{2l+1}}R^{\alpha}_{l-1}
```

We can now distinguish two cases

``` math
\begin{aligned}
R^S \propto \left\{ \begin{array}{ll}
\sqrt{\left(2l+3\right)}R^{\alpha}_{l+1}-2\sqrt{\left(2l+1\right)}R^{\alpha}_{l-1} & \mbox{if $\kappa^L = l$};\\
R^{\alpha}_{l+1} & \mbox{if $\kappa^L = -\left(l+1\right)$}.\end{array} \right.
\end{aligned}
```

The case $`\kappa^L < 0`$ is straightforward, but a bit more care is
needed for the implementation for the case $`\kappa^L>0`$. The modified
spherical Gaussian to be constructed is

``` math
G^*_{lm} = N\left\{\sqrt{\left(2l+1\right)\left(2l-1\right)}R^{\alpha}_l-2\left(2l-1\right)R^{\alpha}_{l-2}\right\}Y_{l-2,m};\quad
N = \frac{1}{\sqrt{\left(2l+1\right)\left(2l-1\right)}}
```

## Constructing modified spherical harmonics for kinetic balance; the gritty details

We write the spherical harmonic $`G^{\alpha}_{l-2,m}`$ as a linear
combination of Cartesian Gaussians

``` math
G^{\alpha}_{l-2,m} = \sum_{i+j+k=l-2}c^{l-2,m}_{ijk}G^{\alpha}_{ijk}
```

In the HERMIT integral code we have selected a transformation such that
the solid harmonics are normalized to unity. From this we obtain

``` math
\begin{aligned}
\begin{array}{lcl}
G^*_{lm}&=&N\left[4\alpha r^2-2\left(2l-1\right)\right]\sum_{i+j+k=l-2}c^{l-2,m}_{ijk}G^{\alpha}_{ijk}\\
        &=&N\sum_{i+j+k=l-2}c^{l-2,m}_{ijk}
           \left[4\alpha
           \left(
           \frac{N^{\alpha}_{ijk}}{N^{\alpha}_{i+2,j,k}}G^{\alpha}_{i+2,j,k}
          +\frac{N^{\alpha}_{ijk}}{N^{\alpha}_{i,j+2,k}}G^{\alpha}_{i,j+2,k}
          +\frac{N^{\alpha}_{ijk}}{N^{\alpha}_{i,j,k+2}}G^{\alpha}_{i,j,k+2}
           \right)
          -2\left(2l-1\right)G^{\alpha}_{ijk}
           \right]
 \end{array}
\end{aligned}
```

The ration of normalization constants in the above relation is given by

``` math
\frac{N^{\alpha}_{ijk}}{N^{\alpha}_{i+2,j,k}}=\frac{1}{4\alpha}\sqrt{\frac{F_{i+2,j,k}}{F_{ijk}}}
```

However, the expression is much simplified by the fact that factors
$`F_{ijk}`$ are set to one in HERMIT, such that

``` math
G^*_{lm}=N\sum_{i+j+k=l-2}c^{l-2,m}_{ijk}
           \left[
           \left(G^{\alpha}_{i+2,j,k}+G^{\alpha}_{i,j+2,k}+G^{\alpha}_{i,j,k+2}\right)
          -2\left(2l-1\right)G^{\alpha}_{ijk}
           \right]
```

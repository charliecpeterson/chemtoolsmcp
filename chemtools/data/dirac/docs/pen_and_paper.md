orphan  

<div id="TCCM_pp_exercises">

In the text full units are used for clarity, but in practice one can
employ SI-based atomic units, setting
$`\hbar=m_{e}=e=4\pi\varepsilon_{0}=1`$.

</div>

# Pauli spin matrices

The Pauli spin matrices

``` math
\begin{aligned}
\sigma_{x}=\left[\begin{array}{cc}
0 & 1\\
1 & 0
\end{array}\right];\quad\sigma_{y}=\left[\begin{array}{cc}
0 & -i\\
i & 0
\end{array}\right];\quad\sigma_{z}=\left[\begin{array}{cc}
1 & 0\\
0 & -1
\end{array}\right]
\end{aligned}
```

are representation matrices of of the operator $`2\mathbf{s}`$ in the
basis
$`\left\{ \left|\alpha\right\rangle ,\left|\beta\right\rangle \right\}`$
of spin-1/2 functions (*spin-up* and *spin-down*, respectively), that is
$`2\mathbf{s}\rightarrow\hbar\boldsymbol{\sigma}`$. These spin functions
obey the following relations

``` math
\begin{aligned}
\begin{array}{lclclcl}
\hat{s}_{z}\left|\alpha\right\rangle  & = & +\frac{1}{2}\hbar\left|\alpha\right\rangle  &  &    \hat{s}_{z}\left|\beta\right\rangle  & = & -\frac{1}{2}\hbar\left|\beta\right\rangle \\
\hat{s}_{+}\left|\alpha\right\rangle  & = & 0 &  & \hat{s}_{+}\left|\beta\right\rangle  & = &  \hbar\left|\alpha\right\rangle \\
\hat{s}_{-}\left|\alpha\right\rangle  & = & \hbar\left|\beta\right\rangle  &  &   \hat{s}_{-}\left|\beta\right\rangle  & = & 0
\end{array}
\end{aligned}
```

where appears the ladder operators $`s_{\pm}=s_{x}\pm is_{y}`$. The
spin-1/2 functions are furthermore orthonormal. Let us see how the Pauli
$`\sigma_{z}`$ matrix is generated:

> - Element (1,1):
>   $`\left\langle \alpha\left|2s_{z}\right|\alpha\right\rangle /\hbar=\left\langle \alpha\left|\right.\alpha\right\rangle =1`$
> - Element (1,2):
>   $`\left\langle \alpha\left|2s_{z}\right|\beta\right\rangle /\hbar=\left\langle \alpha\left|\right.\beta\right\rangle =0`$
> - Element (2,1):
>   $`\left\langle \beta\left|2s_{z}\right|\alpha\right\rangle \hbar=\left\langle \beta\left|\right.\alpha\right\rangle =0`$
> - Element (2,2):
>   $`\left\langle \beta\left|2s_{z}\right|\beta\right\rangle /\hbar=-\left\langle \beta\left|\right.\beta\right\rangle =-1`$
>
> 1.  In the same manner construct representation matrices for the
>     ladder operators $`s_{+}`$ and $`s_{-}`$ and from this
>     $`\sigma_{x}`$ and $`\sigma_{y}`$.
> 2.  Demonstrate, for general vector operators $`\boldsymbol{A}`$ and
>     $`\boldsymbol{B}`$, the Dirac identity
>
> > left(boldsymbol{sigma}cdotboldsymbol{A}right)left(boldsymbol{sigma}
> > cdotboldsymbol{B}right)=boldsymbol{A}cdotboldsymbol{B}+iboldsymbol{sigma}
> > cdotleft(boldsymbol{A}timesboldsymbol{B}right)
>
> 3.  Use the Dirac identity to calculate
>
> > - $`\left(\boldsymbol{\sigma}\cdot\hat{\boldsymbol{p}}\right)\left(\boldsymbol{\sigma}\cdot\hat{\boldsymbol{p}}\right)`$
> > - $`\left(\boldsymbol{\sigma}\cdot\hat{\boldsymbol{\pi}}\right)\left(\boldsymbol{\sigma}\cdot\hat{\boldsymbol{\pi}}\right)`$,
> >   where $`\hat{\boldsymbol{\pi}}`$ is mechanical momentum
> >   $`\hat{\boldsymbol{\pi}}=\hat{\boldsymbol{p}}+e\boldsymbol{A}`$
> >   and $`\boldsymbol{A}`$ is a vector potential. Show how this
> >   expression is simplified in Coulomb gauge:
> >   $`\boldsymbol{\nabla}\cdot\boldsymbol{A}=0`$.
> > - Show that
> >   $`\left(\boldsymbol{\sigma}\cdot\hat{\boldsymbol{p}}\right)V_{eN}\left(\boldsymbol{\sigma}\cdot\hat{\boldsymbol{p}}\right)=\boldsymbol{p}V_{eN}\cdot\boldsymbol{p}+\hbar\left(\frac{Ze^{2}}{2\pi\varepsilon_{0}r^{3}}\right)\hat{\boldsymbol{s}}\cdot\hat{\boldsymbol{\ell}}`$,
> >   where $`V_{eN}`$ is the electron-nucleus interaction
> >   $`V_{en}=-\frac{Ze^{2}}{4\pi\varepsilon_{0}r}`$ in an atom

orphan  

# The $`UF_6`$ anion - another approach

Different and more robust solution for obtaining the SCF convergence of
the $`UF_6`$ anion is provided here. It is based on the damping of the
Fock matrix from SCF iteration to iteration.

## Starting from neutral $`UF_6`$

In the first step we follow the previous approach of starting from the
closed-shell (neutral) $`UF_6`$ system. (We also may utilize a shift in
the virtual space to reduce mixing between occupied and virtual
spinors.) In any case, we get smooth convergence with the default DIIS
method. Files to download are `UF6neutral.inp`, `UF6.mol`, `cc-pVDZ-PP`
and `ECP60MDF`: :

    pam --noarch --inp=UF6neutral.inp  --mol=UF6.mol --get "DFCOEF=DFCOEF.UF6neutral"

## The $`UF_6`$ anion from neutral

In the next step we compare two calculations starting from neutral
molecular spinors of UF6 (the file DFCOEF.UF6neutral). One calculation
uses the DIIS method (not shown here) and the other uses the damping
with two damping factors of 0.90 and 0.50. (File to download is
`UF6_anion_scfdamp.inp`): :

    pam --noarch --inp=UF6_anion_scfdamp.inp --mol=UF6.mol --put "DFCOEF.UF6neutral=DFCOEF" --get "DFCOEF=DFCOEF.UF6anion_damp_pass1" --replace dampfactor=0.9
    pam --noarch --inp=UF6_anion_scfdamp.inp --mol=UF6.mol --put "DFCOEF.UF6anion_damp_pass1=DFCOEF" --get "DFCOEF=DFCOEF.UF6anion_damp_pass2" --replace dampfactor=0.5

Concerning the DIIS speedup method, the SCF iterations initially
converge to ca $`10^{-5}`$ but then began to diverge and after about 150
iterations the error gradient of the Fock matrix is still around
$`10^{-5}`$.

However, with the damping convergence speedup the calculation converges
to ca $`10^{-8}`$ after about 50 iterations. The damping scheme uses
more of the input Fock matrix that the output Fock matrix. Concerning
the damping factor, the lower than 1 it is, we should get faster
convergence. However, when the factor is getting too small, iterations
diverge. The closer we are to the converged states, the smaller the
damping factor can be.

If we begin calculations from the converged solutions from the damping,
the DIIS iterations converge rapidly after few iterations. But this is
only with the deactivated active-active shells interaction (new file to
download is `UF6anion2.inp`): :

    pam --noarch --inp=UF6anion2.inp --mol=UF6.mol --put "DFCOEF.UF6anion_damp_pass2=DFCOEF" --get "DFCOEF=DFCOEF.UF6anion_diis"

## The $`UF_6`$ anion with OPENFAC=1

Unfortunately, with the factor OPENFAC=1 (what is the keyword
`SCF_.OPENFACTOR`) we were not able to get the convergence, neither with
the default DIIS, nor with damping factor (to download, `UF6anion.inp`
and `UF6anion3.inp`): :

    pam --noarch --inp=UF6anion.inp --mol=UF6.mol --put "DFCOEF.UF6anion_diis=DFCOEF"  
    pam --noarch --inp=UF6anion3.inp --mol=UF6.mol --put "DFCOEF.UF6anion_diis=DFCOEF"  --replace dampfactor=0.8

More effort is required to obtain converged molecular spinors of the
$`UF_6`$ anion with the active-active open-shell interaction (that is
OPENFAC=1). We leave this task to the reader.

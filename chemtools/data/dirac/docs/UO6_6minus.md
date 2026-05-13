orphan  

# The $`UO_6^{(6-)}`$ anion

Another (difficult) case study example is the $`UO_6^{(6-)}`$ anion. The
molecule is in squashed geometry (not fully octahedreal). Axial and
equatorial bond distances are not equal.

## The $`UO_6^{(6-)}`$ ground state

The ground state of $`UO_6^{(6-)}`$ converges fine, but not with the
default DIIS and starting from screened bare nucleus. Rather one has to
use the robust damping scheme. Try first the DAMPFC factor of 0.9: :

    pam  --noarch --get "DFCOEF=DFCOEF.UO6_6m_ground_1pass" --replace dampfactor=0.90 --inp=UO6_6minus.x2c.scf_damp.inp --mol=UO6.mol

(Files to download are `UO6_6minus.x2c.scf_damp.inp` and `UO6.mol`.)

Then, after we get the *DFCOEF.UO6_6m_ground_1pass* file, we decrease
the damping level to 0.65. One could try also some lower value: :

    pam  --noarch --put "DFCOEF.UO6_6m_ground_1pass=DFCOEF" --get "DFCOEF=DFCOEF.UO6_6m_ground_2pass" --replace dampfactor=0.50 --inp=UO6_6minus.x2c.scf_damp.inp  --mol=UO6.mol

In this way we get converge molecular spinors saved in the
*DFCOEF.UO6_6m_ground_2pass* file.

## The $`UO_6^{(5-)}`$ anion

Now to the next state - the $`UO_6^{(5-)}`$ system. With $`2p_{3/2}`$
electron out of the U atom there is a difficult convergence. There are
(at least) five siubsequent runs for that.

First run cca 500 iterations with the damping factor of 0.65 and with
the shift of virtual spinors of +0.65. This goes very slowly. The pass
two continues from pass one. Then pass three from previous pass up to
pass 4, which has another 500 iteration. Damping factors of these passes
were decreasing from 0.65 to 0.25. Finally, in pass 5 the convergence
was reached with the default DIIS method: :

    pam  --noarch --put "DFCOEF.UO6_6m_ground_2pass=DFCOEF" --get "DFCOEF=DFCOEF.UO6_5m_2p3_1pass" --replace dampfactor=0.65 --inp=UO6_5m.x2c.scf_2p3_damp.inp --mol=UO6.mol
    pam  --noarch --put "DFCOEF=DFCOEF.UO6_5m_2p3_1pass=DFCOEF" --get "DFCOEF=DFCOEF.UO6_5m_2p3_2pass" --replace dampfactor=0.55 --inp=UO6_5m.x2c.scf_2p3_damp.inp --mol=UO6.mol
    pam  --noarch --put "DFCOEF=DFCOEF.UO6_5m_2p3_2pass=DFCOEF" --get "DFCOEF=DFCOEF.UO6_5m_2p3_3pass" --replace dampfactor=0.45 --inp=UO6_5m.x2c.scf_2p3_damp.inp --mol=UO6.mol
    pam  --noarch --put "DFCOEF=DFCOEF.UO6_5m_2p3_3pass=DFCOEF" --get "DFCOEF=DFCOEF.UO6_5m_2p3_4pass" --replace dampfactor=0.25 --inp=UO6_5m.x2c.scf_2p3_damp.inp --mol=UO6.mol
    pam  --noarch --put "DFCOEF=DFCOEF.UO6_5m_2p3_4pass=DFCOEF" --get "DFCOEF=DFCOEF.UO6_5m_2p3_diis_5pass" --inp=UO6_5m.x2c.scf_2p3.inp --mol=UO6.mol

(For downloading is the file `UO6_5m.x2c.scf_2p3_damp.inp`). Note that
user may encounter difficulties when reproducing SCF convergence of this
tutorial case.

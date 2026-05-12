orphan  

# Examples of open-shell systems

Here we provide few examples of open-shell systems.

## Boron atom: open-shell SCF and small CI relativistic calculations

An example of one electron in two spinors is the Boron atom.

We seek to get the first excited state, 2P<sub>3/2</sub>, of B (X
2P<sub>1/2</sub>), and find out the
2P<sub>1/2</sub>-2P<sub>3/2</sub>fine structure splitting.

We select the Dirac-Coulomb Hamiltonian and neglect the (SS\|SS) class
of integrals.

### Complete open-shell CI

Possible way to get the value is performing small CI over the active
open-shell space, containing six 2p orbitals, which are split due to the
spin-orbital interaction. Note, that for pedagogical purposes we have
employed several computational symmetries.

Starting from the average SCF occupation (one electron distributed over
2p<sub>1/2</sub> and 2p<sub>3/2</sub> shells),

    pam  --mol=B.sto-3g.lsym.mol -inp=B.2Pav.dc_rkb.scf_2fs_cosci.inp

    pam  --mol=B.sto-3g.D2h.mol --inp=B.2Pav.dc_rkb.scf_2fs_cosci.inp

for one-fermion symmetries,

    pam  --mol=B.sto-3g.C2v.mol --inp=B.2Pav.dc_rkb.scf_1fs_cosci.inp

we get the value of 17.179488 cm-1 and in the outputs we see the proper
degeneracy of resulting total CI energies - 2 and 4.

By averaging the COSCI energies of the ground and the states we get the
total SCF energy of the average occupation

    ((2*(-24.249116730099) + 4*(-24.249038454591)))/6 = -24.249064546427
                                        SCF energy:     -24.249064546425263

### MO reordering

To apply the MO reordering scheme, first, one has to obtain the MO
coefficients file of the ground state:

    pam  --outcmo --mol=B.sto-3g.D2h.mol --inp=B.2P12.dc_rkb.scf_2fs.inp
    pam  --outcmo --mol=B.sto-3g.C1.mol  --inp=B.2P12.dc_rkb.scf_1fs.inp

and use them for calculation of the excited state:

    pam --incmo --mol=B.sto-3g.D2h.mol  --inp=B.2P32.dc_rkb.scf_2fs_reord_ovlsel.inp
    pam --incmo --mol=B.sto-3g.C1.mol     --inp=B.2P32.dc_rkb.scf_1fs_reord_ovlsel.inp

For the linear symmetry one can employ the M_J specified occupation of
shells both for the ground and the first excited state. Only the preDHF
reordering works in this case, not the overlap selection.

    pam --outcmo --mol=B.sto-3g.lsym.mol --inp=B.2P12.dc_rkb.scf_2fs_mj.inp
    pam --incmo --mol=B.sto-3g.lsym.mol --inp=B.2P32.dc_rkb.scf_2fs_reord_mj.inp

Finally, in the output one can see proper order of spinors for the
excited state - first 2p<sub>3/2</sub> spinors (four-fold degeneracy),
and then the 2p<sub>1/2</sub> spinors (two-fold degeneracy):

    * Fermion symmetry E1u
      * Open shell #1, f = 0.2500
     -0.28691011961454  ( 4)
      * Virtual eigenvalues, f = 0.0000
      0.12724113329935  ( 2)

Regarding total energies we get

    SCF energy 2_P1/2: -24.249116730391307
    SCF energy 2_P3/2: -24.249038454664991
    FSS : (-24.249116730391307+24.249038454664991)*219474.631280634 cm-1 = 17.179536 cm-1

what is very close to the above given COSCI value of 17.179488 cm-1.

# Correlated calculations

... to add ...

## Conclusion

Further quality improvement of the theoretical data would be in
enlarging basis set. The current one, decontracted STO-3G, is too small
and is chosen for demonstrative purposes only.

## Nitrogen atom open-shell atomic calculations

We set an average occupation of 3 electrons over 6p shells.

    pam  N.sto-3g.C2v.mol  N.3os_aver.dc_rkb.scf_1fs_cosci.inp
    pam  N.sto-3g.lsym.mol N.3os_aver.dc_rkb.scf_2fs_cosci.inp

    Obtained COSCI states are as follows:

    1        0.000000000          0.000000   1   1   1   1   0   0   0   0
    2        2.972174508      23972.206548   1   1   1   1   0   0   0   0
    3        2.972227617      23972.634901   1   1   1   1   1   1   0   0
    4        4.953663077      39953.991305   1   1   0   0   0   0   0   0
    5        4.953762831      39954.795879   1   1   1   1   0   0   0   0

## Hydroxyl radical

Here we present open-shell SCF calculations on the OH.(known as the
hydroxyl radical) system, followed by complete small open shell CI
method (cosci) for getting individual states.

The ground state of the OH readical is X 2_Pi_3/2 resulting from the
electron configuration of

    1s_sig(2) 2s_sig(2) 2pz_sig(2) 2pxy_pi_1/2(2) 2pxy_pi_3/2(1)

## Settig the SCF configuration

What does not converge is the 1 electron in 1 open-shell:

    .CLOSED SHELL
     8
    .OPEN S
    1
    1/2

Therefore for this case one has to resort to the
average-of-configurations (3 electrons over 4 pi shells):

    .CLOSED SHELL
     6
    .OPEN S
    1
    3/4

The Dirac-Coulomb Hamiltonian gives noncorrelated fine structure
splitting of 143.842458 cm-1 as the result of complete open-shell CI
over pi-shells.

    pam OH.ccpVDZ.lsym.mol OH.dc_rkb.scf_os_res.inp
    pam OH.ccpVDZ.C2v.mol OH.dc_rkb.scf_os_res.inp

COSCI energies are :

    -75.446610400724 (   2 * )
    -75.445955006263 (   2 * )

The average SCF energy is

::  
-75.446282703494589

and is equal to the average values of COSCI energies

    ((-75.446610400724-75.445955006263)/2=-75.446282703493

### Spin-free approach

Utilizing the Dirac-Coulomb spin-free Hamiltonian confirms the four-fold
(pi_x=pi_y) degeneracy of the valence open shells:

    pam OH.ccpVDZ.C2v.mol OH.dc_sf.scf_os_res.inp

With the resulting spin-free SCF (and COSCI) energy of

    -75.446280480624 (   4 * ).

## Mj selection

For this heteronuclear diatomic molecule one has the advantage of using
the omega-number occupation of shells. One can even place one electron
in one shell, because the ".MJSELECTION" keyword ensures the convergence
of both "X 2Pi_3/2"

    .CLOSED SHELL
     8
    .OPEN S
    1
    1/2
    # 2Pi_3/2
    .MJSELECTION
    3
    4 0 0
    0 1 0

and of the "A 2Pi_1/2" first excited state:

    .CLOSED SHELL
     8
    .OPEN S
    1
    1/2
    # 2Pi_1/2
    .MJSELECTION
    3
    3 1 0
    1 0 0

    pam OH.ccpVDZ.lsym.mol OH.dc.scf_mj_2Pi32.inp
    pam OH.ccpVDZ.lsym.mol OH.dc.scf_mj_2Pi12.inp

giving total SCF energies of the ground and the excited states:

    X 2Pi_3/2 :  -75.446717125968732
    A 2Pi_1/2 :  -75.446063726708829

The fine-structure splitting then makes

    FSS : (-75.446717125968732+75.446063726708829)*219474.631280634 = 143.404561 cm-1

which is very close to the above given COSCI value of 143.842458 cm-1.

Energy order of spinors for the OH "X 2Pi_3/2" ground state is:

    1: -20.623437096718       (Occupation: f = 1.0000)  m_j=  1/2; 1s_sig(2)
    2: -1.2970494235574       (Occupation: f = 1.0000)  m_j=  1/2; 2s_sig(2)
    3: -0.6455362075645       (Occupation: f = 1.0000)  m_j=  1/2; 2pz_sig(2)
    4: -0.5439526090110       (Occupation: f = 1.0000)  m_j=  1/2; 2pxy_pi_1/2(2)
    5: -0.5863405891346       (Occupation: f = 0.5000)  m_j= -3/2; 2pxy_pi_3/2(1)

While for the first excited "A 2Pi_1/2" state of the OH radical we have
this order:

    1: -20.623901392971       (Occupation: f = 1.0000)  m_j=  1/2 ;  1s_sig(2)
    2: -1.2971646385507       (Occupation: f = 1.0000)  m_j=  1/2 ;  2s_sig(2)
    3: -0.6456515101399       (Occupation: f = 1.0000)  m_j=  1/2 ;  2pz_sig(2)
    4: -0.5431435116553       (Occupation: f = 1.0000)  m_j= -3/2 ;  2pxy_pi_3/2(2)
    5: -0.5874674338778       (Occupation: f = 0.5000)  m_j=  1/2 ;  2pxy_pi_1/2(1)

## Correlated approach

The user can try to approach experimental
[data](http://webbook.nist.gov/cgi/cbook.cgi?ID=C3352576&Units=SI&Mask=1000#Diatomic).

## Hydroxyl radical cation OH+

Again one employs the average-of-configuration:

    .CLOSED SHELL
     6
    .OPEN S
    1
    2/4

Number of electronic states is higher.

Electronic states can be found
[here](http://webbook.nist.gov/cgi/cbook.cgi?ID=C12259299&Units=SI&Mask=1000#Diatomic).

# How to handle open-shell systems using CC methods

Coupled-Cluster Singles, Doubles and Noniterative triples (CCSD(T)) and
Fock-space Coupled-Cluser methods are powerful correlation methods in
the DIRAC program suite.

Coupled Cluster (CC) methods in DIRAC are most powerful ab-initio
correlation methods working upon two/four-component Kramers unrestricted
spinors. They serve as widely employed relativistic analogue with
respect the nonrelativistic realm.

Therefore we feel it is important to present the user few hints on how
to fully exhaust CC capabilities for practical calculations.

Even at the Coupled Cluster correlated level the user has certain
variability in choosing the occupation of spinors. Thus he may alter the
electronic state of a system.

We give you few hints how to employ them in correlated open-shell
calculations.

## CCSD(T) method

Example 1: FO molecule

## Fock space CCSD method

The starting system is always closed shell. One can add one or
two-electrons to the N-electron system to iterate into N+1 and/or N+2
systems.

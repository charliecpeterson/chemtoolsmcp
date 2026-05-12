<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/advanced.examples/ex-rc.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

5.1.6. Solvent models

[previous](<ex-ex.html> "5.1.5. Excited states") | [next](<ex-so.html> "5.1.7. Computing relativistic effects in molecules") | [index](<../genindex.html> "General Index")

# 5.1.6. Solvent models¶

  * Kirkwood model

  * PCM

  * Calculation of solvent effects: Kirkwood model

  * Solvation effects in ground states. PCM model in formaldehyde

  * Solvation effects in excited states. PCM model and acrolein




For isolated molecules of modest size the _ab initio_ methods have reached great accuracy at present both for ground and excited states. Theoretical studies on isolated molecules, however, may have limited value to bench chemists since most of the actual chemistry takes place in a solvent. If solute–solvent interactions are strong they may have a large impact on the electronic structure of a system and then on its excitation spectrum, reactivity, and properties. For these reasons, numerous models have been developed to deal with solute–solvent interactions in _ab initio_ quantum chemical calculations. A microscopic description of solvation effects can be obtained by a supermolecule approach or by combining statistical mechanical simulation techniques with quantum chemical methods. Such methods, however, demand expensive computations. By contrast, at the phenomenological level, the solvent can be regarded as a dielectric continuum, and there are a number of approaches [[328](<../references.html#id208> "V. Barone, M. Cossi. J. Phys. Chem. A, 102 \(1998\) 1995-2001."), [329](<../references.html#id149> "M. Cossi, N. Rega, G. Scalmani, V. Barone. J. Chem. Phys., 114 \(2001\) 5691-5701."), [330](<../references.html#id189> "G. Karlström. J. Phys. Chem., 92 \(1988\) 1315-1318."), [331](<../references.html#id68> "L. Serrano-Andrés, M. P. Fülscher, G. Karlström. Int. J. Quantum Chem., 65 \(1997\) 167-181."), [332](<../references.html#id55> "J. Tomasi, M. Persico. Chem. Rev., 94 \(1994\) 2027-2094.")] based on the classical reaction field concept.

Molcas can model the solvent within the framework of SCF, RASSCF and CASPT2 programs, for the calculation of energies and properties and also for geometry optimizations. The reaction field formalism is based on a sharp partition of the system: the solute molecule (possibly supplemented by some explicit solvent molecules) is placed in a cavity surrounded by a polarizable dielectric. The surrounding is characterized mainly by its dielectric constant and density: an important parameter of the method is the size of the cavity; the dielectric medium is polarized by the solute, and this polarization creates a reaction field which perturbs the solute itself.

Two versions of the model are presently available: one is based on the Kirkwood model [[330](<../references.html#id189> "G. Karlström. J. Phys. Chem., 92 \(1988\) 1315-1318."), [331](<../references.html#id68> "L. Serrano-Andrés, M. P. Fülscher, G. Karlström. Int. J. Quantum Chem., 65 \(1997\) 167-181.")] and uses only spherical cavities; the other is called PCM (polarizable continuum model) [[328](<../references.html#id208> "V. Barone, M. Cossi. J. Phys. Chem. A, 102 \(1998\) 1995-2001."), [329](<../references.html#id149> "M. Cossi, N. Rega, G. Scalmani, V. Barone. J. Chem. Phys., 114 \(2001\) 5691-5701.")] and can use cavities of general shape, modeled on the actual solute molecule. In the former case, the reaction field is computed as a truncated multipolar expansion and added as a perturbation to the one-electron Hamiltonian; in the latter case the reaction field is expressed in terms of a collection of apparent charges (solvation charges) spread on the cavity surface: the PCM reaction field perturbs both one- and two-electron Hamiltonian operators. In both cases, the solvent effects can be added to the Hamiltonian at any level of theory, including MRCI and CASPT2.

## 5.1.6.1. Kirkwood model¶

This version of the model only uses spherical cavities. In addition, it includes Pauli repulsion due to the medium by introducing a repulsive potential representing the exchange repulsion between the solute and the solvent. This is done by defining a penalty function of Gaussian type, generating the corresponding spherical well integrals, and adding them to the one-electron Hamiltonian. When the repulsion potential is used, the size of the cavity should be optimized for the ground state of the molecule (see below). If the repulsive potential is not used and the cavity size is chosen to be smaller (molecular size plus van der Waals radius as is the usual choice in the literature) one must be aware of the consequences: larger solvent effects but also an unknown presence of molecular charge outside the boundaries of the cavity. This is not a consequence of the present model but it is a general feature of cavity models [[331](<../references.html#id68> "L. Serrano-Andrés, M. P. Fülscher, G. Karlström. Int. J. Quantum Chem., 65 \(1997\) 167-181.")].

## 5.1.6.2. PCM¶

The cavities are defined as the envelope of spheres centered on solute atoms or atomic groups (usually hydrogens are included in the same sphere of the atoms they are bonded to). Two selection of radii are presently available, i.e. Pauling radii, and the so-called UATM (united atom topological model) radii: the latter is the default for PCM calculations; sphere radii can also be provided by the user in the input file. The solvation charges are placed in the middle of small tiles (_tesserae_) drawn on the surface; the number of solvation charges can be gauged by changing the average area of tesserae (keyword AAre in SEWARD).

The program prints some information related to the cavity, where one should always check carefully the magnitude of sphere radii: the program adjusts them automatically to the solute topology (each radius depends on hybridization, bonds, etc.), and sometimes this causes some problems (for instance, discontinuities could appear during the scan of a potential energy surface): if this happens, it is preferable to provide the desired radii in the input file, so that they will be kept at all geometries.

When doing state-average RASSCF calculations, one has to specify which root is to be used to generate the solvation charges: this means that the PCM reaction field will be in equilibrium with a specific electronic state, while it perturbs all the states included in the calculation.

In electronic transitions (e.g. photon absorption or emission) one has to include non-equilibrium effects, due to the finite relaxation time of solvent molecules following a sudden change in electronic distribution. This is done by partitioning the reaction field in two components (fast and slow, the former always equilibrated, the latter delayed), whose magnitude is determined by the static dielectric constant and by a “fast” dielectric constant [[333](<../references.html#id148> "M. Cossi, V. Barone. J. Chem. Phys., 112 \(2000\) 2427-2435.")] (for very fast processes, like photon absorption, the fast constant is equal to the square of the refraction index). To perform a non-equilibrium calculation, for example to study a ground-to-excited state transition, one has to perform a regular calculation at equilibrium for the ground state, followed by a calculation for the excited state specifying the keyword NONEQ in the RASSCF program. Failing to include the keyword NONEQ will cause the program to compute equilibrium solvation also for the excited state, what would be appropriate for an adiabatic, instead of a vertical, transition.

CASPT2 calculations can be performed as usual for isolated molecules, specifying the keyword RFPERT. Geometry optimizations can be performed as usual: note that the arrangement of solvation charges around the solute molecule is likely to break the molecular symmetry. If the symmetry was explicitly requested in SEWARD, the system will keep it through the optimization even in the presence of the solvent, otherwise the convergence could be more difficult, and the final geometry could result of a lower symmetry.

## 5.1.6.3. Calculation of solvent effects: Kirkwood model¶

We begin by performing a CASSCF/CASPT2 reaction field calculation on the ground state of a molecule.

To use the Kirkwood model, the keyword
    
    
    REACtion field
    

is needed; if no repulsive potential is going to be used the input simply consists in adding the appropriate data (dielectric constant of the medium, cavity size, and angular quantum number of the highest multipole moment of the charge distribution) into the SEWARD input:
    
    
    &SEWARD &END
    ...
    ...
    RF-Input
    Reaction field
    80 8.0 4
    End of RF-Input
    ...
    ...
    End of Input
    

This will compute the reaction field at those levels. The dielectric constant 80.0 correspond to water as solvent. The radius of the cavity is 8.0 in atomic units. Finally 4 is the maximum angular moment number used in the multipole expansion. The cavity origin is the coordinate origin, thus the molecule must be placed accordingly.

If we want to include the reaction field (either PCM or Kirkwood model) at other levels of theory the keyword RFPErt must be added to the MOTRA or CASPT2 inputs.

We are, however, going to explain the more complicated situation where a repulsive well potential has to be added to the model. In this case it is convenient to optimize the size of the cavity, although in so doing we obtain large cavity sizes and therefore smaller solvent effects. More realistic results can be obtained if additional and specific solvent molecules are added inside the cavity.

To define the well potential we have to add the keyword WELL Integrals to the SEWARD input to compute and add the Pauli repulsion integrals to the bare Hamiltonian.

The requirements considered to build this potential are that it shall reproduce solvation energies for spherical particles, ions, and that it must be wide enough so that the electrons in the excited state of the molecules are also confined to the cavity. Negative ions have the property that their electrons are loosely bound and they are thus suited for parametrizing the repulsive potential. The final result of different calibration calculations [[331](<../references.html#id68> "L. Serrano-Andrés, M. P. Fülscher, G. Karlström. Int. J. Quantum Chem., 65 \(1997\) 167-181."), [334](<../references.html#id41> "A. Bernhardsson, R. Lindh, G. Karlström, B. O. Roos. Chem. Phys. Lett., 251 \(1996\) 141-149.")] is a penalty function which includes four Gaussians. If \\(a\\) is the radius of the cavity the Gaussians are placed at distances \\(a+2.0\\), \\(a+3.0\\), \\(a+5.0\\) and \\(a+7.0\\) \\(a_0\\) from the cavity’s center with exponents 5.0, 3.5, 2.0 and 1.4, respectively.

As an example we will use the N,N-dimethylaminobenzonitrile (DMABN) molecule (see Figure 5.1.6.1). This is a well known system with large dipole moments both in ground and excited states which suffer important effects due to the polar environment.

[](<../_images/dmabn.png>)

Figure 5.1.6.1 N,N-dimethylaminobenzonitrile (DMABN)¶
    
    
    &SEWARD &END
    Title
    para-DMABN molecule. Cavity size: 10 au.
    Symmetry
     X XY
    Basis set
    N.ANO-S...3s2p1d.
    N1             0.0000000000        0.0000000000        4.7847613288
    N2             0.0000000000        0.0000000000       -8.1106617786
    End of basis
    Basis set
    C.ANO-S...3s2p1d.
    C1             0.0000000000        0.0000000000        2.1618352923
    C2             0.0000000000        2.2430930886        0.7747833630
    C3             0.0000000000        2.2317547910       -1.8500321252
    C4             0.0000000000        0.0000000000       -3.1917306021
    C5             0.0000000000        0.0000000000       -5.9242601761
    C6             0.0000000000        2.4377336900        6.0640991723
    End of basis
    Basis set
    H.ANO-S...2s.
    H1             0.0000000000        4.0043085530       -2.8534714086
    H2             0.0000000000        4.0326542950        1.7215314260
    H3             0.0000000000        2.1467175630        8.0879851846
    H4             1.5779129980        3.6622699270        5.5104123361
    End of basis
    
    RF-Input
    reaction field
    38.8 10.0 4
    End of RF-Input
    
    Well Int
    4
    1.0 5.0 12.0
    1.0 3.5 13.0
    1.0 2.0 15.0
    1.0 1.4 17.0
    End of Input
    
    &SCF &END
    TITLE
     DMABN molecule
    OCCUPIED
    20 2 12 5
    ITERATIONS
    50
    END OF INPUT
    
    &RASSCF &END
    TITLE
     p-DMABN
    SYMMETRY
        1
    SPIN
        1
    NACTEL
       10    0    0
    FROZEN
        8    0    3    0
    INACTIVE
       12    1    9    1
    RAS2
        0    2    0    7
    THRS
    1.0E-06,1.0E-03,1.0E-03
    ITER
    50,25
    LUMORB
    END OF INPUT
    

In the SEWARD input the WELL Integrals must include first the number of Gaussians used (four), followed by the coefficient and exponent of the Gaussian and the radius of the cavity in the sequence explained above: first the most compact Gaussian with the radius plus 2.0 \\(a_0\\), and so on to the least compact Gaussian. Here, we have defined a cavity size of 10 \\(a_0\\) (cavity centered at coordinate origin). The RASSCF program will read the RCTFLD input, prepared this time for acetonitrile (\\(\epsilon = 38.8\\)), a cavity size of 10.0 \\(a_0\\) (the same as in the SEWARD input) and a multipole expansion up to the fourth order which is considered sufficient [[331](<../references.html#id68> "L. Serrano-Andrés, M. P. Fülscher, G. Karlström. Int. J. Quantum Chem., 65 \(1997\) 167-181.")]. The active space includes the \\(\pi\\) space over the molecular plane, excluding the \\(\pi\\) orbital of the \\(\ce{CN}\\) group which lies in the molecular plane.

We repeat the calculation for different cavity sizes in order to find the radius which gives the lowest absolute energy at the CASSCF level. The presence of the repulsive terms allows the cavity radius to be computed by energy minimization. For the calculations using different cavity sizes it is not necessary to repeat the calculation of all the integrals, just those related to the well potential. Therefore, the keyword ONEOnly can be included in the SEWARD input. The ONEINT file will be modified and the ORDINT file is kept the same for each molecular geometry. The energies obtained are in Table 5.1.6.1.

Table 5.1.6.1 Ground state CASSCF energies for DMABN with different cavity sizes.¶ Radius (\\(a_0\\)) | CASSCF energies (\\(E_{\text{h}}\\))  
---|---  
no cav. | −455.653242  
10.0 | −455.645550  
11.0 | −455.653486  
12.0 | −455.654483  
14.0 | −455.654369  
16.0 | −455.654063  
  
Taking the gas-phase value (no cav.) as the reference, the CASSCF energy obtained with a 10.0 \\(a_0\\) cavity radius is higher. This is an effect of the repulsive potential, meaning that the molecule is too close to the boundaries. Therefore we discard this value and use the values from 11.0 to 16.0 to make a simple second order fit and obtain a minimum for the cavity radius at 13.8 \\(a_0\\).

Once we have this value we also need to optimize the position of the molecule in the cavity. Some parts of the molecule, especially those with more negative charge, tend to move close to the boundary. Remember than the sphere representing the cavity has its origin in the cartesian coordinates origin. We use the radius of 13.8 \\(a_0\\) and compute the CASSCF energy at different displacements along the coordinate axis. Fortunately enough, this molecule has \\(C_{2v}\\) symmetry. That means that displacements along two of the axis (\\(x\\) and \\(y\\)) are restricted by symmetry. Therefore it is necessary to analyze only the displacements along the \\(z\\) coordinate. In a less symmetric molecule all the displacements should be studied even including combination of the displacements. The result may even be a three dimensional net, although no great accuracy is really required. The results for DMABN in \\(C_{2v}\\) symmetry are compiled in Table 5.1.6.2.

Table 5.1.6.2 Ground state CASSCF energies for different translations with respect to the initial position of of the DMABN molecule in a 13.8 \\(a_0\\) cavity.¶ Disp. in \\(z\\) (\\(a_0\\)) | CASSCF energies (\\(E_{\text{h}}\\))  
---|---  
+0.5 | -455.654325  
0.0 | -455.654400  
−0.5 | -455.654456  
−1.0 | -455.654486  
−1.5 | -455.654465  
  
Fitting these values to a curve we obtain an optimal displacement of −1.0 \\(a_0\\). We move the molecule and reoptimize the cavity radius at the new position of the molecule. The results are listed in Table 5.1.6.3.

Table 5.1.6.3 Ground state CASSCF energies for DMABN with different cavity sizes. The molecule position in the cavity has been optimized.¶ Radius (\\(a_0\\)) | CASSCF energies (\\(E_{\text{h}}\\))  
---|---  
11.8 | −455.653367  
12.8 | −455.654478  
13.8 | −455.654486  
  
There is no significant change. The cavity radius is then selected as 13.8 \\(a_0\\) and the position of the molecule with respect to the cavity is kept as in the last calculation. The calculation is carried out with the new values. The SCF or RASSCF outputs will contain the information about the contributions to the solvation energy. The CASSCF energy obtained will include the reaction field effects and an analysis of the contribution to the solvation energy for each value of the multipole expansion:
    
    
    Reaction field specifications:
    ------------------------------
    
     Dielectric Constant :                  .388E+02
     Radius of Cavity(au):                  .138E+02
     Truncation after    :                 4
    
    Multipole analysis of the contributions to the dielectric solvation energy
    
    --------------------------------------
       l             dE
    --------------------------------------
       0            .0000000
       1           -.0013597
       2           -.0001255
       3           -.0000265
       4           -.0000013
    --------------------------------------
    

Notice that the two-electron integral file is the same independent of the cavity size, well integrals used or translational movement of the molecule. Therefore, if the well parameters are changed only the one-electron integral file ONEINT need to be recomputed using the option ONEOnly in the SEWARD input. In the previous script we have named the ONEINT as $Project.OneInt.$Solvent, but not because it depends on the solvent but on the cavity radius which should be different for each solvent when the well potential is used.

## 5.1.6.4. Solvation effects in ground states. PCM model in formaldehyde¶

The reaction field parameters are added to the SEWARD program input through the keyword
    
    
    RF-Input
    

To invoke the PCM model the keyword
    
    
    PCM-model
    

is required. A possible input is
    
    
    RF-input
    PCM-model
    solvent
    acetone
    AAre
    0.2
    End of rf-input
    

which requests a PCM calculation with acetone as solvent, with tesserae of average area 0.2 Å². Note that the default parameters are solvent = water, average area 0.4 Å²; see the SEWARD manual section for further PCM keywords. By default the PCM adds non-electrostatic terms (i.e. cavity formation energy, and dispersion and repulsion solute-solvent interactions) to the computed free-energy in solution.

A complete input for a ground state CASPT2 calculation on formaldehyde (\\(\ce{H2CO}\\)) in water is
    
    
    &GATEWAY
    Title
    formaldehyde
    Coord
    4
    
    H      0.000000    0.924258   -1.100293    angstrom
    H      0.000000   -0.924258   -1.100293    angstrom
    C      0.000000    0.000000   -0.519589    angstrom
    O      0.000000    0.000000    0.664765    angstrom
    Basis set
     6-31G*
    Group
     X Y
    RF-input
    PCM-model
    solvent
    water
    end of rf-input
    End of input
    
    &SEWARD
    End of input
    
    &SCF
    Title
    formaldehyde
    Occupied
    5 1 2 0
    End of input
    
    &RASSCF
    Title
    formaldehyde
    nActEl
    4 0 0
    Inactive
    4 0 2 0
    Ras2
    1 2 0 0
    LumOrb
    End of input
    
    &CASPT2
    Frozen
    4  0  0  0
    RFPErt
    End of input
    

## 5.1.6.5. Solvation effects in excited states. PCM model and acrolein¶

In the PCM picture, the solvent reaction field is expressed in terms of a polarization charge density \\(\sigma(\vec{s})\\) spread on the cavity surface, which, in the most recent version of the method, depends on the electrostatic potential \\(V(\vec{s})\\) generated by the solute on the cavity according to

\\[\left[ \frac{\epsilon+1}{\epsilon-1} \hat{S} -\frac{1}{2\pi}\hat{S}\hat{D}^* \right] \sigma(\vec{s}) = \left[ - 1 + \frac{1}{2\pi} \hat{D}\right] V(\vec{s})\\]

where \\(\epsilon\\) is the solvent dielectric constant and \\(V(\vec{s})\\) is the (electronic+nuclear) solute potential at point \\(\vec{s}\\) on the cavity surface. The \\(\hat{S}\\) and \\(\hat{D}^*\\) operators are related respectively to the electrostatic potential \\(V^\sigma({\vec{s}})\\) and to the normal component of the electric field \\(E_\perp^\sigma(\vec{s})\\) generated by the surface charge density \\(\sigma(\vec{s})\\). It is noteworthy that in this PCM formulation the polarization charge density \\(\sigma(\vec{s})\\) is designed to take into account implicitly the effects of the fraction of solute electronic density lying outside the cavity.

In the computational practice, the surface charge distribution \\(\sigma(\vec{s})\\) is expressed in terms of a set of point charges vec{q} placed at the center of each surface tessera, so that operators are replaced by the corresponding square matrices. Once the solvation charges (\\(\vec{q}\\)) have been determined, they can be used to compute energies and properties in solution.

The interaction energy between the solute and the solvation charges can be written

\\[E_{\text{int}} = \mat{V}^{\text{T}} \vec{q} = \sum_i^{N_{\text{TS}}} V_i q_i\\]

where \\(V_i\\) is the solute potential calculated at the representative point of tessera \\(i\\). The charges act as perturbations on the solute electron density \\(\rho\\): since the charges depend in turn on \\(\rho\\) through the electrostatic potential, the solute density and the charges must be adjusted until self consistency. It can be shown [[332](<../references.html#id55> "J. Tomasi, M. Persico. Chem. Rev., 94 \(1994\) 2027-2094.")] that for any SCF procedure including a perturbation linearly depending on the electron density, the quantity that is variationally minimized corresponds to a free energy (i.e. \\(E_{\text{int}}\\) minus the work spent to polarize the dielectric and to create the charges). If \\(E^0=E[\rho^0] + V_{\text{NN}}\\) is the solute energy in vacuo, the free energy minimized in solution is

\\[G = E[\rho] + V_{\text{NN}} + \frac{1}{2} E_{\text{int}}\\]

where \\(V_{\text{NN}}\\) is the solute nuclear repulsion energy, \\(\rho^0\\) is the solute electronic density for the isolated molecule, and \\(\rho\\) is the density perturbed by the solvent.

The inclusion of non-equilibrium solvation effects, like those occurring during electronic excitations, is introduced in the model by splitting the solvation charge on each surface element into two components: \\(q_{i,\text{f}}\\) is the charge due to electronic (fast) component of solvent polarization, in equilibrium with the solute electronic density upon excitations, and \\(q_{i,\text{s}}\\), the charge arising from the orientational (slow) part, which is delayed when the solute undergoes a sudden transformation.

The photophysics and photochemistry of acrolein are mainly controlled by the relative position of the \\(^1(n\to\pi^*)\\), \\(^3(n\to\pi^*)\\) and \\(^3(\pi\to\pi^*)\\) states, which is, in turn, very sensitive to the presence and the nature of the solvent. We choose this molecule in order to show an example of how to use the PCM model in a CASPT2 calculation of vertical excitation energies.

The three states we want to compute are low-lying singlet and triplet excited states of the _s-trans_ isomer. The \\(\pi\\) space (4 \\(\pi\\) MOs, 4 \\(\pi\\)-electrons) with the inclusion of the lone-pair MO (\\(n_y\\)) is a suitable choice for the active space in this calculation. For the calculation in aqueous solution, we need first to compute the CASPT2 energy of the ground state in presence of the solvent water. This is done by including in the SEWARD input for the corresponding gas-phase calculation the section
    
    
    RF-input
    
    PCM-model
    solvent
     water
    DIELectric constant
     78.39
    CONDuctor version
    AARE
     0.4
    
    End of rf-input
    

If not specified, the default solvent is chosen to be water. Some options are available. The value of the dielectric constant can be changed for calculations at temperatures other than 298 K. For calculations in polar solvents like water, the use of the conductor model (C-PCM) is recommended. This is an approximation that employs conductor rather than dielectric boundary conditions. It works very well for polar solvents (i.e. dielectric constant greater than about 5), and is based on a simpler and more robust implementation. It can be useful also in cases when the dielectric model shows some convergence problems. Another parameter that can be varied in presence of convergency problem is the average area of the tesserae of which the surface of the cavity is composed. However, a lower value for this parameter may give poorer results.

Specific keywords are in general needed for the other modules to work with PCM, except for the SCF. The keyword NONEquilibrium is necessary when computing excited states energies in RASSCF. For a state specific calculation of the ground state CASSCF energy, the solvent effects must be computed with an equilibrium solvation approach, so this keyword must be omitted. None the less, the keyword RFpert must be included in the CASPT2 input in order to add the reaction field effects to the one-electron Hamiltonian as a constant perturbation.
    
    
    &RASSCF &END
    Title
    Acrolein GS + PCM
    Spin
     1
    Symmetry
     1
    nActEl
     6 0 0
    Frozen
     4 0
    Inactive
     8 0
    Ras2
     1 4
    LUMORB
    THRS
    1.0e-06 1.0e-04 1.0e-04
    ITERation
     100 100
    End of input
    
    &CASPT2 &END
    Title
     ground state + PCM
    RFpert
    End of Input
    

Information about the reaction field calculation employing a PCM-model appear first in the SCF output
    
    
    Polarizable Continuum Model (PCM) activated
    Solvent:water
    Version: Conductor
    Average area for surface element on the cavity boundary: 0.4000 angstrom2
    Minimum radius for added spheres: 0.2000 angstrom
    
    
    
    Polarized Continuum Model Cavity
    ================================
    
     Nord Group  Hybr  Charge Alpha Radius          Bonded to
       1   O     sp2   0.00   1.20  1.590   C   [d]
       2   CH    sp2   0.00   1.20  1.815   O   [d]  C   [s]
       3   CH    sp2   0.00   1.20  1.815   C   [s]  C   [d]
       4   CH2   sp2   0.00   1.20  2.040   C   [d]
     ------------------------------------------------------------------------------
    

The following input is used for the CASPT2 calculation of the \\(^3A''(n\to\pi^*)\\) state. Provided that the same $WorkDir has been using, which contains all the files of of the calculation done for the ground state, the excited state calculation is done by using inputs for the RASSCF and the CASPT2 calculations:
    
    
    &RASSCF &END
    Title
    Acrolein n->pi* triplet state + PCM
    Spin
     3
    Symmetry
     2
    nActEl
     6 0 0
    Frozen
     4 0
    Inactive
     8 0
    Ras2
     1 4
    NONEquilibrium
    LUMORB
    ITERation
     100 100
    End of input
    
    &CASPT2 &END
    Title
     triplet state
    RFpert
    End of Input
    

Note the RASSCF keyword NONEQ, requiring that the slow part of the reaction field be frozen as in the ground state, while the fast part is equilibrated to the new electronic distribution. In this case the fast dielectric constant is the square of the refraction index, whose value is tabulated for all the allowed solvents (anyway, it can be modified by the user through the keyword DIELectric in SEWARD).

The RASSCF output include the line:
    
    
    Calculation type: non-equilibrium (slow component from JobOld)
    
     Reaction field from state:            1
    

This piece of information means that the program computes the solvent effects on the energy of the \\(^3A''(n\to\pi^*)\\) by using a non-equilibrium approach. The slow component of the solvent response is kept frozen in terms of the charges that have been computed for the previous equilibrium calculation of the ground state. The remaining part of the solvent response, due to the fast charges, is instead computed self-consistently for the state of interest (which is state 1 of the specified spatial and spin symmetry in this case).

The vertical excitations to the lowest valence states in aqueous solution for _s-trans_ acrolein are listed in the Table 5.1.6.4 and compared with experimental data. As expected by qualitative reasoning, the vertical excitation energy to the \\(^1A''(n\to\pi^*)\\) state exhibits a blue shift in water. The value of the vertical transition energy computed with the inclusion of the PCM reaction field is computed to be 3.96 eV at the CASPT2 level of theory. The solvatochromic shift is thus of +0.33 eV. Experimental data are available for the excitation energy to the \\(^1A''(n\to\pi^*)\\) state. The band shift in going from isooctane to water is reported to be +0.24 eV which is in fair agreement with the PCM result.

No experimental data are available for the excitation energies to the triplet states of acrolein in aqueous solution. However it is of interest to see how the ordering of these two states depends on solvent effects. The opposing solvatochromic shifts produced by the solvent on these two electronic transitions place the two triplet states closer in energy. This result might suggest that a dynamical interconversion between the \\(n\pi^*\\) and \\(\pi\pi^*\\) may occur more favorable in solution.

Table 5.1.6.4 Vertical excitation energies/eV (solvatochromic shifts) of _s-trans_ acrolein in gas-phase and in aqueous solution.¶ State | Gas-phase | Water | Expt.1  
---|---|---|---  
\\(^1A''(n_y\to\pi^*)\\) | 3.63 | 3.96 (+0.33) | 3.94 (+0.24)2  
\\(T_1\\) \\(^3A''(n_y\to\pi^*)\\) | 3.39 | 3.45 (+0.06) |   
\\(T_2\\) \\(^3A'(\pi\to\pi^*)\\) | 3.81 | 3.71 (−0.10) |   
  
1
    

Ref. [[335](<../references.html#id75> "W. F. Forbes, R. Shilton. J. Am. Chem. Soc., 81 \(1959\) 786-790.")]

2
    

Solvatochromic shifts derived by comparison of the absorption wave lengths in water and isooctane

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<ae.html>)
    * [5.1. Examples](<examples.html>)
      * [5.1.1. Computing high symmetry molecules](<ex-x2.html>)
      * [5.1.2. Geometry optimizations and Hessians](<ex-op.html>)
      * [5.1.3. Computing a reaction path](<ex-rp.html>)
      * [5.1.4. High quality wave functions at optimized structures](<ex-hi.html>)
      * [5.1.5. Excited states](<ex-ex.html>)
      * 5.1.6. Solvent models
      * [5.1.7. Computing relativistic effects in molecules](<ex-so.html>)
    * [5.2. Annexes](<annexes.html>)



### Search

[previous](<ex-ex.html> "5.1.5. Excited states") | [next](<ex-so.html> "5.1.7. Computing relativistic effects in molecules") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/advanced.examples/ex-rc.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

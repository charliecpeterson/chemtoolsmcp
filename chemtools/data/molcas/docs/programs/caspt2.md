<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.3. CASPT2

[previous](<averd.html> "4.2.2. AVERD") | [next](<casvb.html> "4.2.4. CASVB") | [index](<../../genindex.html> "General Index")

# 4.2.3. CASPT2¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




Second order multiconfigurational perturbation theory is used in the program CASPT2 [[32](<../../references.html#id200> "K. Andersson, P.-Å. Malmqvist, B. O. Roos, A. Sadlej, K. Wolinski. J. Phys. Chem., 94 \(1990\) 5483-5486."), [33](<../../references.html#id125> "K. Andersson, P.-Å. Malmqvist, B. O. Roos. J. Chem. Phys., 96 \(1992\) 1218-1226.")] to compute the (dynamic) correlation energy. The reference state is usually of the CAS type, but the program has been extended to also accept RAS reference states [[34](<../../references.html#id173> "P. Å. Malmqvist, K. Pierloot, A. R. Moughal Shahi, C. J. Cramer, L. Gagliardi. J. Chem. Phys., 128 \(2008\) 204109\(1–10\)."), [35](<../../references.html#id180> "V. Sauri, L. Serrano-Andrés, A. R. Moughal Shahi, L. Gagliardi, S. Vancoillie, K. Pierloot. J. Chem. Theory Comput., 7 \(2011\) 153-168.")]. The first step is therefore a RASSCF calculation and the CASPT2 calculation gives a second order estimate of the difference between the RASSCF and the full CI energy. For calculations using a true RAS reference, benchmark calculations were reported by Sauri et al. [[35](<../../references.html#id180> "V. Sauri, L. Serrano-Andrés, A. R. Moughal Shahi, L. Gagliardi, S. Vancoillie, K. Pierloot. J. Chem. Theory Comput., 7 \(2011\) 153-168.")]. For CASSCF references, the CASPT2 method has been tested in a large number of applications [[36](<../../references.html#id289> "B. O. Roos, M. P. Fülscher, P.-Å. Malmqvist, M. Merchán, L. Serrano-Andrés. In Quantum Mechanical Electronic Structure Calculations with Chemical Accuracy, volume 13 of Understanding Chemical Reactivity, pages 357-438. Kluwer Academic Publishers, 1995."), [37](<../../references.html#id293> "B. O. Roos, K. Andersson, M. P. Fülscher, P.-Å. Malmqvist, L. Serrano-Andrés, K. Pierloot, M. Merchán. In New Methods in Computational Quantum Mechanics, volume 93 of Advances in Chemical Physics, pages 213-331. John Wiley & Sons, 1996.")]. Here follows a brief summary of results.

Bond distances are normally obtained with an accuracy of better that 0.01 Å for bonds between first and second row atoms. With the standard Fock matrix formulation, bond energies are normally underestimated with between 2 and 5 kcal/mol for each bond formed. This is due to a systematic error in the method [[38](<../../references.html#id65> "K. Andersson, B. O. Roos. Int. J. Quantum Chem., 45 \(1993\) 591-607.")]. In every process where the number of paired electrons is changed, an error of this size will occur for each electron pair. For example, the singlet-triplet energy difference in the methylene radical (\\(\ce{CH2}\\)) is overestimated with about 3 kcal/mol [[33](<../../references.html#id125> "K. Andersson, P.-Å. Malmqvist, B. O. Roos. J. Chem. Phys., 96 \(1992\) 1218-1226.")]. Heats of reactions for isogyric reactions are predicted with an accuracy of ±2 kcal/mol. These results have been obtained with saturated basis sets and all valence electrons active. The use of smaller basis sets and other types of active spaces may, of course, affect the error.

These systematic errors have recently been considerably reduced by the introduction of a modified zeroth order Hamiltonian [[39](<../../references.html#id52> "G. Ghigo, B. O. Roos, P.-Å. Malmqvist. Chem. Phys. Lett., 396 \(2004\) 142-149.")]. The method introduces a shift (the IPEA shift) that modifies the energies of active orbitals such that they become closer to ionization energies when excited from and closer to electron affinities when excited out of. The approach has been tested for 49 diatomic molecules, reducing the mean error in \\(D_0\\) from 0.2 to 0.1 eV. For the triply bonded molecules \\(\ce{N2}\\), \\(\ce{P2}\\), and \\(\ce{As2}\\) it was reduced from 0.45 eV to less than 0.15 eV. Similar improvements were obtained for excitation and ionization energies. The IPEA modified \\(H_0\\) (with a shift parameter of 0.25) is default in Molcas from version 6.4. If MOLCAS_NEW_DEFAULTS is set to `YES`, the default will be no IPEA shift.

An alternative to IPEA is to use the options, called “\\(g_1\\)”, “\\(g_2\\)”, and “\\(g_3\\)” (See ref. [[40](<../../references.html#id259> "K. Andersson. Theor. Chim. Acta, 91 \(1995\) 31-46.")]), that stabilizes the energies of the active orbitals. The remaining error is no longer systematic, and is generally reduced. For example, the error in the singlet-triplet separation of \\(\ce{CH2}\\) is reduced to 1 kcal/mol [[40](<../../references.html#id259> "K. Andersson. Theor. Chim. Acta, 91 \(1995\) 31-46.")]. This option is, however, not recommended any longer because it has been replaced by the IPEA Hamiltonian.

The CASPT2 method can be used in any case where a valid reference function can be obtained with the CASSCF method. There is thus no restriction in the number of open shells or the spin coupling of the electrons. Excited states can be treated at the same level as ground states. Actually one of the major successes with the method has been in the calculation of excitation energies. A large number of applications have been performed for conjugated organic molecules. Both Rydberg and valence excited states can be treated and the error in computed excitation energies is normally in the range 0.0–0.2 eV. Similar results have been obtained for ligand field and charge-transfer excitations in transition metal compounds. From Molcas-6 it is possible to use the CASPT2 method in conjunction with the Douglas–Kroll–Hess relativistic Hamiltonian, which has made possible calculations on heavy element compounds such a third row transition metal compounds and actinides with accurate results.

The CASPT2 method can also be used in combination with the FFPT program to compute dynamic correlation contributions to properties with good results in most cases. While numerical gradients are available with the SLAPAF module, analytical first-order derivatives, nuclear energy gradients and derivative coupling vector, are also available [[41](<../../references.html#id364> "Y. Nishimoto. J. Chem. Phys., 154\[19\] \(2021\) 194103."), [42](<../../references.html#id363> "Y. Nishimoto, S. Battaglia, R. Lindh. J. Chem. Theory Comput., 18\[7\] \(2022\) 4269–4281.")]. To use the analytical code, it is not sufficient to simply call ALASKA, but additional keywords (GRDT) in the CASPT2 input block are required to precompute the necessary quantities used in MCLR and ALASKA. All quasi-degenerate variants of CASPT2 support the analytical gradients and NAC, as well as RASPT2. However, many of the features are available only in connection with the RI or CD approximation. Most common scenarios for the use of the analytical gradients code are covered in the tests, and we invite the user to have a look at them if they encounter difficulties in setting up the calculation.

The CASPT2 method is based on second order perturbation theory. To be successful, the perturbation should be small. A correct selection of the active space in the preceding CASSCF calculation is therefore of utmost importance. All near-degeneracy effects leading to configurations with large weights must be included at this stage of the calculation. If this is not done, the first order wave function will contain large coefficients. When this occurs, the CASPT2 program issues a warning. If the energy contribution from such a configuration is large, the results is not to be trusted and a new selection of the active space should be made.

Especially in calculations on excited states, intruder states may occur in the first-order wave function. Warnings are then issued by the program that an energy denominator is small or negative. Such intruder states often arise from Rydberg orbitals, which have not been included in the active space. Even if this sometimes leads to large first order CI coefficients, the contribution to the second order energy is usually very small, since the interaction with the intruding Rydberg state is small. It might then be safe to neglect the warning. A safer procedure is to include the Rydberg orbital into the active space. It can sometimes be deleted from the MO space.

Calculations on compounds with heavy atoms (transition metals, actinides, etc.) may yield many virtual orbitals with low energies. The interaction energies for excitations to states where these orbitals are occupied are often very small and the small denominators can then be removed by a suitable technique (see below). Nevertheless, it is always safer to include such orbitals in the active space.

The intruder states problem is thus a fairly common situation, in which weakly coupled intruders cause spurious singularities, “spikes”, in the potential energy surface. Several ways to mitigate this problem are available in :openmolcas:. First, two level shift techniques, activated with keywords SHIFT and IMAGINARY SHIFT, will introduce a shift in the energy denominators, thus avoiding the singularities. The SHIFT keyword adds a real shift, and the use of this procedure is well tested [[43](<../../references.html#id39> "B. O. Roos, K. Andersson. Chem. Phys. Lett., 245 \(1995\) 215-223."), [44](<../../references.html#id271> "B. O. Roos, K. Andersson, M. P. Fülscher, L. Serrano-Andrés, K. Pierloot, M. Merchán, V. Molina. J. Mol. Struct. Theochem, 388 \(1996\) 257-276.")]. However, it does not remove the singularities, just shifts them [[45](<../../references.html#id361> "S. Battaglia, L. Fransén, I. Fdez. Galván, R. Lindh. J. Chem. Theory Comput., 18\[8\] \(2022\) 4814–4825.")]. The IMAGINARY SHIFT adds an imaginary quantity, and then uses the real component of the resulting second-order energy [[46](<../../references.html#id44> "N. Forsberg, P.-Å. Malmqvist. Chem. Phys. Lett., 274 \(1997\) 196-204.")]. This approach is more robust than the real shift, in particular for weak intruder states. An alternative way to remove intruder states is to use \\(\sigma^p\\) regularization [[45](<../../references.html#id361> "S. Battaglia, L. Fransén, I. Fdez. Galván, R. Lindh. J. Chem. Theory Comput., 18\[8\] \(2022\) 4814–4825.")], which is activated by either keyword SIG1 or keyword SIG2. The advantage of regularization over the level shift techniques is its weaker dependence on the input parameter. All intruder state removal techniques are mutually exclusive.

In some cases, where one can expect strong interaction between different CASSCF wave functions, it is advisable to use the multistate (MS) CASPT2 method [[20](<../../references.html#id45> "J. Finley, P.-Å. Malmqvist, B. O. Roos, L. Serrano-Andrés. Chem. Phys. Lett., 288 \(1998\) 299-306.")], the extended multistate (XMS) method [[47](<../../references.html#id338> "A. A. Granovsky. J. Chem. Phys., 134 \(2011\) 214113."), [48](<../../references.html#id339> "T. Shiozaki, W. Győrffy, P. Celani, H.-J. Werner. J. Chem. Phys., 135 \(2011\) 081106.")], extended dynamically weighted (XDW) CASPT2 [[49](<../../references.html#id342> "S. Battaglia, R. Lindh. J. Chem. Theory Comput., 16\[3\] \(2020\) 1555-1567.")] or rotated multistate (RMS) CASPT2 [[50](<../../references.html#id345> "S. Battaglia, R. Lindh. J. Chem. Phys., 154 \(2021\) 034102.")]. A second-order effective Hamiltonian is constructed for a number of CASSCF wave functions obtained in a state-average calculation. This introduces interaction matrix elements at second order between the different CASSCF states. The effective Hamiltonian is diagonalized to obtain the final second-order energies. The program also produces a file, JOBMIX, with the new effective zeroth order wave functions, which are linear combinations of the original CASSCF states. This method has been used successfully to separate artificially mixed valence and Rydberg states and for transition metal compounds with low lying excited states of the same symmetry as the ground state. In the original multistate method, perturbed wave functions are computed for each of several root functions, separately; these are used to compute the effective Hamiltonian. In the XMS-CASPT2 method, the perturbations are computed with one common zeroth-order Hamiltonian. The XDW-CASPT2 and RMS-CASPT2 methods apply the same initial transformation to the CASSCF states as in XMS-CASPT2, but then proceed with state-specific zeroth-order Hamiltonian operators as in the original MS-CASPT2 method, thus retaining advantages from both approaches.

It is clear from the discussion above that it is not a “black box” procedure to perform CASPT2 calculations on excited states. It is often necessary to iterate the procedure with modifications of the active space and the selection of roots in the CASSCF calculation until a stable result is obtained. Normally, the CASSCF calculations are performed as average calculations over the number of electronic states of interest, or a larger number of states. It is imperative that the result is checked before the CASPT2 calculations are performed. The solutions should contain the interesting states. If all of them are not there, the number of roots in the CASSCF calculation has to be increased. Suppose for example, that four states of a given symmetry are required. Two of them are valence excited states and two are Rydberg states. A CASSCF calculation is performed as an average over four roots. Inspection of the solution shows only one valence excited state, the other three are Rydberg states. After several trials it turns out that the second valence excited state occurs as root number seven in the CASSCF calculation. The reason for such a behavior is, of course, the very different dynamic correlation energies of the valence excited states as compared to the Rydberg states. It is important that the AO basis set is chosen to contain a good representation of the Rydberg orbitals, in order to separate them from the valence excited states. For more details on how to perform calculations on excited states we refer to the literature [[43](<../../references.html#id39> "B. O. Roos, K. Andersson. Chem. Phys. Lett., 245 \(1995\) 215-223."), [44](<../../references.html#id271> "B. O. Roos, K. Andersson, M. P. Fülscher, L. Serrano-Andrés, K. Pierloot, M. Merchán, V. Molina. J. Mol. Struct. Theochem, 388 \(1996\) 257-276.")] and [Section 5.1.5](<../../advanced.examples/ex-ex.html#tut-sec-excited>) of the examples manual.

The first order wave function is obtained in the CASPT2 program as an iterative solution to a large set of linear equations. The size of the equation system is approximately \\(n^2 m^2/2\\) where \\(n\\) is the sum of inactive and active orbitals and \\(m\\) is the sum of active and secondary orbitals. Symmetry will reduce the size with approximately a factor \\(g_{\text{sym}}\\), the number of irreps of the point group.

CASPT2 produces a set of molecular orbitals that can be used as start orbitals for other programs or further calculations. A minimal CASSCF and CASPT2 gives orbitals and occupation numbers which can be used to design a proper larger calculation. By default, the orbitals are natural orbitals obtained from the density matrix of the (normalized) wave function through first order. However, the active/active block of that density matrix is not computed exactly. An approximation has been designed in such a way that the trace is correct, and the natural occupation numbers of active orbitals are between zero and two. Due to the approximation, any properties computed using these orbitals are inexact and can be used only qualitatively. An exact first order density matrix can be computed but this is more time-consuming. It is controlled by the keyword DENSity. Use this keyword to compute properties like dipole moments, etc. The most secure accurate way to do that is, however, to use finite field perturbation theory (FFPT).

## 4.2.3.1. Dependencies¶

The CASPT2 program needs the JOBIPH file from a RASSCF calculation, and in addition one- and two-electron integrals and some auxiliary files from SEWARD.

## 4.2.3.2. Files¶

### 4.2.3.2.1. Input files¶

CASPT2 will use the following input files: ONEINT, ORDINT, RUNFILE, JOBIPH (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

### 4.2.3.2.2. Output files¶

PT2ORB
    

Molecular orbitals.

## 4.2.3.3. Input¶

This section describes the input to the CASPT2 program, starting with its name:
    
    
    &CASPT2
    

### 4.2.3.3.1. Keywords¶

TITLe
    

This keyword is followed by one title line.

MULTistate
    

Perform a single-state CASPT2 (SS-CASPT2) or a multi-state CASPT2 (MS-CASPT2) calculation. Enter the total number of states desired followed by a list of which CASSCF roots to include, for example “`3 1 2 4`” would specify a 3-state calculation including the first, second and fourth root. To perform a single-state calculation, simply enter “`1`” followed by the desired root state, for example “`1 3`” would specify the third root. The special value “`all`” can be used if all the states included in the CASSCF calculation (keyword CIRoot in RASSCF) are desired. This keyword is mutually exclusive with XMULtistate and RMULtistate.

XMULtistate
    

Perform an extended MS-CASPT2 (XMS-CASPT2) calculation according to [[47](<../../references.html#id338> "A. A. Granovsky. J. Chem. Phys., 134 \(2011\) 214113."), [48](<../../references.html#id339> "T. Shiozaki, W. Győrffy, P. Celani, H.-J. Werner. J. Chem. Phys., 135 \(2011\) 081106.")]. This keyword works in the same exact way as MULTistate and is mutually exclusive with MULTistate and RMULtistate.

RMULtistate
    

Perform a rotated MS-CASPT2 (RMS-CASPT2) calculation according to [[50](<../../references.html#id345> "S. Battaglia, R. Lindh. J. Chem. Phys., 154 \(2021\) 034102.")]. In this type of calculation the input CASSCF states are rotated to diagonalize the state-average Fock operator and subsequently used in a conventional MS-CASPT2 calculation. This keyword works in the same exact way as MULTistate and is mutually exclusive with MULTistate and XMULtistate.

DWMS
    

Used in conjunction with XMULtistate and DWTYpe it performs an extended dynamically weighted CASPT2 (XDW-CASPT2) calculation according to [[49](<../../references.html#id342> "S. Battaglia, R. Lindh. J. Chem. Theory Comput., 16\[3\] \(2020\) 1555-1567."), [50](<../../references.html#id345> "S. Battaglia, R. Lindh. J. Chem. Phys., 154 \(2021\) 034102.")], thereby rotating the input CASSCF states to diagonalize the state-average Fock operator and constructing the zeroth-order Hamiltonians using dynamically weighted densities. A non-negative real number for the exponential factor \\(\zeta\\) has to be explicitly specified; reasonable values associated to DWTYpe equal to 1, 2 and 3 are 50, 1e-8 and 1, respectively (see [[49](<../../references.html#id342> "S. Battaglia, R. Lindh. J. Chem. Theory Comput., 16\[3\] \(2020\) 1555-1567."), [50](<../../references.html#id345> "S. Battaglia, R. Lindh. J. Chem. Phys., 154 \(2021\) 034102.")] for more details). It is also possible to use this option with MULTistate instead of XMULtistate, in which case the original CASSCF states are used instead of the rotated ones.

DWTYpe
    

This keyword specifies which exponent is used to compute the weights in a XDW-CASPT2 calculation. Three options are available: DWTYpe equal to 1 uses the squared energy difference between the states according to [[49](<../../references.html#id342> "S. Battaglia, R. Lindh. J. Chem. Theory Comput., 16\[3\] \(2020\) 1555-1567.")]. Note that this option might mix states of different symmetry. DWTYpe equal to 2 uses the square of the state total energy divided by the Hamiltonian coupling between the states, while DWTYpe equal to 3 uses the energy difference divided by the square root of the Hamiltonian coupling (see [[50](<../../references.html#id345> "S. Battaglia, R. Lindh. J. Chem. Phys., 154 \(2021\) 034102.")] for more info). We suggest to use the third option associated with a value of 1 in DWMS.

IPEAshift
    

This shift corrects the energies of the active orbitals and is specified in atomic units. It will be weighted by a function of the diagonal density matrix element \\(D_{pp}\\). This option is used to modify the standard definition of the zeroth order Hamiltonian (\\(H_0\\)), which includes an IPEA shift of 0.25 [[39](<../../references.html#id52> "G. Ghigo, B. O. Roos, P.-Å. Malmqvist. Chem. Phys. Lett., 396 \(2004\) 142-149.")]. The modification of \\(H_0\\) has been introduced (Nov 2005) to reduce the systematic error which leads to a relative overestimation of the correlation energy for open shell system. It also reduces the intruder problems. Default is to use an IPEA shift of 0.25, unless MOLCAS_NEW_DEFAULTS is set to `YES`.

SIG2
    

Apply \\(\sigma^2\\) regularization to the first-order amplitudes, removing potential intruder states. See [[45](<../../references.html#id361> "S. Battaglia, L. Fransén, I. Fdez. Galván, R. Lindh. J. Chem. Theory Comput., 18\[8\] \(2022\) 4814–4825.")]. In addition to the conventionally computed second-order energy value, the energy obtained by Hylleraas’ variational formula is computed. This energy is very close to the unregularized one, except close to singularities due to intruders. This option is an alternative to IMAG, but should be preferred over SIG1 and SHIFT. Mutually exclusive with SIG1, IMAG and SHIFT.

SIG1
    

The same as SIG2, but applies \\(\sigma^1\\) regularization instead. This option should be carefully used in case the small denominators change sign due to conformational changes, see [[45](<../../references.html#id361> "S. Battaglia, L. Fransén, I. Fdez. Galván, R. Lindh. J. Chem. Theory Comput., 18\[8\] \(2022\) 4814–4825.")]. Mutually exclusive with SIG2, IMAG and SHIFT.

IMAGinary
    

Add an imaginary shift to the external part of the zeroth-order Hamiltonian. The correlation energy computed is the real part of the resulting complex perturbation energy. As for the regularizers, also in this case a corrected energy is obtained by Hylleraas’ variational formula. See ref. [[46](<../../references.html#id44> "N. Forsberg, P.-Å. Malmqvist. Chem. Phys. Lett., 274 \(1997\) 196-204.")]. This option is used to eliminate intruder states and is comaprable to SIG2 in both effectiveness and accuracy. Mutually exclusive with SIG2, SIG1 and SHIFT.

SHIFt
    

The original level shift technique, which adds a real shift to the external part of the zeroth-order Hamiltonian to shift away weak intruder states. See refs. [[37](<../../references.html#id293> "B. O. Roos, K. Andersson, M. P. Fülscher, P.-Å. Malmqvist, L. Serrano-Andrés, K. Pierloot, M. Merchán. In New Methods in Computational Quantum Mechanics, volume 93 of Advances in Chemical Physics, pages 213-331. John Wiley & Sons, 1996."), [43](<../../references.html#id39> "B. O. Roos, K. Andersson. Chem. Phys. Lett., 245 \(1995\) 215-223."), [46](<../../references.html#id44> "N. Forsberg, P.-Å. Malmqvist. Chem. Phys. Lett., 274 \(1997\) 196-204.")]. As for all other intruder state removal technqiues, the second-order energy is corrected through Hylleraas’ variational formula. SIG2 or IMAG should be preferred over this one. Mutually exclusive with SIG2, SIG1 and IMAG.

AFREeze
    

This keyword is used to select atoms for defining the correlation orbital space for the CASPT2 calculation. Assume that you have a large molecule where the activity takes place in a limited region (the active site). It could be a metal atom with its surrounding ligands. You can then use this option to reduce the size of the CASPT2 calculation by freezing and deleting orbitals that have only a small population in the active site. An example: The cobalt imido complex \\(\ce{Co^{III}(nacnac)(NPh)}\\) has 43 atoms. The active site was cobalt and the surrounding ligand atoms. Using the AFRE option reduces the time for the CASPT2 calculation from 3 h to 3 min with a loss of accuracy in relative energies for 24 electronic states of less than 0.1 eV. The first line after the keyword contains the number of selected atoms then the selection thresholds (the recommended value is 0.1 or less). An additional line gives the names of the atoms as defined in the Seward input. Here is a sample input for the cobalt complex mentioned above.
    
    
    AFREeze
     6 0.10 0.00
     Co N1 N2 C5 C6 C7
    

This input means that inactive orbitals with less than 0.1 of the density on the active sites will be frozen, while no virtual orbitals will be deleted.

LOVCaspt2
    

“Freeze-and-Delete” type of CASPT2, available only in connection with Cholesky or RI. Needs (pseudo)canonical orbitals from RASSCF. An example of input for the keyword LOVC is the following:
    
    
    LovCASPT2
     0.3
    DoMP2  (or DoEnv)
    

In this case, both occupied and virtual orbitals (localized by the program) are divided in two groups: those mainly located on the region determined (automatically) by the spatial extent of the active orbitals (“active site”), and the remaining ones, which are obviously “outside” this region. The value of the threshold (between 0 and 1) is used to perform this selection (in the example, 30% of the gross Mulliken population of a given orbital on the active site). By default, the CASPT2 calculation is performed only for the correlating orbitals associated with the active site. The keyword DoMP2 is optional and forces the program to perform also an MP2 calculation on the “frozen region”. Alternatively, one can specify the keyword VirAll in order to use all virtual orbitals as correlating space for the occupied orbitals of the active site. A third possibility is to use the keyword DoEnv to compute the energy of the environment as total MP2 energy minus the MP2 energy of the active site.

FNOCaspt2
    

Performs a Frozen Natural Orbital (FNO) CASPT2 calculation, available only in combination with Cholesky or RI integral representation. Needs (pseudo)canonical orbitals from RASSCF. An example of input for the keyword FNOC is the following:
    
    
    FNOCaspt2
     0.4
    RegFNOparameter
     0.01
    DoMP2
    

The keyword FNOC has one compulsory argument: a real number in [-1,1] that if in ]0,1] specifies the fraction of virtual orbitals (in each irrep) to be retained in the FNO-CASPT2 calculation; if in [-1,0[ is to be interpreted as an estimate of the percentage of correlation energy that we are willing to trade for the resulting speedup. If the keyword RegFNO is included, the value specified (a real number) is used to regularize the calculation of the density matrix for the selection of the NOs, particularly useful for wavefunction models based on multiple active spaces. The keyword DoMP2 is also optional and used to compute the (estimated) correction for the truncation error.

FOCKtype
    

Use an alternative Fock matrix. The default Fock matrix is described in [[32](<../../references.html#id200> "K. Andersson, P.-Å. Malmqvist, B. O. Roos, A. Sadlej, K. Wolinski. J. Phys. Chem., 94 \(1990\) 5483-5486."), [33](<../../references.html#id125> "K. Andersson, P.-Å. Malmqvist, B. O. Roos. J. Chem. Phys., 96 \(1992\) 1218-1226.")] and the other original CASPT2 references. The three different modifications named G1, G2 and G3 are described in [[40](<../../references.html#id259> "K. Andersson. Theor. Chim. Acta, 91 \(1995\) 31-46.")]. Note: from 6.4 it is not recommended to use this keyword but stay with the IPEA modified \\(H_0\\), which is default.

FROZen
    

This keyword is used to specify the number of frozen orbitals, i.e. the orbitals that are not correlated in the calculation. The next line contain the number of frozen orbitals per symmetry. The default is to freeze the maximum of those that were frozen in the RASSCF calculation and the deep core orbitals. The frozen orbitals are always the first ones in each symmetry.

DELEted
    

This keyword is used to specify the number of deleted orbitals, i.e. the orbitals that are not used as correlating orbitals in the calculation. The next line contain the number deleted orbitals per symmetry. The default is to delete those that were deleted in the RASSCF calculation. The deleted orbitals are always the last ones in each symmetry.

DENSity
    

Computes the full density matrix from the first order wave function, rather than approximated as is the (faster) default option. Used to compute CASPT2 properties, such as dipole moments, etc.

GRDT
    

Enable the calculation of quantities required by MCLR to obtain the Lagrange multipliers for computing the analytical nuclear gradients.

NAC
    

Enable the calculation of quantities required by MCLR to obtain the Lagrange multipliers for computing the analytical non-adiabatic coupling vector (derivative coupling, to be precise; see NOCSf). This keyword expects two 2 integers specifying the two states for which the NAC vector should be computed.

NOCSf
    

By default, NAC activates the computation of the derivative coupling, including the CSF term. However, using this keyword disables the calculation of the CSF term, resulting in either the non-adiabatic or inter-state coupling vector. It should be noted that employing this keyword slightly modifies the CI contribution, due to the fact that the CI and CSF contributions cannot be fully separated in CASPT2, unlike in CASSCF.

SADRef
    

Use the state-averaged density matrix, as weighted in the reference RASSCF, in the generalized Fock operator for all states. This keyword is mainly intended for development purposes and should not be activated in other situations.

DORT or CORT
    

Use a different (canonical) orthonormalization for internally contracted basis. This keyword is required for analytical nuclear gradients with IPEA shift.

RFPErt
    

This keyword makes the program add reaction field effects to the energy calculation. This is done by adding the reaction field effects to the one-electron Hamiltonian as a constant perturbation, i.e. the reaction field effect is not treated self consistently. The perturbation is extracted from RUNOLD, if that file is not present if defaults to RUNFILE.

RLXRoot
    

Specifies which root to be relaxed in a geometry optimization of a multi-state CASPT2 wave function. Defaults to the highest root or root defined by the same keyword in the RASSCF module.

THREsholds
    

On next line, enter two thresholds: for removal of zero-norm components in the first-order perturbed wave function, and for removal of near linear dependencies in the first-order perturbed wave function. Default values are 1.0d-10 and 1.0d-08 respectively.

MAXIter
    

On next line, enter the maximum allowed number of iterations in a procedure for solving a system of linear equations using a conjugate gradient method. Default is 20. A gradient norm is reported. This gradient is a residual error from the CASPT2 equation solution and should be small, else the number of iterations must be increased.

CONVergence
    

On next line, enter the convergence threshold for the procedure described above. The iterative procedure is repeated until the norm of the residual (RNORM) is less than this convergence threshold. Default is 1.0d-06.

NOMIx
    

Normally, an (X)MS-CASPT2 calculation produces a new jobiph file named JOBMIX. It has the same CASSCF wave functions as the original ones, except that those CI vectors that were used in the (Extended) Multi-State CASPT2 calculation have been mixed, using the eigenvectors of the effective Hamiltonian matrix as transformation coefficients. Keyword NOMIX prevents creation of this JOBMIX file.

NOMUlt
    

This keyword removes the multi-state part of the calculation and only runs a series of independent CASPT2 calculations for the roots specified by the MULTistate or XMULtistate keyword. Useful when many roots are required, but multi-state is not needed, or desired. Note that a JOBMIX file is produced anyway, but the vectors will not be mixed, and the energies will be single-state CASPT2 energies. If used with the XMULtistate keyword, the zeroth-order Hamiltonian will be constructed with the state-average density and therefore will be the same for all the states.

ONLY
    

This keyword requires the MULTistate or XMULtistate keyword, and is followed by an integer specifying one of the roots. In a (Extended) Multistate calculation, it requests to compute the energy of only the specified root. However, the effective Hamiltonian coupling terms between this root and all the others included in the (Extended) Multistate treatment will be computed and printed out. This output will be used in a subsequent calculation, in conjunction with the EFFE keyword.

EFFE
    

This keyword requires the MULTistate or XMULtistate keyword. It is followed by the number of states and a matrix of real numbers, specifying the effective Hamiltonian couplings, as provided in a previous calculation using the ONLY keyword. In a (Extended) Multistate calculation over, e.g., 3 states, 3 separate calculations with the ONLY keyword will be performed, possibly on separate computing nodes, so as to speed up the overall process. The three couplings vectors will be given to the EFFE keyword in matrix form, i.e. the first column is made by the couplings of the first computed root, etc. The program will then quickly compute the (Extended) Multistate energies.

NOORbitals
    

In calculations with very many orbitals, use this keyword to skip the printing of the MO orbitals.

PROPerties
    

Normally, a CASPT2 calculation does not produce any density matrix, natural orbitals or properties in order to save time and memory (especially for large calculations). Keyword PROP activates these calculations, at the expense of (some) extra time and memory (especially if used together with the DENS keyword).

TRANsform
    

This keyword specifies that the wave function should be transformed to use pseudo-canonical orbitals, even if this was specified as option to the CASSCF calculation and should be unnecessary. (Default is: to transform when necessary, and not else.)

OFEMbedding
    

Adds an Orbital-Free Embedding potential to the Hamiltonian. Available only in combination with Cholesky or RI integral representation. No arguments required. The runfile of the environment subsystem (AUXRFIL) must be available.

GHOStdelete
    

Excludes from PT2 treatment orbitals localized on ghost atoms. A threshold for this selection must be specified.

OUTPut
    

Use this keyword, followed by any of the words BRIEF, DEFAULT, or LONG, to control the extent of orbital listing. BRIEF gives a very short orbital listing, DEFAULT a normal output, and LONG a detailed listing.

PRWF
    

This keyword is used to specify the threshold for printing the CI coefficients, default is 0.05.

PRSD
    

This keyword is used to request that not only CSFs are printed with the CI coefficients, but also the determinant expansion.

PRHS
    

This keyword expects an integer (1, 2, or 3) or characters (OLD, NEW, or DIRECT) to specify the parallelization strategy for constructing the right-hand-side (RHS; \\(\langle I | \hat{H} | 0 \rangle\\)) vector in connection with Cholesky or RI. Default is 1 (OLD) for serial calculations and 2 (NEW) for parallel calculations using Global Arrays. `PRHS = 3` or `DIRECT` involves more computation but reduces data communication; this option is expected to be efficient for calculations using a large number of computer nodes. The default value is a sensible choice.

FCIQmc
    

Perform a FCIQMC-CASPT2 calculation with the Molcas–M7 interface. The keyword MCM7 and NDPT should be used in RASSCF, the latter being optional if the calculation should be performed in pseudo-canonical orbitals. The program will skip the calculations of the \\(n\\)-particle reduced density matrix and prompt the user for interaction. Multi-state calculations are currently not supported. Always specify `MULTi = 1 iroot`, where `iroot` is the root index.

NDIAgonal
    

Activate FCIQMC-CASPT2 in non-pseudo-canonical orbitals, i.e. using a non-diagonal Fock operator. As a consequence the entire eight-index 4 RDM needs to be computed and contracted with the Fock matrix on the fly. Despite being computationally more costly, very often the number of walkers required to stabilise the dynamics is considerably smaller than running in pseudo-canonical orbitals. This option is therefore highly recommended.

NormalORDer
    

If CASPT2 intermediates are computed from a time-averaged determinant set, one of the 4RDM excitation operators is pre-contracted with the zeroth order wave function. This keyword enables the transformation back to normal order as required for CASPT2.

CHEMps2
    

Activate DMRG-CASPT2 calculation with Molcas–CheMPS2 interface. The keyword 3RDM must be used in RASSCF. The program will skip the calculations of the \\(n\\)-particle reduced density matrix. Note that multi-state calculations are not supported, the calculation will run but produce wrong CASPT2 total energy. Always specify MULTi = 1 _iroot_ , where _iroot_ is the root index.

CUMUlant
    

Activate DMRG-cu(4)-CASPT2 calculation with Molcas–Block interface. The keyword 3RDM must be used in RASSCF. The program will skip the calculations of the 3-particle reduced density matrix and approximate the 4-particle reduced density matrix.

DMRG
    

Activate DMRG-CASPT2 calculation with Molcas–DMRG interface. The program will calculate the \\(n\\)-particle reduced density matrices from the MPS wave functions. Note that only state-specific calculations are supported at the moment.

CMPS
    

Compress MPS to given bond dimension when evaluation the (t)3-RDM in the QCMaquis CASPT2 routines.

The given default values for the keywords Convergence and Thresholds normally give a second order energy which is correct in eight decimal places.

### 4.2.3.3.2. Input example¶
    
    
    &CASPT2
    Title
     The water molecule
    Density matrix
    

The CASPT2 energy and density matrix is computed for the water molecule with the O(1s) orbital frozen. The standard IPEA-\\(H_0\\) is used.

Input example for SS-DMRG-CASPT2 with Molcas–CheMPS2 interface
    
    
    &RASSCF
    Title    = Water molecule. Ground state
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    DMRG     = 500
    LUMOrb
    3RDM
    
    &CASPT2
    CHEMps2
    

Input example for SA-DMRG-CASPT2 with Molcas–CheMPS2 interface
    
    
    &RASSCF
    Title    = Water molecule. Averaging two states
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    CIROot   = 2 2 1
    DMRG     = 500
    
    &RASSCF
    Title    = Ground state
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    CIROot   = 1 1 ; 1
    DMRG     = 500
    LUMOrb
    CIONly
    3RDM
    
    &CASPT2
    CHEMps2
    MULTistate = 1 1
    
    &RASSCF
    Title    = First excited state
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    CIROot   = 1 2 ; 2
    DMRG     = 500
    CIONly
    3RDM
    
    &CASPT2
    CHEMps2
    MULTistate = 1 2
    

### Table of Contents

  * [1\. Introduction](<../../intro.html>)
  * [2\. Installation Guide](<../../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../../tutorials/tut.html>)
  * [4\. User’s Guide](<../ug.html>)
    * [4.1. The Molcas environment](<../env-main.html>)
    * [4.2. Programs](<../programs.html>)
      * [4.2.1. ALASKA](<alaska.html>)
      * [4.2.2. AVERD](<averd.html>)
      * 4.2.3. CASPT2
      * [4.2.4. CASVB](<casvb.html>)
      * [4.2.5. CCSDT](<ccsdt.html>)
      * [4.2.6. CHCC](<chcc.html>)
      * [4.2.7. CHT3](<cht3.html>)
      * [4.2.8. CMOCORR ¤](<cmocorr.html>)
      * [4.2.9. CPF](<cpf.html>)
      * [4.2.10. DIMERPERT ¤](<dimerpert.html>)
      * [4.2.11. DMRGSCF](<dmrgscf.html>)
      * [4.2.12. DYNAMIX](<dynamix.html>)
      * [4.2.13. EMBQ ¤](<embq.html>)
      * [4.2.14. ESPF (+ QM/MM interface)](<espf.html>)
      * [4.2.15. EXPBAS](<expbas.html>)
      * [4.2.16. EXTF](<extf.html>)
      * [4.2.17. FALCON ¤](<falcon.html>)
      * [4.2.18. FALSE](<false.html>)
      * [4.2.19. FFPT](<ffpt.html>)
      * [4.2.20. GATEWAY](<gateway.html>)
      * [4.2.21. GENANO](<genano.html>)
      * [4.2.22. GEO ¤](<geo.html>)
      * [4.2.23. GRID_IT](<grid_it.html>)
      * [4.2.24. GUESSORB](<guessorb.html>)
      * [4.2.25. GUGA](<guga.html>)
      * [4.2.26. GUGACI](<gugaci.html>)
      * [4.2.27. GUGADRT](<gugadrt.html>)
      * [4.2.28. LEVEL](<level.html>)
      * [4.2.29. LOCALISATION](<localisation.html>)
      * [4.2.30. LOPROP](<loprop.html>)
      * [4.2.31. MBPT2](<mbpt2.html>)
      * [4.2.32. MCKINLEY (a.k.a. DENALI)](<mckinley.html>)
      * [4.2.33. MCLR](<mclr.html>)
      * [4.2.34. MCPDFT](<mcpdft.html>)
      * [4.2.35. MKNEMO ¤](<mknemo.html>)
      * [4.2.36. MOTRA](<motra.html>)
      * [4.2.37. MPPROP](<mpprop.html>)
      * [4.2.38. MPSSI](<mpssi.html>)
      * [4.2.39. MRCI](<mrci.html>)
      * [4.2.40. MULA](<mula.html>)
      * [4.2.41. NEMO ¤](<nemo.html>)
      * [4.2.42. NEVPT2](<nevpt2.html>)
      * [4.2.43. NUMERICAL_GRADIENT](<numerical_gradient.html>)
      * [4.2.44. POLY_ANISO](<poly_aniso.html>)
      * [4.2.45. QMSTAT](<qmstat.html>)
      * [4.2.46. QUATER](<quater.html>)
      * [4.2.47. RASSCF](<rasscf.html>)
      * [4.2.48. RASSI](<rassi.html>)
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
      * [4.2.52. SEWARD](<seward.html>)
      * [4.2.53. SINGLE_ANISO](<single_aniso.html>)
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<averd.html> "4.2.2. AVERD") | [next](<casvb.html> "4.2.4. CASVB") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/caspt2.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

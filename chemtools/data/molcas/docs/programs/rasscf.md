<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rasscf.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.47. RASSCF

[previous](<quater.html> "4.2.46. QUATER") | [next](<rassi.html> "4.2.48. RASSI") | [index](<../../genindex.html> "General Index")

# 4.2.47. RASSCF¶

  * States with a core hole

  * Stochastic-CASSCF method

    * Dependencies

    * Input/Output Files

    * Input keywords

    * Input Example

  * GASSCF method

  * MC-PDFT method

  * Non-Orthogonal Configuration Interaction

    * Dependencies

    * Input/Output Files

    * Input keywords

    * Input Example

  * RASSCF output orbitals

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Optional keywords

    * DMRG keywords

    * HCI-CASSCF keywords

    * Input example




The RASSCF program in Molcas performs multiconfigurational SCF calculations using the Complete Active Space (CAS) [[118](<../../references.html#id11> "B. O. Roos, P. R. Taylor, P. E. M. Siegbahn. Chem. Phys., 48 \(1980\) 157-173."), [119](<../../references.html#id276> "B. O. Roos. In Ab Initio Methods in Quantum Chemistry Part II, volume 69 of Advances in Chemical Physics, pages 399-445. John Wiley & Sons, 1987.")], Restricted Active Space (RAS) [[120](<../../references.html#id191> "P.-Å. Malmqvist, A. Rendell, B. O. Roos. J. Phys. Chem., 94 \(1990\) 5477-5482.")] or the Generalized Active Space (GAS) [[18](<../../references.html#id192> "D. Ma, G. Li Manni, L. Gagliardi. J. Chem. Phys., 135 \(2011\) 044128.")] SCF construction of the wave function.

Within the RASSCF program in Molcas Stochastic-CASSCF calculations can be performed [[1](<../../references.html#id195> "G. Li Manni, S. D. Smart, A. Alavi. J. Chem. Theory Comput., 12 \(2016\) 1245-1258.")]. The Stochastic-CASSCF method allows for CASSCF calculations with a large active space selection (calculations with about 40 electrons and 40 orbitals have been reported) [[1](<../../references.html#id195> "G. Li Manni, S. D. Smart, A. Alavi. J. Chem. Theory Comput., 12 \(2016\) 1245-1258."), [2](<../../references.html#id196> "G. Li Manni, A. Alavi. J. Phys. Chem. A, 122\[22\] \(2018\) 4935–4947."), [3](<../../references.html#id197> "G. Li Manni, D. Kats, D. P. Tew, A. Alavi. J. Chem. Theory Comput., 15 \(2019\) 1492-1497.")].

The RASSCF method is based on a partitioning of the occupied molecular orbitals into the following groups:

  * **Inactive orbitals:** Orbitals that are doubly occupied in all configurations.

  * **Active orbitals:** These orbitals are subdivided into three separate groups:

    * **RAS1 orbitals:** Orbitals that are doubly occupied except for a maximum number of holes allowed in this orbital subspace.

    * **RAS2 orbitals:** In these orbitals all possible occupations are allowed (former Complete Active Space orbitals).

    * **RAS3 orbitals:** Orbitals that are unoccupied except for a maximum number of electrons allowed in this subspace.




CASSCF (or Stochastic-CASSCF) calculations can be performed with the program, by allowing orbitals only in the RAS2 space. For building more general GASSCF wave functions read the dedicated section below.

A single reference SDCI wave function is obtained by allowing a maximum of 2 holes in RAS1 and a maximum of 2 electrons in RAS3, while RAS2 is empty (the extension to SDT- and SDTQ-CI is obvious). Multireference CI wave functions can be constructed by adding orbitals also in RAS2.

The RASSCF program is based on the split GUGA formalism. However, it uses determinant based algorithms to solve the configuration interaction problem [[104](<../../references.html#id113> "J. Olsen, B. O. Roos, P. Jørgensen, H. J. Aa. Jensen. J. Chem. Phys., 89 \(1988\) 2185-2192.")]. To ensure a proper spin function, the transformation to a determinant basis is only performed in the innermost loops of the program to evaluate the \\(\sigma\\)-vectors in the Davidson procedure and to compute the two-body density matrices. The upper limit to the size of the CASSCF wave function that can be handled with the present program is about \\(10^7\\) CSFs and is, in general, limited by the dynamic work array available to the program. The GUGA formalism is available both for CAS, RAS and GAS wave functions.

The orbital optimization in the RASSCF program is performed using the super-CI method. The reader is referred to the references [[120](<../../references.html#id191> "P.-Å. Malmqvist, A. Rendell, B. O. Roos. J. Phys. Chem., 94 \(1990\) 5477-5482."), [121](<../../references.html#id61> "B. O. Roos. Int. J. Quantum Chem., 18-S14 \(1980\) 175-189.")] for more details. A quasi-Newton (QN) update method is used in order to improve convergence. No explicit CI-orbital coupling is used in the present version of the program, except for the coupling introduced in the QN update.

Convergence of the orbital optimization procedure is normally good for CASSCF type wave functions, but problems can occur in calculations on excited states, especially when several states are close in energy. In such applications it is better to optimize the orbitals for the average energy of several electronic states. Further, convergence can be slower in some cases when orbitals in RAS1 and RAS3 are included. The program is not optimal for SDCI calculations with a large number of orbitals in RAS1 and RAS3. Also for the GASSCF variant a slower convergence might occur.

As with other program modules, please observe that the input is preprocessed and may therefore differ in some respects to the input file prepared by the user. In most cases, this does not imply any functional changes as compared to the user’s requests. However, when the input has some minor mistakes or contradictory requests, it can be modified when it is felt that the correction is beneficial. Also, see below for the keyword EXPERT. Without this keyword, the program is assuming more flexibility to optimize the calculation, e.g. by using CIRESTART, if the RASSCF module is called during a numerical differentiation, even if the input requested doing CI calculations from scratch. Using keyword EXPERT, such automatic modification of the user’s input is no longer done, and the input is obeyed exactly (when possible).

It is best to provide a set of good input orbitals. (The program can be started from scratch by using CORE, but this should be used only if other possibilities fail). They can either be from some other type of calculation, for example SCF, or generated by GUESSORB, or from a previous RASSCF calculation on the same system. In the first case the orbitals are normally given in formatted form, file INPORB, in the second case they can also be read from a RASSCF input unit JOBOLD. Input provides both possibilities. Some care has to be taken in choosing the input orbitals, especially for the weakly occupied ones. Different choices may lead to convergence to different local minima. One should therefore make sure that the input orbitals have the correct general structure. A good strategy is often to start using a smaller basis set (MB or DZ) and once the orbitals have been defined, increase the basis set and use EXPBAS to generate input orbitals.

When we speak of files like INPORB or JOBIPH, please note that these can be regarded as generic names. You may have various files with different file names available, and when invocating the RASSCF program, these can be linked or copied (See EMIL command LINK and COPY) so that the program treats them as having the names INPORB or JOBIPH. Alternatively, by the keywords FILEORB and IPHNAME, you can instruct the program to use other file names.

There are two kinds of specifications to make for orbitals: One is the coefficient arrays that describe the molecular orbitals, commonly called “CMO data”. The other kind is the number of inactive, RAS1, etc. orbitals of each symmetry type, which will be called “orbital specifications”. The program can take either or both kinds of data from INPORB, JOBIPH or runfile. The program selects where to fetch such data, based on rules and input keywords. Avoid using conflicting keywords: the program may sometimes go ahead and resolve the problem, or it may decide to stop, not to risk wasting the user’s time on a calculation that was not intended. This decision may be in error.

The orbital specification by keyword input is easy: See keywords FROZEN, INACTIVE, etc. If any such keyword is used, then all the orbital specifications are assumed to be by input, and any such input that is lacking is determined by default rules. These are that there are no such orbitals, with the exception of DELETED: If earlier calculations deleted some orbitals for reason of (near) linear dependence, then these will continue being deleted in subsequent calculations, and cannot be “undeleted”. Another special case occurs if both CHARGE and NACTEL are given in the input and there is no symmetry, then the default value of INACTIVE will be automatically determined.

If no such keyword has been given, but keyword LUMORB is used to instruct the program to fetch CMO data from INPORB, then also the orbitals specs are taken from INPORB, if (as is usually the case) this file contains so-called **typeindex** information. The LUSCUS program may have been used to graphically view orbital pictures and pick out suitable active orbitals, etc., producing a file with extension “.GvOrb”. When this is used as INPORB file, the selected orbitals will be picked in the correct order.

An INPORB file with typeindex can also be used to provide orbital specs while the CMO data are taken from another source (JOBOLD, RUNFILE, …). This is achieved by TYPEINDEX, and you can look in the manual for this keyword to see an explanation of how the typeindex is written. (This is usually done by the program generating the file, but since these are ASCII files, you may find it expedient to look at, or edit, the typeindex).

In case both keywords, such as INACTIVE, **and** LUMORB, is given, this is of course the very common case that CMO data are read from INPORB but orbital specs are given by input. This is perhaps the most common usage. However, when the INPORB file is a produced by GV, it happens frequently that also keyword specs are left in the input, since the user knows that these merely duplicate the specs in INPORB. But the latter may also imply a reordering of the orbitals. For this reason, when the keyword input merely duplicates the number of inactive, etc., that is also specified by typeindex, then the typeindex input overrides, to produce the correct ordering. If they do **not** match precisely, then the CMO data are read, without reordering, and the keyword input (as usual) takes precedence.

The CMO data are obtained as follows: With the following keywords, it is assumed that the user knows what he wants.

  * CORE: (A bad choice, but here for completeness). Creates orbitals from scratch.

  * LUMORB or FILEORB: Try INPORB, or fail.

  * JOBIPH: Try JOBOLD, if not usable, try JOBIPH, or fail.




If none of these keywords were used, then the user accepts defaults, namely

  * look for RASSCF orbitals on RUNFILE

  * look for SCF orbitals on RUNFILE

  * look for GUESSORB orbitals on RUNFILE

  * If still nothing found, create orbitals from scratch.




As for earlier versions, notice the possibility to read the orbitals on JOBIPH, at a later time, by using the keywords OUTOrbital and ORBOnly. This results in editable ASCII files, with names like Project.RasOrb (or Project.RasOrb5 for the fifth root). Such orbitals will be produced by default for the lowest roots — up to the tenth, named now, e.g., Project.RasOrb.5. There is a keyword MAXORB to produce more (or fewer) such files.

The RASSCF program has special input options, which will limit the degrees of freedom used in the orbital rotations. It is possible to prevent specific orbitals from rotating with each other. The keyword is Supsym. This can be used, for example, when the molecule has higher symmetry than one can use with the Molcas system. For example, in a linear molecule the point group to be used is \\(C_{2v}\\) or \\(D_{2h}\\). Both \\(\sigma\\)\- and \\(\delta\\)-orbitals will then appear in irrep 1. If the input orbitals have been prepared to be adapted to linear symmetry, the Supsym input can be used to keep this symmetry through the iterations. The program will do this automatically with the use of the input keyword LINEAR. Similarly, for single atoms, spherical symmetry can be enforced by the keyword ATOM.

## 4.2.47.1. States with a core hole¶

For stable calculation of states with a deep hole, e.g. for X-ray transitions, one needs to compute a (number of) states with a core hole, and a (number of) states without the core hole, with quite different orbitals, and then presumably combine these states in a RASSI calculation. The core-hole state(s) cannot be computed in the same calculation as the full-core states, since they will be very highly excited states compared to the states without that hole. There is also the problem of preventing orbital optimization from filling the core hole. In order to make the calculation in a way that is stable, also across calculations with changing geometry, there is a special input to RASSCF. The option CRPR stands for “core projection”, and is followed by two numbers, e.g. as
    
    
    CRPR
      1   33.0
    

which has the effect of selecting one orbital, in this case orbital nr. 1, from the starting orbitals. This orbital should be in symmetry 1, a non-degenerate orbital, doubly occupied in the state without core hole. A projection operator is constructed from the AO basis set, and in the subsequent CI calculations, in each new iteration of the orbital optimizer, this operator is multiplied with a shift, in this case 33.0 a.u., and added to the Hamiltonian. Regardless of the changing orbitals, this operator is defined by the stable AO basis, and any configuration where the core orbital is doubly occupied is shifted upwards in energy, above the core hole states, and are prevented from playing any role in the calculation. The converged solution(s) are used in a subsequent RASSI, for instance (then with the unperturbed Hamiltonian, of course), where it is combined with states without core hole, for energies and transition properties.

The perturbation on the core hole states by the projection shift is small, provided that the basis set is able to to include the core relaxation effect, and the subsequent RASSI is helpful in correcting for any effects of the perturbation since it will anyway compute eigenstates which are non-interacting and orthogonal by state mixing.

## 4.2.47.2. Stochastic-CASSCF method¶

Warning

This program requires an external package to run.

The Stochastic-CASSCF [[1](<../../references.html#id195> "G. Li Manni, S. D. Smart, A. Alavi. J. Chem. Theory Comput., 12 \(2016\) 1245-1258.")] has been developed since 2015 by Li Manni and Alavi, initially into a locally modified version of Molcas and now available in OpenMolcas. The method retains the simplicity of CASSCF, while circumventing the exponential scaling of CAS wave functions. This is obtained by replacing the Davidson diagonalization technique, in its direct-CI implementation (default in Molcas), with the full-CI quantum Monte-Carlo (FCIQMC) algorithm [[122](<../../references.html#id198> "G. H. Booth, A. J. W. Thom, A. Alavi. J. Chem. Phys., 131 \(2009\) 054106.")], whilst the Super-CI method is used for the orbital optimization.

The method is compatible with density fitting techniques available within OpenMolcas, subsequent MC-PDFT calculations to recover correlation outside the active space and state-averaging across multiple multiplicities.

### 4.2.47.2.1. Dependencies¶

In addition to the normal RASSCF dependencies, the Stochastic-CASSCF requires that the NECI program is installed externally (the NECI program can also be embedded into OpenMolcas. This form of installation is however not fully developed and not suggested). For the NECI program a parallel installation (using MPI) is assumed as well as the use of hdf5 libraries to store and process the walker population. Using the FCIQMC algorithm the user will produce one- and two-body density matrix files (ONERDM, TwoRDM_XXXX), that are required for the subsequent orbital optimization step.

### 4.2.47.2.2. Input/Output Files¶

Two files are produced by the RASSCF module at each MCSCF macro-iteration when the Stochastic-CASSCF method is used (in addition to the one produced by default by RASSCF). These files are:

FCIINP
    

The $Project.FciInp (or FCIINP) file contains input keywords for the NECI code. These keywords need to be adjusted depending on the chemical system under investigation for an optimal FCIQMC dynamics.

FCIDMP
    

The $Project.FciDmp (or FCIDMP), also know as FCIDUMP, contains the MO integrals in the active space. It is a ASCII formatted file. Indices are sorted by symmetry (Irrep).

The Input and the FCIDUMP files are the only files necessary to NECI to run a FCIQMC simulation from scratch. For questions about the FCIQMC dynamics we invite to contact its developers.

As accurate density matrices are necessary for a successful Stochastic-CASSCF calculation, users are required to use the dneci.x and mneci.x binaries (this will run the FCIQMC dynamic in replica mode) [[123](<../../references.html#id199> "C. Overy, G. H. Booth, N. S. Blunt, J. J. Shepherd, D. Cleland, A. Alavi. J. Chem. Phys., 141 \(2014\) 244117.")]. The FCIQMC dymanics can be followed in the fciqmc.out output file or in the NECI generated FCIMCStats file. In the fciqmc.out file there are important pieces of information, such as the list of Slater determinants dominating the FCI wave function and the RDM energy. The latter is passed to OpenMolcas as shown in the script below. When a stationary condition is reached and density matrices sampled these are passed to the RASSCF program to continue. This can be achieved by a simple script, such as the following:
    
    
    cp TwoRDM_aaaa.1 $WorkDir/$Project.TwoRDM_aaaa
    cp TwoRDM_abab.1 $WorkDir/$Project.TwoRDM_abab
    cp TwoRDM_abba.1 $WorkDir/$Project.TwoRDM_abba
    cp TwoRDM_bbbb.1 $WorkDir/$Project.TwoRDM_bbbb
    cp TwoRDM_baba.1 $WorkDir/$Project.TwoRDM_baba
    cp TwoRDM_baab.1 $WorkDir/$Project.TwoRDM_baab
    cp   OneRDM.1    $WorkDir/$Project.OneRDM
    grep 'REDUCED D' fciqmc.out | sed "s/^.*: //" > NEWCYCLE
    mv NEWCYCLE $WorkDir/.
    

When performing state-averaging, the user has to ensure that the ordering of all roots is consistent between OpenMolcas and NECI. For instance, consider a SA-CASSCF on a system admitting 2 doublets, 4 quartets, 3 sextets, 2 octets and 1 dectet. Using the FCIDUMP provided by OpenMolcas (multiplicity in the OpenMolcas input is disabled for these calculations, but should nevertheless be provided), one can complete the NECI dynamics, afterwards OpenMolcas will prompt for 12 consecutively numbered density matrices and energies, i.e.:
    
    
    When finished do:
       cp TwoRDM_* /$YOUR_WORKDIR/
       echo $your_RDM_Energy > /$YOUR_WORKDIR/NEWCYCLE
    

A shell script which also takes care of renaming the RDMs might look like this:
    
    
    export CALCBASE="location of your calculation"
    export YOUR_WORKDIR="location of your scratch"
    
    cd $CALCBASE/neci-doublet/  # contains roots 1 and 2
    grep '*REDUCED' $YOUR_INPUT.out | awk '{ print $9 }' >> $CALCBASE/NEWCYCLE
    
    cd $CALCBASE/neci-quartet/  # contains roots 3 to 6
    grep '*REDUCED' mn3o4.out | awk '{ print $9 }' >> $CALCBASE/NEWCYCLE
    # reverse order required, otherwise redundant rename
    rename .4 .6 *.4; rename .3 .5 *.3 ; rename .2 .4 *.2; rename .1 .3 *.1
    
    [for the other roots same procedure]
    
    cd $CALCBASE
    cp neci-doublet/run$1/TwoRDM_* neci-quartet/run$1/TwoRDM_* [other roots] $YOUR_WORKDIR
    cp NEWCYCLE $YOUR_WORKDIR
    

$Project.TwoRDM_XXXX
    

These files are ASCII NECI generated output files. They contain spin-resolved two-body density matrix elements (and one-RDM) and are necessary to Molcas to continue with the Stochastic-CASSSCF calculation.

### 4.2.47.2.3. Input keywords¶

This section describes the input to the Stochastic-CASSCF method in the OpenMolcas program. Two input keywords are strictly required in the RASSCF module for activating the Stochastic-CASSCF:

NECI
    

This keyword is needed to enable the Stochastic-CASSCF method.

Additional keywords like `totalwalkers` have the same meaning as in NECI and are just passed on.

EMBD
    

This keyword enables the embedded version of the Stochastic-CASSCF where NECI runs as subroutine of Molcas.

Optional important keywords are:

DMPO
    

This keyword is used to generate the FCIDUMP file only. The program will deallocate memory and quit in a clean manner.

DEFD
    

This keyword is used to define an initial Slater determinant as starting guess for the FCIQMC dynamics. During the FCIQMC dynamics, if another Slater determinant is more populated (advanced keywords apply) than the guess determinant provided, the more populated determinant is used as reference, overwriting the user choice. Possible changes of reference determinants can be tracked in the FCIQMC output file. An example of the DEFD keyword follows:
    
    
    DEFD
      1 2 3 4 5 13 14 17 18 21 22 27 28 29 30 31 32 39 40 41 42 43 51 52 53 54 55 56 63 64 65 66
    

It contains a list of the occupied spin-orbitals in the order given by the INPORB file (space symmetry sorted).

GUGA
    

Use spin eigenfunctions instead of Slater determinants in the basis for the FCIQMC dynamics to target specific spin states and perhaps benefit from sparsity in this basis.

REOR
    

The user can input a permutation by specifying the number of non fixed point elements, followed by the order of the non fixed point elements.

If the total number of active orbitals is e.g. 6 the following example of the REOR keyword:
    
    
    REOR
      3
      4 5 1
    

leads to an order of `[4 2 3 5 1 6]`.

If GAS is used one can specify -1 as flag:
    
    
    REOR
      -1
    

to follow the order of GAS spaces. This means that the orbitals are ordered by GAS space first and by symmetry second. First all orbitals of GAS1 and within it orbitals of Irrep 1 come first, Irrep 2 next… Once all orbitals of GAS1 are exhausted we continue with orbitals of GAS2 and so on.

### 4.2.47.2.4. Input Example¶

A minimal input example for using state-averaged Stochastic-CASSCF jointly with RICD MC-PDFT is shown below:
    
    
    &GATEWAY
     RICD
     COORD = coor.xyz
     BASIS = ANO-RCC-VTZP
     GROUP = full
    
    &SEWARD
    
    &RASSCF
     CIROOT = 2  2  1   * follows standard &RASSCF syntax
     NECI = ExNe
     NACTEL = 26 0 0
     INACTIVE = 20 17 17 14 0 0 0 0
     RAS2 = 0 0 0 0 7 6 6 5
     SYMMETRY = 1
    
    >>foreach DFT in (T:PBE, T:BLYP, T:LSDA)
    
       &RASSCF
          FileOrb = $CurrDir/converged.RasOrb
          CIONLY
          FUNC = ROKS; $DFT
          NECI = ExNe
          NACTEL = 26 0 0
          INACTIVE = 20 17 17 14 0 0 0 0
          RAS2 = 0 0 0 0 7 6 6 5
          SYMMETRY = 1
    >>enddo
    

## 4.2.47.3. GASSCF method¶

In certain cases it is useful/necessary to enforce restrictions on electronic excitations within the active space beyond the ones accessible by RASSCF. These restrictions are meant to remove configurations that contribute only marginally to the total wave function. In Molcas this is obtained by the GASSCF approach [[18](<../../references.html#id192> "D. Ma, G. Li Manni, L. Gagliardi. J. Chem. Phys., 135 \(2011\) 044128.")]. GASSCF is a further generalization of the active space concept. This method, like RASSCF, allows restrictions on the active space, but they are more flexible than in RASSCF. These restrictions allow GASSCF to be applied to larger and more complex systems at affordable cost. If the active space is well chosen and the restrictions are not too severe, MCSCF methods recover most of the static correlation energy, and part of the dynamic correlation energy. In the GASSCF method, instead of three active spaces, an in-principle arbitrary number of active spaces (GAS1, GAS2…) may be chosen by the user. Instead of a maximum number of holes in RAS1 and particles in RAS3, accumulated minimum and maximum numbers of electrons are specified for GAS1, GAS1+GAS2, GAS1+GAS2+GAS3, etc. in order to define the desired CI expansion (Figure 4.2.47.1). All intra-space excitations are allowed (Full-CI in subspaces). Constraints are imposed by user choice on inter-space excitations.

Figure 4.2.47.1 Pictorial representation of GAS active space.¶

When and how to use the GAS approach? We consider three examples: (1) an organometallic material with separated metal centers and orbitals not delocalized across the metal centers. One can include the near degenerate orbitals of each center in its own GAS space. This implies that one may choose as many GAS spaces as the number of multiconfigurational centers. (2) Lanthanide or actinide metal compounds where the \\(f\\)-electrons require a MC treatment but they do not participate in bonding neither mix with \\(d\\) orbitals. In this case one can put the \\(f\\) orbitals and their electrons into one or more separated GAS spaces and not allow excitations from and/or to other GAS spaces. (3) Molecules where each bond and its correlating anti-bonding orbital could form a separate GAS space as in GVB approach. Finally, if a wave function with a fixed number of holes in one or more orbitals is desired, without interference of configurations where those orbitals are fully occupied the GAS approach is the method of choice instead of the RAS approach. There is no rigorous scheme to choose a GAS partitioning. The right GAS strategy is system-specific. This makes the method versatile but at the same time it is not a black box method. An input example follows:
    
    
    &RASSCF
    nActEl
     6 0 0
    FROZen
    0 0 0 0 0 0 0 0
    INACTIVE
    2 0 0 0 2 0 0 0
    GASScf
    3
     1 0 0 0 1 0 0 0
    2 2
     0 1 0 0 0 1 0 0
    4 4
     0 0 1 0 0 0 1 0
    6 6
    DELEted
    0 0 0 0 0 0 0 0
    

In this example the entire active space counts six active electrons and six active orbitals. These latter are partitioned in three GAS spaces according to symmetry consideration and in the spirit of the GVB strategy. Each subspace has a fixed number of electrons, _two_ , and no interspace excitations are allowed. This input shows clearly the difference with the RAS approach. Also for the GASSCF variant a slower convergence might occur.

## 4.2.47.4. MC-PDFT method¶

The RASSCF module can be used also for Multiconfiguration Pair-Density Functional Theory (MC-PDFT) calculations, as described in [[105](<../../references.html#id193> "G. Li Manni, R. K. Carlson, S. Luo, D. Ma, J. Olsen, D. G. Truhlar, L. Gagliardi. J. Chem. Theory Comput., 10 \(2014\) 3669-3680."), [106](<../../references.html#id194> "R. K. Carlson, G. Li Manni, A. L. Sonnenberger, D. G. Truhlar, L. Gagliardi. J. Chem. Theory Comput., 11 \(2015\) 82-90.")]. The MC-PDFT method involves two steps: (i) a CASSCF, RASSCF, or GASSCF wave function calculation to obtain the kinetic energy, classical Coulomb energy, total electron density, and on-top pair density; (ii) a post-SCF calculation of the remaining energy using an on-top density functional. In the current implementation, the on-top pair density functional is obtained by “translation” (t) of exchange-correlation functionals. Three translated functionals are currently available: tPBE, tLSDA and tBLYP. As multiconfigurational wave functions are used as input quantities, spin and space symmetry are correctly conserved.

## 4.2.47.5. Non-Orthogonal Configuration Interaction¶

Warning

This program requires an external package to run.

OpenMolcas provides an interface to GronOR [[124](<../../references.html#id362> "T. P. Straatsma, R. Broer, A. Sánchez-Mansilla, C. Sousa, C. de Graaf. J. Chem. Theory Comput., 18 \(2022\) 3549-3565.")], a massively parallel and GPU-accelerated implementation of NOCI and its extension to fragments or ensembles of molecules, NOCI-F.

### 4.2.47.5.1. Dependencies¶

Running NOCI and NOCI-F calculations requires the external installation of the GronOR program: <https://gitlab.com/gronor/gronor>.

### 4.2.47.5.2. Input/Output Files¶

One extra file is generated for each electronic state considered in the generation of the many-electron basis functions of the NOCI.

VECDET.x
    

The $Project.VecDet.x (or VECDET.x) file contains the list of determinants of root \\(x\\). The list contains the CI coefficients and the active orbital occupations.

### 4.2.47.5.3. Input keywords¶

The PRSD keyword must be added to the input to expand the CSFs in Slater determinants, which are written to the $Project.VecDet.x file. It is highly recommended to decrease the threshold for writing CSFs to the output file (PRWF) to at least 1e-5.

### 4.2.47.5.4. Input Example¶

A minimal input example to generate the wave functions that describe the ground state and the first excited singlet state of fragment A.
    
    
    &GATEWAY
    coord = fragA.xyz
    basis = ano-s-vdz
    group = c1
    
    * symmetry is not implemented in GronOR
    
    &RASSCF
    nactel = 6
    inactive = 18
    ras2 = 6
    prwf = 1e-5
    prsd
    
    >>>> COPY $Project.RasOrb.1 $CurrDir/benzeneA_S0.orb
    >>>> COPY $Project.VecDet.1 $CurrDir/benzeneA_S0.det
    
    &RASSCF
    nactel = 6
    inactive = 18
    ras2 = 6
    prwf = 1e-5
    prsd
    ciroot = 1 2; 2
    
    >>>> COPY $Project.RasOrb.2 $CurrDir/benzeneA_S1.orb
    >>>> COPY $Project.VecDet.2 $CurrDir/benzeneA_S1.det
    

## 4.2.47.6. RASSCF output orbitals¶

The RASSCF program produces a binary output file called JOBIPH, which can be used in subsequent calculations. Previously, this was usually a link, pointing to whichever file the user wanted, or by default to the file $Project.JobIph if no such links had been made. This default can be changed, see keyword NewIph and IphName. For simplicity, we refer to this as JOBIPH in the manual. The job interface, JOBIPH, contains four different sets of MO’s and it is important to know the difference between the sets:

  1. **Average orbitals:** These are the orbitals produced in the optimization procedure. Before performing the final CI wave function they are modified as follows: inactive and secondary orbitals are rotated (separately) such as to diagonalize an effective Fock operator, and they are then ordered after increasing energy. The orbitals in the different RAS subspaces are rotated (within each space separately) such that the corresponding block of the state-average density matrix becomes diagonal. These orbitals are therefore called “pseudo-natural orbitals”. They become true natural orbitals only for CAS type wave functions. These orbitals are not ordered. The corresponding “occupation numbers” may therefore appear in the output in arbitrary order. The final CI wave function is computed using these orbitals. They are also the orbitals found in the printed output.

  2. **Natural orbitals:** They differ from the above orbitals, in the active subspace. The entire first order density matrix has been diagonalized. Note that in a RAS calculation, such a rotation does not in general leave the RAS CI space invariant. One set of such orbitals is produced for each of the wave functions in an average RASSCF calculation. The main use of these orbitals is in the calculation of one-electron properties. They are extracted by default (up to ten roots) to the working directory from JOBIPH and named $Project.RasOrb.1, $Project.RasOrb.2, etc. Each set of MO’s is stored together with the corresponding occupation numbers. The natural orbitals are identical to the average orbitals only for a single state CASSCF wave function.

  3. **Canonical orbitals:** This is a special set of MO’s generated for use in the CASPT2 and CCSDT programs. They are obtained by a specific input option to the RASSCF program. They are identical to the above orbitals in the inactive and secondary subspaces. The active orbitals have been obtained by diagonalizing an effective one-electron Hamiltonian, a procedure that leaves the CI space invariant only for CAS type wave functions.

  4. **Spin orbitals:** This set of orbitals is generated by diagonalizing the first order spin density matrix and can be used to compute spin properties.

  5. **Improved virtual orbitals:** This refers only to virtual orbitals, when the IVO keyword is employed in the input. In this case, the virtual orbitals are those which diagonalize the core Hamiltonian. Since the energies of virtual orbitals become thus undefined, the obtained RASORB and JOBIPH files can **not** be used for CASPT2 or MRCI or any correlated calculations. The printed virtual orbitals are quite localized and could be used only to decide which ones should be included in an (enlarged) active space in a subsequest RASSCF calculation.




## 4.2.47.7. Dependencies¶

To start the RASSCF module at least the one-electron and two-electron integrals generated by SEWARD have to be available (exception: See keyword ORBONLY). Moreover, the RASSCF requires a suitable start wave function such as the orbitals from a RHF-SCF calculation or produced by GUESSORB.

For MC-PDFT calculations, it is recommended to use a fine grid via the following input specifications (see the SEWARD, [Section 4.2.52](<seward.html#ug-sec-seward>), for details):
    
    
    &SEWARD
    grid input
    grid=ultrafine
    end of grid input
    

CI coefficients are needed to generate one- and two-body density matrices. They are usually pre-optimized vectors passed to the RASSCF module via EMIL command:
    
    
    >>> COPY $WorkDir/$Project.JobIph JOBOLD
    

A pre-optimized CI vector is not compulsory; however, it is recommended to use a pre-optimized CI vector stored in a JOBIPH file. A set of input orbitals is required. They may be stored in JOBIPH or in a formatted INPORB file.

## 4.2.47.8. Files¶

### 4.2.47.8.1. Input files¶

RASSCF will use the following input files: ONEINT, ORDINT, RUNFILE, INPORB, JOBIPH (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

A number of additional files generated by SEWARD are also used by the RASSCF program. The availability of either of the files named INPORB and JOBOLD is optional and determined by the input options LUMORB and JOBIPH, respectively.

### 4.2.47.8.2. Output files¶

JOBIPH
    

This file is written in binary format and carries the results of the wave function optimization such as MO- and CI-coefficients. If several consecutive RASSCF calculations are made, the file names will be modified by appending “01”,”02”, etc.

RUNFILE
    

The RUNFILE is updated with information from the RASSCF calculation such as the first order density and the Fock matrix.

MD_CAS.x
    

Molden input file for molecular orbital analysis for CI root x.

RASORB
    

This ASCII file contains molecular orbitals, occupation numbers, and orbital indices from a RASSCF calculation. The natural orbitals of individual states in an average-state calculation are also produced, and are named RASORB.1, RASORB.2, etc.

MCDENS
    

This ASCII file is generated for MC-PDFT calculations. It contains spin densities, total density and on-top pair density values on grid (coordinates in a.u.).

## 4.2.47.9. Input¶

This section describes the input to the RASSCF program in the Molcas program system. The input starts with the program name
    
    
    &RASSCF
    

There are no compulsory keywords, but almost any meaningful calculation will require some keyword. At the same time, most choices have default settings, and many are able to take relevant values from earlier calculations, from available orbital files, etc.

To run an MC-PDFT calculation in the RASSCF module, the keywords CIONLY, FUNCTIONAL, ROKS and the functional choice are needed. The currently available functionals are tPBE, tBLYP and tLSDA. Also: LUMORB is needed if external orbitals are used. JOBIPH is needed if external orbital stored in JobIph files are used. CIRESTART is needed if a pre-optimized CI vector stored in JOBIPH is to be used.

### 4.2.47.9.1. Optional keywords¶

There is a large number of optional keywords you can specify. They are used to specify the orbital spaces, the CI wave function etc., but also more arcane technical details that can modify e.g. the convergence or precision. The first 4 characters of the keyword are recognized by the input parser and the rest is ignored. If not otherwise stated the numerical input that follows a keyword is read in free format. A list of these keywords is given below:

TITLe
    

Follows the title for the calculation in a single line

SYMMetry
    

Specify the selected symmetry type (the irrep) of the wave function as a number between 1 and 8 (see SYMMETRY keyword in GATEWAY section). Default is 1, which always denote the totally symmetric irrep.

SPIN
    

The keyword is followed by an integer giving the value of spin multiplicity (\\(2S+1\\)). Default is 1 (singlet).

CHARge
    

Specify the total charge on the system as an integer. If this keyword is used, the NACTEL keyword should not be used, unless the symmetry group is C1 and INACTIVE is not used (in this case the number of inactive orbitals will be computed from the total charge and active electrons). Default value: 0

RASScf
    

Specify two numbers: maximum number of holes in RAS1 and the maximum number of electrons occupying the RAS3 orbitals Default values are: 0,0 See also keyword CHARGE and NACTEL. The specification using RASSCF, and CHARGE if needed, together replace the single keyword NACTEL.

NACTel
    

Requires one or three numbers to follow, specifying

  1. the total number of active electrons (all electrons minus twice the number of inactive and frozen orbitals)

  2. the maximum number of holes in RAS1

  3. the maximum number of electrons occupying the RAS3 orbitals




If only one number is given, the maximum number of holes in RAS1 and of electrons in RAS3 are both set to zero. Default values are: x,0,0, where x is the number needed to get a neutral system. See also keywords CHARGE and RASSCF, which offer an alternative specification.

STAVerage
    

Specifies the number of roots to include in a state-average calculation. This is a simplified for of the CIROot keyword, and `STAVerage = n` is equivalent to `CIROot = n n 1` (see below). Only one of STAVerage and CIROot should be given.

CIROot
    

Specifies the CI root(s) and the dimension of the starting CI matrix used in the CI Davidson procedure. This input makes it possible to perform orbital optimization for the average energy of a number of states. The first line of input gives two or three numbers, specifying the number of roots used in the average calculation (NROOTS), the dimension of the small CI matrix in the Davidson procedure (LROOTS), and possibly a non-zero integer IALL. If IALL\\(\ne\\)1 or there is no IALL, the second line gives the index of the states over which the average is taken (NROOTS numbers, IROOT). **Note** that the size of the CI matrix, LROOTS, must be at least as large as the highest root, IROOT. If, **and only if** , NROOTS\\(>\\)1 a third line follows, specifying the weights of the different states in the average energy. If IALL=1 has been specified, no more lines are read. A state average calculation will be performed over the NROOTS lowest states with equal weights. energy. Examples:
    
    
    CIRoot= 3 5; 2 4 5; 1 1 3
    

The average is taken over three states corresponding to roots 2, 4, and 5 with weights 20%, 20%, and 60%, respectively. The size of the Davidson Hamiltonian is 5. Another example is:
    
    
    CIRoot= 19 19 1
    

A state average calculation will be performed over the 19 lowest states each with the weight 1/19 Default values are NROOTS = LROOTS = IROOT = 1 (ground state), which is the same as the input:
    
    
    CIRoot= 1 1; 1
    

PPT2
    

Prepare stochastic CASPT2 in pseudo-canonical orbitals. This keyword will cause a transformation of the output RasOrb to pseudo-canonical orbitals, equivalent to `OUTO = canonical`.

The performance of FCIQMC depends significantly on the orbital basis. In the pseudo-canonical basis, sampling the contraction of the (diagonal) Fock matrix with the 7-index 4RDM is cheaper than in non-canonical orbitals; however, converging FCIQMC may take a (very) high number of walkers. Sampling in a non-diagonal basis is highly recommended, refer to the NDPT keyword for more information.

NDPT
    

Prepare stochastic CASPT2 in any orbital basis. A fockdump.h5 file will be dumped to the WorkDir which can be used with the same FCIDUMP as used for the last CASSCF iteration to perform stochastic-CASPT2.

The performance of FCIQMC depends to a large degree on the orbital basis and working in a non-pseudo-canonical orbital basis may alleviate the burden to converge the dynamics significantly. This comes at the price of higher computational requirements for the contraction of the full 4RDM with the non-diagonal Fockian. In practice, this expense is compensated for by requiring one to two orders of magnitude fewer walkers compared to canonical orbitals.

MCM7
    

Use the M7 package instead of NECI to perform the CI step in the stochastic-CASSCF interface. Currently no multi-root functionality is implemented.

RGRA
    

Compute the norm of the orbital gradient for each CI root instead of just for the root specified in geometry optimization.

WRMA
    

Dump the 1RDM and (anti)symmetrised 2RDM arrays to disk. These matrices can be used in conjunction with the GUGA-FCIQMC interface to create deterministic reference calculations for state-averaged CASSCF across different spin multiplicities. Works only for one root per spin sector per calculation.

SSCR
    

Computes the orbital resolved spin-correlation function between two ranges of orbitals. For physically meaningful results prior localisation and sorting by atomic sites is required. The latter step is not performed by the LOCALISATION module and must be performed manually on the LocOrb file.

At least one integer is required, specifying the length of the orbital vectors. An optional second integer determines whether the vectors are the same (`1`) or different (any other number or no argument). In the latter case, both orbital vectors must be specified in the next two lines.

Consider a triangle with sites A B C, each having three unpaired electrons, corresponding to a CAS(9,9). Below, two practical examples are given:
    
    
    * Spin correlation from orbital 1 to 6, i.e. local spin S_AB expectation value
    * < S_AB > = S_AB (S_AB + 1)
    SSCR = 6 1
    * or
    SSCR = 6
    1 2 3 4 5 6
    1 2 3 4 5 6
    * Spin correlation between sites A (1-3) and C (7-9), i.e < S_A \cdot S_C >
    SSCR = 3
    1 2 3
    7 8 9
    

Notice that the numbering is consecutive and each entry in an orbital range has to be unique.

CISElect
    

This keyword is used to select CI roots by an overlap criterion. The input consists of three lines per root that is used in the CI diagonalization (3*NROOTS lines in total). The first line gives the number of configurations used in the comparison, \\(n_{\text{Ref}}\\), up to five. The second line gives \\(n_{\text{Ref}}\\) reference configuration indices. The third line gives estimates of CI coefficients for these CSF’s. The program will select the roots which have the largest overlap with this input. Be careful to use a large enough value for LROOTS (see above) to cover the roots of interest.

CRPRoject
    

This keyword is followed by two numbers, which define a Hamiltonian shift by a projection operator times a scalar number. For choosing these numbers, please read the section about core hole states above. The shift acts to raise the energy of any configuration where the selected orbital is doubly occupied so the lie far enough above the target core hole states for the duration of the calculation. The purpose is to obtain RASSCF states that are properly optimized, yet with no risk of collapsing the core hole, for use in subsequent RASSI calculations.

ATOM
    

This keyword is used to get orbitals with pure spherical symmetry for atomic calculations (the radial dependence can vary for different irreps though). It causes super-symmetry to be switched on (see SUPSym keyword) and generates automatically the super-symmetry vector needed. Also, at start and after each iteration, it sets to zero any CMO coefficients with the wrong symmetry. Use this keyword instead of SUPSym for atoms.

LINEar
    

This keyword is used to get orbitals with pure rotational symmetry for linear molecules. It causes super-symmetry to be switched on (see SUPSym keyword) and generates automatically the super-symmetry vector needed. Also, at start and after each iteration, it sets to zero any CMO coefficients with the wrong symmetry. Use this keyword instead of SUPSym for linear molecules.

RLXRoot
    

Specifies which root to be relaxed in a geometry optimization of a state average wave function. Thus, the keyword has to be combined with CIRO or STAV. In a geometry optimization the following input
    
    
    CIRoot= 3 5; 2 4 5; 1 1 3
    RLXRoot= 4
    

will relax CI root number 4.

MDRLxroot
    

Selects a root from a state average wave function for gradient computation in the first step of a molecular dynamics simulation. The root is specified in the same way as in the RLXR keyword. In the following steps the trajectory surface hopping can change the root if transitions between the states occur. This keyword is mutually exclusive with the RLXR keyword.

EXPErt
    

This keyword forces the program to obey the input. Normally, the program can decide to change the input requests, in order to optimize the calculation. Using the EXPERT keyword, such changes are disallowed.

RFPErt
    

This keyword will add a constant reaction field perturbation to the Hamiltonian. The perturbation is read from the RUNOLD (if not present defaults to RUNFILE) and is the latest self-consistent perturbation generated by one of the programs SCF or RASSCF.

NONEquilibrium
    

Makes the slow components of the reaction field of another state present in the reaction field calculation (so-called non-equilibrium solvation). The slow component is always generated and stored on file for equilibrium solvation calculations so that it potentially can be used in subsequent non-equilibrium calculations on other states. If the total charge is greater (i.e., fewer electrons) than that of the reference state, for which the slow component was calculated, PCM is initialized with a fake total charge equal to the reference one, thus allowing to calculate ionized states.

RFROot
    

Enter the index of that particular root in a state-average calculation for which the reaction-field is generated. It is used with the PCM model. With RFROOT = 0, the reaction-field is generated for the state-averaged density (this is unphysical). More flexible state-averaging can be defined using the CIROOT-style input; e.g.
    
    
    RFROOT = 3 3 ; 1 2 3 ; 1 1 0
    

specifies the generation of a reaction-field using the averaged density of \\(S_0\\) and \\(S_1\\) from a three-state averaged MCSCF calculation.

CIRFroot
    

Enter the relative index of one of the roots specified in CISElect for which the reaction-field is generated. Used with the PCM model.

NEWIph
    

The default name of the JOBIPH file will be determined by any already existing such files in the work directory, by appending “01”, “02”, etc. so a new unique name is obtained.

FROZen
    

Specifies the number of frozen orbitals in each symmetry. (see below for condition on input orbitals). Frozen orbitals will not be modified in the calculation. Only doubly occupied orbitals can be left frozen. This input can be used for example for inner shells of heavy atoms to reduce the basis set superposition error. Default is 0 in all symmetries.

INACtive
    

Specify on the next line the number of inactive (doubly occupied) orbitals in each symmetry. Frozen orbitals should not be included here. Default is 0 in all symmetries, but if there is no symmetry (C1) and both CHARGE and NACTEL are given, the number of inactive orbitals will be calculated automatically.

RAS1
    

On the next line specify the number of orbitals in each symmetry for the RAS1 orbital subspace. Default is 0 in all symmetries.

RAS2
    

On the next line specify the number of orbitals in each symmetry for the RAS2 orbital subspace. Default is 0 in all symmetries.

RAS3
    

On the next line specify the number of orbitals in each symmetry for the RAS3 orbital subspace. Default is 0 in all symmetries.

DELEted
    

Specify the number of deleted orbitals in each symmetry. These orbitals will not be allowed to mix into the occupied orbitals. It is always the last orbitals in each symmetry that are deleted. Default is 0 in all symmetries, unless orbitals wer already deleted by previous programs due to near-linear dependence.

GASScf
    

Needed to perform a Generalized Active Space (GASSCF) calculation. It is followed by an integer that defines the number of active subspaces, and two lines for each subspace. The first line gives the number of orbitals in each symmetry, the second gives the minimum and maximum number of electrons in the accumulated active space.

An example of an input that uses this keyword is the following:
    
    
    GASSCF
     5
     2 0 0 0 2 0 0 0
     4 4
     0 2 0 0 0 2 0 0
     8 8
     0 0 2 0 0 0 2 0
     12 12
     0 0 0 1 0 0 0 1
     14 14
     4 2 2 1 4 2 2 1
     20 20
    

In the example above (20in32), excitations from one subspace to another are not allowed since the values of MIN and MAX for GSOC are identical for each of the five subspaces.

FUNCtional
    

Needed to perform MC-PDFT calculations. It must be used together with CIONLY keyword (it is a post-SCF method not compatible with SCF) and ROKS keyword. The functional choice follows. Specify the functional by prefixing `T:` or `FT:` to the standard DFT functionals (see keyword FUNCTIONAL of SCF) An example of an input that uses this keyword follows:
    
    
    &RASSCF
    JOBIPH
    CIRESTART
    CIONLY
    Ras2
    1 0 0 0 1 0 0 0
    FUNCTIONAL
    ROKS; T:PBE
    

In the above example, JOBIPH is used to use orbitals stored in JobIph, CIRESTART is used to use a pre-optimized CI vector, CIONLY is used to avoid conflicts between the standard RASSCF module and the MC-PDFT method (not compatible with SCF so far). The functional chosen is the translated-PBE.

JOBIph
    

Input molecular orbitals are read from an unformatted file with FORTRAN file name JOBOLD. **Note** , the keywords Lumorb, Core, and Jobiph are mutually exclusive. If none is given the program will search for input orbitals on the runfile in the order: RASSCF, SCF, GUESSORB. If none is found, the keyword CORE will be activated.

IPHName
    

Override the default choice of name of the JOBIPH file by giving the file name you want. The name will be truncated to 8 characters and converted to uppercase.

LUMOrb
    

Input molecular orbitals are read from a formatted file with FORTRAN file name INPORB. **Note** , the keywords Lumorb, Core, and Jobiph are mutually exclusive. If none is given the program will search for input orbitals on the runfile in the order: RASSCF, SCF, GUESSORB. If none is found, the keyword CORE will be activated.

FILEorb
    

Override the default name (INPORB) for starting orbital file by giving the file name you want.

CORE
    

Input molecular orbitals are obtained by diagonalizing the core Hamiltonian. This option is only recommended in simple cases. It often leads to divergence. **Note** , the keywords Lumorb, Core, and Jobiph are mutually exclusive.

ALPHaOrBeta
    

With UHF orbitals as input, select alpha or beta as starting orbitals. A positive value selects alpha, a negative value selects beta. Default is 0, which fails with UHF orbitals. This keyword does not affect the spin of the wave function (see the SPIN keyword).

TYPEIndex
    

This keyword forces the program to use information about subspaces from the INPORB file.

User can change the order of orbitals by editing of “Type Index” section in the INPORB file. The legend of the types is:

  * **F** — Frozen

  * **I** — Inactive

  * **1** — RAS1

  * **2** — RAS2

  * **3** — RAS3

  * **S** — Secondary

  * **D** — Deleted



ALTEr
    

This keyword is used to change the ordering of MO in INPORB or JOBOLD. The keyword requires first the number of pairs to be interchanged, followed, for each pair, the symmetry species of the pair and the indices of the two permuting MOs. Here is an example:
    
    
    ALTEr= 2; 1 4 5; 3 6 8
    

In this example, 2 pairs of MO will be exchanged: 4 and 5 in symmetry 1 and 6 and 8 in symmetry 3.

ORTH
    

Specify the orthonormalization scheme to apply on the read orbitals. The possibilities are `Gram_Schmidt`, `Lowdin`, `Canonical`, or `no_ON` (no_orthonormalization). For a detailed explanation see [[125](<../../references.html#id341> "A. Szabo, N. S. Ostlund. Modern Quantum Chemistry. McGraw-Hill, 1989.")] (p. 143). The default is Gram_Schmidt.

CLEAnup
    

This input is used to set to zero specific coefficients of the input orbitals. It is of value, for example, when the actual symmetry is higher than given by input and the trial orbitals are contaminated by lower symmetry mixing. The input requires at least one line per symmetry specifying the number of additional groups of orbitals to clean. For each group of orbitals within the symmetry, three lines follow. The first line indicates the number of considered orbitals and the specific number of the orbital (within the symmetry) in the set of input orbitals. Note the input lines cannot be longer than 72 characters and the program expects as many continuation lines as are needed. The second line indicates the number of coefficients belonging to the prior orbitals which are going to be set to zero and which coefficients. The third line indicates the number of the coefficients of all the complementary orbitals of the symmetry which are going to be set to zero and which are these coefficients. Here is an example of what an input would look like:
    
    
    CLEAnup
    2
       3 4 7 9; 3 10 11 13; 4 12 15 16 17
       2 8 11; 1 15; 0
    0; 0; 0
    

In this example the first entry indicates that two groups of orbitals are specified in the first symmetry. The first item of the following entry indicates that there are three orbitals considered (4, 7, and 9). The first item of the following entry indicates that there are three coefficients of the orbitals 4, 7, and 9 to be set to zero, coefficients 10, 11, and 13. The first item of the following entry indicates that there are four coefficients (12, 15, 16, and 17) which will be zero in all the remaining orbitals of the symmetry. For the second group of the first symmetry orbitals 8 and 11 will have their coefficient 15 set to zero, while nothing will be applied in the remaining orbitals. If a geometry optimization is performed the keyword is inactive after the first structure iteration.

CIREstart
    

Starting CI-coefficients are read from a binary file JOBOLD.

ORBOnly
    

This input keyword is used to get a formated ASCII file (RASORB, RASORB.2, etc.) containing molecular orbitals and occupations reading from a binary JobIph file. The program will not perform any other operation. (In this usage, the program can be run without any files, except the JOBIPH file).

CIONly
    

This keyword is used to disable orbital optimization, that is, the CI roots are computed only for a given set of input orbitals.

CHOInput
    

This marks the start of an input section for modifying the default settings of the Cholesky RASSCF. Below follows a description of the associated options. The options may be given in any order, and they are all optional except for ENDChoinput which marks the end of the CHOInput section.

NoLK
    

Available only within ChoInput. Deactivates the “Local Exchange” (LK) screening algorithm [[126](<../../references.html#id169> "F. Aquilante, T. B. Pedersen, R. Lindh. J. Chem. Phys., 126 \(2007\) 194106\(1–11\).")] in computing the Fock matrix. The loss of speed compared to the default algorithm can be substantial, especially for electron-rich systems. Default is to use LK.

DMPK
    

Available only within ChoInput. Modifies the thresholds used in the LK screening. The keyword takes as argument a (double precision) floating point (non-negative) number used as correction factor for the LK screening thresholds. The default value is 1.0d-1. A smaller value results in a slower but more accurate calculation.

**Note:** The default choice of the LK screening thresholds is tailored to achieve as much as possible an accuracy of the converged RASSCF energies consistent with the choice of the Cholesky decomposition threshold.

NODEcomposition
    

Available only within ChoInput. Deactivates the Cholesky decomposition of the inactive AO 1-particle density matrix. The inactive Exchange contribution to the Fock matrix is therefore computed using inactive canonical orbitals instead of (localized) “Cholesky MOs” [[99](<../../references.html#id164> "F. Aquilante, T. B. Pedersen, A. Sánchez de Merás, H. Koch. J. Chem. Phys., 125 \(2006\) 174101\(1–7\).")]. This choice tends to lower the performances of the LK screening. Default is to perform this decomposition in order to obtain the Cholesky MOs.

TIME
    

Activates printing of the timings of each task of the Fock matrix build. Default is to not show these timings.

MEMFraction
    

Set the fraction of memory to use as global Cholesky vector buffer. Default: for serial runs 0.0d0; for parallel runs 0.3d0.

OFEMbedding
    

Performs a Orbital-Free Embedding (OFE)RASSCF calculation, available only in combination with Cholesky or RI integral representation. The runfile of the environment subsystem renamed AUXRFIL is required. An example of input for the keyword OFEM is the following:
    
    
    OFEMbedding
     ldtf/pbe
    dFMD
     1.0   1.0d2
    FTHAw
     1.0d-4
    

The keyword OFEM requires the specification of two functionals in the form fun1/fun2, where fun1 is the functional used for the Kinetic Energy (available functionals: Thomas-Fermi, with acronym LDTF, and the NDSD functional), and where fun2 is the xc-functional (LDA, LDA5, PBE and BLYP available at the moment). The OPTIONAL keyword dFMD has two arguments: first, the fraction of correlation potential to be added to the OFE potential; second, the exponential decay factor for this correction (used in PES calculations). The OPTIONAL keyword dFMD specifies the fraction of correlation potential to be added to the OFE potential. The OPTIONAL keyword FTHA is used in a freeze-and-thaw cycle (EMIL Do While) to specify the (subsystems) energy convergence threshold.

ITERations
    

Specify the maximum number of RASSCF iterations, and the maximum number of iterations used in the orbital optimization (super-CI) section. Default and maximum values are 200,100.

PERI
    

Write the orbital file per iteration. The obtained files are named ${Project}.IterOrb.${iter_number} and if HDF5 is available ${Project}.rasscf.${iter_number}.h5. Note that up until the last iteration all states in a state-averaged calculation are in the same orbital basis.

LEVShft
    

Define a level shift value for the super-CI Hamiltonian. Typical values are in the range 0.0–1.5. Increase this value if a calculation diverges. The default value 0.5, is normally the best choice when Quasi-Newton is performed.

THRS
    

Specify convergence thresholds for: energy, orbital rotation matrix, and energy gradient. Default values are: 1.0e-08, 1.0e-04, 1.0e-04.

TIGHt
    

Convergence thresholds for the Davidson diagonalization procedure. Two numbers should be given: THREN and THFACT. THREN specifies the energy threshold in the first iteration. THFACT is used to compute the threshold in subsequent iterations as THFACT\\(\cdot\\)DE, where DE is the RASSCF energy change. Default values are 1.0d-04 and 1.0d-3.

NOQUne
    

This input keyword is used to switch off the Quasi-Newton update procedure for the Hessian. Pure super-CI iterations will be performed. (Default setting: QN update is used unless the calculation involves numerically integrated DFT contributions.)

QUNE
    

This input keyword is used to switch on the Quasi-Newton update procedure for the Hessian. (Default setting: QN update is used unless the calculation involves numerically integrated DFT contributions.)

CIMX
    

Specify the maximum number of iterations allowed in the CI procedure. Default is 100 with maximum value 200.

HEXS
    

Highly excited states. Will eliminate the maximum occupation in one or more RAS/GAS’s thereby eliminating all roots below. Very helpful for core excitations where the ground-state input can be used to eliminate unwanted roots. Works with RASSI. First input is the number of RAS/GAS where the maximum occupation should be eliminated. Second is the RAS/GAS or RAS/GAS’s where maximum occupation will not be allowed.

DEXS
    

Doubly highly excited states. Will eliminate the maximum and maximum - 1 occupations in one or more RAS/GAS’s thereby eliminating all roots below. Very helpful for double-core excitations where the ground-state input can be used to eliminate unwanted roots. Works with RASSI. First input is the number of RAS/GAS where the maximum and maximum - 1 occupations should be eliminated. Second is the RAS/GAS or RAS/GAS’s where maximum and maximum - 1 occupations will not be allowed.

SDAV
    

Here follows the dimension of the explicit Hamiltonian used to speed up the Davidson CI iteration process. An explicit H matrix is constructed for those configurations that have the lowest diagonal elements. This H matrix is used instead of the corresponding diagonal elements in the Davidson update vector construction. The result is a large saving in the number if CI iterations needed. Default value is the smallest of 100 and the number of configurations. Increase this value if there is problems converging to the right roots.

NKEE
    

Here follows the maximum dimension of the full Davidson Hamiltonian. This Hamiltonian contains the current CI vectors for each state as well as a set of correction vectors from a number of past iterations. Default value is the smallest of 400 and 6 times the number of states, though at least 2 times the number of states. Increasing this size reduces the number of CI iterations but increases memory requirements and can increase the computational cost associated with forming and diagonalizing the Hamiltonian matrix. Very large values can lead to numerical instabilities.

SXDAmp
    

A variable called SXDAMP regulates the size of the orbital rotations. Use keyword SXDAmp and enter a real number. The default value is 0.0002. Larger values can give slow convergence, very low values may give problems e.g. if some active occupations are very close to 0 or 2.

SUPSym
    

This input is used to restrict possible orbital rotations. It is of value, for example, when the actual symmetry is higher than given by input. Each orbital is given a label IXSYM(I). If two orbitals in the same symmetry have different labels they will not be allowed to rotate into each other and thus prevents from obtaining symmetry broken solutions. Note, however, that the starting orbitals must have the right symmetry. The input requires one or more entries per symmetry. The first specifies the number of additional subgroups in this symmetry (a 0 (zero) denotes that there is no additional subgroups and the value of IXSYM will be 0 (zero) for all orbitals in that symmetry ). If the number of additional subgroups is not zero there are additional entries for each subgroup: The dimension of the subgroup and the list of orbitals in the subgroup counted relative to the first orbital in this symmetry. Note, the input lines cannot be longer than 180 characters and the program expects continuation lines as many as there are needed. As an example assume an atom treated in \\(C_{2v}\\) symmetry for which the d\\(_{z^2}\\) orbitals (7 and 10) in symmetries 1 may mix with the s orbitals. In addition, the d\\(_{z^2}\\) and d\\(_{x^2-y^2}\\) orbitals (8 and 11) may also mix. Then the input would look like:
    
    
    SUPSym
    2
       2 7 10; 2 8 11
    0; 0; 0
    

In this example the first entry indicates that we would like to specify two additional subgroups in the first symmetry (total symmetric group). The first item in the following two entries declares that each subgroup consists of two orbitals. Orbitals 7 and 10 constitute the first group and it is assumed that these are orbitals of d\\(_{z^2}\\) character. The second group includes the d\\(_{x^2-y^2}\\) orbitals 8 and 11. The following three entries denote that there are no further subgroups defined for the remaining symmetries. Ordering of the orbitals according to energy is deactivated when using SUPSym. If you activate ordering using ORDEr, the new labels will be printed in the output section. If a geometry optimization is performed the reordered matrix will be stored in the RUNFILE file and read from there instead of from the input in each new structure iteration.

HOME
    

With this keyword, the root selection in the Super-CI orbital update is by maximum overlap rather than lowest energy.

IVO
    

The RASSCF program will diagonalize the core Hamiltonian in the space of virtual orbitals, before printing them in the output. The resulting orbitals are only suitable to select which ones should enter the active space in a subsequent RASSCF calculation. The RASSCF wave function and orbitals are not suitable for CASPT2, MRCI or any other correlated methods, because the energies of the virtual orbitals are undefined. This keyword is equivalent to the IVO keyword of the SCF program.

VB
    

Using this keyword, the CI optimization step in the RASSCF program will be replaced by a call to the CASVB program, such that fully variational valence bond calculations may be carried out. The VB keyword can be followed by any of the directives described in [Section 4.2.4](<casvb.html#ug-sec-casvb>) and should be terminated by ENDVB. Energy-based optimization of the VB parameters is the default, and the output level for the main CASVB iterations is reduced to \\(-1\\), unless the print level for CASVB print option 6 is \\(\geq\\)2.

PRINt
    

The keyword is followed by a line giving the print levels for various logical code sections. It has the following structure: IW IPR IPRSEC(I), I=1,7

  * IW — logical unit number of printed output (not used).

  * IPR — is the overall print level (normally 2).

  * IPRSEC(I) — gives print levels in different sections of the program.

    1. Input section

    2. Transformation section

    3. CI section

    4. Super-CI section

    5. Not used

    6. Output section

    7. Population analysis section




The meaning of the numbers: 0=Silent, 1=Terse, 2=Normal, 3=Verbose, 4=Debug, and 5=Insane. If input is not given, the default (normally=2) is determined by a global setting which can be altered bubroutine call. (Programmers: See programming guide). The local print level in any section is the maximum of the IPRGLB and IPRSEC() setting, and is automatically reduced e.g. during structure optimizations or numerical differentiation. Example:
    
    
    Print= 6 2 2 2 3 2 2 2 2
    

MAXOrb
    

Maximum number of RasOrb files to produce, one for each root up to the maximum.

OUTOrbitals
    

This keyword is used to select the type of orbitals to be written in a formated ASCII file. By default a formated RASORB file containing average orbitals and subsequent RASORB.1, RASORB.2, etc., files containing natural orbitals for each of the computed (up to ten) roots will be generated in the working directory. An entry follows with an additional keyword selecting the orbital type. The possibilities are:

AVERage orbitals: this is the default option. This keyword is used to produce a formated ASCII file of orbitals (RASORB) which correspond to the final state average density matrix obtained by the RASSCF program. The inactive and secondary orbitals have been transformed to make an effective Fock matrix diagonal. Corresponding diagonal elements are given as orbital energies in the RASSCF output listing. The active orbitals have been obtained by diagonalizing the sub-blocks of the average density matrix corresponding to the three different RAS orbital spaces, thereby the name pseudo-natural orbitals. They will be true natural orbitals only for a CASSCF wave function.

CANOnical orbitals: Use this keyword to produce the canonical orbitals. They differ from the natural orbitals, because also the active part of the Fock matrix is diagonalized. Note that the density matrix is no longer diagonal and the CI coefficients have not been transformed to this basis. This option substitutes the previous keyword CANOnical.

NATUral orbitals: Use this keyword to produce the true natural orbitals. The keyword should be followed by a new line with an integer specifying the maximum CI root for which the orbitals and occupation numbers are needed. One file for each root will be generated up to the specified number. In a one state RASSCF calculation this number is always 1, but if an average calculation has been performed, the NO’s can be obtained for all the states included in the energy averaging. Note that the natural orbitals main use is as input for property calculations using SEWARD. The files will be named RASORB, RASORB.2, RASORB.3, etc. This keyword is on by default for up to ten roots.

SPIN orbitals. This keyword is used to produce a set of spin orbitals and is followed by a new line with an integer specifying the maximum CI root for which the orbitals and occupation numbers are needed. One file for each root will be generated up to the specified number. Note, for convenience the doubly occupied and secondary orbitals have been added to these sets with occupation numbers 0 (zero). The main use of these orbitals is to act as input to property calculations and for graphical presentations. This keyword is on by default for all roots.

An example input follows in which five files are requested containing natural orbitals for roots one to five of a RASSCF calculation. The files are named RASORB.1, RASORB.2, RASORB.3, RASORB.4, and RASORB.5, respectively for each one of the roots. Although this is the default, it can be used complemented by the ORBOnly keyword, and the orbitals will be read from a JobIph file from a previous calculation, in case the formated files were lost or you require more than ten roots. As an option the MAXOrb can be also used to increase the number of files over ten.
    
    
    OUTOrbital= Natural; 15
    

ORBListing
    

This keyword is followed with a word showing how extensive you want the orbital listing to be in the printed output. The alternatives are:

  * **NOTHing:** No orbitals will be printed (suggested for numerical CASPT2 optimization). (Also, the old VERYbrief will be accepted).

  * **FEW:** The program will print the occupied orbitals, and any secondary with less than 0.15 a.u. orbital energy. (Old BRIEF also accepted).

  * **NOCOre:** The program will print the active orbitals, and any secondary with less than 0.15 a.u. orbital energy.

  * **ALL:** All orbitals are printed. (Old LONG also accepted).




(unless other limits are specified by the PROR keyword).

ORBAppear
    

This keyword requires an entry with a word showing the appearance of the orbital listing in the printed output. The alternatives are:

  * **COMPact:** The format of the orbital output is changed from a tabular form to a list giving the orbital indices and MO-coefficients. Coefficients smaller than 0.1 will be omitted.

  * **FULL:** The tabular form will be chosen.



PROR
    

This keyword is used to alter the printout of the MO-coefficients. Two numbers must be given of which the first is an upper boundary for the orbital energies and the second is a lower boundary for the occupation numbers. Orbitals with energy higher than the threshold or occupation numbers lower that the threshold will not be printed. By default these values are set such that all occupied orbitals are printed, and virtual orbitals with energy less than 0.15 au. However, the values are also affected by use of OUTPUT.

PRSD
    

This keyword is used to request that not only CSFs are printed with the CI coefficients, but also the determinant expansion.

ORDEr
    

This input keyword is used to deactivate or activate ordering of the output orbitals according to energy. One number must be given: 1 if you want ordering and 0 if you want to deactivate ordering. Default is 1 and with SUPSym keyword default is 0.

PRSP
    

Use this keyword to get the spin density matrix for the active orbitals printed.

PRWF
    

Enter the threshold for CI coefficients to be printed (Default: 0.05).

TDM
    

If this keyword is given, and if HDF5 support is enabled, the active 1-electron transition density matrix between every pair of states in the current calculation (and transition spin density matrix for non-singlet states) will be computed and stored in the HDF5 file.

XMSInter
    

This keyword can be used in an XMS-PDFT calculation (which needs RASSCF and MCPDFT modules). This keyword stands for XMS Intermediate states. It rotates the CASSCF, CASCI, RASSCF or RASCI states into the XMS intermediate states. This keyword generates a file named Do_Rotate.txt that stores the rotation vector and another file named H0_Rotate.txt that stores the Hamiltonian matrix, called the intermediate Hamiltonian matrix, for the XMS intermediate states. The intermediate Hamiltonian matrix is the XMS-PDFT effective Hamiltonian matrix before one replaces the diagonal elements with the MC-PDFT energies. This keyword currently does not work for wave functions optimized with the DMRG algorithm. This keyword performs the functions called by ROSTate; therefore one does not need to use ROSTate when this keyword is used. More information regarding XMS-PDFT can be found on the Minnesota OpenMolcas page1.

1(1,2,3)
    

<https://comp.chem.umn.edu/openmolcas/>

CMSInter
    

This keyword can be used in a CMS-PDFT calculation (which needs RASSCF and MCPDFT modules). This keyword stands for CMS Intermediate states. It rotates the CASSCF, CASCI, RASSCF or RASCI states into the CMS intermediate states. This keyword generates a file named Do_Rotate.txt that stores the rotation vector and another file named H0_Rotate.txt that stores the Hamiltonian matrix, called intermediate the Hamiltonian matrix, for the CMS intermediate states. The intermediate Hamiltonian matrix is the CMS-PDFT effective Hamiltonian matrix before one replaces the diagonal elements with the MC-PDFT energies. This keyword currently does not work for wave functions optimized with the DMRG algorithm. This keyword performs the functions called by ROSTate; therefore one does not need to use ROSTate when this keyword is used. More information regarding CMS-PDFT can be found on the Minnesota OpenMolcas page1.

CMSStart
    

This keyword gives the file that stores the starting rotation matrix for finding the CMS intermediate states (see CMSInter). The file has the same format as Do_Rotate.txt. The default is to use the XMS intermediate states (see XMSInter).

CMSOpt
    

This keyword defines the maximization algorithm to find the CMS intermediate states (see CMSInter). The allowed values are:

  * **Newton:** Newton’s method. The Hessian and the gradient of the sum-over-states of the active–active classical Coulomb energies (Q_a-a) are computed. This is the default for calculations with more than two states. Note that Q_a-a may decrease within the minimum number of cycles defined by CMMI if a step is too big. After the minimum number of cycles, a smaller step will be taken to ensure that Q_a-a increases, and an extra cycle will always be taken if a smaller step is used.

  * **Jacobi:** Jacobi’s method. States are rotated in pairwise succession, and a trigonometric function is used to fit such rotation to find the maximum. This is the default for calculations with two states.



CMMAx
    

This keyword defines the maximum number of cycles to find the CMS intermediate states (see CMSInter). The default value is 100.

CMMIn
    

This keyword defines the minimum number of cycles to find the CMS intermediate states (see CMSInter). The default value is 5.

CMTHreshold
    

This keyword defines the threshold for the change in the sum over states of the classical Coulomb energy for CMS intermediate states to converge (see CMSInter). The default value is 1.0d-8.

ROSTate
    

This keyword can be used in an MS-PDFT calculation. This keyword stands for ROtate STates, and it rotate the states after the last diagonalization of the CASSCF, CASCI, RASSCF or RASCI calculation. This keyword is only effective when there is a file named Do_Rotate.txt present in the scratch directory; otherwise the states will not be rotated. The file Do_Rotate.txt stores the rotation vector that rotates the states; the rotation vector is stored in a format such that the first line of the file records the first row of the rotation matrix, and so on. This keyword writes a file called H0_Rotate.txt in the scratch directory; H0_Rotate.txt contains the Hamiltonian matrix of the rotated states. This keyword currently does not work for wave functions optimized with the DMRG algorithm. More information regarding XMS-PDFT or CMS-PDFT can be found on the Minnesota OpenMolcas page1.

### 4.2.47.9.2. DMRG keywords¶

Warning

The DMRG keyword has different meanings for QCMaquis, Block and CheMPS2 DMRG interfaces.

DMRG
    

For QCMaquis interface, this keyword is used standalone and activates the DMRG calculation with QCMaquis. In this case, the input should also contain RGINPUT block with parameters controlling the DMRG optimization settings in QCMaquis.

For Block and CheMPS2 interfaces, it should be followed by an integer \\(m\\) Specify maximum number of renormalized states in the DMRG calculation, also known as (virtual) bond dimension \\(m\\) in each microiteration in DMRG calculations. \\(m\\) should be at least 500. This keyword is supported in both CheMPS2 and Block interfaces. Note that DMRG-CASSCF calculations for excited states are not fully supported by the Block interface.

Keywords for the QCMaquis DMRG interface:

Warning

Using DMRG with QCMaquis interface is deprecated. It is advised to use the DMRGSCF module for QCMaquis DMRG calculations.

RGInput
    

This block, terminated by EndRG, is mandatory and contains parameters to QCMaquis which control the DMRG wavefunction optimization. This block is equivalent to the DMRGSettings..EndDMRGSettings block of the DMRGSCF module (see [Section 4.2.11.1.1](<dmrgscf.html#ug-sec-dmrgsettings-input>)).

SOCCupy
    

Initial electronic configuration for the calculated state(s). This keyword is equivalent to the hf_occ card in the **QCMaquis** input (see Table 8 of the QCMaquis manual), but allows input for multiple states. The occupation is inserted as a string (strings) of aliases of occupations of the active (RAS2) orbitals with the aliases `2` = full, `u` = up, `d` = down, `0` = empty. For several states, the occupation strings for each state are separated by newlines.

NEVPT2prep
    

Prepare for a subsequent DMRG-NEVPT2 or CASPT2 calculation. Then the four- and transition three-particle density matrices (4- and t-3RDMs), required for the MRPT2 calculations, will be evaluated and stored on disk in $WorkDir. **QCMaquis** input files for the 4- and t-3RDMs evaluation are prepared and the RDM evaluation may be performed externally or directly in the NEVPT2 program. More about external RDM evaluation in Section 6.3 of the QCMaquis manual.

Keywords for the CheMPS2 DMRG interface:

3RDM
    

Use this keyword to get the 3-particle and Fock matrix contracted with the 4-particle reduced density matrices (3-RDM and F.4-RDM) for DMRG-CASPT2. OUTOrbitals = `CANOnical` is automatically activated. In CheMPS2 interface, both 3-RDM and F.4-RDM are calculated. In Block interface, only 3-RDM is calculated while F.4-RDM is approximated in the CASPT2 module.

CHBLb
    

Specify a threshold for activating restart in CheMPS2. After each macroiteration, if the max BLB value is smaller than CHBLb, activate partial restart in CheMPS2. If the max BLB value is smaller than CHBLb/10.0, activate full restart in CheMPS2. Default value is: 0.5d-2.

DAVTolerance
    

Specify value for Davidson tolerance in CheMPS2. Default value is 1.0d-7.

NOISe
    

Specify value for noise pre-factor in CheMPS2. This noise is set to 0.0 in the last instruction. Default value (recommended) is: 0.05.

MXSWeep
    

Maximum number of sweeps. in the last instruction in CheMPS2. Default value is: 8. In the last iteration of DMRG-SCF, MXSW is increased by five times (default 40).

MXCAnonical
    

Maximum number of sweeps in the last instruction with pseudocanonical orbitals in CheMPS2. Default value is: 40.

CHREstart
    

Use this keyword to activate restart in the first DMRG iteration from a previous calculation. The working directory must contain molcas_natorb_fiedler.txt and CheMPS2_natorb_MPSx.h5 (`x`=0 for the ground state, 1 for the first excited state, etc.). If these files are not in the working directory, a warning is printed at the beginning of the calculation and restart is skipped (start from scratch).

DMREstart
    

Use this keyword to activate restart in the last DMRG iteration from the previous iteration or calculation. This keyword only works when using OUTOrbitals = `CANOnical` or 3RDM. DMREstart = `0` (default): start from scratch to calculate 3-RDM and F.4-RDM.

DMREstart = `1`: start form user-supplied checkpoint files. The working directory must contain molcas_canorb_fiedler.txt and CheMPS2_canorb_MPSx.h5 (`x`=0 for the ground state, 1 for the first excited state, etc.). If these files are not in the working directory, a warning is printed at the beginning of the calculation and restart is skipped (start from scratch).

DMREstart = `2` (Not recommended): start form previous checkpoint files with natural orbitals. DMREstart = `2` is not recommended since this may produce non-optimal energy because the orbital ordering is not optimized.

A general comment concerning the input orbitals: The orbitals are ordered by symmetry. Within each symmetry block the order is assumed to be: frozen, inactive, active, external (secondary), and deleted. Note that if the Spdelete option has been used in a preceding SCF calculation, the deleted orbitals will automatically be placed as the last ones in each symmetry block.

For calculations of a molecule in a reaction field see [Section 4.2.20.1.5](<gateway.html#ug-sec-rfield>) of the present manual and [Section 5.1.6](<../../advanced.examples/ex-rc.html#tut-sec-cavity>) of the examples manual.

### 4.2.47.9.3. HCI-CASSCF keywords¶

Warning

An external package (DICE) is required to run HCI-CASSCF.

DICE
    

Use this keyword to activate Heat-Bath Configuration Interaction (HCI)-CASSCF, calculated with the Dice–Molcas interface. The perturbative component will be calculated deterministically.

EPSIlons
    

Array of two thresholds. \\(\epsilon_1\\): the threshold for adding determinants to the Fock space during the variational calculation; and \\(\epsilon_2\\): the threshold for the second-order perturbative energy correction to the variational energy. \\(\epsilon_2 < \epsilon_1\\). Lower thresholds will give lower HCI energy, but increase the computational cost. Default values are: 1.0d-4, 1.0d-5.

DITErations
    

Maximum number of iterations in the variational step. Default value is: 20.

DIREstart
    

Use this keyword to activate restart in the first HCI iteration from a previous calculation.

DIOCcupy
    

Initial electronic configuration for the HCI-CASSCF calculations. This keyword is required. The keyword requires first the number of configurations \\(n\\), followed by \\(n\\) configuration. Each configuration is inserted as a string of aliases of occupations/couplings of the active (RAS2) orbitals with the aliases `2` = full, `u` = up, `d` = down, `0` = empty.
    
    
    DIOCcupy
    3
    u u 2 0 2 0
    2 0 u u 2 0
    2 u d 0 u u
    

In this CAS(6,6) example, three initial configurations will be read. The first configuration is \\(\ket{\mathord{\uparrow\uparrow}2020}\\).

### 4.2.47.9.4. Input example¶

The following example shows the input to the RASSCF program for a calculation on the water molecule. The calculation is performed in \\(C_{2v}\\) symmetry (symmetries: \\(a_1\\), \\(b_2\\), \\(b_1\\), \\(a_2\\), where the two last species are antisymmetric with respect to the molecular plane). Inactive orbitals are \\(1a_1\\) (oxygen 1s) \\(2a_1\\) (oxygen 2s) and \\(1b_1\\) (the \\(\pi\\) lone-pair orbital). Two bonding and two anti-bonding OH orbitals are active, \\(a_1\\) and \\(b_2\\) symmetries. The calculation is performed for the \\(^1A_1\\) ground state. Note that no information about basis set, geometry, etc. has to be given. Such information is supplied by the SEWARD integral program via the one-electron integral file ONEINT.
    
    
    &RASSCF
    Title= Water molecule. Active orbitals OH and OH* in both symmetries
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    

The following input is an example of how to use the RASSCF program to run MC-PDFT calculations:
    
    
    &RASSCF
    Ras2
    1 0 0 0 1 0 0 0
    
    >>COPY $CurrDir/$Project.JobIph JOBOLD
    
    &RASSCF
    JOBIPH
    CIRESTART
    CIONLY
    Ras2
    1 0 0 0 1 0 0 0
    FUNC
    ROKS; T:PBE
    

The first RASSCF run is a standard CASSCF calculation that leads to variationally optimized orbitals and CI coefficients. The second call to the RASSCF input will use the CI vector and the orbitals previously optimized. The second RASSCF will require the CIONLY keyword as the MC-PDFT is currently not compatible with SCF. FUNCTIONAL ROKS and the functional choice will provide MC-PDFT energies.

More advanced examples can be found in the tutorial section of the manual.

Input example for DMRG-CASSCF with CheMPS2–Molcas interface:
    
    
    &RASSCF
    Title= Water molecule. Active orbitals OH and OH* in both symmetries
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    DMRG     = 500
    3RDM
    

Input example for HCI-CASSCF with Dice–Molcas interface:
    
    
    &RASSCF
    Title= Water molecule. Active orbitals OH and OH* in both symmetries
    Spin     = 1
    Symmetry = 1
    Inactive = 2 0 1 0
    Ras2     = 2 2 0 0
    THRS     = 1.0e-07 1.0e-03 1.0e-03
    DICE
    EPSIlons = 1.0d-4 1.0d-5
    DITErations = 30
    DIOCuppy
    1
    2 0  2 0
    

### Table of Contents

  * [1\. Introduction](<../../intro.html>)
  * [2\. Installation Guide](<../../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../../tutorials/tut.html>)
  * [4\. User’s Guide](<../ug.html>)
    * [4.1. The Molcas environment](<../env-main.html>)
    * [4.2. Programs](<../programs.html>)
      * [4.2.1. ALASKA](<alaska.html>)
      * [4.2.2. AVERD](<averd.html>)
      * [4.2.3. CASPT2](<caspt2.html>)
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
      * 4.2.47. RASSCF
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

[previous](<quater.html> "4.2.46. QUATER") | [next](<rassi.html> "4.2.48. RASSI") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/rasscf.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

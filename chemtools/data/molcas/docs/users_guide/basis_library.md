<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/basis_library.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

4.2.59. The Basis Set Libraries

[previous](<programs/wfa.html> "4.2.58. WFA") | [next](<tools.html> "4.3. GUI") | [index](<../genindex.html> "General Index")

# 4.2.59. The Basis Set Libraries¶

  * Dummy atoms

  * The All Electron Basis Set Library

    * Small ANO basis sets – ANO-S

    * Large ANO basis sets – ANO-L

    * Relativistic ANO basis sets — ANO-RCC

    * Polarized basis sets

    * Structure of the all electron basis set library

  * The ECP Library

    * Core AIMP’s

    * Structure of the ECP libraries




The basis sets library contains both all-electron and effective core potentials. They will be briefly described below and we refer to the publications for more details. The user can also add new basis sets to the basis directory and the structure of the file will therefore be described below.

## 4.2.59.1. Dummy atoms¶

Note that to use dummy atoms the user should employ the basis set label “`X....`”. This will signify centers associated with no charge and no basis functions.

## 4.2.59.2. The All Electron Basis Set Library¶

The basis set library of Molcas contains an extensive set of basis sets both segmented and generally contracted. The files in the basis directory are named in upper case after the basis type label (see below). Three sets of generally contracted basis sets have been especially designed for Molcas. They are based on the Atomic Natural Orbital (ANO) concept and are labeled ANO-X (X=S, L, or RCC). They have been designed to give a balanced description of the atoms in ground, excited, and ionized states. A more detailed description of these basis sets is given below. A fourth basis set, which is especially designed for the calculation of electric properties of molecules (POL) will also be described.

In addition to this, an subset of segmented standard basis sets are included, for example, STO-3G, 3-21G 4-31G, 6-31G, 6-31G*, 6-31G**, cc-pVXZ (X=D,T,Q), and aug-cc-pVXZ (X=D,T). In addition, the library also contains different variants of the Turbomole RI basis sets. For additional all electron basis set we recommend a visit to the EMSL Basis Set Exchange (<https://bse.pnl.gov/bse/portal>). All basis sets are stored in the directory basis_library. The different types of available basis sets can be found in the file basistype.tbl in this directory. Aliases for the names are listed in the file basis.tbl. However, the best way to find out which basis sets are available is to issue the command molcas help basis X where X is the atom. Note that a short hand notation can be used for most basis sets: for example ANO-L-VTZP will give a basis set of valence triple zeta accuracy with polarization functions.

### 4.2.59.2.1. Small ANO basis sets – ANO-S¶

The smallest of the Atomic Natural Orbital (ANO) basis sets are available for the atoms H–Kr. They have been constructed as eigenfunctions of a density matrix averaged over several electronic configurations. The ground state of the atom was included for all atoms, and dependent on the particular atom one or more of the following states were included: valence excited states, ground state for the anion and ground state for the cation. The density matrices were obtained by the SCF, SDCI or MCPF methods for 1 electron, 2 electron and many electron cases respectively. The emphasis have been on obtaining good structural properties such as bond-lengths and -strengths with as small contracted sets as possible. The quality for electric properties such as polarizabilities have been sacrificed for the benefit of the properties mentioned above. See [[91](<../references.html#id260> "K. Pierloot, B. Dumez, P.-O. Widmark, B. O. Roos. Theor. Chim. Acta, 90 \(1995\) 87-114.")] for further discussions. These basis sets are recommended for large molecules where the more extended ANO-L basis sets require to much computational times. One should, however, remember that for a given contraction it is only the time needed to generate the integrals (or Cholesky vectors) that is affected and it is usually preferred to use the more accurate ANO-L (or ANO-RCC) basis sets.

For information about the primitive basis set we refer to the library. The maximum number of ANO’s given in the library is:

  * 4s3p for H–He.

  * 6s4p3d for Li–Be.

  * 7s6p3d for B–Ne.

  * 7s5p3d for Na–Mg.

  * 7s7p4d for Al–Ar.

  * 7s7p4d for K–Ca.

  * 8s7p7d4f for Sc–Zn.

  * 9s9p5d for Ga–Kr.




However, such contractions are unnecessarily large. Almost converged results (compared to the primitive sets) are obtained with the basis sets:

  * 3s2p for H–He.

  * 4s3p2d for Li–Ne.

  * 5s4p3d for Na–Ar.

  * 6s5p4d for K–Ca.

  * 7s5p4d3f for Sc–Zn.

  * 6s5p4d for Ga–Kr.




The results become more approximate below the DZP size:

  * 2s1p for H–He.

  * 3s2p1d for Li–Ne.

  * 4s3p2d for Na–Ar.

  * 5s4p3d for K–Ca.

  * 6s4p3d2f for Sc–Zn.

  * 5s4p3d for Ga–Kr.




### 4.2.59.2.2. Large ANO basis sets – ANO-L¶

The large ANO basis sets for atoms H–Zn, excluding K and Ca, have been constructed by averaging the corresponding density matrix over several atomic states, positive and negative ions and the atom in an external electric field [[88](<../references.html#id249> "P.-O. Widmark, P.-Å. Malmqvist, B. O. Roos. Theor. Chim. Acta, 77 \(1990\) 291."), [89](<../references.html#id251> "P.-O. Widmark, B. J. Persson, B. O. Roos. Theor. Chim. Acta, 79 \(1991\) 419-432."), [90](<../references.html#id258> "R. Pou-Amérigo, M. Merchán, I. Nebot-Gil, P.-O. Widmark, B. O. Roos. Theor. Chim. Acta, 92 \(1995\) 149-181.")]. The different density matrices have been obtained from correlated atomic wave functions. Usually the SDCI method has been used. The exponents of the primitive basis have in some cases been optimized. The contracted basis sets give virtually identical results as the corresponding uncontracted basis sets for the atomic properties, which they have been optimized to reproduce. The design objective has been to describe the ionization potential, the electron affinity, and the polarizability as accurately as possible. The result is a well balanced basis set for molecular calculations.

For information about the primitive basis set we refer to the library. The maximum number of ANO’s given in the library is:

  * 6s4p3d for H.

  * 7s4p3d for He.

  * 7s6p4d3f for Li–Be.

  * 7s7p4d3f for B–Ne.

  * 7s7p5d4f for Na–Ar.

  * 8s7p6d5f4g for Sc–Zn.




However, such contractions are unnecessarily large. Almost converged results (compared to the primitive sets) are obtained with the VQZP basis sets:

  * 3s2p1d for H–He.

  * 5s4d3d2f for Li–Ne.

  * 6s5p4d3f for Na–Ar.

  * 7s6p5d4f3g for Sc–Zn.




The results become more approximate below the size:

  * 3s2p for H–He.

  * 4s3p2d for Li–Ne.

  * 5s4p2d for Na–Ar.

  * 6s5p4d3f for Sc–Zn.




It is recommended to use at least two polarization (3d/4f) functions, since one of them is used for polarization and the second for correlation. If only one 3d/4f-type function is used one has to decide for which purpose and adjust the exponents and the contraction correspondingly. Here both effects are described jointly by the two first 3d/4f-type ANO’s (The same is true for the hydrogen 2p-type ANO’s). For further discussions regarding the use of these basis sets we refer to the literature [[88](<../references.html#id249> "P.-O. Widmark, P.-Å. Malmqvist, B. O. Roos. Theor. Chim. Acta, 77 \(1990\) 291."), [89](<../references.html#id251> "P.-O. Widmark, B. J. Persson, B. O. Roos. Theor. Chim. Acta, 79 \(1991\) 419-432."), [90](<../references.html#id258> "R. Pou-Amérigo, M. Merchán, I. Nebot-Gil, P.-O. Widmark, B. O. Roos. Theor. Chim. Acta, 92 \(1995\) 149-181.")].

### 4.2.59.2.3. Relativistic ANO basis sets — ANO-RCC¶

Extended relativistic ANO-type basis sets are available for the atoms H–Cm. These basis sets have been generated using the same principles as described above for the ANO-L basis sets with the difference that the density matrices have been computed using the CASSCF/CASPT2 method. The basis have been contracted using the Douglas–Kroll Hamiltonian and should therefore only be used in calculations where scalar relativistic effects are included. Seward will automatically recognize this and turn on the DK option when these basis sets are used [[4](<../references.html#id266> "B. O. Roos, V. Veryazov, P.-O. Widmark. Theor. Chem. Acc., 111 \(2004\) 345-351."), [5](<../references.html#id211> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark. J. Phys. Chem. A, 108 \(2004\) 2851-2858."), [6](<../references.html#id212> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark. J. Phys. Chem. A, 109 \(2005\) 6575-6579."), [7](<../references.html#id53> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark. Chem. Phys. Lett., 409 \(2005\) 295-299.")]. The basis sets contain functions for correlation of the semi-core electrons. The new basis sets are called ANO-RCC. More details about the construction and performance is given in the header for each basis set in the ANO-RCC library. Basis sets are available for all atoms up to Cm.

Scalar relativistic effect become important already in the second row of the periodic systems. It is therefore recommended to use these basis sets instead of ANO-L in all calculations.

For information about the primitive basis set we refer to the library. The maximum number of ANOs given in the library is:

  * 6s4p3d1f for H.

  * 7s4p3d2f for He.

  * 8s7p4d2f1g for Li–Be.

  * 8s7p4d3f2g for Be–Ne.

  * 17s12p5d4f for Na.

  * 9s8p5d4f for Mg–Al.

  * 8s7p5d4f2g for Si–Ar.

  * 10s9p5d3f for K.

  * 10s9p6d2f for Ca.

  * 10s10p8d6f4g2h for Sc–Zn.

  * 9s8p6d4f2g for Ga–Kr.

  * 10s10p5d4f for Rb–Sr.

  * 10s9p8d5f3g for In–Xe.

  * 12s10p8d4f for Cs–Ba.

  * 11s10p8d5f3g for La.

  * 12s11p8d7f4g2h for Ce–Lu.

  * 11s10p9d8f4g2h for Hf–Hg.

  * 11s10p9d6f4g for Tl–Rn.

  * 12s11p8d5f for Fr–Ra.

  * 13s11p10d8f6g3h for Ac–Pa.

  * 12s10p9d7f5g3h for U–Cm.




However, such contractions are unnecessarily large. Almost converged results (compared to the primitive sets) are usually obtained with basis sets of QZP quality. You can get a feeling for the convergence from the test results presented in the header of each basis set in the library. One should also remember that larger basis sets are needed for the correlation of semi-core electrons.

Below is a list of the core electrons correlated for each atom.

Li–B | 1s  
---|---  
C–Ne | No core correlation  
Na | 2s,2p  
Mg–Al | 2p  
Si–Ar | No core correlation  
K | 3s,3p  
Ca–Zn | 3p  
Ga–Ge | 3d  
As–Kr | No core correlation  
Rb–Sr | 4p  
In–Xe | 4d  
Cs–Ba | 5p  
La–Lu | 5s,5p  
Hf–Re | 4f,5s,5p  
Os–Hg | 5s,5p  
Tl–Rn | 5d  
Fr–Ra | 6p  
Ac–Cm | 6s,6p  
  
Basis set label in input:

The general label is given as for the other ANO basis sets:

> _Atom.ANO-RCC…contracted set._ (Note the last dot!).

A short hand notation is also possible:

> _Atom.ANO-RCC-label_ , where _label_ is one of MB, VDZ, VDZP, VTZP, or VQZP.

A translation between the two possibilities can be found in file: $MOLCAS/basis_library/basis.tbl

### 4.2.59.2.4. Polarized basis sets¶

The so-called polarized basis sets are purpose oriented, relatively small GTO/CGTO sets devised for the purpose of accurate calculations of dipole electric properties of polyatomic molecules [[237](<../references.html#id9> "A. J. Sadlej. Collect. Czech. Chem. Commun., 53 \(1988\) 1995-2016."), [238](<../references.html#id250> "A. J. Sadlej. Theor. Chim. Acta, 79 \(1991\) 123-140."), [239](<../references.html#id270> "A. J. Sadlej, M. Urban. J. Mol. Struct. Theochem, 234 \(1991\) 147-171."), [240](<../references.html#id252> "A. J. Sadlej. Theor. Chim. Acta, 81 \(1991\) 45-63."), [241](<../references.html#id253> "A. J. Sadlej. Theor. Chim. Acta, 81 \(1992\) 339-354.")]. For each row of the periodic table the performance of the basis sets has been carefully examined in calculations of dipole moments and dipole polarizabilities of simple hydrides at both the SCF and correlated levels of approximation [[237](<../references.html#id9> "A. J. Sadlej. Collect. Czech. Chem. Commun., 53 \(1988\) 1995-2016."), [238](<../references.html#id250> "A. J. Sadlej. Theor. Chim. Acta, 79 \(1991\) 123-140."), [239](<../references.html#id270> "A. J. Sadlej, M. Urban. J. Mol. Struct. Theochem, 234 \(1991\) 147-171."), [240](<../references.html#id252> "A. J. Sadlej. Theor. Chim. Acta, 81 \(1991\) 45-63."), [241](<../references.html#id253> "A. J. Sadlej. Theor. Chim. Acta, 81 \(1992\) 339-354.")]. The corresponding results match within a few percent the best available experimental data. Also the calculated molecular quadrupole moments turn out to be fairly close to those computed with much larger basis sets. According to the present documentation the polarized basis GTO/CGTO sets can be used for safe accurate predictions of molecular dipole moments, dipole polarizabilities, and also molecular quadrupole moments by using high-level correlated computational methods. The use of the polarized basis sets has also been investigated in calculations of weak intermolecular interactions. The interaction energies, corrected for the basis set superposition effect (BSSE), which is rather large for these basis sets, turn out to be close to the best available data. In calculations for molecules involving the 4th row atoms, the property data need to be corrected for the relativistic contribution. The corresponding finite perturbation facility is available [[242](<../references.html#id120> "V. Kellö, A. J. Sadlej. J. Chem. Phys., 93 \(1990\) 8122-8132."), [243](<../references.html#id33> "A. J. Sadlej, M. Urban. Chem. Phys. Lett., 176 \(1991\) 293-302.")].

It is recommended to use these basis sets with the contraction given in the library. It is of course possible to truncate them further, for example by deleting some polarization functions, but this will lead to a deterioration of the computed properties.

### 4.2.59.2.5. Structure of the all electron basis set library¶

The start of a given basis set in the library is given by the line
    
    
    /label
    

where “label” is the basis set label, as defined below in the input description to SEWARD. Then follows two lines with the appropriate literature reference for that basis set. These cards are read by SEWARD and must thus be included in the library, and may not be blank. Next is a set of comment lines, which begin with an asterisk in column 1, giving some details of the basis sets. A number of lines follow, which specifies the basis set:

  1. Charge of the atom and the highest angular momentum. For each angular momentum (l) then follows.

  2. Number of primitives and contracted functions for angular momentum l (must be identical to those given in the basis set label).

  3. Exponents of the primitive functions.

  4. The contraction matrix (with one CGTO per column). Note that all basis sets are given in the generally contracted format, even if they happen to be segmented. Note that the number of CGTOs must correspond to the data given in the label.




The following is an example of an entry in a basis set library.
    
    
    * This is the Huzinaga 5s,2p set contracted to 3s,2p        -- Comment
    * according to the Dunning paper.                           -- Comment
    /H.TZ2P.Dunning.5s2p.3s2p.                                  -- Label
    Exponents  : S. Huzinaga, J. Chem. Phys., 42, 1293(1965).   -- First ref line
    Coefficients: T. H. Dunning, J. Chem. Phys., 55, 716(1971). -- Second ref line
     1.0 1                                                      -- Charge, sp
     5 3                                                        -- 5s->3s
     52.56 7.903 1.792 0.502 0.158                              -- s-exponents
     0.025374  0.0  0.0                                       -- contr. matrix
     0.189684  0.0  0.0                                       -- contr. matrix
     0.852933  0.0  0.0                                       -- contr. matrix
     0.0     1.0  0.0                                         -- contr. matrix
     0.0     0.0  1.0                                         -- contr. matrix
     2  2                                                       -- 2p->2p
     1.5  0.5                                                   -- p-exponents
     1.0 0.0                                                  -- contr. matrix
     0.0 1.0                                                  -- contr. matrix
    

## 4.2.59.3. The ECP Library¶

Molcas is able to perform _effective core potential_ (ECP) calculations and _embedded cluster_ calculations. In ECP calculations, only the _valence_ electrons of a molecule are explicitly handled in a quantum mechanical calculation, at a time that the _core_ electrons are kept frozen and are represented by ECP’s. (An example of this is a calculation on \\(\ce{HAt}\\) in which only the 5d, 6s and 6p electrons of Astatine and the one of Hydrogen are explicitly considered.) Similarly, in _embedded cluster_ calculations, only the electrons assigned to a piece of the whole system (the _cluster_) are explicitly handled in the quantum mechanical calculation, under the assumption that they are the only ones relevant for some local properties under study; the rest of the whole system (the _environment_) is kept frozen and represented by embedding potentials which act onto the _cluster_. (As an example, calculations on a \\(\ce{TlF12^{11-}}\\) cluster embedded in a frozen lattice of \\(\ce{KMgF3}\\) can be sufficient to calculate spectroscopical properties of \\(\ce{Tl^+}\\)-doped \\(\ce{KMgF3}\\) which are due to the \\(\ce{Tl+}\\) impurity.)

In order to be able to perform ECP calculations in molecules, as well as _embedded cluster_ calculations in ionic solids, with the Ab Initio Model Potential method (AIMP) [[244](<../references.html#id110> "S. Huzinaga, L. Seijo, Z. Barandiarán, M. Klobukowski. J. Chem. Phys., 86 \(1987\) 2132-2145."), [245](<../references.html#id115> "Z. Barandiarán, L. Seijo. J. Chem. Phys., 89 \(1988\) 5739-5746."), [246](<../references.html#id119> "Z. Barandiarán, L. Seijo, S. Huzinaga. J. Chem. Phys., 93 \(1990\) 5843-5850."), [247](<../references.html#id15> "C. Wittborn, U. Wahlgren. Chem. Phys., 201 \(1995\) 357-362."), [248](<../references.html#id146> "F. Rakowitz, C. M. Marian, L. Seijo, U. Wahlgren. J. Chem. Phys., 110 \(1999\) 3678-3686."), [249](<../references.html#id145> "F. Rakowitz, C. M. Marian, L. Seijo. J. Chem. Phys., 111 \(1999\) 10436-10443.")] Molcas is provided with the library ECP which includes nonrelativistic and relativistic _core_ ab initio model potentials and _embedding_ ab initio model potentials representing both complete-cations and complete-anions in ionic lattices [[245](<../references.html#id115> "Z. Barandiarán, L. Seijo. J. Chem. Phys., 89 \(1988\) 5739-5746."), [250](<../references.html#id283> "Z. Barandiarán, L. Seijo. In Computational Chemistry: Structure, Interactions and Reactivity, volume 77B of Studies in Physical and Theoretical Chemistry, pages 435-461. Elsevier, 1992.")].

Before we continue we should comment a little bit on the terminology used here. Strictly speaking, ECP methods are all that use the frozen-core approximation. Among them, we can distinguish two families: the “pseudopotential” methods and the “model potential” methods. The pseudopotential methods are ultimately based on the Phillips–Kleinman equation [[251](<../references.html#id233> "J. C. Phillips, L. Kleinman. Phys. Rev., 116 \(1959\) 287-294.")] and handle valence nodeless pseudo orbitals. The model potential methods are based on the Huzinaga equation [[252](<../references.html#id94> "S. Huzinaga, A. A. Cantu. J. Chem. Phys., 55 \(1971\) 5543-5549."), [253](<../references.html#id6> "S. Huzinaga, D. McWilliams, A. A. Cantu. Adv. Quantum Chem., 7 \(1973\) 187-220.")] and handle node-showing valence orbitals; the AIMP method belongs to this family. Here, when we use the general term ECP we will be referring to the more particular of AIMP. According to its characteristics, the AIMP method can be also applied to represent frozen-ions in ionic lattices in embedded cluster calculations; in this case, we will not be very strict in the nomenclature and we will also call ECP’s to the frozen-ion (embedding) _ab initio_ model potentials.

The effective potentials in the libraries include the effects of the atomic core wave functions (embedding ion wave functions) through the following operators:

  * a local representation of the core (ion) Coulomb operator,

  * a non-local spectral representation of the core (ion) exchange operator,

  * a core (ion) projection operator,

  * a spectral representation of the relativistic mass-velocity and Darwin operators corresponding to the valence orbitals, if the Cowan–Griffin-based scalar relativistic CG-AIMP method [[246](<../references.html#id119> "Z. Barandiarán, L. Seijo, S. Huzinaga. J. Chem. Phys., 93 \(1990\) 5843-5850.")] is used.

  * a spectral representation of the relativistic no-pair Douglas–Kroll operators, if the scalar relativistic no-pair Douglas–Kroll NP-AIMP method [[247](<../references.html#id15> "C. Wittborn, U. Wahlgren. Chem. Phys., 201 \(1995\) 357-362."), [248](<../references.html#id146> "F. Rakowitz, C. M. Marian, L. Seijo, U. Wahlgren. J. Chem. Phys., 110 \(1999\) 3678-3686."), [249](<../references.html#id145> "F. Rakowitz, C. M. Marian, L. Seijo. J. Chem. Phys., 111 \(1999\) 10436-10443.")] is used.




Given the quality and non-parametric nature of the operators listed above, the flexibility of the basis sets to be used with the AIMP’s is crucial, as in any _ab initio_ method.

The valence basis sets included in the libraries have been obtained by energy minimization in atomic valence-electron calculations, following standard optimization procedures. All the experience gathered in the design of molecular basis sets starting from all-electron atomic basis sets, and in particular from segmented minimal ones, is directly applicable to the AIMP valence basis sets included in the libraries. They are, for non-relativistic and relativistic Cowan–Griffin AIMPs, minimal basis sets with added functions, such as polarization and diffuse functions; in consequence, the minimal sets should be split in molecular calculations in order to get reasonable sets (a splitting pattern is recommended in the library for every set); the splitting can be done by means of “the basis set label”. For the relativistic no-pair Douglas–Kroll AIMPs contracted valence basis sets are given directly in a form which is recommended in molecular calculations, i.e. they are of triple zeta quality in the outer shells and contain polarization functions. In both cases these _valence_ basis sets contain very _inner_ primitive GTF’s: They are necessary since, typical to a model potential method, the valence orbitals will show correct nodal structure. Finally, it must be noted that the core AIMP’s can be safely mixed together with all-electron basis sets.

In AIMP _embedded cluster calculations_ , the cluster basis set, which must be decided upon by the user, should be designed following high quality standard procedures. Very rigid cluster basis sets should not be used. In particular, the presence of the necessary embedding projection operators, which prevent the cluster densities from collapsing onto the crystal lattice, demands flexible cluster bases, including, eventually, components outside the cluster volume [[254](<../references.html#id133> "J. L. Pascual, L. Seijo, Z. Barandiarán. J. Chem. Phys., 98 \(1993\) 9715-9724.")]. The use of flexible cluster basis sets is then a necessary requirement to avoid artificial frontier effects, not ascribable to the AIMP embedding potentials. This requirement is unavoidable, anyway, if good correlated wave functions are to be calculated for the cluster. Finally, one must remember that the AIMP method does exclude any correlation between the cluster electronic group and the embedding crystal components; in other words, only intra-cluster correlation effects can be accounted for in AIMP embedded cluster calculations. Therefore the cluster-environment partition and the choice of the cluster wave function must be done accordingly. In particular, the use of one-atom clusters is not recommended.

Core- and embedding-AIMP’s can be combined in a natural way in valence-electron, embedded cluster calculations. They can be used with any of the different types of wave functions that can be calculated with Molcas.

### 4.2.59.3.1. Core AIMP’s¶

The list of core potentials and valence basis sets available in the AIMP library follows. Although AIMP’s exist in the literature for different core sizes, this library includes only those recommended by the authors after numerical experimentation. Relativistic CG-AIMP’s and NP-AIMP’s, respectively, and nonrelativistic NR-AIMP’s are included. Each entry of the CG-AIMP’s and the NR-AIMP’s in the list is accompanied with a recommended contraction pattern (to be used in the fifth field). The NP-AIMP basis sets are given explicitly in the recommended contraction pattern. For the third-row transition metals two NP-AIMP basis sets are provided which differ in the number of primitive and contracted f GTFs. For further details, please refer to the literature [[249](<../references.html#id145> "F. Rakowitz, C. M. Marian, L. Seijo. J. Chem. Phys., 111 \(1999\) 10436-10443.")]. For more information about a particular entry consult the AIMP library.

The ECP libraries have also been extended to include the so-called nodeless ECPs or pseudo potentials based on the Phillips–Kleinman equation [[251](<../references.html#id233> "J. C. Phillips, L. Kleinman. Phys. Rev., 116 \(1959\) 287-294.")]. These are included both as explicit and implicit operators. Following the work by M. Pelissier and co-workers [[255](<../references.html#id87> "M. Pelissier, N. Komiha, J.-P. Daudey. J. Comput. Chem., 9 \(1988\) 298-302.")] the operators of nodeless ECPs can implicitly be fully expressed via spectral representation of operators. In the list of nodeless ECPs the Hay and Wadt’s family of ECPs (LANL2DZ ECPs) [[256](<../references.html#id103> "P. J. Hay, W. R. Wadt. J. Chem. Phys., 82 \(1985\) 270-283."), [257](<../references.html#id104> "P. J. Hay, W. R. Wadt. J. Chem. Phys., 82 \(1985\) 284-298."), [258](<../references.html#id105> "P. J. Hay, W. R. Wadt. J. Chem. Phys., 82 \(1985\) 299-310.")] has been included in addition to the popular set of the so-called Stoll and Dolg ECPs [[259](<../references.html#id26> "P. Fuentealba, H. Preuss, H. Stoll, L. Von Szentpály. Chem. Phys. Lett., 89 \(1982\) 418-422."), [260](<../references.html#id186> "P. Fuentealba, L. von Szentpály, H. Preuss, H. Stoll. J. Phys. B: At. Mol. Phys., 18 \(1985\) 1287-1296."), [261](<../references.html#id219> "G. Igel-Mann, H. Stoll, H. Preuss. Mol. Phys., 65 \(1988\) 1321-1328."), [262](<../references.html#id222> "A. Bergner, M. Dolg, W. Küchle, H. Stoll, H. Preuß. Mol. Phys., 80 \(1993\) 1431-1441."), [263](<../references.html#id185> "P. Fuentealba, H. Stoll, L. von Szentpály, P. Schwerdtfeger, H. Preuss. J. Phys. B: At. Mol. Phys., 16 \(1983\) L323-L328."), [264](<../references.html#id121> "M. Kaupp, P. v. R. Schleyer, H. Stoll, H. Preuss. J. Chem. Phys., 94 \(1991\) 1360-1366."), [265](<../references.html#id112> "M. Dolg, U. Wedig, H. Stoll, H. Preuss. J. Chem. Phys., 86 \(1987\) 866-872."), [266](<../references.html#id274> "U. Wedig, M. Dolg, H. Stoll, H. Preuss. In Quantum Chemistry: The Challenge of Transition Metals and Coordination Chemistry, volume 176 of NATO ASI Series, pages 79-89. D. Reidel, 1986."), [267](<../references.html#id27> "L. Von Szentpály, P. Fuentealba, H. Preuss, H. Stoll. Chem. Phys. Lett., 93 \(1982\) 555-559."), [268](<../references.html#id248> "D. Andrae, U. Häußermann, M. Dolg, H. Stoll, H. Preuß. Theor. Chim. Acta, 77 \(1990\) 123-141."), [269](<../references.html#id102> "H. Stoll, P. Fuentealba, P. Schwerdtfeger, J. Flad, L. v. Szentpály, H. Preuss. J. Chem. Phys., 81 \(1984\) 2732-2736."), [270](<../references.html#id220> "W. Küchle, M. Dolg, H. Stoll, H. Preuss. Mol. Phys., 74 \(1991\) 1245-1263."), [271](<../references.html#id277> "G. Igel-Mann. PhD thesis, Universität Stuttgart, Institut für Theoretische Chemie, 1987."), [272](<../references.html#id255> "M. Dolg, H. Stoll, H. Preuss. Theor. Chim. Acta, 85 \(1993\) 441-450."), [273](<../references.html#id116> "M. Dolg, H. Stoll, H. Preuss. J. Chem. Phys., 90 \(1989\) 1730-1734."), [274](<../references.html#id122> "M. Dolg, P. Fulde, W. Küchle, C.-S. Neumann, H. Stoll. J. Chem. Phys., 94 \(1991\) 3011-3017."), [275](<../references.html#id247> "M. Dolg, H. Stoll, A. Savin, H. Preuss. Theor. Chim. Acta, 75 \(1989\) 173-194."), [276](<../references.html#id124> "M. Dolg, H. Stoll, H.-J. Flad, H. Preuss. J. Chem. Phys., 97 \(1992\) 1162-1173."), [277](<../references.html#id203> "M. Dolg, H. Stoll, H. Preuss, R. M. Pitzer. J. Phys. Chem., 97 \(1993\) 5852-5859."), [278](<../references.html#id117> "P. Schwerdtfeger, M. Dolg, W. H. E. Schwarz, G. A. Bowmaker, P. D. W. Boyd. J. Chem. Phys., 91 \(1989\) 1762-1774."), [279](<../references.html#id221> "U. Häussermann, M. Dolg, H. Stoll, H. Preuss, P. Schwerdtfeger, R. M. Pitzer. Mol. Phys., 78 \(1993\) 1211-1224."), [280](<../references.html#id136> "W. Küchle, M. Dolg, H. Stoll, H. Preuss. J. Chem. Phys., 100 \(1994\) 7535-7542."), [281](<../references.html#id139> "A. Nicklass, M. Dolg, H. Stoll, H. Preuss. J. Chem. Phys., 102 \(1995\) 8942-8952."), [282](<../references.html#id140> "T. Leininger, A. Nicklass, H. Stoll, M. Dolg, P. Schwerdtfeger. J. Chem. Phys., 105 \(1996\) 1052-1059."), [283](<../references.html#id151> "X. Cao, M. Dolg, H. Stoll. J. Chem. Phys., 118 \(2003\) 487-496.")]. Both of them in either the explicit form labeled as LANL2DZ and STUTTGART, or in the implicit form labeled as LANL2DZ_NL and STUTTGART_NL. The latter include the recently developed ANO-basis sets for actinides [[283](<../references.html#id151> "X. Cao, M. Dolg, H. Stoll. J. Chem. Phys., 118 \(2003\) 487-496.")].

### 4.2.59.3.2. Structure of the ECP libraries¶

The start of a given basis set and AIMP is identified by the line
    
    
    /label
    

where “label” is defined below, in the input description to SEWARD. Then, comment lines, effective charge, and basis set follow, with the same structure that the all-electron Basis Set Library (see items 1. to 4. in Section 4.2.59.2.5.) Next, the AIMP/ECP/PP is specified as follows:

  5. The pseudo potential approach [[284](<../references.html#id20> "L. R. Kahn, W. A. Goddard, III. Chem. Phys. Lett., 2 \(1968\) 667-670."), [285](<../references.html#id97> "P. A. Christiansen, Y. S. Lee, K. S. Pitzer. J. Chem. Phys., 71 \(1979\) 4445-4450."), [286](<../references.html#id246> "P. Durand, J.-C. Barthelat. Theor. Chim. Acta, 38 \(1975\) 283-302.")], see eqs. (3) and (4) in Ref. [[287](<../references.html#id46> "C.-K. Skylaris, L. Gagliardi, N. C. Handy, A. G. Ioannou, S. Spencer, A. Willetts, A. M. Simper. Chem. Phys. Lett., 296 \(1998\) 445-451.")], with the following lines:

     1. The keyword PP On the same line follows the atomic symbol of the element, the number of core electrons (\\(N_c\\)) and \\(L\\), where \\(L-1\\) is the largest angular momentum orbital belonging to the core. This line is followed by \\(L+1\\) identical sections. The first of these sections is the so-called \\(L\\) potential and the subsequent sections corresponds to the S-\\(L\\), P-\\(L\\), D-\\(L\\), etc. potentials. Each sections start with a line specifying the number of Gaussian terms in the potential. This line is then followed by a single line for each Gaussian specifying the powers (\\(n_{kl}\\)), the Gaussian exponent (\\(\zeta_{kl}\\)), and the associated coefficient (\\(d_{kl}\\)).

Note that the pseudo potential input is mutually exclusive to the M1, M2, COREREP, and PROJOP keywords!

  6. The Coulomb local model potential, eq.(6) in Ref. [[244](<../references.html#id110> "S. Huzinaga, L. Seijo, Z. Barandiarán, M. Klobukowski. J. Chem. Phys., 86 \(1987\) 2132-2145.")] with the following lines:

     1. The keyword M1, which identifies the terms with \\(n_k=0\\).

     2. The number of terms. If greater than 0, lines 3 and 4 are read.

     3. The exponents \\(\alpha_k\\).

     4. The coefficients \\(A_k\\). (divided by the negative of the effective charge).

     5. The keyword M2, which identifies the terms with \\(n_k=1\\).

     6. The number of terms. If greater than 0, lines 7 and 8 are read.

     7. The exponents \\(\alpha_k\\).

     8. The coefficients \\(A_k\\). (divided by the negative of the effective charge).

  7. A line with the keyword COREREP followed by another one with a real constant. This is not used now but it is reserved for future use.

  8. The projection operator, eq.(3) in Ref. [[244](<../references.html#id110> "S. Huzinaga, L. Seijo, Z. Barandiarán, M. Klobukowski. J. Chem. Phys., 86 \(1987\) 2132-2145.")] with the following lines:

     1. The keyword PROJOP.

     2. The maximum angular momentum (\\(l\\)) of the frozen core (embedding) orbitals. Lines 3 to 6 are repeated for each angular momentum \\(l\\).

     3. The number of primitives and the number of orbitals (more properly, degenerate sets of orbitals or \\(l\\)-shells) for angular momentum \\(l\\). As an option, these two integers can be followed by the occupation numbers of the \\(l\\)-shells; default values are 2 for \\(l=0\\), 6 for \\(l=1\\), etc.

     4. The projection constants, \\(-2\varepsilon_c\\).

     5. The exponents of the primitive functions.

     6. The coefficients of the orbitals, one per column, using general contraction format.

  9. The spectral representation operator, eq.(7) in Ref. [[244](<../references.html#id110> "S. Huzinaga, L. Seijo, Z. Barandiarán, M. Klobukowski. J. Chem. Phys., 86 \(1987\) 2132-2145.")] for NR-AIMP, eq.(3) in Ref. [[246](<../references.html#id119> "Z. Barandiarán, L. Seijo, S. Huzinaga. J. Chem. Phys., 93 \(1990\) 5843-5850.")] for relativistic CG-AIMP, and eqs.(1) and (7) in Ref. [[249](<../references.html#id145> "F. Rakowitz, C. M. Marian, L. Seijo. J. Chem. Phys., 111 \(1999\) 10436-10443.")] for relativistic NP-AIMP, with the following lines:

     1. The keyword Spectral Representation Operator.

     2. One of the keywords Valence, Core, or External. Valence indicates that the set of primitive functions specified in the basis set data will be used for the spectral representation operator; this is the standard for ab initio _core_ model potentials. Core means that the set of primitives specified in the PROJOP section will be used instead; this is the standard for complete-ion ab initio _embedding_ model potentials. External means that a set of primitives specific for the spectral representation operator will be provided in the next lines. In this case the format is one line in which an integer number specifies the highest angular momentum of the external basis sets; then, for each angular momentum the input is formated as for lines 2, 3, and 4 in Section 4.2.59.2.5.

     3. The keyword Exchange.

     4. For relativistic AIMPs one of the keywords NoPair or 1stOrder Relativistic Correction. NoPair indicates that scalar relativistic no-pair Douglas-Kroll AIMP integrals are to be calculated. 1stOrder Relativistic Correction means that Cowan-Griffin-based scalar relativistic AIMP, CG-AIMP’s, are used. In the latter case, in the next line a _keyword_ follows which, in the library QRPLIB, identifies the starting of the numerical mass-velocity plus Darwin potentials (eq.(2) in Ref. [[246](<../references.html#id119> "Z. Barandiarán, L. Seijo, S. Huzinaga. J. Chem. Phys., 93 \(1990\) 5843-5850.")]). (In QRPLIB a line with “ _keyword_ mv&dw potentials start” must exist, followed by the number of points in the radial logarithmic grid, the values of the radial coordinate r, and, for each valence orbital, its label (2S, 4P, etc.), and the values of the mass-velocity plus Darwin potentials at the corresponding values of r; these data must end up with a line “ _keyword_ mv&dw potentials end”.)

     5. The keyword End of Spectral Representation Operator.




Below is an example of an entry in the AIMP library.
    
    
    /S.CG-AIMP.Barandiaran.7s6p1d.1s1p1d.ECP.6el.       -- label (note that 6th field is ECP)
    Z.Barandiaran and L.Seijo, Can.J.Chem. 70(1992)409. -- 1st ref. line
    core[Ne] val[3s,3p]  (61/411/1*)=2s3p1d recommended -- 2nd ref. line
    *SQR-SP(7/6/1)                 (61/411/1)           -- comment line
      6.000000         2                                -- eff. charge & highest ang.mom.
                                                        -- blank line
        7    1                                          -- 7s -> 1s
       1421.989530                                      -- s-exponent
       211.0266560                                      -- s-exponent
       46.72165060                                      -- s-exponent
       4.310564040                                      -- s-exponent
       1.966475840                                      -- s-exponent
       .4015383790                                      -- s-exponent
       .1453058790                                      -- s-exponent
       .004499703540                                    -- contr. coeff.
       .030157124800                                    -- contr. coeff.
       .089332590700                                    -- contr. coeff.
      -.288438151000                                    -- contr. coeff.
      -.279252515000                                    -- contr. coeff.
       .700286615000                                    -- contr. coeff.
       .482409523000                                    -- contr. coeff.
        6    1                                          -- 6p -> 1p
       78.08932440                                      -- p-exponent
       17.68304310                                      -- p-exponent
       4.966340810                                      -- p-exponent
       .5611646780                                      -- p-exponent
       .2130782690                                      -- p-exponent
       .8172415400E-01                                  -- p-exponent
      -.015853278200                                    -- contr. coeff.
      -.084808963800                                    -- contr. coeff.
      -.172934245000                                    -- contr. coeff.
       .420961662000                                    -- contr. coeff.
       .506647309000                                    -- contr. coeff.
       .200082121000                                    -- contr. coeff.
        1    1                                          -- 1d -> 1d
       .4210000000                                      -- d-exponent
      1.000000000000                                    -- contr. coeff.
    *                                                   -- comment line
    * Core AIMP: SQR-2P                                 -- comment line
    *                                                   -- comment line
    * Local Potential Parameters : (ECP convention)    -- comment line
    *                            A(AIMP)=-Zeff*A(ECP)   -- comment line
    M1                                                  -- M1 operator
        9                                               -- number of M1 terms
       237485.0100                                      -- M1 exponent
       24909.63500                                      -- M1 exponent
       4519.833100                                      -- M1 exponent
       1082.854700                                      -- M1 exponent
       310.5610000                                      -- M1 exponent
       96.91851000                                      -- M1 exponent
       26.63059000                                      -- M1 exponent
       9.762505000                                      -- M1 exponent
       4.014487500                                      -- M1 exponent
                                                        -- blank line
       .019335998333                                    -- M1 coeff.
       .031229360000                                    -- M1 coeff.
       .061638463333                                    -- M1 coeff.
       .114969451667                                    -- M1 coeff.
       .190198283333                                    -- M1 coeff.
       .211928633333                                    -- M1 coeff.
       .336340950000                                    -- M1 coeff.
       .538432350000                                    -- M1 coeff.
       .162593178333                                    -- M1 coeff.
    M2                                                  -- M2 operator
        0                                               -- number of M2 terms
    COREREP                                             -- CoreRep operator
       1.0                                              -- CoreRep constant
    PROJOP                                              -- Projection operator
        1                                               -- highest ang. mom.
        8    2                                          -- 8s -> 2s
      184.666320      18.1126960                        -- 1s,2s proj. op. constants
       3459.000000                                      -- s-exponent
       620.3000000                                      -- s-exponent
       171.4000000                                      -- s-exponent
       58.53000000                                      -- s-exponent
       22.44000000                                      -- s-exponent
       6.553000000                                      -- s-exponent
       2.777000000                                      -- s-exponent
       1.155000000                                      -- s-exponent
       .018538249000   .005054826900                    -- contr. coeffs.
       .094569248000   .028197248000                    -- contr. coeffs.
       .283859290000   .088959130000                    -- contr. coeffs.
       .454711270000   .199724180000                    -- contr. coeffs.
       .279041370000   .158375340000                    -- contr. coeffs.
       .025985763000  -.381198090000                    -- contr. coeffs.
      -.005481472900  -.621887210000                    -- contr. coeffs.
       .001288714400  -.151789890000                    -- contr. coeffs.
        7    1                                          -- 7p -> 1p
      13.3703160                                        -- 2p proj. op. constant
       274.0000000                                      -- p-exponent
       70.57000000                                      -- p-exponent
       24.74000000                                      -- p-exponent
       9.995000000                                      -- p-exponent
       4.330000000                                      -- p-exponent
       1.946000000                                      -- p-exponent
       .8179000000                                      -- p-exponent
       .008300916100                                    -- cont. coeff.
       .048924254000                                    -- cont. coeff.
       .162411660000                                    -- cont. coeff.
       .327163550000                                    -- cont. coeff.
       .398615170000                                    -- cont. coeff.
       .232548200000                                    -- cont. coeff.
       .034091088000                                    -- cont. coeff.
    *                                                   -- comment line
    Spectral Representation Operator                    -- SR operator
    Valence primitive basis                             -- SR basis specification
    Exchange                                            -- Exchange operator
    1stOrder Relativistic Correction                    -- mass-vel + Darwin oper.
    SQR-2P                                              -- label in QRPLIB
    End of Spectral Representation Operator             -- end of SR operator
    

Below is an example of an entry in the STUTTGART file for a pseudo potential.
    
    
    /Hg.Stuttgart.Kuchle.4s4p1d.2s2p1d.ECP.2el.         -- label (note the 6th field is ECP)
    W. Kuechle, M. Dolg, H. Stoll, H. Preuss, Mol. Phys.-- ref. line 1
    74, 1245 (1991); J. Chem. Phys. 94, 3011 (1991).    -- ref. line 2
        2.00000    2                                    -- eff. charge & highest ang.mom.
    *s functions                                        -- comment line
      4  2                                              -- 4s -> 2s
      0.13548420E+01                                    -- s-exponent
      0.82889200E+00                                    -- s-exponent
      0.13393200E+00                                    -- s-exponent
      0.51017000E-01                                    -- s-exponent
      0.23649400E+00  0.00000000E+00                    -- contr. coeff.
     -0.59962800E+00  0.00000000E+00                    -- contr. coeff.
      0.84630500E+00  0.00000000E+00                    -- contr. coeff.
      0.00000000E+00  0.10000000E+01                    -- contr. coeff.
    *p functions                                        -- comment line
      4  2                                              -- 4p -> 2p
      0.10001460E+01                                    -- p-exponent
      0.86645300E+00                                    -- p-exponent
      0.11820600E+00                                    -- p-exponent
      0.35155000E-01                                    -- p-exponent
      0.14495400E+00  0.00000000E+00                    -- contr. coeff.
     -0.20497100E+00  0.00000000E+00                    -- contr. coeff.
      0.49030100E+00  0.00000000E+00                    -- contr. coeff.
      0.00000000E+00  0.10000000E+01                    -- contr. coeff.
    *d functions                                        -- comment line
      1  1                                              -- 1d -> 1d
      0.19000000E+00                                    -- d-exponent
      0.10000000E+01                                    -- contr. coeff.
    *                                                   -- comment line
    PP,Hg,78,5;                                         -- PP operator, label, # of core elec., L
    1; ! H POTENTIAL                                    -- # number of exponents in the H potential
    2, 1.00000000,.000000000;                           -- power, exponent and coeff.
    3; ! S-H POTENTIAL                                  -- # number of exponents in the S-H potential
    2,0.227210000,-.69617800;                           -- power, exponent and coeff.
    2, 1.65753000,27.7581050;                           -- power, exponent and coeff.
    2, 10.0002480,48.7804750;                           -- power, exponent and coeff.
    2; ! P-H POTENTIAL                                  -- # number of exponents in the P-H potential
    2,0.398377000,-2.7358110;                           -- power, exponent and coeff.
    2,0.647307000,8.57563700;                           -- power, exponent and coeff.
    2; ! D-H POTENTIAL                                  -- # number of exponents in the D-H potential
    2,0.217999000,-.01311800;                           -- power, exponent and coeff.
    2,0.386058000,2.79286200;                           -- power, exponent and coeff.
    1; ! F-H POTENTIAL                                  -- # number of exponents in the F-H potential
    2,0.500000000,-2.6351640;                           -- power, exponent and coeff.
    1; ! G-H POTENTIAL                                  -- # number of exponents in the G-H potential
    2,0.800756000,-13.393716;                           -- power, exponent and coeff.
    *                                                   -- comment line
    Spectral Representation Operator                    -- SR operator
    End of Spectral Representation Operator             -- end of SR operator
    

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../tutorials/tut.html>)
  * [4\. User’s Guide](<ug.html>)
    * [4.1. The Molcas environment](<env-main.html>)
    * [4.2. Programs](<programs.html>)
      * [4.2.1. ALASKA](<programs/alaska.html>)
      * [4.2.2. AVERD](<programs/averd.html>)
      * [4.2.3. CASPT2](<programs/caspt2.html>)
      * [4.2.4. CASVB](<programs/casvb.html>)
      * [4.2.5. CCSDT](<programs/ccsdt.html>)
      * [4.2.6. CHCC](<programs/chcc.html>)
      * [4.2.7. CHT3](<programs/cht3.html>)
      * [4.2.8. CMOCORR ¤](<programs/cmocorr.html>)
      * [4.2.9. CPF](<programs/cpf.html>)
      * [4.2.10. DIMERPERT ¤](<programs/dimerpert.html>)
      * [4.2.11. DMRGSCF](<programs/dmrgscf.html>)
      * [4.2.12. DYNAMIX](<programs/dynamix.html>)
      * [4.2.13. EMBQ ¤](<programs/embq.html>)
      * [4.2.14. ESPF (+ QM/MM interface)](<programs/espf.html>)
      * [4.2.15. EXPBAS](<programs/expbas.html>)
      * [4.2.16. EXTF](<programs/extf.html>)
      * [4.2.17. FALCON ¤](<programs/falcon.html>)
      * [4.2.18. FALSE](<programs/false.html>)
      * [4.2.19. FFPT](<programs/ffpt.html>)
      * [4.2.20. GATEWAY](<programs/gateway.html>)
      * [4.2.21. GENANO](<programs/genano.html>)
      * [4.2.22. GEO ¤](<programs/geo.html>)
      * [4.2.23. GRID_IT](<programs/grid_it.html>)
      * [4.2.24. GUESSORB](<programs/guessorb.html>)
      * [4.2.25. GUGA](<programs/guga.html>)
      * [4.2.26. GUGACI](<programs/gugaci.html>)
      * [4.2.27. GUGADRT](<programs/gugadrt.html>)
      * [4.2.28. LEVEL](<programs/level.html>)
      * [4.2.29. LOCALISATION](<programs/localisation.html>)
      * [4.2.30. LOPROP](<programs/loprop.html>)
      * [4.2.31. MBPT2](<programs/mbpt2.html>)
      * [4.2.32. MCKINLEY (a.k.a. DENALI)](<programs/mckinley.html>)
      * [4.2.33. MCLR](<programs/mclr.html>)
      * [4.2.34. MCPDFT](<programs/mcpdft.html>)
      * [4.2.35. MKNEMO ¤](<programs/mknemo.html>)
      * [4.2.36. MOTRA](<programs/motra.html>)
      * [4.2.37. MPPROP](<programs/mpprop.html>)
      * [4.2.38. MPSSI](<programs/mpssi.html>)
      * [4.2.39. MRCI](<programs/mrci.html>)
      * [4.2.40. MULA](<programs/mula.html>)
      * [4.2.41. NEMO ¤](<programs/nemo.html>)
      * [4.2.42. NEVPT2](<programs/nevpt2.html>)
      * [4.2.43. NUMERICAL_GRADIENT](<programs/numerical_gradient.html>)
      * [4.2.44. POLY_ANISO](<programs/poly_aniso.html>)
      * [4.2.45. QMSTAT](<programs/qmstat.html>)
      * [4.2.46. QUATER](<programs/quater.html>)
      * [4.2.47. RASSCF](<programs/rasscf.html>)
      * [4.2.48. RASSI](<programs/rassi.html>)
      * [4.2.49. RHODYN](<programs/rhodyn.html>)
      * [4.2.50. RPA](<programs/rpa.html>)
      * [4.2.51. SCF](<programs/scf.html>)
      * [4.2.52. SEWARD](<programs/seward.html>)
      * [4.2.53. SINGLE_ANISO](<programs/single_aniso.html>)
      * [4.2.54. SLAPAF](<programs/slapaf.html>)
      * [4.2.55. SURFACEHOP](<programs/surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<programs/symmetrize.html>)
      * [4.2.57. VIBROT](<programs/vibrot.html>)
      * [4.2.58. WFA](<programs/wfa.html>)
      * 4.2.59. The Basis Set Libraries
    * [4.3. GUI](<tools.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<programs/wfa.html> "4.2.58. WFA") | [next](<tools.html> "4.3. GUI") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/users.guide/basis_library.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/introduction.html -->

[ ](<index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<index.html>)

1.1.1. Molcas, Quantum Chemistry Software

[previous](<intro.html> "1. Introduction") | [next](<aboutthismanual.html> "1.1.2. The Molcas Manual") | [index](<genindex.html> "General Index")

# 1.1.1. Molcas, Quantum Chemistry Software¶

Molcas is a Quantum Chemistry software package developed by scientists to be used by scientists. It is neither a commercial product nor is it sold for significant profit gain by its owner, Lund University. The authors of Molcas have assembled their collected experience and knowledge in computational Quantum Chemistry to produce a research product which is used as a platform by the scientists in the Molcas network to develop new and improved computational tools in Quantum Chemistry. Several of the codes in the Molcas software have newly developed leading-edge features. Along with these new capabilities, users should not be surprised to ocasionally discover bugs when using Molcas.

The basic philosophy behind Molcas is to develop methods that allow accurate _ab initio_ treatment of very general electronic structure problems for molecular systems in both ground and excited states which is not an easy task. Nowadays, knowledge about how to obtain accurate properties for single-reference dominated ground states is well developed, and Molcas contains a number of codes that can perform such calculations (MP2, CC, CPF, DFT etc). All these methods treat the electron correlation starting from a single determinant (closed or open shell) reference state. Such codes are today’s standard in most Quantum Chemistry program.

However, Molcas is to be able to treat, highly degenerate states, such as those occurring in excited states, transition states in chemical reactions, diradicaloid systems, heavy metal systems, as well as other chemically important problems, all at the same level of accuracy. This is a much more difficult problem, since a single-determinant approach does not work well in these cases. The key feature of Molcas is the multiconfigurational approach. Molcas contains codes for general and effective multi-configurational SCF calculations at the Complete Active Space (CASSCF) level, but also employs more restricted MCSCF wave functions such as the Restricted Active Space, RASSCF, and the Generalized Active Space, GASSCF. It is also possible using either CASSCF or RASSCF to employ optimization techniques and obtain equilibrium geometries, transition-state structures, force fields, and vibrational energies using gradient techniques.

Since Molcas-8 the Stochastic-CASSCF method [[1](<references.html#id195> "G. Li Manni, S. D. Smart, A. Alavi. J. Chem. Theory Comput., 12 \(2016\) 1245-1258.")] is also available to treat systems requiring large active spaces. Routine calculations have been presented with active spaces of size of 40 electrons in 40 orbitals [[2](<references.html#id196> "G. Li Manni, A. Alavi. J. Phys. Chem. A, 122\[22\] \(2018\) 4935–4947."), [3](<references.html#id197> "G. Li Manni, D. Kats, D. P. Tew, A. Alavi. J. Chem. Theory Comput., 15 \(2019\) 1492-1497.")] and larger active space are currently tested.

Although the RASSCF approach is known to give reasonable structures for degenerate systems both in ground and excited states, in general it is not capable of recovering more than a fraction of the correlation energy. Therefore, it becomes necessary to supplement the multi-configurational SCF treatment with a calculation of dynamic correlation effects. In the earliest version of Molcas, this was achieved by means of the multi-reference (MR) CI method. This method has severe limitations in the number of electrons that can be correlated and the size of the reference space which limits study to excited states of small molecules. However, the MRCI code in Molcas does have the capacity to produce very accurate wave functions and potential energy surfaces, and is used by many groups for this purpose. In fact, it is also possible to run the COLUMBUS MRCI code together with Molcas.

During the period of 1986–90, a new method called CASPT2 was developed, which computes dynamic electron correlation effects for multi-configurational wave functions based on second order perturbation theory and was included into the second version of Molcas. From the beginning it was not clear whether or not the CASPT2 method would be sufficiently accurate to be useful in practice, but it turned out to be surprisingly accurate in a number of different types of chemical applications. The CASPT2 approach has become especially important in studies of excited states and spectroscopic properties of large molecules, where no other _ab initio_ method has, so far, been applicable. Since the CASPT2 method is based on second order perturbation theory and has, therefore, limitations in accuracy, the error limits have been investigated in a large number of applications. The relative energy errors are small in almost all cases leading to results which can be used for conclusive predictions of molecular properties in ground and excited states. Important application areas for the CASPT2 method are potential energy surfaces for chemical reactions, photochemistry, transition metal chemistry, and heavy element chemistry.

Several quasi-degenerate versions of CASPT2 are available, which allow for the simultaneous study of several electronic states, including their interaction to second order. This code is especially useful in cases where two or more energy surfaces are close in energy. An analytical CASPT2 gradient code (nuclear energy gradients and derivative coupling vector) supporting all these variants is available, allowing automatic geometry optimization at the CASPT2 level of theory. It is applicable to all states and systems for which the CASPT2 energy can be computed, whereas the calculation of vibrational frequencies can be carried out with the numerical procedure. It is important to note that the CASPT2 method is under constant development.

Since Molcas-8 the Multiconfiguration Pair-Density Functional Theory, MC-PDFT, is also available to treat dynamical correlation.

If only a few electrons are correlated, the active space can be quite large without too many configurations being generated, but in most cases the number of active electrons is comparable to the number of active orbitals. Occasionally, a larger active space would be preferred, but would result in too many configurations (more than a few million CSF’s). The more general RASSCF scheme can be useful, at the price of less efficient calculations, and the risk of bad convergence properties in the orbital optimization. The CASPT2 program handles also such wave functions, but will not include correlation within the active space, i.e., interaction with states that would have been included in the full CASSCF but are excluded by the RASSCF restrictions. The RASSCF wave function is regarded as an accurate approximation to the full CASSCF wave function, and the CASPT2 program evaluates only dynamic correlation that involves at least one non-active orbital.

Molcas not only contains the ability to produce various types of wave functions, but also can compute molecular properties using formulas of expectation values or finite perturbation theory using the RASSI program. The RASSI program has the capacity to compute the interaction between several RASSCF wave functions based on different orbitals which are generally non-orthonormal (i.e. a non-orthogonal CI). RASSI is routinely used to compute transition dipole moments in spectroscopy, to study electron transfer, and to obtain eigenstates of a relativistic Hamiltonian with inclusion of spin-orbit interaction.

Scalar, i.e. spin-averaged, relativistic effects are typically included in any calculations by using Douglas–Kroll–Hess transformation of one-electron integrals. The standard basis set library ANO-RCC [[4](<references.html#id266> "B. O. Roos, V. Veryazov, P.-O. Widmark. Theor. Chem. Acc., 111 \(2004\) 345-351."), [5](<references.html#id211> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark. J. Phys. Chem. A, 108 \(2004\) 2851-2858."), [6](<references.html#id212> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark. J. Phys. Chem. A, 109 \(2005\) 6575-6579."), [7](<references.html#id53> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark. Chem. Phys. Lett., 409 \(2005\) 295-299."), [8](<references.html#id214> "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, A. C. Borin. J. Phys. Chem. A, 112 \(2008\) 11431-11435.")] is optimized for use with these integrals and to include correlation of semi-core orbitals, and to to have uniform quality across the periodic system up to element 96, Curium. For heavier elements, typical calculations include the spin-orbit interaction by using CASSCF wave functions as a many-electron basis set, letting RASSI compute a Hamiltonian matrix over the set of all spin-components of these functions, correct for dynamic correlation using CASPT2 and include a one-electron spin-orbit Hamiltonian. This procedure has been shown to give accurate results in a number of studies for actinides and other heavy atom systems [[9](<references.html#id297> "B. O. Roos, P.-Å. Malmqvist, L. Gagliardi. In Fundamental World of Quantum Chemistry. Vol. II, pages 425-442. Kluwer Academic Publishers, 2003.")].

It is also possible to model solvent effects by adding a reaction field Hamiltonian (PCM). A new QM/MM model is also included in Molcas.

The release of Molcas-7 leads to many important enhancements. The sizes of the systems that can be treated with Molcas were previously restricted because of limitations in storing two-electron integrals for large basis sets. This system size limitation has been substantially reduced by the introduction of a Cholesky decomposition of the two-electron integrals. This feature is used in Molcas-7 at all levels of theory [[10](<references.html#id167> "F. Aquilante, R. Lindh, T. B. Pedersen. J. Chem. Phys., 127 \(2007\) 114107\(1–7\)."), [11](<references.html#id177> "F. Aquilante, P.-Å. Malmqvist, T. B. Pedersen, A. Ghosh, B. O. Roos. J. Chem. Theory Comput., 4 \(2008\) 694-702."), [12](<references.html#id171> "F. Aquilante, T. B. Pedersen, B. O. Roos, A. Sánchez de Merás, H. Koch. J. Chem. Phys., 129 \(2008\) 024113\(1–8\).")] and speeds up calculations by orders of magnitude, extending the size of basis sets that can be used. Accuracy can be controlled by the threshold used in the decomposition. The same approach can be used to generate RI auxiliary basis sets on the fly, allowing the calculation of energy derivatives for HF, MP2, DFT, and CASSCF levels of theory.

It is important to emphasize that important problems in Quantum Chemistry cannot be solved by simply applying black box techniques. Nor is Molcas a black box tool. A typical Molcas user should be someone with a high degree of chemical insight, who has some knowledge of different Quantum Chemical models in use today, and, most importantly, is able to apply these models to the appropriate chemical problem while understanding the inherent accuracy of these methods. The typical Molcas user should also apply critical analysis of results, take nothing for granted, and always check that the results are consistent with the model that was used. The skill to use Molcas effectively will not come immediately, but the user has several resources including this manual and examples which explain how different key projects were solved using Molcas. Users are certain to find them helpful in their own attempts to master the software for use in chemical applications. The Molcas group also arranges regular workshops, which provide a more intimate environment on learning how to use Molcas.

### Table of Contents

  * [1\. Introduction](<intro.html>)
    * [1.1. Introduction to Molcas](<intro.html#introduction-to-molcas>)
      * 1.1.1. Molcas, Quantum Chemistry Software
      * [1.1.2. The Molcas Manual](<aboutthismanual.html>)
      * [1.1.3. New features and updates](<news.html>)
      * [1.1.4. Parallelization efforts for Molcas modules](<parallelization.html>)
      * [1.1.5. Acknowledgment](<ack.html>)
      * [1.1.6. Citation for OpenMolcas and Molcas](<citation.html>)
      * [1.1.7. Web Site](<web.html>)
      * [1.1.8. Disclaimer](<disclaimer.html>)
  * [2\. Installation Guide](<installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<tutorials/tut.html>)
  * [4\. User’s Guide](<users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<advanced.examples/ae.html>)



### Search

[previous](<intro.html> "1. Introduction") | [next](<aboutthismanual.html> "1.1.2. The Molcas Manual") | [index](<genindex.html> "General Index")

[Get PDF](<../Manual.pdf>) | [Show Source](<_sources/introduction.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

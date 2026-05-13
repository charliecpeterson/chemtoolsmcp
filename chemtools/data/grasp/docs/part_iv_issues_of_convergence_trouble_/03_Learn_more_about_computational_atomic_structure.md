<!-- Source: GRASP2018-manual.pdf, pages 321–327 -->
<!-- Part: IV Issues of convergence, trouble shooting and non-default options -->

# Learn more about computational atomic structure

## Appendix A
## Learn more about computational
## atomic structure
### A.1
### The Computational Atomic Structure Group
To meet the demands for atomic data the Computational Atomic Structure (CompAS) group has been formed. The group is involved in developing state of the art computer codes for atomic calcu- lations in the non-relativistic scheme with relativistic corrections in the Breit-Pauli approximation ATSP2K, as well as in the fully relativistic scheme GRASP2018. The codes rely on multicon- guration methods and the wave function for an atomic state is expanded in conguration state functions (CSFs). In addition to the code development itself, the group includes members with expertise in the meth- ods and is constantly developing computational techniques for the evaluation of atomic properties of the highest quality. To learn more about CompAS group and its activities go to http://ddwap.mah.se/tsjoek/compas/index.php.
### A.2
### Suggested reading
In this section we suggest books and articles that provide the theoretical background to multi- conguration methods, electron correlation, and the systematic computation of dierent atomic properties.
1. B. Swirles
Relativistic self-consistent elds Proc. Roy. Soc. A 152, 625, (1935). The rst attempt to formulate DHF equations. This investigation was suggested in a con- versation between D R Hartree and Bertha Swirles on a railway station while returning to Cambridge from a conference in Manchester.
2. B. Swirles
Proc. Roy. Soc. A 157, 680, (1936).
3. I. P. Grant
Relativistic self-consistent elds. Proc. Roy. Soc. A 262 555-576 (1961). Formulated DHF equations with the use of Racah methods to exploit the internal symmetry of Dirac central eld orbitals. The mathematical tools were not available to Swirles in 1935. 321

322 APPENDIX A. LEARN MORE ABOUT COMPUTATIONAL ATOMIC STRUCTURE
4. I. P. Grant
Relativistic self-consistent elds. Proc. Phys. Soc. (Lond.) 86, 523-527 (1965) Sum rules for Breit interaction of a single electron with a closed subshell.
5. I. P. Grant and V. M. Burke
The eect of relativity on atomic wavefunctions Proc. Phys. Soc. (Lond.) 90, 297-314 (1967) First published comparison of Schrödinger and Dirac charge densities for Hg79+
6. C. Froese
Hartree-Fock Procedure for Some nsn′s 1S Congurations Phys. Rev. 150.1 (1966): 1-6: doi: /10.1103/PhysRev.150. The 1s2s 1S state was of considerable theoretical interest in the 1960's. This paper explores the trade-os between orthogonal and non-orthogonal orbitals.
7. I. P. Grant
Relativistic calculation of atomic structures Adv. Phys. 19, 747-811 (1970). Review of progress in relativistic atomic structure calculations to 1970. This survey revealed how quickly the reformulation of DHF had been adopted after 1961 and suggested the desirability of developing MCDHF.
8. I. P. Grant
Gauge invariance and relativistic radiative transitions J. Phys. B 7, 1458-1475 (1974). This paper began as an attempt to reconcile two alternative relativistic expressions for electric multipole transitions, equivalent to the length and velocity forms of nonrelativistic theory. These give the same numerical result for one-electron transitions in a local potential but can be wildly dierent when using HF-type orbitals. The electric and longitudinal radiative transition multipole operators have the same selection rules so that the transition matrix element is an arbitrary linear combination of the two. Coulomb and Babushkin expressions correspond to dierent multiples of the longitudinal operator whose contribution is only null if there is local charge conservation. The dierence of Coulomb and Babushkin results is now used to assess the accuracy of the wavefunctions describing initial and nal states. Papers 9-11 formulate the reduction of Breit and transverse interaction to radial integrals used in the original GRASP
9. I. P. Grant and N. C. Pyper
Breit interaction in multi-conguration relativistic atomic structure J. Phys. B 9, 761-774 (1976).
10. I. P. Grant and B. J. McKenzie
The transverse electron-electron interaction in atomic structure calculations J. Phys. B 13, 2671-2681 (1980).
11. B. J. McKenzie, I. P. Grant and P. H. Norrington
A program to calculate transverse Breit and QED corrections to energy levels in a MCDF environment Comput. Phys. Commun. 21, 233-246 (1980); ibid 23, 222 (1980). Papers 12-22 explore the use of MCDHF+B calculations in dierent contexts. The outputs were limited by the computing resources then available.

A.2. SUGGESTED READING 323
12. I. P. Grant, D. F. Mayers and N. C. Pyper
Studies in multi-conguration Dirac-Fock theory. I: The low-lying spectrum of Hf III J. Phys. B 9, 2777-2796 (1976).
13. S. J. Rose, N. C. Pyper and I. P. Grant
Studies in multi-conguration Dirac-Fock theory. II: The even-parity low-lying spectrum of Ba I J. Phys. B 11, 755-768 (1978).
14. N. C. Pyper and I. P. Grant
Studies in multi-conguration Dirac-Fock theory. III: Interpretation of the electronic struc- ture of the neutral and ionized states of uranium J. C. S. Faraday II 74, 1885-1900 (1978).
15. S. J. Rose, N. C. Pyper and I. P. Grant
Studies in multi-conguration Dirac-Fock theory. IV: The low-lying spectrum of Bi I. J. Phys. B 11, 3499-3512 (1978).
16. S. J. Rose, N. C. Pyper and I. P. Grant
the direct and indirect eects in the relativistic modication of atomic valence orbitals J. Phys. B 11, 1171-1176 (1978).
17. S. J. Rose, I. P. Grant and J.-P. Connerade
Fully relativistic analysis of 5p-excitation in atomic barium Phil. Trans. Roy. Soc. (Lond.) A296, 527-544 (1980).
18. J.-P. Connerade, S. J. Rose and I. P. Grant
Two-step autoionization and the double ionization anomaly on Ba I J. Phys. B 12, L53-L55 (1979).
19. N. C. Pyper, S. J. Rose and I. P. Grant
Analysis of ne structure excitation energies in Dirac-Fock and perturbation theories J. Phys. B 14, 1319-1331 (1982).
20. I. P. Grant and N. C. Pyper
Theoretical chemistry of superheavy elements E116 and E114 Nature 265, 715-717 (1977).
21. N. C. Pyper and I. P. Grant
On the interpretation of Hund's rules in atomic spectra J. Phys. B 10, 1803-1814 (1977).
22. N. Beatham, I. P. Grant, B. J. McKenzie and S. J. Rose
Spectroscopic studies with a MCDF program Physica Scripta 21, 423-431 (1980).
23. I. P. Grant
Many electron eects in the theory of nuclear volume isotope shift Physica Scripta 21, 443-447 (1980).
24. Relativistic eects in atoms, molecules and solids (ed. G. L. Malli) 1983 (Plenum: NATO
ASI Series B, Vol. 87) contain IPG's lectures: Incidence of relativistic eects in atoms pp. 55-72 Formulation of the relativistic N-electron problem. pp. 73-88 Techniques for open-shell calculations for atoms. pp. 89-100 Self-consistency and numerical problems. pp. 101-114

324 APPENDIX A. LEARN MORE ABOUT COMPUTATIONAL ATOMIC STRUCTURE
25. K. G. Dyall and I. P. Grant
Phase conventions, quasi-spin and the jj −LS transformation coecients J. Phys. B 15, L371-L373 (1982).
26. J. R. Lemen, K. J. H. Phillips, R. D. Cowan, J. Hata and I. P. Grant
Inner shell transitions of Fe XXIII and Fe XXIV in the X-ray spectra of solar ares Astron. & Astrophys. 135, 313-324 (1984). Program package publications: 27 was the rst time that previously published programs were assembled in package; 28 upgraded it and was superseded by GRASP92.
27. I. P. Grant, B. J. McKenzie, P. H. Norrington, D. F. Mayers and N. C. Pyper
An atomic MCDF package. Comput. Phys. Commun. 21, 207-232 (1980)
28. K. G. Dyall, I. P. Grant, C. T. Johnson, F. A. Parpia and E. P. Plummer
GRASP  a general-purpose relativistic atomic structure package Comput. Phys. Commun. 55, 425-456 (1989).
29. M. Tong, P. Jönsson, and C. Froese Fischer
Convergence studies of atomic properties from variational methods: total energy, ionization energy, specic mass shift, and hyperne parameters for Li Physica Scripta 48.4 (1993): 446; doi:10.1088/0031-8949/48/4/009. Discusses convergence of SDT excitations for a 3-electron system by partial waves, each expanded by n. Extrapolation procedures are discussed.
30. C. Froese Fischer
Convergence studies of MCHF calculations for Be and Li-. Journal of Physics B: 26.5 (1993): 855-862; doi:10.1088/0953-4075/26/5/009. Studies SDTQ excitations of 4-electron systems motivating what is now the MR-SD method. Introduces systematic methods, n-expansions, and extrapolations.
31. T. Brage and C. Froese Fischer.
Systematic calculations of correlation in complex ions. Physica Scripta 1993.T47 (1993): 18-28; doi:10.1088/0031-8949/1993/T47/002. Introduces a systematic approach with independent theoretical tests of atomic properties along with comparison with experiment.
32. T. Brage, C. Froese Fischer, and Per Jönsson.
Eects of core-valence and core-core correlation on the line strength of the resonance lines in Li I and Na I. Phys. Rev. A 49.3 (1994): 2181-2184; doi:10.1103/PhysRevA.49.2181 . Shows the importance of core-core correlation for accurate transition rates which, in time, agreed with experiment when experiment improved.
33. Z. Cai, V. M. Umar, and C. Froese Fischer. Large-scale relativistic correlation calculations:
Levels of Pr+3. Physical Review Letters 68.3 (1992): 297; doi:10.1103/PhysRevLett.68.297 Describes correlation studies for 4f 2 in the lanthanides using GRASP92.
34. A. Ynnerman and C. Froese Fischer
Multicongurational-Dirac-Fock calculation of the 2s2 1S0 −2s2p 3P1 spin-forbidden transi- tion for the Be-like isoelectronic sequence. Phys. Rev. A 51.3 (1995): 2020-2030; doi:10.1103/PhysRevA.51.2020. Introduced optimization by layers with GRASP92 using active set expansions. Discussed length/velocity discrepancy in the transition rate and the numerical cancellation in the cal- culation of the transition rate.

A.2. SUGGESTED READING 325
35. J. Olsen, M. Godefroid, P. Jönsson, P.Å. Malmqvist and C. Froese Fischer
Transition probability calculations for atoms using nonorthogonal orbitals. Physical Review E 52, 4499 (1995). Describes the transformation method that makes it possible to compute transition rates between separately optimized initial and nal states.
36. M. R. Godefroid, P. Jönsson and C. Froese Fischer
Atomic Structure Variational Calculations in Spectroscopy Physica Scripta. Vol. T78, 3346, 1998. The paper gives examples on how computational atomic structure can be used in atomic spectroscopy for testing theoretical models or experimental results, predicting properties or interpreting them in terms of electron correlation. The eects inherent in the multicong- uration Hartree-Fock method due to its variational nature are emphasized through some simple analysis of the wave function spatial distribution in correlation with the model used.
37. T. Brage, D. S. Leckrone, and C. Froese Fischer.
Core-valence and core-core correlation eects on hyperne-structure parameters and oscilla- tor strengths in Tl II and Tl III. Phys. Rev. A 53.1 (1996): 192-200; doi:10.1103/PhysRevA.53.192. Applies systematic procedures to the study of hyperne-structure parameters, including zero-and rst-order CSFs using GRASP92.
38. C. Froese Fischer.
Correlation and relativistic eects on transitions in lighter atoms. Physica Scripta 1999.T83 (1999): 49; doi:10.1238/Physica.Topical.083a00049 Discusses Breit-Pauli and MCDHF results for the calculation of transition data with regard to spectrum calculations.
39. Yu Zou, and C. Froese Fischer.
Resonance transition energies and oscillator strengths in lutetium and lawrencium. Physical Review Letters 88.18 (2002): 183001; doi: 10.1103/PhysRevLett.88.183001. Describes correlation in a complex system with two unlled shells. Though Lu and Lr are homologous systems, the conguration of the ground state changes.
40. J. Biero«, C. Froese Fischer, P. Indelicato, P. Jönsson, and P. Pyykkö
Complete Active Space multiconguration Dirac-Hartree-Fock calculations of hyperne struc- ture constants of the gold atom Phys. Rev. A 79, 052502 (2009) arXiv:physics/0902.4307 Describes calculations where correlation eects deep down in the core are accounted for.
41. C. Froese Fischer.
Relativistic Variational Calculations for Complex Atoms. Advances in the Theory of Atomic and Molecular Systems Springer Netherlands, 2009. 115- 128; doi:10.1007/978-90-481-2596-8-7 Reviews relativistic calculations for complex heavy atoms.
42. G. Gaigalas, E. Gaidamauskas, Z. R. Rudzikas, N. Magnani, R. Caciuo
Theoretical studies of spectroscopic properties of Cm4+ and Am3+ Physical Review A, Atomic, molecular, and optical physics. ISSN 1050-2947. 2009, Vol. 79, iss. 2, p. 022511-1-8.

326 APPENDIX A. LEARN MORE ABOUT COMPUTATIONAL ATOMIC STRUCTURE
43. G. Gaigalas, E. Gaidamauskas, Z.R. Rudzikas, N. Magnani, R. Cacciuo
Correlation, relativistic, and quantum electrodynamics eects on the atomic structure of eka-thorium Physical Review A, Atomic, molecular, and optical physics. ISSN 1050-2947. 2010, Vol. 81, Issue 2, p. 022508.
44. G. Gaigalas, Z. Rudzikas, E. Gaidamauskas, P. Rynkun, A. Alkauskas
Peculiarities of spectroscopic properties of W24+ Physical Review A, Atomic, molecular, and optical physics. ISSN 1050-2947. 2010, Vol. 82, issue 1, p. 014502-4.
45. C. Froese Fischer and G. Gaigalas
Multiconguration Dirac-Hartree-Fock energy levels and transition probabilities for W XXXVIII Physical Review A, Atomic, molecular, and optical physics. ISSN 1050-2947. 2012, Vol. 85, p. 042501.
46. S. Verdebout, P. Rynkun, P. Jönsson, G. Gaigalas, C. Froese Fischer, M. Godefroid
A Partitioned Correlation Function Interaction approach for describing electron correlation in atoms Journal of Physics B 46, 085003 (2013). This paper includes a discussion about electron correlation eects and the limitations of cur- rent methodologies for describing these eects. The paper gives an outline of how improved methods can be implemented based on a biorthonormal orbital transformation.
47. P. Jönsson, P. Bengtsson, J. Ekman, S. Gustafsson, L.B. Karlsson, G. Gaigalas, C. Froese
Fischer, D. Kato, I. Murakami, H.A. Sakaue, H. Hara, T. Watanabe, N. Nakamura, and N. Yamamoto Relativistic CI calculations of spectroscopic data for the 2p6 and 2p53l congurations in Ne-like ions between Mg III and Kr XXVII. Atomic Data and Nuclear Data Tables 100, 1 (2014). Discusses accurate spectrum calculations giving energies and transition rates for hundreds of levels.
48. J. Ekman, P. Jönsson, S. Gustafsson, H. Hartman, G. Gaigalas, M.R. Godefroid, and C.
Froese Fischer Calculations with spectroscopic accuracy: energies, Landé gJ-factors, and transition rates in the carbon isoelectronic sequence from Ar XIII to Zn XXV Astronomy & Astrophysics 564, A24 (2014). Discusses accurate spectrum calculations giving energies and transition rates for hundreds of levels.
49. S. Verdebout, C. Nazé, P. Jönsson, P. Rynkun, M. Godefroid, G. Gaigalas
Hyperne structures and Landé: gJ-factors for n = 2 states in beryllium-, boron-, carbon-, and nitrogen-like ions from relativistic conguration interaction calculations Atomic Data and Nuclear Data Tables 100, 1111 (2014). This paper discusses hyperne structure and the computation of Landé gJ-factors.
50. C. Nazé, S. Verdebout, P. Rynkun, G. Gaigalas, M. Godefroid, P. Jönsson
Isotope Shifts in Beryllium-, Boron-, Carbon-, and Nitrogen-like Ions from Relativistic Con- guration Interaction Calculations Atomic Data and Nuclear Data Tables 100, 1197 (2014). The paper gives examples of accurate IS calculations including relativistic corrections.

A.2. SUGGESTED READING 327
51. J. Ekman, M. R. Godefroid and H. Hartman
Validation and Implementation of Uncertainty Estimates of Calculated Transition Rates Atoms 2014, 2(2), 215-224; doi:10.3390/atoms2020215. The paper discusses how and if uncertainties can be estimated for calculated transition rates in length and velocity gauge.
52. L. Radºi�ut
e, D. Kato, G. Gaigalas, P. Jönsson, P. Rynkun, V. Jonauskas and S. Kucas
Energy level structure of Er3+ Journal of Quantitative Spectroscopy and Radiative Transfer, ISSN 0022-4073, 2015, Vol. 152, pp. 94-106. Describes how calculations can be done for atoms with very complex shells structures.
53. G. Gaigalas, P. Rynkun, Ch. Froese Fischer
Lifetimes of 4p54d levels in highly ionized atoms Physical Review A, Atomic, molecular, and optical physics. ISSN 1050-2947. 2015, Vol. 91, p. 022509-10. Shows how, for levels (identied by their label) that cross along an iso-electronic sequence, when viewed instead by J, parity, and position, their composition changes smoothly along an isoelectronic sequence and transition matrix elements can be approximated by formulas in Z.
54. L. Radºi�ut
e, J. Ekman, P. Jönsson, and G. Gaigalas
Extended calculations of level and transition properties in the nitrogen isoelectronic se- quence: Cr XVIII, Fe XX, Ni XXII, and Zn XXIV Astronomy & Astrophysics 2015, 582, A61 Discusses accurate spectrum calculations giving energies and transition rates for hundreds of levels.
55. S. Gustafsson, P. Jönsson, C. Froese Fischer and I.P. Grant
Combining multiconguration and perturbation methods: perturbative estimates of core- core electron correlation contributions to excitation energies in Mg-like iron Atoms, 5(1), 3 (2017); doi:10.3390/atoms5010003 Summarizes how perturbative techniques (zero- and rst-order partitions) can be used to include electron correlation.
56. P. Jönsson, G. Gaigalas, P. Rynkun, L. Radºi�ut
e, J. Ekman, S. Gustafsson, H. Hartman, K.
Wang, M. Godefroid, C. Froese Fischer, I. Grant, T. Brage, G. Del Zanna Multiconguration Dirac-Hartree-Fock calculations with spectroscopic accuracy: applica- tions to astrophysics Atoms 5(2) , 16 (2017); doi:10.3390/atoms5020016 Summarizes a number of very accurate calculations on astrophysically important ions attain- ing spectroscopic accuracy, i.e. accuracy high enough to directly support line identications in spectra.
<!-- Source: GRASP2018-manual.pdf, pages 23–30 -->
<!-- Part: I Overview of GRASP2018 -->

# Program structure and file flow

## Chapter 3
## Program structure and le ow
### 3.1
### Program naming conventions
In multiconguration calculations the wave function for an atomic state is approximated by an atomic state function (ASF). The ASF, in turn, is given as an expansion over conguration state functions (CSFs) Ψ(γPJ) = N � i=1 ciΦ(γiPJ). (3.1) Here {γi} denote the congurations together with the angular coupling trees, P is parity, J is the nal angular quantum number, and {ci} are the expansion (mixing) coecients. The CSFs are given as coupled anti-symmetric products of one-electron orbitals φ(x) = 1 r � P(nκ; r)χκm(θ, ϕ) i Q(nκ; r)χ−κm(θ, ϕ) � , (3.2) where the radial parts of the orbitals (the radial wave functions) P(nκ; r), Q(nκ; r) are numerically represented on a grid.1 Given this description we identify three main concepts:
• lists of CSFs dening the ASF
• mixing coecients
• radial parts (wave functions) of orbitals
These concepts are the basis for the program naming conventions: programs generating or manip- ulating lists of CSFs have names starting with rcsf, programs generating or manipulating mixing coecients have names starting with rmix, programs generating or manipulating the radial parts of the orbitals (radial wave functions) have names starting with rwfn. Other programs are named according to the atomic properties they compute. There are also a number of programs that produce output tables in LaTeX format. These programs all have names starting with rtab. Finally, there are programs that create GNU Octave and Matlab M-les for plotting properties along iso-electronic sequences. These programs have names starting with rseq. 1In the guide the three terms radial orbital, radial part of the orbital, and radial wave function will be used intermixed meaning the same thing. Sometimes we will also loosely speak about the orbitals meaning the radial parts of the orbitals. 23

24 CHAPTER 3. PROGRAM STRUCTURE AND FILE FLOW
### 3.2
### Application programs and tools
Below is a partial list of programs in the package:
1. rnucleus
 dene nuclear data.
2. Routines that generate and manipulate lists of CSFs:
(a) rcsfgenerate  generate a list of CSFs using rules for excitations. (b) rcsfinteract  reduce a list of CSFs by retaining only CSFs that interact with CSFs of a reference list (former jjreduce). (c) rcsfsplit  split a list of CSFs into a number of lists with CSFs that can be formed from dierent sets of active orbitals. (d) rcsfzerofirst  rearrange a list of CSFs in such a way that the most important CSFs come at the beginning, dening the zero-order space, and the less important come at the end, dening the rst-order space.
3. rangular, rangular_mpi  perform angular integration and compute angular coecients
(former mcp).
4. rwfnestimate  estimate the radial parts of the orbitals - radial wave functions (former
erwf).
5. rmcdhf, rmcdhf_mpi  determine radial parts of the orbitals and mixing coecients of the
CSFs in a relativistic self-consistent-eld (SCF) procedure.
6. rci, rci_mpi  perform relativistic conguration interaction (RCI) calculation with trans-
verse photon (Breit) interaction and vacuum polarization and self-energy (QED) corrections.
7. jj2lsj  a program for converting a portion of the wave function expansion in jj-coupled
CSFs to LSJ-coupled CSFs for labeling purposes. Includes a unique labeling feature.
8. Routines for computing transition probabilities:
(a) rbiotransform, rbiotransform_mpi  perform biorthonormal transformations of wave functions. (b) rtransition, rtransition_mpi  compute transition properties from transformed wave functions. If the program jj2lsj has been run, the labels of the states in the output les are in LSJ-coupling.
9. rhfs  compute diagonal and o-diagonal hyperne interaction constants and Landé gJ-
factors.
10. rsms  compute isotope shift.
A number of generally short programs have been developed as tools to facilitate computational procedures.
1. rmixaccumulate  accumulate CSFs corresponding to a specied fraction of the total wave
function.
2. rmixextract
 extract and print the numerical values of the expansion coecients above a cut-o value along with the corresponding CSFs, in descending order of magnitude, if requested (former extmix).
3. rcsfmr  analyse the wave function expansion in LSJ-coupled CSFs and determine a mul-
tireference.

3.2. APPLICATION PROGRAMS AND TOOLS 25
4. hf  perform a non-relativistic Hartree-Fock (HF) calculation to produce a radial wave
function le wfn.out. The le wfn.out should be copied to wfn.inp for further processing.
5. rwfnmchfmcdf  convert a non-relativistic Hartree-Fock radial wave function le wfn.inp
le to a grasp2018 radial wave function le rwfn.out that can be used with rwfnestimate.
6. rwfnplot  extract radial wave functions from a radial wave function le and generate a
GNU Octave/Matlab M-le that plots the radial wave functions as functions of √r or r.
7. wfnplot  extract radial wave functions from the non-relativistic radial wave function le
as produced by the hf program and generate a GNU Octave/Matlab M-le that plots the radial wave functions as functions of √r or r.
8. rwfnrotate  a routine that rotates radial orbitals, useful for debugging purposes (former
rotate_pair).
9. rlevels  list the levels in a series of mixing les, in the order of increasing energy and
report levels in cm−1 relative to the lowest. If the program jj2lsj has been run, the levels are given in LSJ-coupling notation.
10. rlevelseV  list the levels in a series of mixing les, in the order of increasing energy and
report levels in eV relative to the lowest. If the program jj2lsj has been run, the levels are given in LSJ-coupling notation.
11. rtablevels  produce LaTeX and ASCII tables of energies from energy les produced by
rlevels.
12. lscomp.pl  perl script to produce LaTeX tables with LSJ composition and energies from
energy les rlevels.
13. rtabtransE1  produce LaTeX and ASCII tables of transition parameters from les produced
by rtransition (E1 transitions only).
14. rtabtrans1 and rtabtrans2  produce LaTeX tables of transition parameters and lifetimes
from les produced by rtransition.
15. rhfs_lsj  give the output from the rhfs program and its variants in LSJ-coupling notation.
16. rtabhfs  produce LaTeX tables of hyperne interaction constants.
17. rseqenergy  produce GNU Octave/Matlab M-les that plot energies as functions of Z
along an iso-electronic sequence.
18. rseqhfs  produce GNU Octave/Matlab M-les that plot hyperne interaction constants
and Landé gJ-factors as functions of Z along an iso-electronic sequence.
19. rseqtrans  produce GNU Octave/Matlab M-les that plot transition parameters as func-
tions of Z along an iso-electronic sequence.
20. rsave  a script le such that the command rsave name moves rwfn.out to name.w,
rmix.out to name.m, rcsf.inp to name.c, rmcdhf.sum to name.sum, rangular.log to name.alog and rmcdhf.log to name.log.
21. rasfsplit  splits the les dening a number of ASFs of dierent symmetry blocks (J and
parity) into groups of les, one for each symmetry block.
22. jjgen  generates a list of CSFs in non-block form.
23. rcsfblock  splits the list produced by jjgen into block-form.

26 CHAPTER 3. PROGRAM STRUCTURE AND FILE FLOW
### 3.3
### File naming convention, program and data ow
Passing information between dierent programs is done through les. This process is greatly sim- plied by a le naming convention. Grasp2018 uses a convention similar to the one for Atsp2K [1]. A name is associated with the results from a calculation and an extension denes the content and format of a le. Thus the le name becomes name.extension. Common extensions are listed in Table 3.1. The tool rsave makes use of these default extensions to save the output les from an rmcdhf calculation. Most programs produce a le that keeps a record of the input data. This le is called a log-le. Table 3.1: Table of common extensions. Extension Type of le c List of CSFs. w Binary le of radial wave functions. m Binary le of expansion or mixing coecients produced by rmcdhf. sum File containing information from an rmcdhf run. cm Binary le of mixing coecients produced by rci. bw A .w le after biorthonormal transformation using rbiotransform. csum File containing information from an rci run. bm A .m le after biorthonormal transformation using rbiotransform. cbm A .cm le after biorthonormal transformation using rbiotransform. lsj.lbl File containing composition of wave functions in LSJ-coupling. uni.lsj.lbl File containing composition of wave functions in LSJ-coupling but arranged to give unique labels of all states. t Transition probability data from rmcdhf mixing coecients. t.lsj Transition probability data from rmcdhf mixing coecients. Labels in in LSJ-coupling. ct Transition probability data from rci mixing coecients. ct.lsj Transition probability data from rci mixing coecients. Labels in in LSJ-coupling. h Hyperne structure data and Landé gJ-factors from rmcdhf mixing coecients. ch Hyperne structure data and Landé gJ-factors from rci mixing coecients. hoffd O-diagonal hyperne structure data from rmcdhf mixing coecients. choffd O-diagonal hyperne structure data from rci mixing coecients. i Isotope shift data from rmcdhf mixing coecients. ci Isotope shift data from rci mixing coecients. log Log-le that keeps a record of program input. To perform a calculation a number of programs needs to be run in a pre-determined sequence. Figure 3.1 displays a typical sequence of program calls to compute wave functions and dierent expectation values. The resulting ow of les is displayed in Figure 3.2.

3.3. FILE NAMING CONVENTION, PROGRAM AND DATA FLOW 27 Sequence of program calls to compute expectation values rnucleus Generation of nuclear data ? rcsfgenerate Generation of list of CSFs based on rules for excitations ? rcsfinteract Reduction of a list to CSFs interacting with the multireference ? rangular Angular integration ? rwfnestimate Initial estimates of radial orbitals ? rmcdhf Self-consistent eld procedure ? rci Relativistic RCI with optional transverse photon (Breit) interaction and vacuum polarization and self-energy (QED) corrections ? jj2lsj Transform representation from jj- to LSJ-coupling � � � � � � � �  PPPPPPPPPPP q rbiotransform Biorthonormal transf. ? rtransition Eval. of expect. values rhfs, rsms Eval. of expect. values Figure 3.1: Typical sequence of program calls to compute wave functions and dierent expectation values.

28 CHAPTER 3. PROGRAM STRUCTURE AND FILE FLOW Flow of les between dierent programs rnucleus ? Output; isodata rcsfgenerate ? Output; rcsf.out, rcsfgenerate.log rcsfinteract ? Input; rcsfmr.inp, rcsf.inp Output; rcsf.out rangular ? Input; rcsf.inp Output; rangular.log, mcp.30, mcp.31, ... rwfnestimate ? Input; isodata, rcsf.inp, optional radial wave function le(s) Output; rwfn.inp rmcdhf ? Input; isodata, rcsf.inp, rwfn.inp, mcp.30, mcp.31, ... Output; rmix.out, rwfn.out, rmcdhf.sum, rmcdhf.log Use rsave to move output les to name.c, name.m, name.w, name.sum, name.log rci ? Input; name.c, name.w Output; name.cm, name.csum, name.clog, rci.res (restart le) jj2lsj ? Input; name.c, name.(c)m Output; name.lsj.lbl, name.uni.lsj.lbl (optional) rbiotransform ? Input; name1.c, name1.(c)m, name1.w, name2.c, name2.(c)m, name2.w Input; name1.TB, name2.TB (if available) Output; name1.(c)bm, name1.(c)bw, name2.(c)bm, name2.(c)bw Output; name1.TB, name2.TB rtransition Input; name1.c, name1.(c)bm, name1.(c)bw name2.c, name2.(c)bm, name2.(c)bw Input; name1.name2.xT (if available) Output; name1.name2.(c)t, name1.name2.(c)t.lsj (in LSJ-coupling) Output; name1.name2.xT Figure 3.2: Flow of les for a normal sequence of program runs. Extensions (c) indicate data les based on rci mixing coecients. For rtransition the extension x denotes the multipole.

3.4. OLD AND NEW PROGRAM NAMES 29
### 3.4
### Old and new program names
Below is a conversion table between old and new program names. Table 3.2: Table of name conversions Grasp2K old version Grasp2018 version bioscl rtransition bioscl_mpi rtransition_mpi biotra rbiotransform biotra_mpi rbiotransform_mpi erwf rwfnestimate extmix rmixextract iso rnucleus jj2lsj jj2lsj jjgen jjgen jjreduce rcsfinteract jsplit rcsfblock obsolete, blocks handled automatically mchfmcdf rwfnmchfmcdf mcp rangular mcp_mpi rangular_mpi plotmcdf rwfnplot rci rci rci_mpi rci_mpi rhfs rhfs rlevels rlevels, rlevelseV rotate_pair rwfnrotate rsave rsave rscf rmcdhf rscf_mpi rmcdhf_mpi sms rsms
### 3.5
### Test data set under bash
Along with Grasp2018 there is a test data set. Under the grasptest directory the following sub-directories reside example1 : script, output example2 : script, output example3 : script, output example4 : script, output, tmp_mpi example5 : script, output case1 : script case1_mpi: script, tmp_mpi case2 : script case2_mpi: script, tmp_mpi case3 : script The output directories contain output les from the ve examples in chapter 7. To validate program operations these output les can be used as references. Script les for the examples and the three case studies can be found in the script directories. In example 4 and in the rst two cases studies the MPI codes are used and the temporary les from each process are written to

30 CHAPTER 3. PROGRAM STRUCTURE AND FILE FLOW the tmp_mpi sub-directories. Carefully look at the README le before running the scripts in these three cases. The scripts have been tested under bash.
<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/ccsdt.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.5. CCSDT

[previous](<casvb.html> "4.2.4. CASVB") | [next](<chcc.html> "4.2.6. CHCC") | [index](<../../genindex.html> "General Index")

# 4.2.5. CCSDT¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

  * How to run closed shell calculations using ROHF CC codes




CCSDT performs the iterative single determinant CCSD procedure for open shell systems and the noniterative triple contribution calculation to the CCSD energy. For further details the reader is referred to [Sections 3.3.16](<../../tutorials/tut_ccsdt.html#tut-sec-ccsdt>) and [5.1.4](<../../advanced.examples/ex-hi.html#tut-sec-rp-wf>) of the tutorials and examples manual.

## 4.2.5.1. Dependencies¶

CCSDT requires a previous run of the RASSCF program to produce orbital energies, Fock matrix elements, wave function specification, and some other parameters stored in file JOBIPH. The RASSCF program should be run with the options that produce canonical output orbitals, which is not default. CCSDT also requires transformed integrals produced by MOTRA and stored in the files TRAONE and TRAINT.

It is well known that the CCSD procedure brings the spin contamination into the final wave function \\(\ket{\Psi}\\) even in the case where the reference function \\(\ket{\Phi}\\) is the proper spin eigenfunction. The way how to reduce the spin contamination and mainly the number of independent amplitudes is to introduce the spin adaptation. Besides the standard nonadapted (spinorbital) CCSD procedure this program allows to use different levels of spin adaptation of CCSD amplitudes (the recommended citations are Refs. [[63](<../../references.html#id134> "P. Neogrády, M. Urban, I. Hubač. J. Chem. Phys., 100 \(1994\) 3706-3716."), [64](<../../references.html#id127> "P. Neogrády, M. Urban, I. Hubač. J. Chem. Phys., 97 \(1992\) 5074-5080.")]):

  * DDVV T2 adaptation.

This is the most simple and most universal scheme, in which only the dominant part of T2 amplitudes, namely those where both electrons are excited from _doubly occupied (inactive)_ to _virtual (secondary)_ orbitals, are adapted. The remaining types of amplitudes are left unadapted, i.e. in the spinorbital form. This alternative is an excellent approximation to the full adaptation and can be used for any multiplet.

  * Full T1 and T2 adaptation (only for doublet states yet).

In this case full spin adaptation of all types of amplitudes is performed. In the present implementation this version is limited to systems with the single unpaired electrons, i.e. to the doublet states only.




Besides these two possibilities there are also available some additional partial ones (see keyword ADAPTATION in Section 4.2.5.3). These adaptations are suitable only for some specific purposes. More details on spin adaptation in the CCSD step can be found in Refs. [[63](<../../references.html#id134> "P. Neogrády, M. Urban, I. Hubač. J. Chem. Phys., 100 \(1994\) 3706-3716."), [64](<../../references.html#id127> "P. Neogrády, M. Urban, I. Hubač. J. Chem. Phys., 97 \(1992\) 5074-5080."), [65](<../../references.html#id129> "P. J. Knowles, C. Hampel, H.-J. Werner. J. Chem. Phys., 99 \(1993\) 5219-5227.")]. The current implementation of the spin adaptation saves no computer time. A more efficient version is under development.

The noniterative triples calculation can follow these approaches:

  * CCSD + T(CCSD) — according to Urban et al. [[66](<../../references.html#id106> "M. Urban, J. Noga, S. J. Cole, R. J. Bartlett. J. Chem. Phys., 83 \(1985\) 4041-4046.")]

  * CCSD(T) — according to Raghavachari et al. [[67](<../../references.html#id32> "K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon. Chem. Phys. Lett., 157 \(1989\) 479-483.")]

  * CCSD(T) — according e.g. to Watts et al. [[27](<../../references.html#id132> "J. D. Watts, J. Gauss, R. J. Bartlett. J. Chem. Phys., 98 \(1993\) 8718-8733.")]




Actual implementation and careful analysis and discussion of these methods is described in Ref. [[28](<../../references.html#id66> "P. Neogrády, M. Urban. Int. J. Quantum Chem., 55 \(1995\) 187-203.")], which is a recommended reference for this program.

The first alternative represents the simplest noniterative T3 treatment and contains only pure \\(\braket{T3}{W T2}\\) term. Second possibility represents the well known extension to the first one by the \\(\braket{T3}{W T1}\\) term (\\(W\\) is the two electron perturbation). For closed shell systems this is the most popular and most frequently used noniterative triples method. For single determinant open shell systems, described by the ROHF reference function standard (Raghavachari et. al.) method needs to be extended by the additional fourth order energy term, namely \\(\braket{T3}{U T2}\\) (\\(U\\) is the off-diagonal part of the Fock operator).

In contrast to the iterative CCSD procedure, noniterative approaches are not invariant with respect to the partitioning of the Hamiltonian. Hence, we obtain different results using orbital energies, Fock matrix elements or some other quantities in the denominator. According to our experiences [[28](<../../references.html#id66> "P. Neogrády, M. Urban. Int. J. Quantum Chem., 55 \(1995\) 187-203.")], diagonal Fock matrix elements in the denominator represent the best choice. Using of other alternatives requires some experience. Since the triple excitation contribution procedure works strictly within the restricted formalism, resulting noniterative triples contributions depend also on the choice of the reference function. However, differences between this approach (with the reference function produced by a single determinant RASSCF procedure and the diagonal Fock matrix elements considered in the denominator) and the corresponding invariant treatment (with the semicanonical orbitals) are found to be chemically negligible.

For noniterative T3 contribution both non-adapted (spin-orbital) and spin-adapted CCSD amplitudes can be used. For more details, see Ref. [[28](<../../references.html#id66> "P. Neogrády, M. Urban. Int. J. Quantum Chem., 55 \(1995\) 187-203.")].

## 4.2.5.2. Files¶

### 4.2.5.2.1. Input files¶

CCSDT will use the following input files: TRAONE, TRAINT, RUNFILE, JOBIPH, (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

### 4.2.5.2.2. Output files¶

RSTART
    

file with CC amplitudes and CC energy. The name of the file can be changed using keyword RESTART. It contains restart information, like T1aa, T1bb, T2aaaa, T2bbbb, T2abab, CC energy and the number of iterations.

T3hfxyy
    

These files contain integrals of \\(\braket{ia}{bc}\\) type where _x_ represents the symmetry and _yy_ the value of the given index \\(i\\). The number of these files is equal to the number of \\(\alpha\\) occupied orbitals (_inactive + active_).

## 4.2.5.3. Input¶

The input for each module is preceded by its name like:
    
    
    &CCSDT
    

TITLe
    

This keyword should be followed by precisely one title line. It should not begin with a blank (else it will not be printed!) This keyword is _optional_.

CCSD
    

This keyword specifies that only CCSD calculation will follow and the integrals will be prepared for the CCSD procedure only. This keyword is _optional_. (Default=OFF)

CCT
    

This keyword specifies that after CCSD calculation also noniterative T3 step will follow. For such calculations this key must be switched on. The integrals for the triple contribution calculation will then be prepared. This keyword is _optional_. (Default=ON)

ADAPtation
    

The parameter on the following line defines the type of spin adaptations of CCSD amplitudes.

0 — no spin adaptation — full spinorbital formalism

1 — T2 DDVV spin adaptation

2 — T2 DDVV + T1 DV spin adaptation (only recommended for specific purposes, since the adaptation of T1 included incompletely)

3 — full T2 and T1 spin adaptation (in current implementations limited to doublets only)

4 — full T2 adaptation without SDVS coupling (for doublets only)

This keyword is _optional_. (Default=0)

DENOminators
    

The parameter on the following line specifies the type of denominators that will be used in the CCSD procedure.

0 — diagonal Fock matrix elements (different for \\(\alpha\\) and \\(\beta\\) spins)

1 — spin averaged diagonal Fock matrix elements — \\(\frac{f_{\alpha\alpha}+f_{\beta\beta}}{2}\\)

2 — orbital energies

In some cases alternatives 1 and 2 are identical. For nonadapted CCSD calculations the resulting CCSD energy is invariant with respect to the selection of denominators. However, convergence may be affected.

In the present implementation a symmetric denominators (i.e. the input 1 or 2) should be used for spin adapted CCSD calculations. This keyword is _optional_. (Default=0)

SHIFts
    

Following line contains _socc_ and _svirt_ levelshift values for occupied and virtual orbitals respectively. Typical values are in the range 0.0–0.5 (in _a.u._)
    
    
    dp(occ)=dp(occ)-socc
    dp(virt)=dp(virt)+svirt
    

For spin adaptations 3 and 4 only inactive (D) and active (V) orbitals will be shifted, due to the character of the adaptation scheme. For other cases all orbitals are shifted.

This keyword is _optional_. (Defaults: _socc_ = 0.0, _svirt_ = 0.0)

TRIPles
    

The parameter on the following line specifies the type of noniterative triples procedure. There are three different types of perturbative triples available (see Section 4.2.5).

0 — CCSD approach (no triples step)

1 — CCSD+T(CCSD) according to Urban et. al [[66](<../../references.html#id106> "M. Urban, J. Noga, S. J. Cole, R. J. Bartlett. J. Chem. Phys., 83 \(1985\) 4041-4046.")]

2 — CCSD(T) according to Raghavachari et. al. [[67](<../../references.html#id32> "K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon. Chem. Phys. Lett., 157 \(1989\) 479-483.")]

3 — CCSD(T) according e.g. to Watts et. al. [[27](<../../references.html#id132> "J. D. Watts, J. Gauss, R. J. Bartlett. J. Chem. Phys., 98 \(1993\) 8718-8733.")]

This keyword is _optional_. (Default=3)

T3DEnominators
    

The parameter on the following line specifies the type of denominators that will be used in noniterative triples procedure.

0 — diagonal Fock matrix elements (different for \\(\alpha\\) and \\(\beta\\) spins)

1 — spin averaged diagonal Fock matrix elements — \\(\frac{f_{\alpha\alpha}+f_{\beta\beta}}{2}\\)

2 — orbital energies

In some cases alternatives 1 and 2 are identical. This keyword is _optional_. (Default=0)

T3SHifts
    

The following line contains _socc_ and _svirt_ levelshift values for occupied and virtual orbitals respectively. Typical values are in the range 0.0–0.5 (in _a.u._)
    
    
    dp(occ)=dp(occ)-socc
    dp(virt)=dp(virt)+svirt
    

In contrast to the iterative CCSD procedure, in noniterative T3 step results are not invariant with respect to the denominator shifting. It is extremely dangerous to use any other than 0.0 0.0 shifts here, since resulting T3 energy may have no physical meaning. This keyword may be useful only in estimating some trends in resulting energy, however, using of default values is strongly recommended.

This keyword is _optional_. (Defaults: _socc_ = 0.0, _svirt_ = 0.0)

ITERations
    

This keyword is followed on the next line by the maximum number of iterations in the CCSD procedure. In the case of the RESTART run this is the number of last allowed iteration, since counting of iterations in RESTART run starts from the value taken from the RSTART file. This keyword is _optional_. (Default=30)

ACCUracy
    

The real value on the following line defines the convergence criterion on CCSD energy. This keyword is _optional_. (Default=1.0d-7)

END of input
    

This keyword indicates that there is no more input to be read. This keyword is _optional_.

EXTRapolation
    

This keyword switches on the DIIS extrapolation. This keyword is followed by two additional parameters on the next line _n1_ and _n2_.

_n1_ — specifies the first iteration, in which DIIS extrapolation procedure will start for the first time. This value must not be less then _n2_ , recommended value is 5–7.

_n2_ — specifies the size of the DIIS procedure, i.e. the number of previous CCSD steps which will be used for new prediction. In the present implementation _n2_ is limited to 2–4.

This keyword is _optional_. (Default=OFF)

PRINt
    

The parameter on the next line specifies the level of output printing

0 — minimal level of printing

1 — medium level of printing

2 — full output printing (useful for debugging purposes)

This keyword is _optional_. (Default=0)

LOAD
    

This keyword is followed by the line which specifies the name of the CCSD amplitudes and energy file. The default name is RSTART, but it can be changed in CCSD step using RESTART keyword. This keyword is _optional_. (Default=:file:RSTART)

RESTart
    

This keyword defines the restart conditions and modifies the name of the file, in which restart information (CC amplitudes, CC energy and the number of iterations) is saved. On the following two lines there are control key _nn_ and the name of restart information storing file _name_.

_nn_ — restart status key

0 — restart informations will be not saved

1 — restart informations will be saved after each iteration in _name_.

2 — restart run. CC amplitudes and energy will be taken from _name_ file and the CCSD procedure will continue with these values as an estimate.

_name_ — specifies the restart information storing key. The name is limited to 6 characters.

This keyword is _optional_. (Defaults: _nn_ = 1, _name_ = RSTART)

IOKEy
    

This keyword specifies the input-output file handling.

1 — Internal Fortran file handling

2 — Molcas DA file handling

The default (1) is recommended in majority of cases, since when calculating relatively large systems with low symmetry, the size of some intermediate files produced may become large, what could cause some troubles on 32-bit machines (2 GB file size limit).

MACHinetyp
    

This keyword specifies which type of matrix multiplication is preferred on a given machine. The following line contains two parameters _nn_ , _limit_.

_nn_ = 1 — standard multiplication \\(A B\\) is preferred

_nn_ = 2 — transposed multiplication \\(A^{\text{T}} B\\) is preferred

Parameter _limit_ specifies the limit for using \\(A^{\text{T}} B\\) multiplication, when _nn_ = 2. (It has no meaning for _nn_ = 1.)

If _size(A)/size(B)_ \\(\geq\\) _limit_ — standard multiplication is performed, _size(A)/size(B)_ \\(<\\) _limit_ — transposed multiplication is performed.

(_size(A,B)_ — number of elements in matrix A,B).

Recommended value for _limit_ is 2–3.

Using of transposed matrix (_nn_ = 2) multiplication may bring some computer time reduction only in special cases, however, it requires some additional work space. Default is optimal for absolute majority of cases.

This keyword is _optional_. (Default=1).

Note, that CCSD and CCT keywords are mutually exclusive.

## 4.2.5.4. How to run closed shell calculations using ROHF CC codes¶

First of all it should be noted here, that it is not advantageous to run closed shell calculations using ROHF CC codes, since in the present implementation it will require the same number of arithmetical operations and the core and disk space like corresponding open shell calculations.

Since ROHF CC codes are connected to the output of RASSCF code (through the JOBIPH file), it is necessary to run closed shell Hartree–Fock using the RASSCF program. This can be done by setting the number of active orbitals and electrons to zero (also by including only doubly occupied orbitals into the active space; this has no advantage but increases the computational effort). to guarantee the single reference character of the wave function.

The CC program will recognize the closed shell case automatically and will reorganize all integrals in a required form. For more information the reader is referred to the tutorials and examples manual.

Below is an input file for \\(\ce{HF+}\\) CCSD(T) calculation.
    
    
    &CCSDT
    Title
     HF(+) CCSD(T) input example
    CCT
    Triples
    3
    

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
      * 4.2.5. CCSDT
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

[previous](<casvb.html> "4.2.4. CASVB") | [next](<chcc.html> "4.2.6. CHCC") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/ccsdt.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

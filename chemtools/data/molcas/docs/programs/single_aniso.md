<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/single_aniso.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.53. SINGLE_ANISO

[previous](<seward.html> "4.2.52. SEWARD") | [next](<slapaf.html> "4.2.54. SLAPAF") | [index](<../../genindex.html> "General Index")

# 4.2.53. SINGLE_ANISO¶

  * Dependencies

  * Files

    * Input files

    * Restart files & options

    * Output files

  * Input

    * Optional general keywords to control the input

    * An input example




The SINGLE_ANISO program is a routine which allows the non-perturbative calculation of effective spin (pseudospin) Hamiltonians and static magnetic properties of mononuclear complexes and fragments completely _ab initio_ , including the spin-orbit interaction. As a starting point it uses the results of RASSI calculation for the ground and several excited spin-orbital multiplets. A short description of methodology and applications can be found in [[194](<../../references.html#id4> "L. F. Chibotaru, L. Ungur, A. Soncini. Angew. Chem. Int. Ed., 47 \(2008\) 4126-4129.")], [[195](<../../references.html#id82> "L. F. Chibotaru, L. Ungur, Christophe Aronica, H. Elmoll, G. Pillet, D. Luneau. J. Am. Chem. Soc., 130 \(2008\) 12445-12455.")]. The second version of the SINGLE_ANISO program is able to calculate the following quantities:

  * Parameters of pseudospin magnetic Hamiltonians (the methodology is described in [[196](<../../references.html#id176> "L. F. Chibotaru, L. Ungur. J. Chem. Phys., 137 \(2012\) 064112\(1–22\).")]):

    1. First rank (linear after pseudospin) Zeeman splitting tensor \\(g_{\alpha\beta}\\), its main values, including the sign of the product \\(g_{X} \cdot g_{Y} \cdot g_{Z}\\), and the main magnetic axes.

    2. Second rank (bilinear after pseudospin) zero-field splitting tensor \\(D_{\alpha\beta}\\), its main values and the anisotropy axes. The anisotropy axes are given in two coordinate systems: a) in the initial Cartesian coordinate system (\\(x, y, z\\)) and b) in the coordinate system of the main magnetic axes (\\(X_{\text{m}}, Y_{\text{m}}, Z_{\text{m}}\\)).

    3. Higher rank ZFS tensors (\\(D^4\\), \\(D^6\\), etc.) and Zeeman splitting tensors (\\(G^3\\), \\(G^5\\), etc.) for complexes with moderate and strong spin-orbit coupling.

    4. Angular moments along the main magnetic axes.

  * All (27) parameters of the _ab initio_ Crystal field acting on the ground atomic multiplet of lanthanides, and the decomposition of the CASSCF/RASSI wave functions into functions with definite projections of the total angular moment on the quantization axis.

  * Static magnetic properties:

    1. Van Vleck susceptibility tensor \\(\chi_{\alpha\beta}(T)\\).

    2. Powder magnetic susceptibility function \\(\chi(T)\\).

    3. Magnetization vector \\(\vec M (\vec H)\\) for specified directions of the applied magnetic field \\(\vec H\\).

    4. Powder magnetization \\(M_{\text{mol}}(H)\\).




The magnetic Hamiltonians are defined for a desired group of \\(N\\) electronic states obtained in RASSI calculation to which a pseudospin \\(\tilde{S}\\) (it reduces to a true spin \\(S\\) in the absence of spin-orbit coupling) is subscribed according to the relation \\(N=2\tilde{S}+1\\). For instance, the two wave functions of a Kramers doublet correspond to \\(\tilde{S}=1/2\\). The implementation is done for \\(\tilde{S}=1/2, 1, 3/2, \ldots ,15/2\\).

The calculation of magnetic properties takes into account the contribution of excited states (the ligand-field and charge transfer states of the complex or mononuclear fragment included in the RASSI calculation) via their thermal population and Zeeman admixture. The intermolecular exchange interaction between magnetic molecules in a crystal can be taken into account during the simulation of magnetic properties by a phenomenological parameter \\(z_J\\) specified by the user (see keyword MLTP).

## 4.2.53.1. Dependencies¶

The SINGLE_ANISO program take, by default, all needed _ab initio_ information from the RUNFILE: i.e. matrix elements of angular momentum, spin-orbit energy spectrum and mixing coefficients, number of mixed states and their multiplicity, etc. In order to find the necessary information in the RUNFILE, the keywords MEES and SPIN are mandatory for RASSI. The SEWARD keyword ANGM is also compulsory.

As an alternative, the SINGLE_ANISO program may read an ASCII datafile obtained in a previous successful run $project.aniso. For more information see the DATA below.

## 4.2.53.2. Files¶

### 4.2.53.2.1. Input files¶

RUNFILE
    

The file of communication between different modules in Molcas. Its presence is mandatory when the calculation is not a restart one from a data file.

### 4.2.53.2.2. Restart files & options¶

RUNFILE
    

The file of communication between different modules in Molcas. Normally it is already present in $WorkDir. The SINGLE_ANISO may be restarted as many times as necessary in the same working directory where the previous RASSI was succesfully executed. The RUNFILE contains then all necessary data.

ANISOINPUT
    

(Obsolescent. This option is being surpassed by the DATA keyword and the ANISO-DATA-FILE, see below) The program may be restarted from the ASCII text file $Project.old.aniso generated by default in a previous succesful run of the SINGLE_ANISO (the name of this file may be specified during restart execution, see REST keyword below). This file contains all necessary data for SINGLE_ANISO as well as for the POLY_ANISO. In this case the initial $WorkDir may be empty (RUNFILE is not necessary).

ANISO-DATA-FILE
    

The SINGLE_ANISO may be restarted from the ASCII formatted file $Project.aniso produced in a previous successful run. This file can be saved at the end of any successful run of SINGLE_ANISO, and may be given as input under any name. The initial $WorkDir may be empty and RUNFILE is not necessary. This option is used with the keyword DATA.

Example:
    
    
    DATA
      ANISO_DATA_FILE
    

### 4.2.53.2.3. Output files¶

$Project.aniso
    

This formatted file is recommended for restart. It is produced by any successful run of the code by default. This file will be used by POLY_ANISO.

ANISO
    

(Obsolescent) This file is intended to be as input for the POLY_ANISO module in Molcas. It is an ASCII formated file. It is produced by any successful run of the code.

zeeman_energy_xxx.txt
    

Zeeman eignestates for the applied field in the direction # _xxx_ are placed in the corresponding text file. It may be used directly with external plotting programs like gnuplot to visualize the data.

XT_compare.txt
    

In case TEXP is employed (experimental XT(T) data points), the SINGLE_ANISO produces a data file used to directly plot the comparison between experimental and calculated magnetic susceptibility.

MH_compare_xxx.txt
    

In case HEXP is employed (experimental M(H,T) data points), the SINGLE_ANISO produces one or several data file(s) used to directly plot the comparison(s) between experimental and calculated molar magnetization at each temperature.

## 4.2.53.3. Input¶

Normally SINGLE_ANISO runs without specifying any of the following keywords. The only unknown variable for SINGLE_ANISO is the dimension (multiplicity) of the pseudospin. By default one multiplet is selected, which has the dimension equal to the multiplicity of the ground term. For example, in cases where spin-orbit coupling is weak, the multiplicity of the effective spin Hamiltonian is usually the same as the multiplicity of the lowest term, while in the cases with strong anisotropy (lanthanide or actinide complexes, \\(\ce{Co^{2+}}\\) complexes, etc…) the lowest energy levels of the complexes form a group of states which can differ quite strong from the spin multiplicity of the lowest term. In these cases the user should specify the multiplicity corresponding to a chosen value of pseudospin \\((2\tilde{S}+1)\\). For instance, in \\(\ce{Dy^{3+}}\\) the spin of the ground state term is \\(S=5/2\\), but in many situations only the ground Kramers doublet is considered; then the user should set the multiplicity of the pseudospin equal to 2 (see MLTP keyword). The calculation of the parameters of the crystal field corresponding to the ground atomic multiplet for lanthanides should be requested by the CRYS keyword.
    
    
    &SINGLE_ANISO
    

Argument(s) to a keyword are always supplied on the next line of the input file.

### 4.2.53.3.1. Optional general keywords to control the input¶

TITLe
    

One line following this one is regarded as title.

TYPE
    

This keyword is obsolete.

MLTP
    

The number of molecular multiplets (i.e. groups of spin-orbital eigenstates) for which \\(g\\), \\(D\\) and higher magnetic tensors will be calculated (default MLTP=1). The program reads two lines: the first is the number of multiplets (\\(n_{\text{mult}}\\)) and the second the array of \\(n_{\text{mult}}\\) numbers specifying the dimension of each multiplet. By default, the code will first analyze the energy spectra by itself and will compute the \\(g\\) and \\(D\\) tensors for ten low-lying groups of states. By using this keyword the user overwrites the default.

Example:
    
    
    MLTP
    4
    2 4 4 2
    

SINGLE_ANISO will compute the \\(g\\) tensor for four groups of states: the second and third groups have the effective spin \\(\tilde{S}=\ket{3/2}\\) each, while the first and last groups of states being doublets.

DATA
    

This keyword will read on the next line the filename of an ANISO data file. The ANISO data file is produced by default by any successful run of SINGLE_ANISO under the name $Project.aniso

Example:
    
    
    DATA
       project_aniso_saved_before.aniso
    

SINGLE_ANISO will fetch the file name project_aniso_saved_before.aniso from the input file and will attempt to open such file from $WorkDir. The file contains all required data for a successful run of SINGLE_ANISO.

TINT
    

Specifies the temperature points for the evaluation of the magnetic susceptibility. The program will read three numbers: \\(T_{\text{min}}\\), \\(T_{\text{max}}\\), \\(n_T\\).

\\(T_{\text{min}}\\) — the minimal temperature (Default 0.0 K)

\\(T_{\text{max}}\\) — the maximal temperature (Default 300.0 K)

\\(n_T\\) — number of temperature points (Default 101)

Example:
    
    
    TINT
    0.0  330.0  331
    

SINGLE_ANISO will compute temperature dependence of the magnetic susceptibility in 331 points evenly distributed in temperature interval: 0.0 K – 330.0 K.

HINT
    

Specifies the field points for the evaluation of the magnetization in a certain direction. The program will read four numbers: \\(H_{\text{min}}\\), \\(H_{\text{max}}\\), \\(n_H\\).

\\(H_{\text{min}}\\) — the minimal field (Default 0.0 T)

\\(H_{\text{max}}\\) — the maximal filed (Default 10.0 T)

\\(n_H\\) — number of field points (Default 101)

Example:
    
    
    HINT
    0.0  20.0  201
    

SINGLE_ANISO will compute the molar magnetization in 201 points evenly distributed in field interval: 0.0 T – 20.0 T.

TMAG
    

Specifies the temperature(s) at which the field-dependent magnetization is calculated. The program will read the number of temperature points (\\(N_{\text{temp}}\\)) and then an array of real numbers specifying the temperatures (in kelvin) at which magnetization is to be computed. Default is to compute magnetization at one temperature point (2.0 K). Example:
    
    
    TMAG
    5   1.8  2.0  3.4  4.0  5.0
    

SINGLE_ANISO will compute the molar magnetization at 5 temperature points (1.8 K, 2.0 K, 3.4 K, 4.0 K, and 5.0 K).

ENCU
    

This flag is used to define the cut-off energy for the lowest states for which Zeeman interaction is taken into account exactly. The contribution to the magnetization arising from states that are higher in energy than \\(E\\) (see below) is done by second-order perturbation theory. The program will read two integer numbers: \\(N_K\\) and \\(M_G\\). Default values are: \\(N_K\\)=100, \\(M_G\\)=100.

\\[E=N_K \cdot k_{\text{Boltz}} \cdot \text{TMAG} + M_G \cdot \mu_{\text{Bohr}} \cdot H_{\text{max}}\\]

The field-dependent magnetization is calculated at the temperature value TMAG. Example:
    
    
    ENCU
    250  150
    

If \\(H_{\text{max}}\\) = 10 T and TMAG = 1.8 K, then the cut-off energy is:

\\[E=100 \cdot 250 \cdot k_{\text{Boltz}} \cdot 1.8\,\text{K} + 150 \cdot \mu_{\text{Bohr}} \cdot 10\,\text{T} = 1013.06258\,\text{cm}^{-1}\\]

This means that the magnetization coming from all spin-orbit states with energy lower than \\(E=1013.06258\,\text{cm}^{-1}\\) will be computed exactly. The contribution from the spin-orbit states with higher energy is accounted by second-order perturbation.

NCUT
    

This flag is used to define the cut-off energy for the lowest states for which Zeeman interaction is taken into account exactly. The contribution to the magnetization arising from states that are higher in energy than lowest \\(N_{\text{CUT}}\\) states, is done by second-order perturbation theory. The program will read one integer number. In case the number is larger than the total number of spin-orbit states(\\(N_{\text{SS}}\\), then the \\(N_{\text{CUT}}\\) is set to \\(N_{\text{SS}}\\) (which means that the molar magnetization will be computed exactly, using full Zeeman diagonalization for all field points). The field-dependent magnetization is calculated at the temperature value(s) defined by TMAG.

Example:
    
    
    NCUT
    32
    

MVEC
    

Defines the number of directions for which the magnetization vector will be computed. On the first line below the keyword, the number of directions should be mentioned (\\(N_{\text{DIR}}\\). Default 0). The program will read \\(N_{\text{DIR}}\\) lines for Cartesian coordinates specifying the direction \\(i\\) of the applied magnetic field (\\(\theta_i\\) and \\(\phi_i\\)). These values may be arbitrary real numbers. The direction(s) of applied magnetic field are obtained by normalizing the length of each vector to one. Example:
    
    
    MVEC
    4
    0.0000  0.0000   0.1000
    1.5707  0.0000   2.5000
    1.5707  1.5707   1.0000
    0.4257  0.4187   0.0000
    

The above input requests computation of the magnetization vector in four directions of applied field. The actual directions on the unit sphere are:
    
    
    4
    0.00000  0.00000  1.00000
    0.53199  0.00000  0.84675
    0.53199  0.53199  0.33870
    0.17475  0.17188  0.00000
    

MAVE
    

This keyword specifies the grid density used for the computation of powder molar magnetization. The program uses Lebedev–Laikov distribution of points on the unit sphere. The program reads two integer numbers: \\(n_{\text{sym}}\\) and \\(n_{\text{grid}}\\). The \\(n_{\text{sym}}\\) defines which part of the sphere is used for averaging. It takes one of the three values: 1 (half-sphere), 2 (a quarter of a sphere) or 3 (an octant of the sphere). \\(n_{\text{grid}}\\) takes values from 1 (the smallest grid) till 32 (the largest grid, i.e. the densest). The default is to consider integration over a half-sphere (since \\(M(H)=-M(-H)\\)): \\(n_{\text{sym}}=1\\) and \\(n_{\text{grid}}=15\\) (i.e. 185 points distributed over half-sphere). In case of symmetric compounds, powder magnetization may be averaged over a smaller part of the sphere, reducing thus the number of points for the integration. The user is responsible to choose the appropriate integration scheme. Note that the program’s default is rather conservative.

TEXP
    

This keyword allows computation of the magnetic susceptibility \\(\chi T(T)\\) at experimental points. On the line below the keyword, the number of experimental points \\(N_T\\) is defined, and on the next \\(N_T\\) lines the program reads the experimental temperature (in kelvin) and the experimental magnetic susceptibility (in \\(\text{cm}^3\,\text{K}\,\text{mol}^{-1}\\)). TEXP and TINT keywords are mutually exclusive. The magnetic susceptibility routine will also print the total average standard deviation from the experiment.

HEXP
    

This keyword allows computation of the molar magnetization \\(M_{\text{mol}} (H)\\) at experimental points. On the line below the keyword, the number of experimental points \\(N_H\\) is defined, and on the next \\(N_H\\) lines the program reads the experimental field strength (in tesla) and the experimental magnetization (in \\(\mu_{\text{Bohr}}\\)). HEXP and HINT are mutually exclusive. The magnetization routine will print the standard deviation from the experiment.

ZJPR
    

This keyword specifies the value (in \\(\text{cm}^{-1}\\)) of a phenomenological parameter of a mean molecular field acting on the spin of the complex (the average intermolecular exchange constant). It is used in the calculation of all magnetic properties (not for pseudo-spin Hamiltonians) (Default is 0.0)

XFIE
    

This keyword specifies the value (in \\(\text{T}\\)) of applied magnetic field for the computation of magnetic susceptibility by \\(\mathrm{d}M/\mathrm{d}H\\) and \\(M/H\\) formulas. A comparison with the usual formula (in the limit of zero applied field) is provided. (Default is 0.0)

PRLV
    

This keyword controls the print level.

2 — normal. (Default)

3 or larger (debug)

POLY
    

(Obsolescent. The formatted $Project.aniso is generated and saved by default) The keyword is obsolete. The SINGLE_ANISO creates by default one ASCII formated text file named ANISOINPUT and also a binary file named $Project.Aniso. Both may be used to restart (or re-run again) the SINGLE_ANISO calculation.

CRYS
    

This keyword will enables the computation of the parameters of the crystal-field acting on the ground atomic multiplet of a lanthanide from the _ab initio_ calculation performed. The implemented methodology is described [[197](<../../references.html#id323> "L. Ungur, L. F. Chibotaru. Chem. Eur. J., 23\[15\] \(2017\) 3708-3718.")] and [[196](<../../references.html#id176> "L. F. Chibotaru, L. Ungur. J. Chem. Phys., 137 \(2012\) 064112\(1–22\).")]. Two types of crystal field parametererization are implemented:

  1. Parameterisation of the ground \\(\ket{J,M_J}\\) group of spin-orbit states (e.g. parameterisation of the ground \\(J=15/2\\) of a \\(\ce{Dy^{3+}}\\) complex).

  2. Parameterisation of the ground \\(\ket{L,M_L}\\) group of spin-free states (e.g. parameterisation of the ground \\(^6H\\) multiplet of a \\(\ce{Dy^{3+}}\\)).




For each of the above cases, the parameters of the crystal field are given in terms of irreducible tensor operators defined in [[196](<../../references.html#id176> "L. F. Chibotaru, L. Ungur. J. Chem. Phys., 137 \(2012\) 064112\(1–22\).")], in terms of Extended Stevens Operators defined in [[198](<../../references.html#id324> "C. Rudowicz. J. Phys. C: Solid State Phys., 18\[7\] \(1985\) 1415."), [199](<../../references.html#id325> "C. Rudowicz, C. Y. Chung. J. Phys.: Condens. Matter, 16\[32\] \(2004\) 5825."), [200](<../../references.html#id326> "C. Rudowicz, M. Karbowiak. Coord. Chem. Rev., 287 \(2015\) 28.")] and also employed in the EasySpin function of MATLAB. On the next line the program will read the chemical symbol of the metal ion. The code understands the labels of: lanthanides, actinides and first-row transition metal ions. For transition metal ions, the oxidation state should be indicated as well. By default the program will not compute the parameters of the crystal-field.

QUAX
    

This keyword controls the quantization axis for the computation of the Crystal-Field parameters acting on the ground atomic multiplet of a lanthanide. On the next line, the program will read one of the three values: 1, 2 or 3.

1 — quantization axis is the main magnetic axis \\(Z_{\text{m}}\\) of the ground pseudospin multiplet, whose size is specified within the MLTP keyword. (Default)

2 — quantization axis is the main magnetic axis \\(Z_{\text{m}}\\) of the entire atomic multiplet \\(\ket{J,M_J}\\).

3 — the direction of the quantization axis is given by the user: on the next line the program will read three real numbers: the projections (\\(p_x\\), \\(p_y\\), \\(p_z\\)) of the specified direction on the initial Cartesian axes. Note that \\(p_x^2 + p_y^2 + p_z^2 = 1\\).

UBAR
    

This keyword allows estimation of the structuere of the blocking barier of a single-molecule magnet. The default is not to compute it. The method prints transition matrix elements of the magnetic moment according to the Figure 4.2.53.1.

[](<../../_images/ubar.png>)

Figure 4.2.53.1 Pictorial representation of the low-lying energy structure of a single-molecule magnet. A qualitative performance picture of the investigated single-molecular magnet is estimated by the strengths of the transition matrix elements of the magnetic moment connecting states with opposite magnetizations (\\(n{+} \to n{-}\\)). The height of the barrier is qualitatively estimated by the energy at which the matrix element (\\(n{+} \to n{-}\\)) is large enough to induce significant tunnelling splitting at usual magnetic fields (internal) present in the magnetic crystals (0.01 – 0.1 tesla). For the above example, the blocking barrier closes at the state (\\(8{+} \to 8{-}\\)).¶

All transition matrix elements of the magnetic moment are given as (\\((\vert\mu_{X}\vert+\vert\mu_{Y}\vert+\vert\mu_{Z}\vert)/3\\)). The data is given in Bohr magnetons (\\(\mu_{\text{Bohr}}\\)). The keyword is used with no arguments.

ABCC
    

This keyword will enable computation of magnetic and anisotropy axes in the crystallographic \\(abc\\) system. On the next line, the program will read six real values, namely \\(a\\), \\(b\\), \\(c\\), \\(\alpha\\), \\(\beta\\), and \\(\gamma\\), defining the crystal lattice. On the second line, the program will read the Cartesian coordinates of the magnetic center. The computed values in the output correspond to the crystallographic position of three “dummy atoms” located on the corresponding anisotropy axes, at the distance of 1 ångström from the metal site.
    
    
    ABCC
    20.17   19.83   18.76    90  120.32  90
    12.329  13.872  1.234
    

PLOT
    

This keyword will generate a few plots (png or eps format) via an interface to the linux program _gnuplot_. The interface generates a datafile, a gnuplot script and attempts execution of the script for generation of the image. The plots are generated only if the respective function is invoked. The magnetic susceptibility, molar magnetisation and blocking barrier (UBAR) plots are generated. The files are named: $Project.XT_no_field.dat, $Project.XT_no_field.plt, $Project.XT_no_field.png, $Project.XT_no_field.dat, $Project.XT_no_field.plt, $Project.XT_with_field_M_over_H.dat, $Project.XT_with_field_M_over_H.png, $Project.XT_with_field_dM_over_dH.plt, $Project.XT_with_field_dM_over_dH.dat, $Project.XT_with_field_M_over_H.png, $Project.MH.dat, $Project.MH.plt, $Project.MH.png, $Project.BARRIER_TME.dat, $Project.BARRIER_ENE.dat, $Project.BARRIER.plt and $Project.BARRIER.png. In case the version of the GNUPLOT is older than 5.0, then the eps images are generated.

### 4.2.53.3.2. An input example¶
    
    
    &SINGLE_ANISO
    MLTP
    3
    4 4 2
    ZJPR
    -0.2
    ENCU
    250 400
    HINT
    0.0  20.0  100
    TINT
    0.0  330.0  331
    MAVE
    1  12
    PLOT
    

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
      * [4.2.47. RASSCF](<rasscf.html>)
      * [4.2.48. RASSI](<rassi.html>)
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
      * [4.2.52. SEWARD](<seward.html>)
      * 4.2.53. SINGLE_ANISO
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<seward.html> "4.2.52. SEWARD") | [next](<slapaf.html> "4.2.54. SLAPAF") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/single_aniso.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

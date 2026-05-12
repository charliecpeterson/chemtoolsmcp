<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/poly_aniso.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.44. POLY_ANISO

[previous](<numerical_gradient.html> "4.2.43. NUMERICAL_GRADIENT") | [next](<qmstat.html> "4.2.45. QMSTAT") | [index](<../../genindex.html> "General Index")

# 4.2.44. POLY_ANISO¶

  * Files

    * Input files

    * Output files

  * Input

    * Mandatory keywords defining the calculation

    * Optional general keywords to control the input




The POLY_ANISO program is a routine which allows a semi-ab initio description of the (low-lying) electronic structure and magnetic properties of polynuclear compounds. It is based on the localized nature of the magnetic orbitals (i.e. the d or f orbitals containing unpaired electrons [[110](<../../references.html#id327> "P. W. Anderson. Phys. Rev., 115\[1\] \(1959\) 2-13."), [111](<../../references.html#id328> "P. W. Anderson. In Solid State Physics, volume 14, pages 99-214. Academic Press, 1963.")]). For many compounds of interest, the localized character of magnetic orbitals leads to very weak character of the interactions between magnetic centers. Due to this weakness of the interaction, the metals’ orbitals and corresponding localized ground and excited states may be optimized in the absence of the magnetic interaction at all. For this purpose, various fragmentation models may be applied. The most commonly used fragmentation model is exemplified in Figure 4.2.44.1.

[](<../../_images/fragment.png>)

Figure 4.2.44.1 Fragmentation model of a polynuclear compound. The upper scheme shows a schematic overview of a tetranuclear compound and the resulting four mononuclear fragments obtained by _diamagnetic atom substitution_ method. By this scheme, the neighboring magnetic centers, containing unpaired electrons are computationally replaced by their diamagnetic equivalents. As example, transition metal sites TM(II) are best replaced by either diamagnetic \\(\ce{Zn(II)}\\) or \\(\ce{Sc(III)}\\), in function which one is the closest. For lanthanides \\(\ce{Ln(III)}\\) the same principle is applicable, \\(\ce{La(III)}\\) or \\(\ce{Lu(III)}\\) are best suited to replace a given magnetic lanthanide. Individual mononuclear metal framgents are then investigated by common CASSCF/CASPT2/RASSI/SINGLE_ANISO computational method. A single file for each magnetic site, produced by the SINGLE_ANISO run, is needed by the POLY_ANISO code as input.¶

Magnetic interaction between metal sites is very important for accurate description of low-lying states and their properties. It can be considered as a sum of various interaction mechanisms: magnetic exchange, dipole-dipole interaction, antisymmetric exchange, etc. In the POLY_ANISO code we have implemented several mechanisms.

The description of the magnetic exchange interaction is done within the Lines model [[112](<../../references.html#id329> "M. E. Lines. J. Chem. Phys., 55\[6\] \(1971\) 2977-2984.")]. This model is exact in three cases:

  1. interaction between two isotropic spins (Heisenberg),

  2. interaction between one Ising spin (only \\(S_z\\) component) and one isotropic (i.e. usual) spin, and

  3. interaction between two Ising spins.




In all other cases of interaction between magnetic sites with intermediate anisotropy, the Lines model represents an approximation. However, it was succesfully applied for a wide variety of polynuclear compounds so far.

In addition to the magnetic exchange, magnetic dipole-dipole interaction can be accounted exactly, by using the information about each metal site already computed _ab initio_. In the case of strongly anisotropic lanthanide compounds, the dipole-dipole interaction is usualy the dominant one. Dipolar magnetic coupling is one kind of long-range interaction between magnetic moments. For example, a system containing two magnetic dipoles \\(\mu_1\\) and \\(\mu_2\\), separated by distance \\(\vec{r}\\) have a total energy:

\\[E_{\text{dip}} = \frac{\mu_{\text{Bohr}}^{2}}{r^3} [\vec{\mu}_1 \cdot \vec{\mu}_2 - 3(\vec{\mu}_1 \vec{n}_{12}) \cdot (\vec{\mu}_2 \vec{n}_{12})],\\]

where \\(\vec{\mu}_{1,2}\\) are the magnetic moments of sites 1 and 2, respectively; \\(r\\) is the distance between the two magnetic dipoles, \\(\vec{n}_{12}\\) is the directional vector connecting the two magnetic dipoles (of unit length). \\(\mu_{\text{Bohr}}^2\\) is the square of the Bohr magneton; with an approximative value of 0.43297 in \\(\text{cm}^{-1}/\text{T}\\). As inferred from the above Equation, the dipolar magnetic interaction depends on the distance and on the angle between the magnetic moments on magnetic centers. Therefore, the Cartesian coordinates of all non-equivalent magnetic centers must be provided in the input (see the keyword COOR).

## 4.2.44.1. Files¶

### 4.2.44.1.1. Input files¶

The program POLY_ANISO needs the following files:

aniso_XX.input
    

This is an ASCII text file generated by the Molcas/SINGLE_ANISO program. It should be provided for POLY_ANISO aniso_i.input (\\(i=1, 2, 3\\), etc.): one file for each magnetic center. In cases when the entire polynuclear cluster or molecule has exact point group symmetry, only aniso_i.input files for crystallographically non-equivalent centers should be given.

chitexp.input
    

set directly in the standard input (key TEXP)

magnexp.input
    

set directly in the standard input (key HEXP)

### 4.2.44.1.2. Output files¶

zeeman_energy_xxx.txt
    

A series of files named zeeman_energy_xxx.txt is produced in the $WorkDir only in case keyword ZEEM is employed (see below). Each file is an ASCII text formated and contains Zeeman spectra of the investigated compound for each value of the applied magnetic field.

chit_compare.txt
    

A text file contining the experimental and calculated magnetic susceptibility data.

magn_compare.txt
    

A text file contining the experimental and calculated powder magnetisation data.

Files chit_compare.txt and chit_compare.txt may be used in connection with a simple gnuplot script in order to plot the comparison between experimental and calculated data.

## 4.2.44.2. Input¶

This section describes the keywords used to control the standard input file. Only two keywords NNEQ, PAIR (and SYMM if the polynuclear cluster has symmetry) are mandatory for a minimal execution of the program, while the other keywords allow customization of the execution of the POLY_ANISO.

### 4.2.44.2.1. Mandatory keywords defining the calculation¶

_Keywords defining the polynuclear cluster_

NNEQ
    

This keyword defines several important parameters of the calculation. On the first line after the keyword the program reads 2 values: 1) the number of types of different magnetic centers (NON-EQ) of the cluster and 2) a letter `T` or `F` in the second position of the same line. The number of NON-EQ is the total number of magnetic centers of the cluster which cannot be related by point group symmetry. In the second position the answer to the question: _Have all NON-EQ centers been computed ab initio?_ is given: `T` for _True_ and `F` for _False_. On the third position, the answer to the question: _Are the rassi.h5 files to be read for input?_ is given. For the current status, the letter `F` is the only option. On the following line the program will read NON-EQ values specifying the number of equivalent centers of each type. On the following line the program will read NON-EQ integer numbers specifying the number of low-lying spin-orbit functions from each center forming the local exchange basis.

Some examples valid for situations where all sites have been computed _ab initio_ (case `T`, _True_):
    
    
    NNEQ
    2  T  F
    1  2
    2  2
    

| 
    
    
    NNEQ
    3  T  F
    2  1  1
    4  2  3
    

| 
    
    
    NNEQ
    6  T  F
    1  1  1  1  1  1
    2  4  3  5  2  2
      
  
---|---|---  
There are two kinds of magnetic centers in the cluster; both have been computed ab initio; the cluster consists of 3 magnetic centers: one center of the first kind and two centers of the second kind. From each center we take into the exchange coupling only the ground doublet. As a result the \\(N_{\text{exch}}=2^1 \times 2^2=8\\) aniso_1.input (for — type 1) and aniso_2.input (for — type 2) files must be present. | There are three kinds of magnetic centers in the cluster; all three have been computed ab initio; the cluster consists of four magnetic centers: two centers of the first kind, one center of the second kind and one center of the third kind. From each of the centers of the first kind we take into exchange coupling four spin-orbit states, two states from the second kind and three states from the third center. As a result the \\(N_{\text{exch}}=4^2 \times 2^1 \times 3^1=96\\). Three files aniso_i.input for each center (\\(i=1,2,3\\)) must be present. | There are 6 kinds of magnetic centers in the cluster; all six have been computed ab initio; the cluster consists of 6 magnetic centers: one center of each kind. From the center of the first kind we take into exchange coupling two spin-orbit states, four states from the second center, three states from the third center, five states from the fourth center and two states from the fifth and sixth centers. As a result the \\(N_{\text{exch}}=2^1 \times 4^1 \times 3^1 \times 5^1 \times 2^1 \times 2^1=480\\). Six files aniso_i.input for each center (\\(i=1,2,\ldots,6\\)) must be present.  
  
Only in cases when some centers have NOT been computed ab initio (i.e. for which no aniso_i.input file exists), the program will read an additional line consisting of NON-EQ letters (`A` or `B`) specifying the type of each of the NON-EQ centers: `A` — the center is computed ab initio and `B` — the center is considered isotropic. On the following `number-of-B-centers` line(s) the isotropic \\(g\\) factors of the center(s) defined as `B` are read. The spin of the `B` center(s) is defined: \\(S=(N-1)/2\\), where \\(N\\) is the corresponding number of states to be taken into the exchange coupling for this particular center.

Some examples valid for mixed situations: the system consists of centers computed _ab initio_ and isotropic centers (case `F`, _False_):
    
    
    NNEQ
    2  F  F
    1  2
    2  2
    A  B
    2.3
    

| 
    
    
    NNEQ
    3  F  F
    2  1  1
    4  2  3
    A  B  B
    2.1
    2.0
    

| 
    
    
    NNEQ
    6  T  F
    1  1  1  1  1  1
    2  4  3  5  2  2
    B  B  A  A  B  A
    2.12
    2.43
    2.00
      
  
---|---|---  
There are two kinds of magnetic centers in the cluster; the center of the first type has been computed _ab initio_ , while the centers of the second type are considered isotropic with \\(g=2.3\\); the cluster consists of three magnetic centers: one center of the first kind and two centers of the second kind. Only the ground doublet state from each center is considered for the exchange coupling. As a result the \\(N_{\text{exch}}=2^1 \times 2^2=8\\). File aniso_1.input (for — type 1) must be present. | There are three kinds of magnetic centers in the cluster; the first center type has been computed _ab initio_ , while the centers of the second and third types are considered isotropic with \\(g=2.1\\) (second type) and \\(g=2.0\\) (third type); the cluster consists of four magnetic centers: two centers of the first kind, one center of the second kind and one center of the third kind. From each of the centers of the first kind, four spin-orbit states are considered for the exchange coupling, two states from the second kind and three states from the center of the third kind. As a result the \\(N_{\text{exch}}=4^2 \times 2^1 \times 3^1=96\\). The file aniso_1.input must be present. | There are six kinds of magnetic centers in the cluster; only three centers have been computed _ab initio_ , while the other three centers are considered isotropic; the \\(g\\) factor of the first center is 2.12 (\\(S=1/2\\)); of the second center 2.43 (\\(S=3/2\\)); of the fifth center 2.00 (\\(S=1/2\\)); the entire cluster consists of six magnetic centers: one center of each kind. From the center of the first kind, two spin-orbit states are considered in the exchange coupling, four states from the second center, three states from the third center, five states from the fourth center and two states from the fifth and sixth centers. As a result the \\(N_{\text{exch}}=2^1 \times 4^1 \times 3^1 \times 5^1 \times 2^1 \times 2^1=480\\). Three files aniso_3.input and aniso_4.input and aniso_6.input must be present.  
  
There is no maximal value for NNEQ, although the calculation becomes quite heavy in case the number of exchange functions is large.

SYMM
    

Specifies rotation matrices to symmetry equivalent sites. This keyword is mandatory in the case more centers of a given type are present in the calculation. This keyword is mandatory when the calculated polynuclear compound has exact crystallographic point group symmetry. In other words, when the number of equivalent centers of any kind \\(i\\) is larger than 1, this keyword must be employed. Here the rotation matrices from the one center to all the other of the same type are declared. On the following line the program will read the number `1` followed on the next lines by as many \\(3\times3\\) rotation matrices as the total number of equivalent centers of type `1`. Then the rotation matrices of centers of type `2`, `3` and so on, follow in the same format. When the rotation matrices contain irrational numbers (e.g. \\(\sin{\frac{\pi}{6}}=\frac{\sqrt{3}}{2}\\)), then more digits than presented in the examples below are advised to be given: \\(\frac{\sqrt{3}}{2}=0.86602540378\\).

Examples:
    
    
    NNEQ
    2  F  F
    1  2
    2  2
    A  B
    2.3
    
    SYMM
    1
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    2
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    -1.0 0.0 0.0
    0.0 -1.0 0.0
    0.0 0.0 -1.0
    

| 
    
    
    NNEQ
    3  F  F
    2  1  1
    4  2  3
    A  B  B
    2.1
    2.0
    2.0
    
    SYMM
    1
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    0.0 -1.0 0.0
    1.0 0.0  0.0
    0.0 0.0  1.0
    2
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    3
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    

| 
    
    
    NNEQ
    6  F  F
    1  1  1  1  1  1
    2  4  3  5  2  2
    B  B  A  A  B  A
    2.12
    2.43
    2.00
      
  
---|---|---  
The cluster computed here is a trinuclear compound, with one center computed ab initio, while the other two centers, related to each other by inversion, are considered isotropic with \\(g_x=g_y=g_z=2.3\\). The rotation matrix for the first center is \\(I\\) (identity, unity) since the center is unique. For the centers of type 2, there are two matrices \\(3\times3\\) since we have two centers in the cluster. The rotation matrix of the first center of type 2 is Identity while the rotation matrix for the equivalent center of type 2 is the inversion matrix. | In this input a tetranuclear compound is defined, all centers are computed ab initio. There are two centers of type “1”, related one to each other by \\(C_2\\) symmetry around the Cartesian Z axis. Therefore the SYMM keyword is mandatory. There are two matrices for centers of type 1, and one matrix (identity) for the centers of type 2 and type 3. | In this case the computed system has no symmetry. Therefore, the SYMM keyword may be skipped.  
  
More examples are given in the _Tutorial_ section.

_Keywords defining the magnetic exchange interactions_

This section defines the keywords used to set up the interacting pairs of magnetic centers and the corresponding exchange interactions.

A few words about the numbering of the magnetic centers of the cluster in the POLY_ANISO. First all equivalent centers of the type 1 are numbered, then all equivalent centers of the type 2, etc. These labels of the magnetic centers are used further for the declaration of the magnetic coupling. The pseudo-code is:
    
    
    k=0
    Do i=1, number-of-non-equivalent-sites
      Do j=1, number-of-equivalent-sites-of-type(i)
         k=k+1
         site-number(i,j)=k
      End Do
    End Do
    

PAIR or LIN1
    

Specifies the Lines interaction(s) between metal pairs. One parameter per interacting pair is required.
    
    
    LIN1
       READ number-of-interacting-pairs
       Do i=1, number-of-interacting-pairs
          READ site-1, site-2,   J
       End Do
    

ALIN or LIN3
    

Specifies the anisotropic interactions between metal pairs. Three parameters per interacting pair are required.
    
    
    LIN3
       READ number-of-interacting-pairs
       Do i=1, number-of-interacting-pairs
          READ site-1, site-2,   Jxx, Jyy, Jzz
       End Do
    

\\(J_{\alpha\beta}\\), where \\(\alpha\\) and \\(\beta\\) are main values of the Cartesian components of the (\\(3\times3\\)) matrix defining the exchange interaction between site-1 and site-2.

LIN9
    

Specifies the full anisotropic interaction matrices between metal pairs. Nine parameters per interacting pair is required.
    
    
    LIN9
       READ number-of-interacting-pairs
       Do i=1, number-of-interacting-pairs
          READ site-1, site-2,   Jxx, Jxy, Jxz,   Jyx, Jyy, Jyz,  Jzx, Jzy, Jzz
       End Do
    

\\(J_{\alpha\beta}\\), where \\(\alpha\\) and \\(\beta\\) are main values of the Cartesian components of the (\\(3\times3\\)) matrix defining the exchange interaction between site-1 and site-2.

COOR
    

Specifies the symmetrized coordinates of the metal sites. This keyword enables computation of dipole-dipole magnetic interaction between metal sites defined in the keywords PAIR, ALIN, LIN1, LIN3 or LIN9.
    
    
    COOR
       Do i=1, number-of-non-equivalent-sites
          READ coordinates of center 1
          READ coordinates of center 2
          ...
       End Do
    

_Other keywords_

Normally POLY_ANISO runs without specifying any of the following keywords.

Argument(s) to a keyword are always supplied on the next line of the input file.

### 4.2.44.2.2. Optional general keywords to control the input¶

MLTP
    

The number of molecular multiplets (i.e. groups of spin-orbital eigenstates) for which \\(g\\), \\(D\\) and higher magnetic tensors will be calculated (default MLTP=1). The program reads two lines: the first is the number of multiplets (\\(N_{\text{MULT}}\\)) and the second the array of \\(N_{\text{MULT}}\\) numbers specifying the dimension (multiplicity) of each multiplet.

Example:
    
    
    MLTP
    10
    2 4 4 2 2   2 2 2 2 2
    

POLY_ANISO will compute the \\(g\\) and \\(D{-}\\) tensors for 10 groups of states. The groups 1 and 4–10 are doublets (\\(\tilde{S}=\ket{1/2}\\)), while the groups 2 and 3 are quadruplets, having the effective spin \\(\tilde{S}=\ket{3/2}\\). For the latter cases, the ZFS (\\(D{-}\\)) tensors will be computed.

TINT
    

Specifies the temperature points for the evaluation of the magnetic susceptibility. The program will read three numbers: \\(T_{\text{min}}\\), \\(T_{\text{max}}\\), \\(n_T\\).

\\(T_{\text{min}}\\) — the minimal temperature (Default 0.0 K)

\\(T_{\text{max}}\\) — the maximal temperature (Default 300.0 K)

\\(n_T\\) — number of temperature points (Default 101)

Example:
    
    
    TINT
    0.0  330.0  331
    

POLY_ANISO will compute temperature dependence of the magnetic susceptibility in 331 points evenly distributed in temperature interval: 0.0 K – 330.0 K.

HINT
    

Specifies the field points for the evaluation of the magnetization in a certain direction. The program will read four numbers: \\(H_{\text{min}}\\), \\(H_{\text{max}}\\) and \\(n_H\\)

\\(H_{\text{min}}\\) — the minimal field (Default 0.0 T)

\\(H_{\text{max}}\\) — the maximal filed (Default 10.0 T)

\\(n_H\\) — number of field points (Default 101)

Example:
    
    
    HINT
    0.0  20.0  201
    

POLY_ANISO will compute the molar magnetization in 201 points evenly distributed in field interval: 0.0 T – 20.0 T.

TMAG
    

Specifies the temperature(s) at which the field-dependent magnetization is calculated. Default is one temperature point, \\(T\\)=2.0 K. Example:
    
    
    TMAG
    6   1.8 2.0 2.4  2.8 3.2 4.5
    

ENCU
    

This flag is used to define the cut-off energy for the lowest states for which Zeeman interaction is taken into account exactly. The contribution to the magnetization coming from states that are higher in energy than \\(E\\) (see below) is done by second order perturbation theory. The program will read two integer numbers: \\(N_K\\) and \\(M_G\\). Default values are: \\(N_K\\)=100, \\(M_G\\)=100.

\\[E=N_K \cdot k_{\text{B}} \cdot T_{\text{MAG}} + M_G \cdot \mu_{\text{Bohr}} \cdot H_{\text{max}}\\]

The field-dependent magnetization is calculated at the (highest) temperature value defined in either TMAG or HEXP. Example:
    
    
    ENCU
    250  150
    

If \\(H_{\text{max}}\\) = 10 T and TMAG = 1.8 K, then the cut-off energy is:

\\[E=100 \cdot 250 \cdot k_{\text{B}} \cdot 1.8 + 150 \cdot \mu_{\text{Bohr}} \cdot 10 = 1013.06258\,\text{cm}^{-1}\\]

This means that the magnetization coming from all spin-orbit states with energy lower than \\(E=1013.06258\,\text{cm}^{-1}\\) will be computed exactly. ERAT, NCUT and ENCU are mutually exclusive.

ERAT
    

This flag is used to define the cut-off energy for the lowest states for which Zeeman interaction is taken into account exactly. The contribution to the molar magnetization coming from states that are higher in energy than \\(E\\) (see below) is done by second order perturbation theory. The program reads one real number in the domain (0.0–1.0). Default is 1.0 (all exchange states are included in the Zeeman interaction).

\\[E = \text{ERAT} \cdot \text{Maximal-spread-of-exchange-splitting}\\]

The field-dependent magnetization is calculated at all temperature points defined in either TMAG or HEXT. Example:
    
    
    ERAT
    0.75
    

ERAT, NCUT and ENCU are mutually exclusive.

NCUT
    

This flag is used to define the number of low-lying exchange states for which Zeeman interaction is taken into account exactly. The contribution to the magnetization coming from the remaining exchange states is done by second order perturbation theory. The program will read one integer number. The field-dependent magnetization is calculated at all temperature points defined in either TMAG or HEXT. Example:
    
    
    NCUT
    125
    

In case the defined number is larger than the total number of exchange states in the calculation (\\(N_{\text{exch}}\\)), then \\(n_{\text{Cut}}\\) is set to be equal to \\(N_{\text{exch}}\\). ERAT, NCUT and ENCU are mutually exclusive.

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
    

ZEEM
    

Defines the number of directions for which the Zeeman energy will be computed/saved/plotted. On the first line below the keyword, the number of directions should be mentioned (\\(N_{\text{DIR}}\\). Default 0). The program will read \\(N_{\text{DIR}}\\) lines for Cartesian coordinates specifying the direction \\(i\\) of the applied magnetic field (\\(\theta_i\\) and \\(\phi_i\\)). These values may be arbitrary real numbers. The direction(s) of applied magnetic field are obtained by normalizing the length of each vector to one. Example:
    
    
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
    

This keyword allows computation of the magnetic susceptibility \\(\chi T(T)\\) at experimental points. On the line below the keyword, the number of experimental points \\(N_T\\) is defined, and on the next \\(N_T\\) lines the program reads the experimental temperature (in K) and the experimental magnetic susceptibility (in \\(\text{cm}^3\text{K}\text{mol}^{-1}\\)). TEXP and TINT keywords are mutually exclusive. The magnetic susceptibility routine will also print the standard deviation from the experiment.
    
    
    TEXP
       READ  number-of-T-points
       Do i=1, number-of-T-points
          READ ( susceptibility(i, Temp), TEMP = 1, number-of-T-points )
       End Do
    

HEXP
    

This keyword allows computation of the molar magnetization \\(M_{\text{mol}} (H)\\) at experimental points. On the line below the keyword,the number of experimental points \\(N_H\\) is defined, and on the next \\(N_H\\) lines the program reads the experimental field intensity (tesla) and the experimental magnetization (in \\(\mu_{\text{Bohr}}\\)). HEXP and HINT are mutually exclusive. The magnetization routine will print the standard deviation from the experiment.
    
    
    HEXP
       READ  number-of-T-points-for-M,  all-T-points-for-M-in-K
       READ  number-of-field-points
       Do i=1, number-of-field-points
          READ ( Magn(i, iT), iT=1, number-of-T-points-for-M )
       End Do
    

ZJPR
    

This keyword specifies the value (in \\(\text{cm}^{-1}\\)) of a phenomenological parameter of a mean molecular field acting on the spin of the complex (the average intermolecular exchange constant). It is used in the calculation of all magnetic properties (not for spin Hamiltonians) (Default is 0.0)

ABCC
    

This keyword will enable computation of magnetic and anisotropy axes in the crystallographic \\(abc\\) system. On the next line, the program will read six real values, namely \\(a\\), \\(b\\), \\(c\\), \\(\alpha\\), \\(\beta\\), and \\(\gamma\\), defining the crystal lattice. On the second line, the program will read the Cartesian coordinates of the magnetic center. The computed values in the output correspond to the crystallographic position of three “dummy atoms” located on the corresponding anisotropy axes, at the distance of 1 ångström from the metal site.
    
    
    ABCC
    20.17   19.83   18.76    90  120.32  90
    12.329  13.872  1.234
    

XFIE
    

This keyword specifies the value (in \\(\text{T}\\)) of applied magnetic field for the computation of magnetic susceptibility by \\(\mathrm{d}M/\mathrm{d}H\\) and \\(M/H\\) formulas. A comparison with the usual formula (in the limit of zero applied field) is provided. (Default is 0.0)

PRLV
    

This keyword controls the print level.

2 — normal. (Default)

3 or larger (debug)

PLOT
    

This keyword will generate a few plots (png or eps format) via an interface to the linux program _gnuplot_. The interface generates a datafile, a gnuplot script and attempts execution of the script for generation of the image. The plots are generated only if the respective function is invoked. The magnetic susceptibility, molar magnetisation and blocking barrier (UBAR) plots are generated. The files are named: XT.dat, XT.plt, XT.png, MH.dat, MH.plt, MH.png, BARRIER_TME.dat, BARRIER_ENE.dat, BARRIER.plt and BARRIER.png.

OLDA
    

This keyword requests to use the old-formatted ANISOINPUT files produced by SINGLE_ANISO run. Please make use of the new DATAFILE $Project.aniso produced in any successful run of SINGLE_ANISO.

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
      * 4.2.44. POLY_ANISO
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

[previous](<numerical_gradient.html> "4.2.43. NUMERICAL_GRADIENT") | [next](<qmstat.html> "4.2.45. QMSTAT") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/poly_aniso.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 

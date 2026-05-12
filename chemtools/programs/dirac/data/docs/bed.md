orphan  

# Beyond the electric dipole approximation: the full light-matter interaction

## Introduction

write me ...

## Case study 1: The radon atom

## Case study 2: Titanium tetrachloride

As a molecular example we will consider titanium tetrachloride. We use
the structure employed in Lestrange_JChemPhys2015

<div class="literalinclude">

TiCl4.xyz

</div>

The full molecular symmetry is $`T_d`$, but DIRAC will work in the
$`D_2`$ subgroup. We focus on excitations from the Cl $`1s`$ to vacant
Ti $`3d`$ orbitals. The relation between irreps of $`T_d`$ and $`D_2`$
is

|  |  |  |  |
|----|----|----|----|
| $`T_d`$ | $`D_2`$ |  |  |
| $`A_1`$ | $`A`$ |  | $`x^2+y^2+z^2`$ |
| $`A_2`$ | $`A`$ |  |  |
| $`E`$ | $`A\oplus A`$ |  | $`(2z^2-x^2-y^2,x^2-y^2)`$ |
| $`T_1`$ | $`B_1\oplus B_2\oplus B_3`$ | $`(R_x,R_y,R_z)`$ |  |
| $`T_2`$ | $`B_1\oplus B_2\oplus B_3`$ | $`(x,y,z)`$ | $`(xy,xz,yz)`$ |

The table also shows that the components of the electric dipole operator
transforms as $`T_2`$.

We first perform a spin-orbit free DFT/PBE0 calculation of the molecule
using the input file

<div class="literalinclude">

TlCl4_PBE0.inp

</div>

Curiously the DFT calculation does not converge, but by first running HF
and restart we converge (HOMO-LUMO gap 0.219 $`E_h`$). The Cl $`1s`$
orbitals span $`A_1\oplus T_2`$. Looking at the Mulliken population
analysis:

    * Electronic eigenvalue no.  2: -102.11926181928       (Occupation : f = 1.0000)  sym= B2  
    ==========================================================================================

    Gross     Total   |    L B2 Cl s      A  Cl _small   B3 Cl _small   B1 Cl _small
    -----------------------------------------------------------------------------------
     alpha    0.0024  |      0.0000         0.0012         0.0000         0.0012
     beta     0.9976  |      0.9963         0.0000         0.0012         0.0000

    * Electronic eigenvalue no.  3: -102.11926181926       (Occupation : f = 1.0000)  sym= B3  
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L B3 Cl s      A  Cl _small   B2 Cl _small   B1 Cl _small
    -----------------------------------------------------------------------------------
     alpha    0.0024  |      0.0000         0.0012         0.0000         0.0012
     beta     0.9976  |      0.9963         0.0000         0.0012         0.0000

    * Electronic eigenvalue no.  4: -102.11926181917       (Occupation : f = 1.0000)  sym= B1  
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L B1 Cl s      A  Cl _small   B2 Cl _small   B3 Cl _small
    -----------------------------------------------------------------------------------
     alpha    0.9976  |      0.9963         0.0012         0.0000         0.0000
     beta     0.0024  |      0.0000         0.0000         0.0012         0.0012

    * Electronic eigenvalue no.  5: -102.11926179218       (Occupation : f = 1.0000)  sym= A   
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L A  Cl s      B2 Cl _small   B3 Cl _small   B1 Cl _small
    -----------------------------------------------------------------------------------
     alpha    0.9976  |      0.9963         0.0000         0.0000         0.0012
     beta     0.0024  |      0.0000         0.0012         0.0012         0.0000

we find from degeneracy and symmetry that orbitals 2,3 and 4 span
$`T_2`$, whereas orbital 5 span $`A_1`$. Likewise, looking at the
Mulliken population analysis for the lower virtuals:

    * Electronic eigenvalue no. 46: -0.1347912984154       (Occupation : f = 0.0000)  sym= A   
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L A  Ti dxx    L A  Ti dyy    L A  Ti dzz    L A  Ti g202   L A  Ti g022   L A  Cl px     L A  Cl py  
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha    0.9999  |      0.3412         0.4428         0.0066         0.0004         0.0003           0.0748         0.0971
     beta     0.0001  |      0.0000         0.0000         0.0000         0.0000         0.0000           0.0000         0.0000

    Gross  | L A  Cl pz     L A  Cl dxx    L A  Cl dxy    L A  Cl dxz    L A  Cl dyy    L A  Cl dyz    L A  Cl dzz    L A  Cl fxxx
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.0014         0.0074         0.0001         0.0092         0.0096         0.0071           0.0001        -0.0003
     beta  |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000           0.0000         0.0000

    Gross  | L A  Cl fxxy   L A  Cl fxxz   L A  Cl fxyy   L A  Cl fxzz   L A  Cl fyyy   L A  Cl fyyz   L A  Cl fyzz
    -----------------------------------------------------------------------------------------------------------------
     alpha |   0.0009         0.0003         0.0011        -0.0005        -0.0004         0.0007        -0.0003
     beta  |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000

    * Electronic eigenvalue no. 47: -0.1347912984122       (Occupation : f = 0.0000)  sym= A   
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L A  Ti dxx    L A  Ti dyy    L A  Ti dzz    L A  Ti g220   L A  Ti g022   L A  Cl px     L A  Cl py  
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha    0.9999  |      0.1859         0.0843         0.5205         0.0004         0.0002           0.0407         0.0185
     beta     0.0001  |      0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000

    Gross  | L A  Cl pz     L A  Cl dxx    L A  Cl dxy    L A  Cl dxz    L A  Cl dyy    L A  Cl dyz    L A  Cl dzz    L A  Cl fxxx
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.1141         0.0040         0.0109         0.0018         0.0018         0.0039         0.0113        -0.0002
     beta  |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000

    Gross  | L A  Cl fxxy   L A  Cl fxxz   L A  Cl fxyy   L A  Cl fxzz   L A  Cl fyzz   L A  Cl fzzz
    --------------------------------------------------------------------------------------------------
     alpha |  -0.0002         0.0004        -0.0004         0.0012         0.0011        -0.0004
     beta  |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000


    * Electronic eigenvalue no. 48: -0.1090420939155       (Occupation : f = 0.0000)  sym= B2  
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L B2 Ti py     L B2 Ti dxz    L B2 Ti fxxy   L B2 Ti fyyy   L B2 Ti fyzz   L B2 Ti g301   L B2 Ti g121
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha    0.0001  |      0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
     beta     0.9999  |      0.0565         0.6691         0.0023        -0.0014         0.0023         0.0002        -0.0017

    Gross  | L B2 Ti g103   L B2 Cl s      L B2 Cl px     L B2 Cl py     L B2 Cl pz     L B2 Cl dxx    L B2 Cl dxy    L B2 Cl dxz 
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
     beta  |   0.0002         0.0048         0.0107         0.2054         0.0107         0.0040         0.0091         0.0012

    Gross  | L B2 Cl dyy    L B2 Cl dyz    L B2 Cl dzz    L B2 Cl fxxx   L B2 Cl fxxy   L B2 Cl fxyz   L B2 Cl fyyy   L B2 Cl fyzz
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
     beta  |   0.0122         0.0091         0.0040         0.0004         0.0004         0.0004        -0.0007         0.0004

    Gross  | L B2 Cl fzzz
    -----------------------
     alpha |   0.0000
    beta  |   0.0004

    * Electronic eigenvalue no. 49: -0.1090420939055       (Occupation : f = 0.0000)  sym= B1  
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L B1 Ti pz     L B1 Ti dxy    L B1 Ti fxxz   L B1 Ti fyyz   L B1 Ti fzzz   L B1 Ti g310   L B1 Ti g130
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha    0.9999  |      0.0565         0.6691         0.0023         0.0023        -0.0014         0.0002         0.0002
     beta     0.0001  |      0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000

    Gross  | L B1 Ti g112   L B1 Cl s      L B1 Cl px     L B1 Cl py     L B1 Cl pz     L B1 Cl dxx    L B1 Cl dxy    L B1 Cl dxz 
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |  -0.0017         0.0048         0.0107         0.0107         0.2054         0.0040         0.0012         0.0091
     beta  |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000

    Gross  | L B1 Cl dyy    L B1 Cl dyz    L B1 Cl dzz    L B1 Cl fxxx   L B1 Cl fxxz   L B1 Cl fxyz   L B1 Cl fyyy   L B1 Cl fyyz
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.0040         0.0091         0.0122         0.0004         0.0004         0.0004         0.0004         0.0004
     beta  |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000

    Gross  | L B1 Cl fzzz
    -----------------------
     alpha |  -0.0007
     beta  |   0.0000

    * Electronic eigenvalue no. 50: -0.1090420938994       (Occupation : f = 0.0000)  sym= B3  
    ==========================================================================================

    * Gross populations greater than 0.00010

    Gross     Total   |    L B3 Ti px     L B3 Ti dyz    L B3 Ti fxxx   L B3 Ti fxyy   L B3 Ti fxzz   L B3 Ti g211   L B3 Ti g031
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha    0.0001  |      0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
     beta     0.9999  |      0.0565         0.6691        -0.0014         0.0023         0.0023        -0.0017         0.0002

    Gross  | L B3 Ti g013   L B3 Cl s      L B3 Cl px     L B3 Cl py     L B3 Cl pz     L B3 Cl dxx    L B3 Cl dxy    L B3 Cl dxz 
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
     beta  |   0.0002         0.0048         0.2054         0.0107         0.0107         0.0122         0.0091         0.0091

    Gross  | L B3 Cl dyy    L B3 Cl dyz    L B3 Cl dzz    L B3 Cl fxxx   L B3 Cl fxyy   L B3 Cl fxyz   L B3 Cl fxzz   L B3 Cl fyyy
    ---------------------------------------------------------------------------------------------  -----------------------------------
     alpha |   0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         0.0000
     beta  |   0.0040         0.0012         0.0040        -0.0007         0.0004         0.0004         0.0004         0.0004

    Gross  | L B3 Cl fzzz
    -----------------------
     alpha |   0.0000
     beta  |   0.0004

    we find that the orbitals 46 and 47 span :math:`E`, whereas orbitals 48, 49 and 50 span :math:`T_2`.

We are now ready to consider the simulation of ligand K-edge X-ray
spectroscopy. We consider excitations from 4 occupied orbitals to 5
virtual ones. Each excitation gives rise to a singlet and a triplet, so
there will be in all $`4\times 5 \times (1+3) = 80`$ excitations. The
singlets are totally symmetric ($`A_1`$), whereas the triplets span the
rotations ($`T_1`$). From a $`T_d`$ direct product table (see for
instance [here](http://www.webqc.org/symmetrypointgroup-td.html) ) we
find the symmetries of the singlet excitations to be

``` math
\begin{aligned}
\begin{array}{lcll}T_2&\rightarrow&E:\quad&T_1\oplus T_2\\T_2&\rightarrow&T_2:\quad&A_1\oplus E\oplus T_1\oplus T_2\\A_1&\rightarrow&E:\quad&E\\A_1&\rightarrow&T_2:\quad&T_2\end{array}
\end{aligned}
```

For triplet excitations we get

``` math
\begin{aligned}
\begin{array}{lcll}T_2&\rightarrow&E:\quad&A_1\oplus A_2\oplus 2E\oplus 2T_1\oplus 2T_2\\T_2&\rightarrow&T_2:\quad&A_1\oplus A_2\oplus 2E\oplus 4T_1\oplus 3T_2\\A_1&\rightarrow&E:\quad&T_1\oplus T_2\\A_1&\rightarrow&T_2:\quad&A_2\oplus E\oplus T_1\oplus T_2\end{array}
\end{aligned}
```

Translated to $`D_2`$ we find that we need 20 excitations per symmetry
and so we set up the following input

<div class="literalinclude">

TiCl4_exc.inp

</div>

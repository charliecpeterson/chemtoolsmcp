orphan  

<div class="index">

\*\*VISUAL

</div>

# \*\*VISUAL

This section describes the functionalities of the
<span class="title-ref">VISUAL</span> module, which calculates various
molecular densities from the 1-electron density matrices and atomic
orbitals. These densities (here called <span class="title-ref">grid
functions</span>) are calculated on numerical grids and can be
integrated or exported as scalar, vector, or tensor fields (e.g., for
visualization or further processing). The VISUAL module is not activated
by default.

The main setup for these calculations should be provided by the
<span class="title-ref">.GRIDS</span> and
<span class="title-ref">.GRIDFUNCTIONS</span> blocks, as described
below.

## GRIDS

<div class="index">

.GRIDS

</div>

### .GRIDS

Grid(s) specification block. Next line should contain the number of
grids to be used. Then, the following lines should provide information
about each grid (as a <span class="title-ref">key=value</span> list).

For example, to import the grid from the
<span class="title-ref">grid.h5</span> file, use:

    .GRIDS
    1
    id=1 input=import file_inp=grid.h5

### Available <span class="title-ref">key=value</span> options

- <span class="title-ref">id=int</span>

  > The ID of a grid. Required, used as a reference for
  > `VISUAL_.GRIDFUNCTIONS` specification.

- <span class="title-ref">npoints=\[Nx,Ny,nz\]</span>

  > Specify the number of grid points in
  > <span class="title-ref">x</span>, <span class="title-ref">y</span>,
  > and <span class="title-ref">z</span> directions. Either
  > <span class="title-ref">npoints</span> or
  > <span class="title-ref">spacing</span> is required for grid
  > generation.

- <span class="title-ref">spacing=\[Sx,Sy,Sz\]</span> or
  <span class="title-ref">spacing=S</span>

  > Specify the grid resolution, i.e., the distance between two
  > consecutive grid points. Use
  > <span class="title-ref">spacing=\[Sx,Sy,Sz\]</span> to specify the
  > distance between grid points in <span class="title-ref">x</span>,
  > <span class="title-ref">y</span>, and
  > <span class="title-ref">z</span> directions, or
  > <span class="title-ref">spacing=S</span> to specify the same
  > distance between grid points for all directions. Either
  > <span class="title-ref">npoints</span> or
  > <span class="title-ref">spacing</span> is required for grid
  > generation.

- <span class="title-ref">input=option</span>

  > Specify grid input. Available options:
  >
  > > - <span class="title-ref">input=create</span> : create grid; the
  > >   default is a 3D rectilinear grid; requires the specification of
  > >   <span class="title-ref">npoints</span> or
  > >   <span class="title-ref">spacing</span>
  > > - <span class="title-ref">input=import</span> : import grid from
  > >   an external file
  > > - <span class="title-ref">input=dft</span> : use the DFT numerical
  > >   grid; requires <span class="title-ref">numerical_grid</span>
  > >   file
  > > - <span class="title-ref">input=list</span> : use the list of grid
  > >   points; should be provided in the input (see `VISUAL_.LIST`)
  > > - <span class="title-ref">input=line</span> : create grid points
  > >   on line; should be provided in the input (see `VISUAL_.LINE`
  > >   keyword)
  > > - <span class="title-ref">input=2d</span> : create grid points on
  > >   a 2D plane; should be provided in the input (see `VISUAL_.2D`
  > >   keyword)
  > > - <span class="title-ref">input=2d_int</span> : create grid points
  > >   on a 2D plane for integration; should be provided in the input
  > >   (see `VISUAL_.2D_INT` keyword)
  > > - <span class="title-ref">input=3d</span> : create grid points on
  > >   a 3D surface; should be provided in the input (see `VISUAL_.3D`
  > >   keyword)
  > > - <span class="title-ref">input=3d_int</span> : create grid points
  > >   on a 3D surface for integration; should be provided in the input
  > >   (see `VISUAL_.3D_INT` keyword)
  > > - <span class="title-ref">input=radial</span> : create radial
  > >   grid; should be provided in the input (see `VISUAL_.RADIAL`
  > >   keyword)

- <span class="title-ref">margin=f</span>

  > Add space around the 3D rectilinear grid. Applies only to generated
  > 3D rectilinear grids. Default is
  > <span class="title-ref">f=0.0</span> (value in a.u.), which creates
  > grid point spanning the space between the minimum and maximum
  > coordinates of all atoms, i.e.,
  > $`x_{grid}=[min(x_{a}),max(x_{a})]_{a\in atoms}`$,
  > $`y_{grid}=[min(y_{a}),max(y_{a})]_{a\in atoms}`$,
  > $`z_{grid}=[min(z_{a}),max(z_{a})]_{a\in atoms}`$. Therefore, for
  > typical applications, choose some $`f>0`$ (e.g.,
  > <span class="title-ref">margin=4.0</span> is typically OK for the
  > electron density visualization).

- <span class="title-ref">export=option</span>

  > If the grid should be exported to a file, this allows to define the
  > file format for export. Supported formats:
  >
  > > - <span class="title-ref">export=hdf5</span> : export to hdf5 file
  > > - <span class="title-ref">export=txt</span> : export to text file,
  > >   no header, space as a separator (for backward compatibility with
  > >   old VISUAL code)
  > > - <span class="title-ref">export=csv</span> : export to text file
  > >   (csv): header line, coma as a separator

- <span class="title-ref">file_inp=string</span>

  > The name of a file, from which the grid should be read.

- <span class="title-ref">file_out=string</span>

  > The name of a file, to which the grid should be exported

Notes:

- *current limitations:*  
  - make sure there is one space between each
    <span class="title-ref">key=value</span> entry in the grid
    specification line.
  - the maximum line length in the input file is 200 characters

- for the specific examples, see the corresponding test in
  <span class="title-ref">DIRAC</span> test suite:
  <span class="title-ref">test/visual_grid_options</span>

### Additional keywords

The selected grid inputs (see
<span class="title-ref">input=option</span>) require additional
specification. This should be provided as a separate keyword block
**following** the <span class="title-ref">.GRID</span> keyword. Most of
these keywords are the same as used in the old VISUAL code.

<div class="index">

.LIST

</div>

### .LIST

Calculate various densities in few points. Scalar and vector densities
are written to the standard output file. Example (3 points; coordinates
in bohr):

    .LIST
     3
     1.0 0.0 0.0
     0.0 1.0 0.0
     0.0 0.0 1.0

In the <span class="title-ref">DIRAC</span> input file, this should be
provided as:

    .GRIDS
    id=1 input=list
    .LIST
     3
     1.0 0.0 0.0
     0.0 1.0 0.0
     0.0 0.0 1.0

<div class="index">

.LINE

</div>

### .LINE

Calculate various densities along a line. Example (line connecting two
points; 200 steps; coordinates in bohr):

    .LINE
     0.0 0.0 0.0
     0.0 0.0 5.0
     200

<div class="index">

.RADIAL

</div>

### .RADIAL

Compute radial distributions

``` math
f(r) = \int_{0}^{2\pi}\int_{0}^{\pi}f(\mathbf{r})r^2\sin\theta d\theta d\phi
```

by performing Lebedev angular integration over a specified number of
even-spaced radial shells out to some specified distance from a
specified initial point. Example (coordinates and distance in bohr):

    .RADIAL
    0.0 0.0 0.0
    10.0
    200

The first line after the keyword specifies the initial point, here
chosen to be the origin. The second and third line is the distance and
step size, respectively.

<div class="index">

.2D

</div>

### .2D

Calculate various densities in a plane. The plane is specified using 3
points that have to form a right angle. Example (coordinates in bohr):

    .2D
     0.0  0.0  0.0     !origin
     0.0  0.0 10.0     !"right"
     200               !nr of points origin-"right"
     0.0 10.0  0.0     !"top"
     200               !nr of points origin-"top"

<div class="index">

.2D_INT

</div>

### .2D_INT

Integrate various densities in a plane using Gauss-Lobatto quadrature.
The plane is specified using 3 points that have to form a right angle.
Example (coordinates in bohr):

    .2D_INT
     0.0  0.0  0.0     !origin
     0.0  0.0 10.0     !"right"
     10                !nr of tiles to the "right"
     0.0 10.0  0.0     !"top"
     10                !nr of tiles to the "right"
     5                 !order of the Legendre polynomial for each tile

<div class="index">

.3D

</div>

### .3D

Calculate various densities in 3D and write to cube file format. Example
(coordinates in bohr):

    .3D
     40 40 40          ! 40 x 40 x 40 points

Note: this is the same as using
<span class="title-ref">npoints=\[40,40,40\]</span> in the grid
specification line.

<div class="index">

.3D_INT

</div>

### .3D_INT

Integrate various densities in a volume.

## Grid functions

<div class="index">

.GRIDFUNCTIONS

</div>

### .GRIDFUNCTIONS

Grid function(s) specification block. Next line should contain the
number of grid functions to be calculated. Then, the following lines
should provide information about each grid function (as a
<span class="title-ref">key=value</span> list).

For example, to calculate the values of the electron density (*ed*), its
laplacian (*ed_laplacian*), and the reduced density gradient (*rdg*) in
the grid points of the grid supplied with the index $`id=1`$ (specified
in the `VISUAL_.GRIDS` section of the input) and exporting them to the
respective files, use:

    .GRIDFUNCTIONS
    3
    name=ed id_grid=1 purpose=visualization export=cube file_out=ed.cube
    name=ed_laplacian id_grid=1 purpose=visualization export=hdf5 file_out=ed_laplacian.h5
    name=rdg id_grid=1 purpose=visualization export=csv file_out=rdg.csv

### Available grid functions

The table below summarizes the grid functions currently implemented in
<span class="title-ref">DIRAC</span> with their respective test
directories in <span class="title-ref">DIRAC</span> test suite.

| name | property | <span class="title-ref">DIRAC</span> tests directories |
|----|----|----|
| <span class="title-ref">ed</span> | electron density, \$rho(vec{r})\$ | <span class="title-ref">visual_3d_electronden\*</span> |
| <span class="title-ref">ed_gradient</span> | electron density gradient, \$nablarho(vec{r})\$ | <span class="title-ref">visual_3d_electronden\*</span> |
| <span class="title-ref">ed_hessian</span> | electron density Hessian, \$Hrho(vec{r})\$ | <span class="title-ref">visual_3d_electronden\*</span> |
| <span class="title-ref">ed_laplacian</span> | electron density Laplacian, \$nabla^2rho(vec{r})\$ | <span class="title-ref">visual_3d_electronden\*</span> |
| <span class="title-ref">ed_sign_lambda2</span> | \$sign(lambda_2)rho(vec{r})\$ | <span class="title-ref">visual_3d_electronden\*</span> |
| <span class="title-ref">rdg</span> | reduced density gradient | <span class="title-ref">visual_3d_electronden\*</span> |
| <span class="title-ref">elf</span> | electron localization function | <span class="title-ref">visual_3d_elf\*</span> |
| <span class="title-ref">kin</span> | kinetic energy density (1) | <span class="title-ref">visual_3d_kinden\*</span> |
| <span class="title-ref">kinls</span> | kinetic energy density (2) | <span class="title-ref">visual_3d_kinden\*</span> |
| <span class="title-ref">kinsl</span> | kinetic energy density (2) | <span class="title-ref">visual_3d_kinden\*</span> |
| <span class="title-ref">kinlap</span> | the Laplacian of the kinetic energy density | <span class="title-ref">visual_3d_kinden\*</span> |
| <span class="title-ref">kintau</span> | kinetic energy density (3) | <span class="title-ref">visual_3d_kinden\*</span> |
| <span class="title-ref">kinnr</span> | the non-relativistic kinetic energy density | <span class="title-ref">visual_3d_kinden\*</span> |
| <span class="title-ref">s</span> | spin density, \$rho_s(vec{r})\$ | <span class="title-ref">visual_3d_spinden\*</span> |
| <span class="title-ref">divs</span> | \$Delta cdotrho_s(vec{r})\$ | <span class="title-ref">visual_3d_spinden\*</span> |
| <span class="title-ref">rots</span> | \$Delta times rho_s(vec{r})\$ | <span class="title-ref">visual_3d_spinden\*</span> |
| <span class="title-ref">gamma5</span> | \$gamma^5\$ density | <span class="title-ref">visual_3d_gamma5\*</span> |
| <span class="title-ref">edip</span> | the electric dipole density | <span class="title-ref">visual_3d_rspE\*</span> |
| <span class="title-ref">bdip</span> | the magnetic dipole density | <span class="title-ref">visual_3d_rspB\*</span> |
| <span class="title-ref">ndip</span> | the nuclear magnetic dipole density | <span class="title-ref">visual_3d_rspB\*</span> |
| <span class="title-ref">j</span> | the probability current density, \$vec{j}(vec{r})\$ | <span class="title-ref">visual_3d_rspB\*</span> |
| <span class="title-ref">rotj</span> | \$nabla times vec{j}(vec{r})\$ | <span class="title-ref">visual_3d_rspB\*</span> |
| <span class="title-ref">divj</span> | \$nabla cdot vec{j}(vec{r})\$ | <span class="title-ref">visual_3d_rspB\*</span> |
| <span class="title-ref">gradj</span> | \$Delta vec{j}(vec{r})\$ | <span class="title-ref">visual_3d_rspB\*</span> |

DIRAC properties

Comments:

- the kinetic energy density (1) and its LS- and SL-components (2) are
  calculated as \$c psi^dagger(r) (alpha cdot p) psi(r)\$
- the kinetic energy density (3) calculated as \$tau = nabla_i phi_k
  nabla_i phi_k\$

### Available <span class="title-ref">key=value</span> options

Notes:

- *current limitations:*  
  - make sure there is one space between each
    <span class="title-ref">key=value</span> entry in the grid function
    specification line.
  - the maximum line length in the input file is 200 characters

#### General <span class="title-ref">key=value</span> options

- <span class="title-ref">name=string</span>

  > The name of the grid function to calculate; see the available
  > options in the table above.

- <span class="title-ref">id_grid=int</span>

  > The ID of a grid on which the grid function should be calculated
  > (see the grid ID specification in `VISUAL_.GRIDS`.

- <span class="title-ref">input=option</span>

  > Specify grid function input. Available options:
  >
  > - <span class="title-ref">input=calculate</span> : calculate grid
  >   function; the default option
  > - <span class="title-ref">input=import</span> : import grid function
  >   from file

- <span class="title-ref">purpose=option</span>

  > Specify the type of calculations. Available options:
  >
  > - <span class="title-ref">purpose=visualization</span> : calculate
  >   grid function in grid points; possible export as
  >   scalar/vector/tensor field for a subsequent visualization; the
  >   default option
  > - <span class="title-ref">purpose=integration</span> : integrate
  >   grid function; for the specific examples, see the corresponding
  >   test in <span class="title-ref">DIRAC</span> test suite:
  >   <span class="title-ref">test/visual_integration</span>.

- <span class="title-ref">outpri=option</span>

  > Specify whether the grid function values should also be printed out
  > to <span class="title-ref">DIRAC</span> output. Available options:
  > <span class="title-ref">outpri=no</span> (default if
  > <span class="title-ref">purpose=visualization</span>),
  > <span class="title-ref">outpri=yes</span> (default if
  > <span class="title-ref">purpose=integration</span>).

- <span class="title-ref">export=option</span>

  > If the grid function should be exported to a file, this allows to
  > define the file format for export. Supported formats:
  >
  > > - <span class="title-ref">export=hdf5</span> : export to hdf5 file
  > > - <span class="title-ref">export=txt</span> : export to text file,
  > >   no header, space as a separator (for backward compatibility with
  > >   old VISUAL code)
  > > - <span class="title-ref">export=cube</span> : export to gaussian
  > >   cube file; available only for 3D rectilinear grids and scalar
  > >   fields (for backward compatibility with old VISUAL code)
  > > - <span class="title-ref">export=csv</span> : export to text file
  > >   (csv): header line, coma as a separator

- <span class="title-ref">file_inp=string</span>

  > The name of a file, from which the grid should be read.

- <span class="title-ref">file_out=string</span>

  > The name of a file, to which the grid should be exported

#### <span class="title-ref">key=value</span> options allowing for pointwise modification of grid functions

- <span class="title-ref">dscale=f</span>

  > Scale densities *down* by a factor <span class="title-ref">f</span>.
  > Default: <span class="title-ref">dscale=1.0</span>

- <span class="title-ref">uscale=f</span>

  > Scale densities *up* by a factor <span class="title-ref">f</span>.
  > Default: <span class="title-ref">uscale=1.0</span>

- <span class="title-ref">carpow=\[x,y,z\]</span>

  > Scale densities by Cartesian product $`x^iy^jz^k`$. The
  > <span class="title-ref">x,y,z</span> values are three integers
  > specifying the exponents $`(i,j,k)`$. For example,
  > <span class="title-ref">carpow=\[1,0,0\]</span> is equivalent to
  > calculating the <span class="title-ref">x</span>-component of the
  > electric dipole moment density (specification
  > <span class="title-ref">name=edip</span>).

- <span class="title-ref">radpow=f</span>

  > Scale densities by a radial power $`r^{n}`$. The keyword is followed
  > by three integers specifying the exponent $`n`$. Example:
  > <span class="title-ref">radpow=1</span> allows to the calculation of
  > radial expectation values $`<r>`$ with respect to the origin.

Notes:

- for the specific examples, see the corresponding test in DIRAC test
  suite:
  <span class="title-ref">test/visual_grid_function_modifications</span>

### Additional keywords affecting grid functions

<div class="index">

.OCCUPATION

</div>

### .OCCUPATION

Specify occupation of orbitals. Example (neon atom):

    .OCCUPATION
     2
     1 1-2 1.0
     2 1-3 1.0

The first line after the keyword gives the number of subsequent lines to
read. In each line, the first number is the fermion ircop. In molecules
with inversion symmetry there are two fermion ircops: gerade (1) and
ungerade (2). Otherwise there is a single fermion ircop (1). The
specification of the fermion ircop is followed by the range of selected
orbitals and their occupation. If a single orbital is specified a single
number is given instead of the range.

Another example (water):

    .OCCUPATION
     1
     1 1-5 1.0

Another example (nitrogen atom):

    .OCCUPATION
     2
     1 1-2 1.0
     2 1-3 0.5

*Warning:* this keyword affects *all* grid functions specified in the
input.

<div class="index">

.CVALUE

</div>

### .CVALUE

Set the speed of light to be used in the
<span class="title-ref">VISUAL</span> module.

*Warning:* this keyword affects *all* grid functions specified in the
input.

### General setup

<div class="index">

.PRINTL

</div>

### .PRINTL

Control the print level:

    .PRINTL
    print_level

Predefined <span class="title-ref">print_level</span> values:

> - <span class="title-ref">print_level</span> = 0: ony the basic
>   information from the <span class="title-ref">VISUAL</span>
>   calculations is printed out; the default option
> - <span class="title-ref">print_level</span> =
>   <span class="title-ref">1</span> : more details on the methods used
>   are printed out
> - <span class="title-ref">print_level</span> =
>   <span class="title-ref">2</span> : additional print of matrices
>   (warning: large output!)
> - <span class="title-ref">print_level</span> \>
>   <span class="title-ref">2</span> : the development prints (warning:
>   large output!)

All information is printed out to <span class="title-ref">DIRAC</span>
output.

## Additional notes (untested functionalities; notes from the old VISUAL code)

> [!WARNING]
> Only the calculation of the density is tested for open shell
> configurations (and relies on a correct .OCCUPATION). All other
> properties are only tested for closed shell systems and should not be
> trusted for open shell systems without a thorough testing.

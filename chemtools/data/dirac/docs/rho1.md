orphan  

<div class="index">

\*RHO1

</div>

# \*RHO1

For each unique pair (A,B) of centers a formatted file is created,
containing four columns of numbers: The origin is placed at atom A, and
the first column gives the coordinate in Angstroms along the line
between centers A and B. The following columns contain the total, large
and small component density, respectively (in atomic units). The file is
named 'RH',NAMN(ICENTA),NAMN(ICENTB),IDEGB where ICENTB is the name of
symmetry-independent center B and IDEGB the numbering of the dependent
center. Underscores are inserted wherever there are blanks. The
generation of the density requires the coefficient file DFCOEF. Note
that one can in principle obtain the density along any line simply by
introducing ghost centers.

For uranyl one may use the command:

    pam --incmo --get ``RHU___O____1  RHU___O____2 ...

<div class="index">

.PRINT

</div>

## .PRINT

Print level. Default:

    .PRINT
     0

<div class="index">

.MESH

</div>

## .MESH

Step length for grid. Default:

    .MESH
     0.01

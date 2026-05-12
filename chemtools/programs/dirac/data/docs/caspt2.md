orphan  

<div class="index">

\*CASPT2

</div>

# \*CASPT2

Relativistic IVO/CASCI/CASPT2 and IVO/RASCI/RASPT2 module, Ref.
Masuda2025, Ref. Abe2006, Ref. Anderson1990, Ref. Huzinaga1970

- Initial implementation: Minori Abe
- Parallelisation, DIRAC integration, and RASCI/RASPT2 extension: Kohei
  Noda
- `**MOLTRA` interface and IVO functionality: Sumika Iwamuro
- Code optimisation: Kohei Noda and Yasuto Masuda
- RAS3 minimum electron option: Taichi Inoue

The program is based on the CASPT2D formalism reported in Anderson1990.

> [!NOTE]
> Currently, IVO is not supported when the AO coefficients in
> checkpoint.\[no\]h5 are complex or quaternion numbers, such as for
> certain C₂ or C₁ symmetry molecules.

In the input file of IVO, CASCI, and CASPT2 programs, orbitals must be
numbered in ascending order of orbital energy, and if multiple orbitals
share the same energy, they are numbered by increasing symmetry index.
Since DIRAC’s SCF output often lists orbitals grouped by symmetry rather
than energy, it is convenient to use the following input generator to
prepare your CASCI/CASPT2 input.

## **GUI program to support creating CASPT2 input**

You can use the CASPT2 input support GUI program
<span class="title-ref">dcaspt2_input_generator
\<https://github.com/RQC-HU/dcaspt2_input_generator\></span>\
if your DIRAC output was generated using an input that includes the
`ANALYZE_.PRIVEC` keyword in the `**ANALYZE` module.\
This program creates the necessary DIRAC input fragment for the
`*CASPT2` section.\
If you are not familiar with writing CASPT2 input files, we recommend
using this tool.\
For detailed instructions, see
<https://github.com/RQC-HU/dcaspt2_input_generator/wiki> .

## **Required keywords**

<div class="index">

.NINACT

</div>

### .NINACT

Specify the number of inactive spinors.

<div class="index">

.NACT

</div>

### .NACT

Specify the number of active (CAS or RAS) spinors.

<div class="index">

.NELEC

</div>

### .NELEC

Specify the number of electrons in the active space.

<div class="index">

.NSEC

</div>

### .NSEC

Specify the number of secondary spinors.

<div class="index">

.SUBPROGRAMS

</div>

### .SUBPROGRAMS

Specify the subprograms to be used in the CASPT2 calculation.\
Currently the following subprograms are available:\
- IVO\
- CASCI (perform RASCI or CASCI calculation based on your input)\
- CASPT2 (perform RASPT2 or CASPT2 calculation based on your input)\
\
For example, if you want to carry out an IVO/CASCI/CASPT2 calculation
sequentially, you must specify the sub-programs as follows:

    .SUBPROGRAMS
    IVO
    CASCI
    CASPT2

## **Required keywords for CASCI/CASPT2 and RASCI/RASPT2**

<div class="index">

.CASPT2_CIROOTS

</div>

### .CASPT2_CIROOTS

Specify the total symmetry index and the roots to be included in the
CASCI/CASPT2 calculation.\
For example, to compute the Ag symmetry of a C₂h molecule (symmetry
index 5) and request the 1st, 2nd, 4th, 5th, and 6th lowest-energy
roots,\
you need to set the keyword as follows:

    .CASPT2_CIROOTS
    5 1 2 4..6

The first number is the total-symmetry index; the remaining numbers
specify the roots to be calculated.\
Additional lines can be added to define roots for other symmetries.\
For example:

    .CASPT2_CIROOTS
    5 1 2 4..6
    6 1 2

## **Required keywords for IVO**

<div class="index">

.NOCCG

</div>

### .NOCCG

Specify the number of occupied gerade Kramers‐pair orbitals (count
pairs, not individual spinors).\
This keyword must be specified for molecules that possess an inversion
center.\
Conversely, it should not be used for molecules without inversion
symmetry.

<div class="index">

.NOCCU

</div>

### .NOCCU

Specify the number of occupied ungerade Kramers‐pair orbitals (count
pairs, not individual spinors).\
This keyword must be specified for molecules that possess an inversion
center.\
Conversely, it should not be used for molecules without inversion
symmetry.

<div class="index">

.NOCC

</div>

### .NOCC

Specify the number of occupied Kramers‐pair orbitals (count pairs, not
individual spinors).\
This keyword must be specified for molecules that lack an inversion
center;\
it should not be used for molecules with inversion symmetry.

## **Required keyword for RASCI/RASPT2**

<div class="index">

.RAS2

</div>

### .RAS2

Specify the RAS2 spinor indices.\
For example, to assign the spinors ranked 7th, 8th, 11th, 12th, 13th,
and 14th lowest in energy to the RAS2 space,\
set the keyword as follows:

    .RAS2
    7,8,11..14

By using this keyword, you can select spinors that are not contiguous in
energy for inclusion in the CAS of a CASCI/CASPT2 or RASCI/RASPT2
calculation.\
For example, to include the spinors ranked 7th, 8th, 11th, 12th, 13th,
and 14th lowest in energy,\
set the keyword as follows:

    .NINACT
    6
    .NACT
    6
    .RAS2
    7,8,11..14
    .SECONDARY
    6

In this case, spinors 1–6 are inactive;\
spinors 7, 8, 11, 12, 13, and 14 are active;\
and spinors 9, 10, 15-18 are secondary.

## **Optional keywords for RASCI/RASPT2**

<div class="index">

.RAS1

</div>

### .RAS1

Specify the RAS1 spinor indices on row 1 and the maximum number of holes
allowed in RAS1 on row 2.\
For example, to assign spinors 3, 4, 5, and 6 (counted in ascending
energy) to RAS1 and to allow up to two holes,\
set the keyword as follows:

    .RAS1
    3..6
    2

<div class="index">

.RAS3

</div>

### .RAS3

Specify the RAS3 spinor indices on row 1 and the maximum number of
electrons allowed in RAS3 on row 2.\
For example, to assign spinors 15, 16, 17, and 18 (counted in ascending
energy) to RAS3 and to allow up to two electrons,\
set the keyword as follows:

    .RAS3
    15..18
    2

## **Optional keywords for IVO**

<div class="index">

.NHOMO

</div>

### .NHOMO

Specify the number of HOMO-like spinors.\
By default (NHOMO = 0), this value is determined automatically by
counting all occupied spinors whose orbital energies lie within 0.1 a.u.
of the HOMO energy.\
If desired, you can override the default by specifying NHOMO explicitly
in the input file (e.g., 2, 4, or 6).

*Default:*

    .NHOMO
    0

<div class="index">

.NVCUTG

</div>

### .NVCUTG

Specify the number of gerade virtual Kramers-pair orbitals to be
excluded in the IVO scheme.\
This keyword is applicable only to molecules with an inversion center;
do not use it for molecules without inversion symmetry.

*Default:*

    .NVCUTG
    0

<div class="index">

.NVCUTU

</div>

### .NVCUTU

Specify the number of ungerade virtual Kramers-pair orbitals to be
excluded in the IVO scheme.\
This keyword is applicable only to molecules with an inversion center;
do not use it for molecules without inversion symmetry.

*Default:*

    .NVCUTU
    0

<div class="index">

.NVCUT

</div>

### .NVCUT

Specify the number of virtual Kramers-pair orbitals to be excluded in
the IVO scheme.\
This keyword should be used only for molecules without an inversion
center; do not use it for molecules with inversion symmetry.

*Default:*

    .NVCUT
    0

## **general optional keywords for this module**

<div class="index">

.ESHIFT

</div>

### .ESHIFT

Specify the energy shift for the CASPT2 calculation.

*Default:* :

    .ESHIFT
    0.0

<div class="index">

.MINHOLERAS1

</div>

### .MINHOLERAS1

Specify the minimum number of holes allowed in RAS1.\
This keyword is used only in RASCI/RASPT2 calculations.\
Note: Setting this parameter may be helpful for excited-state
calculations (including core excitations) performed with RASCI/RASPT2.

*Default:* :

    .MINHOLERAS1
    0

<div class="index">

.MINELECRAS3

</div>

### .MINELECRAS3

Specify the minimum number of electrons in RAS3.\
This keyword is used only in RASCI/RASPT2 calculations.

*Default:* :

    .MINELECRAS3
    0

<div class="index">

.SCHEME

</div>

### .SCHEME

Specify the MOLTRA scheme to be used for the CASPT2 calculation.\
In most cases, you can simply reuse the scheme defined under `**MOLTRA`
(See `MOLTRA_.SCHEME`).

*Default:*

    .SCHEME
    4

<div class="index">

.COUNTNDET

</div>

### .COUNTNDET

Count and print the number of determinants within each irreducible
representation as defined by the current CASPT2 input,\
skipping all other calculations in this module.\
(Default: false — no determinant counting is performed.)

<div class="index">

.DEBUGPRINT

</div>

### .DEBUGPRINT

Print debugging information.\
(Default: false — no debugging information is printed.)

<div class="index">

.RESTART

</div>

### .RESTART

Restart the CASPT2 calculation from an existing checkpoint file.\
(Default: false — the calculation starts from scratch.)\
Note: Restart capability is currently implemented only for the CASPT2 or
RASPT2 sub-programs.\
To create a valid restart file, run the helper script\
 \`gen_caspt2_restart
\<https://raw.githubusercontent.com/RQC-HU/dirac_caspt2/refs/heads/main/tools/gen_dcaspt2_restart\>\`\_
.\
The typical workflow is:

    $ gen_caspt2_restart <prev_output_file>
    $ pam --inp <input_file> --mol <molfile> --put "caspt2_restart_*"

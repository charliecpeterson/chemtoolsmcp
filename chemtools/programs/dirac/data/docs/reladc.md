orphan  

# RELADC, FANOADC and LANCZOS

This section lists the available keywords for RELADC, FanoADC and the
closely connected iterative LANCZOS diagonalizer in DIRAC.

If you want to perform RELADC/LANCZOS calculations for the one- and
two-particle propagator or a FanoADC calculation you invoke all of them
by setting the .RELADC keyword in the \*\*WAVE FUNCTION Section.

The RELADC/LANCZOS calculation is then individually controlled in the
\*\*RELADC and \*\*LANCZOS input sections.

The FanoADC calculation is a special case of a one-particle propagator
calculation. Therefore the keywords are part of the \*\*RELADC section.

<div class="index">

\*\*RELADC

</div>

## \*\*RELADC

<div class="index">

.DOSIPS

</div>

### .DOSIPS

Do Single Ionization Potentials. If you intend to do single ionization
spectra calculations then set this keyword. Closely related to SIP
calculations is the keyword

<div class="index">

.ADCLEVEL

</div>

### .ADCLEVEL

Here you determine the perturbational order for a SIP calculation.

    ADCLEVEL = 1  strict second order
    ADCLEVEL = 2  extended second order
    ADCLEVEL = 3  third order + constant diagrams

*Default:*

    ADCLEVEL = 3 (including constant diagrams)

*Input example:*

    .ADCLEVEL
    2

<div class="index">

.SIPREPS

</div>

### .SIPREPS

Integer array of length 32. Specifies the symmetries of the one-hole
final states (SIP) to be calculated. If you do not enter this array all
symmetries are checked for possible final states and those are
calculated in the respective perturbational order. You can also specify
symmetries individually if you are interested only in a few by listing
the number of requested symmetries followed by the individual irrep
numbers.

*Default:*

    SIPREPS(1:32) = 0  (no symmetries preselected, all are calculated)

*Input example:*

    .SIPREPS
    8
    1,3,5,7,17,19,21,23

<div class="index">

.READQKL

</div>

### .READQKL

Read back previously calculated constant diagrams in SIP runs. This is a
restart option to avoid the most time consuming step in SIP runs. The
constant diagrams for all symmetries are stored in the file QKLVAL.

<div class="index">

.NOCONS

</div>

### .NOCONS

Block calculation of constant diagrams in third-order SIP calculations.
ADC is still executed up to third order but in the hole/hole block the
(time consuming) constant diagrams are omitted. Skipping constant
diagrams reduces accuracy but increases speed considerably. Not
recommended for production runs.

<div class="index">

.VCONV

</div>

### .VCONV

Determines convergence of the inverse iteration in the calculation of
the constant diagrams. If these iterations take much time you can reduce
tightness by setting VCONV to 1.0E-05, 1.0E-04 asf. Check accuracy of
results when you activate this option.

*Default:*

    VCONV=1.0E-06

*Input example:*

    .VCONV
    1.0E-04

<div class="index">

.DOFULL

</div>

### .DOFULL

Lanczos is automatically invoked in a single and double ionization
calculation. If you want to invoke an additional full diagonalizer set
this keyword. Attention: ADC matrices can become very large. A full
diagonalization is therefore supported only for matrix dimensions up to
5000 x 5000 and is useful for test purposes only.

<div class="index">

.DODIPS

</div>

### .DODIPS

Do Double Ionization Potentials. If you intend to do double ionization
spectra calculations then set this keyword. The keyword .ADCLEVEL does
not apply to DIP runs because the perturbational order is set fixed to
'extended second order'.

*Default:*

    DODIPS = F

<div class="index">

.DIPREPS

</div>

### .DIPREPS

Integer array of length 32. Specifies the symmetries of the two-hole
final states (DIP) to be calculated. If you do not enter this array all
symmetries are checked for possible final states and those are
calculated in the respective perturbational order. You can also specify
symmetries individually if you are interested only in a few. Input is
analogous to SIPREPS.

*Default:*

    DIPREPS(1:32) = 0  (no symmetries preselected, all are calculated)

<div class="index">

.ADCTHR

</div>

### .ADCTHR

In DIP calculations only matrix elements whose amount is larger than
ADCTHR will be written to disk. This is due to the large matrices
occurring in DIP runs. Tests have shown that you can reduce matrix size
by a factor of two setting ADCTHR to 1.0E-05 with an accuracy loss in
the meV region. Perform accuracy tests for production runs since the
behavior is system-dependent.

*Default:*

    ADCTHR = 0.0 (all nonzero matrix elements are written to disk)

*Input example:*

    .ADCTHR
    1.0E-05

<div class="index">

.FANO

</div>

### .FANO

This keyword activates the FanoADC module. It requires the specification
of the initial and final states to be investigated. By running the
default calculation settings, specified ionization spectra are
calculated as well. This can be turned off by using the keyword
.FANOONLY.

*Default:*

    False

For a Fano calculation it is crucial to use a proper (and normally
large) basis set. It has been shown, that it is beneficial to add KBJ
exponents to the atoms themselves or to ghost atoms surrounding the
system in order to span a basis for the interaction region of the final
state and the outgoing electron. By this the user should take care not
to loose symmetry by specifying the positions of ghost atoms.

More information about the proper selection of basis sets can be found
in the PhD thesis of Elke Fasshauer (link).

<div class="index">

.FANOIN

</div>

### .FANOIN

Here the initially ionized state is specified by symmetry and relative
spinor number. The following example chooses the first spinor in the
first symmetry to be the initial state:

    .FANOIN
     1 # symmetry
     1 # relative spinor number

You can find a table of spinors, their symmetries and relative and
absolute spinor numbers at the beginning of every ADC calculation with
the *same* active space. For the case of the Auger process of a neon
atom, this looks like:

    Spinor   Abelian Rep.         Energy   Recalc. Energy
     O    1    1    1g      -32.8174679811  -32.8174680470
     O    2    2    1g       -1.9358495110   -1.9358495652
     O    1    3   -1g      -32.8174679811  -32.8174680470
     O    2    4   -1g       -1.9358495110   -1.9358495652
     O    1    5    1u       -0.8528295909   -0.8528295926
     O    2    6    1u       -0.8482685119   -0.8482685445
     O    1    7   -1u       -0.8528295909   -0.8528295926
     O    2    8   -1u       -0.8482685119   -0.8482685445
     O    1    9    3u       -0.8482685155   -0.8482685461
     O    1   10   -3u       -0.8482685155   -0.8482685461
     .
     .
     virtual orbitals

which means, that the first spinor of the first symmetry (1g) which
represents the Ne1s orbital is chosen as the initially ionized state.

<div class="index">

.FANOCHNL

</div>

### .FANOCHNL

Autoionization processes decay via several channels into different final
states. These are specified in this section. In the first line the
number of channels , let us call it N, needs to be given. In the second
line the number of two-hole configurations for each channel is
specified. If you chose 2 channels, you need to put two numbers here!

The following lines give details about the channels. For every channel a
maximum 4 letter shortcut has to be specified followed by the two-hole
configurations of describing the final state. They have to be pairs of
absolute spinor numbers.

*Take care: The number of 2-hole-configurations of a channel has to be
the same as you specified in the upper part. The 4 letter shortcut has
to start at the beginning of the line. Do not include a space.*

This part is at the moment no black box method and the user is required
to think carefully about the selection of these final states and not to
forget a possible channel. On the other hand, this gives a lot of
control to the user to specify exactly what he/she wants to calculate.

Again the Neon atom as example:

    .FANOCHNL
     3       # number of channels N
     1,12,15 # number of hole configurations of the different channels
    s2       # descriptor of 2s-2 final state
     2 4     # two-hole configuration of 2s-2 final state
    sp
     2 5
     2 6
     2 7
     2 8
     2 9
     2 10
     4 5
     4 6
     4 7
     4 8
     4 9
     4 10
    p2
     5 6
     5 7
     5 8
     5 9
     5 10
     6 7
     6 8
     6 9
     6 10
     7 8
     7 9
     7 10
     8 9
     8 10
     9 10

If you change the Hamiltonian you will have to change the channel
specification as well!

<div class="index">

.FANOONLY

</div>

### .FANOONLY

This keyword enforces to run only a FanoADC run and overwrite every
other calculation of the one-particle propagator.

*Default:*

    False

<div class="index">

\*\*LANCZOS

</div>

## \*\*LANCZOS

Once the ADC matrices are stored in packed form on disk they are
diagonalized by the iterative Lanczos algorithm. The spectral
information is written to the files SSPEC.#irrep (SIPs) and DSPEC.#irrep
(DIPs). Hereby the ionization potential, the pole strength and the error
estimate are written in a line terminated by the '@' for grep purposes.
Immediately after this line follows the (indented) configuration
information belonging to this final state. This is imaginable as a
one-hole or two-hole Slater-determinant forming this state in zeroth
order. Strictly speaking, the configuration coefficients refer to the
intermediate state irepresentation basis.

### .SIPITER, .DIPITER

Determines the number of Lanczos iterations in a SIP or DIP calculation
for all symmetries. There is no need to specify the number of iterations
per symmetry because the convergence behaviour is similar within a
specific ionization class. However, DIP calculations can require
substantially more iterations for a comparable accuracy. Due to the
iterative nature of the Lanczos diagonalizer the edge values converge
very fast and some may be reproduced if SIPITER or DIPITER are set to
high values. These reproduced eigenvalues are spurious and will be
projected out from the final result. If one observes very many spurious
solutions (mainly in the SIP case) it is recommended to reduce SIPITER
accordingly.

*Default:*

    SIPITER = 500, DIPITER = 500.

*Input example:*

    .SIPITER
    1000
    .DIPITER
    2500

### .SIPPRNT, .DIPPRNT

Real values. These two variables only control screen output of the
calculated eigenvalues and have no influence on the results in the
(SD)SPEC files. You can enter the threshold in eV up to which computed
IPs (SIPs, DIPs) will be printed on screen. Sometimes one is only
interested in a few lowest IPs and the screen output suffices.

*Default:*

    SIPPRNT = 50.0, DIPPRNT = 50.0.

*Input example:*

    .SIPPRNT
    20.0
    .DIPPRNT
    100.0

### .SIPEIGV, .DIPEIGV

For each chosen symmetry selected by the XXXREPS array a lower and upper
energy boundary value for the corresponding method are specified. Only
within this energy range the long eigenvectors are calculated. This is
more user-friendly since one can not anticipate the number of
eigenvectors to be expected in a certain energy range.

*Default:*

    0.0 0.0   (no eigenvectors calculated)

*Input example:*

    .SIPEIGV   #(same for .DIPEIGV)
    4 # number of lines of ranges to follow
    10.0 20.0
    20.0 30.0
    0.0 0.0
    10.0 15.0

<div class="index">

.DOINCORE

</div>

### .DOINCORE

This keyword activates an incore Lanczos diagonalization. If you run
jobs on machines with large core memory you can speed up the
diagonalization considerably by transferring large parts of the ADC
matrix to memory. This is especially noticeable in DIP jobs because the
matrices are much larger than in the single ionization case. Attention:
Be aware that the operating system allows you to allocate as much memory
as you want. If there is not enough physical memory the OS starts to
swap portions of the memory to disk. You have to avoid this situation
since your job will not terminate in time. Check carefully that the
memory you request is physically available!

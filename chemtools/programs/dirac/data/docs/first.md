orphan  

# ECP calculation

In DIRAC, the effective core potential (ECP) method is implemented and
various subsequent correlation methods are available within the
two-component or one-component effective Hamiltonian.

From the inclusion (exclusion) of spin-orbit potential parameters in the
input file, molecular spinors (orbitals) are obtained and this is the
starting point of several ground and excited state calculation with the
ECP method.

The first example is the SCF calculation of Hydrogen Iodide.

## SOREP calculation

The simplest SCF input file (`HF.inp <HF.inp>`) for ECP calculation is

<div class="literalinclude">

HF.inp

</div>

together with the molecular input (`HI_sorep.mol <HI_sorep.mol>`)

<div class="literalinclude">

HI_sorep.mol

</div>

This is the two-component spin-orbit relativistic effective core
potential (SOREP) calulation. (For the detail of input, see `ecp_input`)

In the calculation output, the two-component molecular spinors (see also
the `full output <HF_HI_sorep.out>`) are listed as :

    * Fermion symmetry E1
      * Closed shell, f = 1.0000
     -0.91849028398625  ( 2)    -0.53547268879208  ( 2)    -0.40006484189588  ( 2)    -0.37307836548798  ( 2)
      * Virtual eigenvalues, f = 0.0000
      0.03426904787073  ( 2)     0.05875502261229  ( 2)     0.08417741978544  ( 2)     0.08661765422042  ( 2)     0.08729608075284  ( 2)
      0.13645335183505  ( 2)     0.20146339739788  ( 2)     0.20209264550882  ( 2)     0.23363818798538  ( 2)     0.23500980856689  ( 2)
      0.26681617711053  ( 2)     0.33365537629186  ( 2)     0.34280937791647  ( 2)     0.34322959805083  ( 2)     0.45833702543722  ( 2)
      0.48328741534790  ( 2)     0.50649128877099  ( 2)     0.51736684713002  ( 2)     0.61447310303433  ( 2)     0.61538818610103  ( 2)
      0.61831033924315  ( 2)     0.61894855200797  ( 2)     0.63389639932323  ( 2)     0.63480881100894  ( 2)     0.63698525034339  ( 2)
      0.67886357619745  ( 2)     0.68325244000339  ( 2)     0.75638614817881  ( 2)     0.75745722311601  ( 2)     0.82161646959771  ( 2)
      0.92201376113948  ( 2)     0.98455089264787  ( 2)     0.98468688973278  ( 2)     1.01721143371678  ( 2)     1.01778243972275  ( 2)
      1.19559448686781  ( 2)     1.32033135104656  ( 2)     1.32175612476556  ( 2)     1.47246800149637  ( 2)     1.47312441361493  ( 2)
      1.48573327678144  ( 2)     1.48673373305796  ( 2)     1.52583503659726  ( 2)     1.66340941889701  ( 2)     1.66434543208265  ( 2)
      1.68305757276943  ( 2)     1.99128262601425  ( 2)     1.99405394098767  ( 2)     2.00507423191293  ( 2)     2.00643391158823  ( 2)
      2.23720017057984  ( 2)     2.96761200756551  ( 2)     3.86994566085372  ( 2)     3.87002312971009  ( 2)     4.03165192324828  ( 2)
      4.03197238122657  ( 2)     4.26133522183310  ( 2)     4.26149442194385  ( 2)     4.35237329819399  ( 2)     4.68664685801378  ( 2)
      6.91601218905285  ( 2)     7.39252180916936  ( 2)     7.61998005503501  ( 2)    21.10496588735050  ( 2)

In this output, the molecular spinors are generated with the inclusion
of spin-orbit coupling in the SCF step. The orbital degeneracies showing
up in AREP calculation are broken with SOREP (for comparison, see AREP
calculation results below).

Note: Compared to all-electron calculations, only small number of
occupied molecular spinors are listed in the above example. Here, only
four occupied molecular spinors are shown. This is because only eight
(upper) electrons are described - seven electrons on the iodine atom and
one electron on the hydrogen atom. Remaining forty-six electrons in
iodine are omitted and are substituted by relativistic effective core
potentials.

## AREP calculation

We set the one-component scalar relativistic effective core potential
(AREP) calculation by omitting the SO parameters in SOREP
(`HI_arep.mol <HI_arep.mol>`).

<div class="literalinclude">

HI_arep.mol

</div>

After the SCF step, the following molecular orbitals are generated (see
also the `full output <HF_HI_arep.out>`) :

    * Fermion symmetry E1
      * Closed shell, f = 1.0000
     -0.91858300709907  ( 2)    -0.53332424176610  ( 2)    -0.38664035068657  ( 4)
      * Virtual eigenvalues, f = 0.0000
      0.03415914639101  ( 2)     0.05878598572499  ( 2)     0.08511594399606  ( 2)     0.08655831374568  ( 4)     0.13616239596081  ( 2)
      0.20167198225135  ( 4)     0.23444552560595  ( 4)     0.26659777181880  ( 2)     0.33379422568499  ( 2)     0.34298998830327  ( 4)
      0.46036400731232  ( 2)     0.49614437446042  ( 4)     0.51225813066935  ( 2)     0.61491923213700  ( 4)     0.61871900761802  ( 4)
      0.63458123096312  ( 4)     0.63671447698655  ( 2)     0.68126113881102  ( 4)     0.75708271906160  ( 4)     0.82156075161775  ( 2)
      0.92185115859998  ( 2)     0.98434317399476  ( 4)     1.01698231678790  ( 4)     1.19528790938055  ( 2)     1.32084962430606  ( 4)
      1.47297468375170  ( 4)     1.48649707416567  ( 4)     1.52558291015577  ( 2)     1.66381720482403  ( 4)     1.68291895964979  ( 2)
      1.99293721990332  ( 4)     2.00580508554321  ( 4)     2.23700102551750  ( 2)     2.96678284298137  ( 2)     3.86917720594751  ( 4)
      4.03098540560673  ( 4)     4.26054689341582  ( 4)     4.35208715121292  ( 2)     4.68579647949884  ( 2)     7.21075466518969  ( 4)
      7.50571888682965  ( 2)    21.10478953048176  ( 2)

Compared to molecular spinors from the SOREP calculation above, the
spin-orbit coupling is neglected in AREP-based molecular orbitals. As a
result, some degeneracies in molecular orbitals appear in the
relativistic scalar (spin-free, AREP) only treatment. Detailed
comparison of molecular orbitals can be done using the `**ANALYZE`
functionality.

## Exercises

1.  Identify and compare few occupied and virtual orbitals (including
    HOMO and LUMO) obtained without (AREP) and with (SOREP) spin-orbital
    relativistic effects.
2.  Optimize (numerically, see `*OPTIMIZE`) the internuclear distance
    with (SOREP) and without (AREP) spin-orbital interaction. What is
    effect of spin-orbital coupling on the bond distance ? Compare
    against other published works (cite:\`Park2012\`). Help - the input
    file, `geopt_HF.inp <geopt_HF.inp>`.
3.  Compute the dipole moment, electric polarizability and first
    hyperpolarizability of the HI molecule with (SOREP) and without
    (AREP) spin-orbital relativistic effects. How do spin-orbital
    effects influence this property ? Compare with two-component (X2C)
    all electron calculations. Input files
    `HI.acv2z.mol <HI.acv2z.mol>`, `x2c.scf_prop.inp <x2c.scf_prop.inp>`
    and `x2c_sf.scf_prop.inp <x2c_sf.scf_prop.inp>`.
4.  At the X2c-level evaluate the effect of the picture change for the
    dipole moment, electric polarizability and the first
    hyperpolarizability (input file
    `x2c.scf_prop_nopctr.inp <x2c.scf_prop_nopctr.inp>`). What is the
    size of the picture change effect affecting the properties ? Apply
    these findings to the previous ECP property calculations.
5.  To evaluate the size of spin-orbital effects on properties, you
    might use (in addition to the X2C approach of point 3.) the
    four-component Dirac-Coulomb Hamiltonian.
